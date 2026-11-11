"""Tests of the comparators of the metrics on their own, with scipy as the oracle."""

import warnings
from typing import Callable, NamedTuple, Optional

import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance

from pairwise_compare import (
    CosineSimilarity,
    EuclideanDistance,
    ManhattanDistance,
    PearsonsCorrelation,
    SpearmanCorrelation,
)

# rows with ties, an exact copy (rows 1 and 0), rows with the same value everywhere (the second one with a mean that is not
# exact), a row of zeros, a missing value, and a perfectly correlated and anticorrelated row of integers
MESSY = np.array(
    [
        [1, 2, 2, 3],
        [1, 2, 2, 3],
        [4, 4, 4, 4],
        [0.1, 0.1, 0.1, 0.1],
        [0, 0, 0, 0],
        [3, 1, np.nan, 2],
        [5, 1, 4, 1],
        [2, 2, 1, 1],
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [4, 3, 2, 1],
    ],
    dtype=float,
)



class Metric(NamedTuple):
    """A comparator, the same metric from scipy for two rows, and the warning for a row that has no value for it."""

    comparator: type
    oracle: Callable
    name: str  # the default name of the column of the results
    warning: Optional[str]  # None when every row has a value


# the warning is for a row with the same value everywhere (correlations) or of zeros (cosine)
METRICS = [
    pytest.param(Metric(PearsonsCorrelation, lambda a, b: 1 - distance.correlation(a, b), "pearsons_correlation", "same value in every feature"), id="Pearson"),
    pytest.param(Metric(SpearmanCorrelation, lambda a, b: stats.spearmanr(a, b).statistic, "spearman_correlation", "same value in every feature"), id="Spearman"),
    pytest.param(Metric(CosineSimilarity, lambda a, b: 1 - distance.cosine(a, b), "cosine_similarity", "zero in every feature"), id="cosine"),
    pytest.param(Metric(EuclideanDistance, distance.euclidean, "euclidean_distance", None), id="Euclidean"),
    pytest.param(Metric(ManhattanDistance, distance.cityblock, "manhattan_distance", None), id="Manhattan"),
]


@pytest.fixture(params=METRICS)
def metric(request):
    """Each metric in turn."""
    return request.param


def scipy_values(oracle, values, rows0, rows1):
    """The values of scipy for the pairs of rows, missing (NaN) for a row with a missing value (scipy refuses those)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(
            [np.nan if np.isnan(values[a]).any() or np.isnan(values[b]).any() else oracle(values[a], values[b]) for a, b in zip(rows0, rows1)]
        )


def compare(comparator, values, rows0, rows1):
    """The values of a comparator for the pairs of rows of `values`."""
    return comparator.compare(comparator.prepare(np.asarray(values, dtype=float)), np.asarray(rows0), np.asarray(rows1))


@pytest.mark.parametrize("block_elements", [10_000_000, 16], ids=["one_block", "small_blocks"])
@pytest.mark.parametrize(
    "n_rows, pairs",
    [
        pytest.param(8, lambda n: np.triu_indices(n, 1), id="matrix_products"),
        pytest.param(300, lambda n: (np.arange(n // 2), np.arange(n // 2, n)), id="gathered_pairs"),
    ],
)
def test_matches_scipy_for_every_pair(monkeypatch, metric, block_elements, n_rows, pairs):
    """Matches scipy for every pair, with ties, for pairs picked one by one and for matrix products, in blocks or not."""
    monkeypatch.setattr("pairwise_compare.comparators.Comparator._BLOCK_ELEMENTS", block_elements)
    values = np.random.default_rng(0).integers(0, 5, size=(n_rows, 6)).astype(float)  # small integers, so there are ties
    rows0, rows1 = pairs(n_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values_found = compare(metric.comparator(), values, rows0, rows1)

    np.testing.assert_allclose(values_found, scipy_values(metric.oracle, values, rows0, rows1), atol=1e-12, equal_nan=True)


def test_rows_without_a_value_give_nan_and_one_warning_and_the_others_match_scipy(metric):
    """Same-valued, zero and missing rows, copies and ties: scipy's values, with NaN for the rows that have none, and one warning."""
    rows0, rows1 = np.triu_indices(len(MESSY), 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = compare(metric.comparator(), MESSY, rows0, rows1)

    np.testing.assert_allclose(values, scipy_values(metric.oracle, MESSY, rows0, rows1), atol=1e-12, equal_nan=True)
    assert [w.category for w in caught] == ([UserWarning] if metric.warning else [])
    assert metric.warning is None or metric.warning in str(caught[0].message)

    # no pairs, no values
    assert compare(metric.comparator(), MESSY[8:], np.empty(0, dtype=int), np.empty(0, dtype=int)).shape == (0,)


def test_default_and_custom_names_of_the_columns(metric):
    """The default name of the column of the results, and a custom one."""
    assert metric.comparator().name == metric.name
    assert metric.comparator(name="other").name == "other"


def test_the_metrics_measure_what_the_documentation_says():
    """Spearman follows any increasing relation, cosine does not center the features, and a distance is 0 for a copy."""
    increasing = [[1.0, 2.0, 3.0, 4.0], [1.0, 10.0, 100.0, 1000.0]]
    assert compare(SpearmanCorrelation(), increasing, [0], [1]) == pytest.approx([1.0])
    assert compare(PearsonsCorrelation(), increasing, [0], [1])[0] < 0.9

    # rows with a large common value point the same way, but have no correlation
    large = [[100.0, 101.0, 100.0, 101.0], [101.0, 100.0, 100.0, 101.0]]
    assert compare(CosineSimilarity(), large, [0], [1])[0] > 0.999
    assert compare(PearsonsCorrelation(), large, [0], [1])[0] == pytest.approx(0.0)

    assert compare(EuclideanDistance(), [[1.0, 2.0], [1.0, 2.0]], [0], [1]) == [0.0]
    assert compare(ManhattanDistance(), [[1.0, 2.0], [4.0, 6.0]], [0], [1]) == [7.0]
