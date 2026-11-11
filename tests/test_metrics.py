"""Tests of the comparators of the metrics on their own, with scipy as the oracle."""

import warnings

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

# rows with ties, an exact copy (a row 1 and 0), a row with the same value everywhere, a row of zeros, and a missing value
MESSY = np.array(
    [[1, 2, 2, 3], [1, 2, 2, 3], [4, 4, 4, 4], [0, 0, 0, 0], [3, 1, np.nan, 2], [5, 1, 4, 1], [2, 2, 1, 1]], dtype=float
)

# each metric: its comparator, the same metric from scipy for two rows, its default column name, and the warning it gives
# for a row that has no value for it (a row with the same value everywhere, or of zeros)
METRICS = [
    pytest.param(PearsonsCorrelation, lambda a, b: 1 - distance.correlation(a, b), "pearsons_correlation", "same value in every feature", id="Pearson"),
    pytest.param(SpearmanCorrelation, lambda a, b: stats.spearmanr(a, b).statistic, "spearman_correlation", "same value in every feature", id="Spearman"),
    pytest.param(CosineSimilarity, lambda a, b: 1 - distance.cosine(a, b), "cosine_similarity", "zero in every feature", id="cosine"),
    pytest.param(EuclideanDistance, distance.euclidean, "euclidean_distance", None, id="Euclidean"),
    pytest.param(ManhattanDistance, distance.cityblock, "manhattan_distance", None, id="Manhattan"),
]


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
@pytest.mark.parametrize("metric, oracle, name, warning", METRICS)
def test_matches_scipy_for_every_pair(monkeypatch, metric, oracle, name, warning, block_elements, n_rows, pairs):
    """Matches scipy for every pair, with ties, for pairs picked one by one and for matrix products, in blocks or not."""
    monkeypatch.setattr("pairwise_compare.comparators.Comparator._BLOCK_ELEMENTS", block_elements)
    values = np.random.default_rng(0).integers(0, 5, size=(n_rows, 6)).astype(float)  # small integers, so there are ties
    rows0, rows1 = pairs(n_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values_found = compare(metric(), values, rows0, rows1)

    np.testing.assert_allclose(values_found, scipy_values(oracle, values, rows0, rows1), atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("metric, oracle, name, warning", METRICS)
def test_rows_without_a_value_give_nan_and_one_warning_and_the_others_match_scipy(metric, oracle, name, warning):
    """Same-valued, zero and missing rows, and copies, ties: scipy's values, with NaN for the rows that have none, and one warning."""
    rows0, rows1 = np.triu_indices(len(MESSY), 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = compare(metric(), MESSY, rows0, rows1)

    np.testing.assert_allclose(values, scipy_values(oracle, MESSY, rows0, rows1), atol=1e-12, equal_nan=True)
    assert [w.category for w in caught] == ([UserWarning] if warning else [])
    assert warning is None or warning in str(caught[0].message)


@pytest.mark.parametrize("metric, oracle, name, warning", METRICS)
def test_default_and_custom_names_of_the_columns(metric, oracle, name, warning):
    """The default name of the column of the results, and a custom one."""
    assert metric().name == name
    assert metric(name="other").name == "other"


def test_pearson_gives_perfect_correlations_for_integers_and_ignores_the_rounding_of_the_mean():
    """Perfect correlations for integers, and a same-valued row whose mean is not exact (0.1) is detected."""
    correlations = compare(PearsonsCorrelation(), [[1, 2, 3], [2, 4, 6], [3, 2, 1]], [0, 0], [1, 2])
    assert correlations == pytest.approx([1.0, -1.0])

    with pytest.warns(UserWarning, match="same value in every feature") as caught:
        values = compare(PearsonsCorrelation(), [[0.1, 0.1, 0.1], [3.0, 2.0, 1.0]], [0], [1])

    assert np.isnan(values[0]) and len(caught) == 1


def test_missing_feature_values_give_nan_for_that_row_only():
    """Missing feature values give nan for that row only."""
    correlations = compare(PearsonsCorrelation(), [[1.0, np.nan, 3.0], [1.0, 2.0, 4.0], [1.0, 5.0, 2.0]], [0, 1], [2, 2])

    assert np.isnan(correlations[0])
    assert correlations[1] == pytest.approx(np.corrcoef([1.0, 2.0, 4.0], [1.0, 5.0, 2.0])[0, 1])


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

    # rows that have a value for every metric give no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for metric in (PearsonsCorrelation, SpearmanCorrelation, CosineSimilarity, EuclideanDistance, ManhattanDistance):
            assert compare(metric(), [[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]], [0], [1]).shape == (1,)
