"""Tests of the Pearson comparator on its own."""

import numpy as np
import pytest

from pairwise_compare import PearsonsCorrelation


def compare(values, rows0, rows1):
    """The correlations of the pairs of rows, with the default comparator."""
    comparator = PearsonsCorrelation()

    return comparator.compare(comparator.prepare(np.asarray(values, dtype=float)), np.array(rows0), np.array(rows1))


@pytest.mark.parametrize("block_elements", [10_000_000, 16], ids=["one block", "small blocks"])
@pytest.mark.parametrize(
    "n_rows, pairs",
    [
        pytest.param(8, lambda n: np.triu_indices(n, 1), id="every pair of 8 rows (matrix products)"),
        pytest.param(300, lambda n: (np.arange(n // 2), np.arange(n // 2, n)), id="150 pairs of 300 rows (gathered)"),
    ],
)
def test_matches_numpy_for_every_pair(monkeypatch, block_elements, n_rows, pairs):
    """Matches numpy for every pair, for pairs that are picked one by one and for matrix products, in blocks or not."""
    monkeypatch.setattr("pairwise_compare.comparators.PearsonsCorrelation._BLOCK_ELEMENTS", block_elements)
    values = np.random.default_rng(0).normal(size=(n_rows, 6))
    rows0, rows1 = pairs(n_rows)

    expected = [np.corrcoef(values[a], values[b])[0, 1] for a, b in zip(rows0, rows1)]
    assert compare(values, rows0, rows1) == pytest.approx(expected, abs=1e-12)


def test_perfect_correlations_with_integers_and_a_custom_column_name():
    """Perfect correlations with integers and a custom column name."""
    comparator = PearsonsCorrelation(name="correlation")
    prepared = comparator.prepare(np.array([[1, 2, 3], [2, 4, 6], [3, 2, 1]]))

    assert comparator.name == "correlation"
    assert comparator.compare(prepared, np.array([0, 0]), np.array([1, 2])) == pytest.approx([1.0, -1.0])


def test_rows_without_variance_give_nan_and_a_warning():
    """Rows without variance give NaN and one warning, also when their mean is not exact (rounding error)."""
    # the third row is the same value in every feature too, but its mean is not exactly 0.1
    with pytest.warns(UserWarning, match="same value in every feature") as caught:
        correlations = compare([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1], [3.0, 2.0, 1.0]], [0, 1, 2], [3, 3, 3])

    first, second, third = correlations
    assert np.isnan(first) and np.isnan(third)
    assert second == pytest.approx(-1.0)
    assert len(caught) == 1


def test_missing_feature_values_give_nan_for_that_row_only():
    """Missing feature values give nan for that row only."""
    correlations = compare([[1.0, np.nan, 3.0], [1.0, 2.0, 4.0], [1.0, 5.0, 2.0]], [0, 1], [2, 2])

    assert np.isnan(correlations[0])
    assert correlations[1] == pytest.approx(np.corrcoef([1.0, 2.0, 4.0], [1.0, 5.0, 2.0])[0, 1])
