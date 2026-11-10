"""Tests of the Pearson comparator on its own."""

import numpy as np
import pandas as pd
import pytest

from pairwise_compare import PearsonsCorrelation


def random_groups(n0, n1, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n0, n_features))), pd.DataFrame(rng.normal(size=(n1, n_features)))


def test_matches_numpy_for_every_pair_of_rows():
    group0, group1 = random_groups(3, 2)
    comparator = PearsonsCorrelation()
    comparator(group0, group1)

    # one value for each row of group0 with each row of group1, one group0 row after another
    expected = [np.corrcoef(row0, row1)[0, 1] for _, row0 in group0.iterrows() for _, row1 in group1.iterrows()]
    assert comparator.comparisons["pearsons_correlation"] == pytest.approx(expected, abs=1e-12)


def test_perfect_correlations_with_integers_and_a_custom_column_name():
    comparator = PearsonsCorrelation(name="correlation")
    comparator(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[2, 4, 6], [3, 2, 1]]))

    assert list(comparator.comparisons) == ["correlation"]
    assert comparator.comparisons["correlation"] == pytest.approx([1.0, -1.0])


def test_row_without_variance_gives_nan_and_a_warning():
    group0 = pd.DataFrame([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    comparator = PearsonsCorrelation()

    with pytest.warns(UserWarning, match="same value in every feature"):
        comparator(group0, pd.DataFrame([[3.0, 2.0, 1.0]]))

    first, second = comparator.comparisons["pearsons_correlation"]
    assert np.isnan(first)
    assert second == pytest.approx(-1.0)


def test_missing_feature_values_give_nan_for_that_row_only():
    # numpy missing values, and the missing values of pandas
    group0 = pd.DataFrame({"a": pd.array([1, None, 1], dtype="Int64"), "b": [np.nan, 3.0, 2.0], "c": [3, 5, 4]})
    comparator = PearsonsCorrelation()
    comparator(group0, pd.DataFrame({"a": [1.0], "b": [5.0], "c": [2.0]}))

    first, second, third = comparator.comparisons["pearsons_correlation"]
    assert np.isnan(first) and np.isnan(second)
    assert third == pytest.approx(np.corrcoef([1.0, 2.0, 4.0], [1.0, 5.0, 2.0])[0, 1])


def test_group_values_are_repeated_for_every_comparison():
    comparator = PearsonsCorrelation()
    comparator(*random_groups(3, 2))
    comparator.save_groups(["well", "plate"], group0=("W1", "P1"), group1=("W2", "P1"))

    # 3 x 2 = 6 comparisons in the call
    assert comparator.comparisons["well__group0"] == ["W1"] * 6
    assert comparator.comparisons["well__group1"] == ["W2"] * 6
    assert comparator.comparisons["plate__group0"] == ["P1"] * 6
    assert len(comparator.comparisons["pearsons_correlation"]) == 6
