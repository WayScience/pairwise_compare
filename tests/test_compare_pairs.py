"""Tests of which rows are compared, the values, and the columns of the results, on small and real datasets."""

import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from helpers import expected_pairs, numpy_correlations, result_pairs
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pairwise_compare import PearsonsCorrelation, compare_pairs
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare
from pairwise_compare.comparison_tools.PairwiseCompareManager import PairwiseCompareManager


def compare(df, features, **arguments):
    """Compare the wells of `df` with the defaults of these tests (different wells of the same plate)."""
    arguments = {"different": ["well"], "features": features, "same": ["plate"], **arguments}

    return compare_pairs(df, **arguments)


def test_real_datasets_give_the_pairs_and_the_correlations_of_the_slow_checks(real_dataset):
    """Real datasets (NF1 plate 4, a PCCMA plate, iris, wine) give the pairs of the oracle and the correlations of numpy."""
    df, features, comparisons = real_dataset

    for same, different in comparisons:
        result = compare_pairs(df, different=different, features=features, same=same)

        assert len(result) > 0
        assert len(result_pairs(result)) == len(result), "a pair was compared more than once"
        assert result_pairs(result) == expected_pairs(df, different, same)
        assert result["pearsons_correlation"].to_numpy() == pytest.approx(numpy_correlations(df, features, result), abs=1e-9)


def test_columns_of_the_results(small_data):
    """The columns of the results, and what they mean for the two rows of a pair."""
    df, features = small_data
    result = compare(df, features, different=["well", "line"])

    assert list(result.columns) == [
        "pearsons_correlation",
        "plate__group0",
        "plate__group1",
        "well__group0",
        "well__group1",
        "line__group0",
        "line__group1",
    ]
    # both rows of a pair have the same value in a `same` column and a different value in a `different` column
    assert (result["plate__group0"] == result["plate__group1"]).all()
    assert (result["well__group0"] != result["well__group1"]).all()
    assert (result["line__group0"] != result["line__group1"]).all()


def test_input_is_not_changed_and_the_order_of_the_rows_does_not_matter(small_data):
    """The dataframe is not changed, and shuffling its rows gives the same pairs."""
    df, features = small_data
    before = df.copy()
    first = compare(df, features, different=["well", "line"])
    second = compare(df.sample(frac=1, random_state=0), features, different=["well", "line"])

    pd.testing.assert_frame_equal(df, before)

    key = ["well__group0", "well__group1"]
    pd.testing.assert_frame_equal(
        first.sort_values(key).reset_index(drop=True), second.sort_values(key).reset_index(drop=True)
    )


def test_a_new_comparator_is_used_unless_one_is_given(small_data):
    """Each call starts with a new comparator, and a comparator that is used again keeps its earlier results."""
    df, features = small_data
    pd.testing.assert_frame_equal(compare(df, features), compare(df, features))

    comparator = PearsonsCorrelation()
    for _ in range(2):
        result = compare(df, features, comparator=comparator)

    assert len(result) == 2 * 12


def test_manager_gives_the_same_results(small_data):
    """The manager gives the same results as `compare_pairs`."""
    df, features = small_data
    manager = PairwiseCompareManager(df, different=["well"], features=features, same=["plate"])

    pd.testing.assert_frame_equal(manager(), compare(df, features))


def test_engine_compares_outer_groups_that_differ_in_every_outer_column(small_data):
    """Used directly, the engine can have several outer columns: (P1, A) is compared with (P2, B) but not with (P2, A)."""
    df, features = small_data
    comparator = PearsonsCorrelation()
    PairwiseCompare(
        df, comparator, outer_columns=["plate", "line"], inner_columns=["well"], features=features
    ).inter_comparisons()
    result = pd.DataFrame(comparator.comparisons)

    # 2 pairs of outer groups, with 2 x 2 wells each
    assert len(result) == 8
    assert (result["plate__group0"] != result["plate__group1"]).all()
    assert (result["line__group0"] != result["line__group1"]).all()


@st.composite
def comparison_cases(draw):
    """A random dataframe (with missing values) and a valid choice of `same`, `different` and `drop` columns."""
    n_rows, n_features = draw(st.integers(2, 30)), draw(st.integers(3, 6))
    values = draw(arrays(np.float64, (n_rows, n_features), elements=st.integers(-50, 50).map(float)))
    with_missing = draw(st.booleans())
    for row, column in draw(st.lists(st.tuples(st.integers(0, n_rows - 1), st.integers(0, n_features - 1)), max_size=3 if with_missing else 0)):
        values[row, column] = np.nan

    features = [f"f{number}" for number in range(n_features)]
    df = pd.DataFrame(values, columns=features)
    df["well"] = [f"W{number}" for number in range(n_rows)]
    for column in "abc":
        domain = draw(st.sampled_from([[0, 1], [0, 1, 2], ["x", "y"], ["x", "y", "z"]]))
        df[column] = draw(st.lists(st.sampled_from(domain), min_size=n_rows, max_size=n_rows))
        for row in draw(st.lists(st.integers(0, n_rows - 1), max_size=3 if with_missing else 0)):
            df[column] = df[column].astype(object)
            df.loc[row, column] = None

    columns = draw(st.permutations(["a", "b", "c", "well"]))
    n_same = draw(st.integers(0, 2))
    same, different = columns[:n_same], columns[n_same:n_same + draw(st.integers(1, 4 - n_same))]
    assume(same or len(different) >= 2)
    drop = draw(st.lists(st.sampled_from(same + different), unique=True))

    return df, features, same, different, drop


def pearson_of_rows(first, second):
    """The Pearson correlation of two rows from numpy, missing (NaN) when a row has a missing value or no variance."""
    if np.isnan(first).any() or np.isnan(second).any() or np.ptp(first) == 0 or np.ptp(second) == 0:
        return np.nan

    return float(np.corrcoef(first, second)[0, 1])


@settings(max_examples=100, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(case=comparison_cases())
def test_any_valid_comparison_gives_the_pairs_and_correlations_of_the_slow_checks(case):
    """Random dataframes and arguments: every pair that the rules allow is compared once, with numpy's correlation."""
    df, features, same, different, drop = case
    used = same + different
    kept = [column for column in used if column not in drop]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare_pairs(df, different=different, features=features, same=same or None, drop=drop or None)

    # the slow way: every pair of rows without a missing group value, tested one by one
    valid = df.dropna(subset=used)
    columns = {column: valid[column].to_numpy() for column in used}
    values = valid[features].to_numpy()
    expected = defaultdict(list)
    for first, second in combinations(range(len(valid)), 2):
        if all(columns[c][first] != columns[c][second] for c in different) and all(columns[c][first] == columns[c][second] for c in same):
            labels = tuple(sorted(tuple(str(valid[c].iloc[row]) for c in kept) for row in (first, second)))
            expected[labels].append(pearson_of_rows(values[first], values[second]))

    found = defaultdict(list)
    for row in result.to_dict("records"):
        labels = tuple(sorted(tuple(str(row[f"{c}__group{group}"]) for c in kept) for group in (0, 1)))
        found[labels].append(row["pearsons_correlation"])

    assert found.keys() == expected.keys()
    for labels in expected:
        np.testing.assert_allclose(np.sort(found[labels]), np.sort(expected[labels]), atol=1e-9, equal_nan=True)

    if len(result) > 0:
        assert list(result.columns) == ["pearsons_correlation"] + [f"{c}__group{g}" for c in kept for g in (0, 1)]
