"""Tests of which rows are compared, the values, and the columns of the results, on small and real datasets."""

import warnings

import numpy as np
import pandas as pd
import pytest
from helpers import expected_order, numpy_correlations
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pairwise_compare import PearsonsCorrelation, compare_pairs
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare, _sort_order
from pairwise_compare.comparison_tools.PairwiseCompareManager import PairwiseCompareManager


def compare(df, features, **arguments):
    """Compare the wells of `df` with the defaults of these tests (different wells of the same plate)."""
    arguments = {"different": ["well"], "features": features, "same": ["plate"], **arguments}

    return compare_pairs(df, **arguments)


def test_real_datasets_give_the_pairs_order_labels_and_correlations_of_the_slow_checks(real_dataset):
    """Real datasets (NF1 plate 4, a PCCMA plate, iris, wine) give the pairs of the oracle, in its order, with numpy's correlations."""
    df, features, comparisons = real_dataset
    correlations = np.corrcoef(df[features].to_numpy(dtype=float))

    for same, different in comparisons:
        rows0, rows1 = expected_order(df, different, same)
        result = compare_pairs(df, different=different, features=features, same=same)

        assert len(result) == len(rows0) > 0
        # which row is first, and the order of the pairs, as well as the pairs themselves
        for column in same + different:
            assert list(result[f"{column}__group0"]) == list(df[column].iloc[rows0])
            assert list(result[f"{column}__group1"]) == list(df[column].iloc[rows1])
        assert result["pearsons_correlation"].to_numpy() == pytest.approx(correlations[rows0, rows1], abs=1e-9)


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
    """The dataframe is not changed, and shuffling its rows gives the same results in the same order."""
    df, features = small_data
    before = df.copy()
    first = compare(df, features, different=["well", "line"])
    second = compare(df.sample(frac=1, random_state=0), features, different=["well", "line"])

    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(first, second)


def test_a_comparator_can_be_used_again(small_data):
    """A comparator keeps no results, so using it again gives the same results as a new one."""
    df, features = small_data
    comparator = PearsonsCorrelation()

    pd.testing.assert_frame_equal(compare(df, features, comparator=comparator), compare(df, features, comparator=comparator))
    pd.testing.assert_frame_equal(compare(df, features), compare(df, features, comparator=comparator))


def test_manager_gives_the_same_results(small_data):
    """The manager gives the same results as `compare_pairs`."""
    df, features = small_data
    manager = PairwiseCompareManager(df, different=["well"], features=features, same=["plate"])

    pd.testing.assert_frame_equal(manager(), compare(df, features))


def test_engine_compares_outer_groups_that_differ_in_every_outer_column(small_data, monkeypatch):
    """Used directly, the engine can have several outer columns: (P1, A) is compared with (P2, B), not with (P2, A)."""
    df, features = small_data
    engine = PairwiseCompare(df, PearsonsCorrelation(), outer_columns=["plate", "line"], inner_columns=["well"], features=features)
    result = engine.inter_comparisons()

    # 2 pairs of outer groups, with 2 x 2 wells each
    assert len(result) == 8
    assert (result["plate__group0"] != result["plate__group1"]).all()
    assert (result["line__group0"] != result["line__group1"]).all()

    # the pairs are found in blocks of rows, and a block of 2 rows gives the same pairs as one block of all the rows
    monkeypatch.setattr("pairwise_compare.comparison_tools.PairwiseCompare._PAIR_BLOCK", 16)
    pd.testing.assert_frame_equal(engine.inter_comparisons(), result)


@pytest.mark.parametrize(
    "top_values",
    [(3, 3), (2**40, 2**40), (2**29 - 1, 2**29 - 1)],
    ids=["keys that fit in one number", "keys that are too large for it", "keys of exactly 64 bits together"],
)
def test_pairs_are_sorted_by_the_first_key_then_the_next(top_values):
    """Pairs are sorted by the first key, then the next, whether or not the keys fit in one number."""
    # small values with many ties, so that the next key decides, and the largest value gives the size of the key
    keys = [np.random.default_rng(seed).integers(0, 3, 50) for seed in (0, 1)]
    for key, top in zip(keys, top_values):
        key[0] = top
    # the last key (the row) makes every combination of keys different, as in the engine
    keys.append(np.arange(50))

    assert list(_sort_order(keys)) == list(np.lexsort(keys[::-1]))


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
def test_any_valid_comparison_gives_the_pairs_order_and_correlations_of_the_slow_checks(case):
    """Random dataframes and arguments: the pairs the rules allow are compared once, in the expected order, with numpy's correlation."""
    df, features, same, different, drop = case
    used = same + different
    kept = [column for column in used if column not in drop]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compare_pairs(df, different=different, features=features, same=same or None, drop=drop or None)

    # the slow way: every pair of rows without a missing group value, tested one by one
    valid = df.dropna(subset=used)
    rows0, rows1 = expected_order(valid, different, same)
    values = valid[features].to_numpy()
    expected = [pearson_of_rows(values[first], values[second]) for first, second in zip(rows0, rows1)]

    assert len(result) == len(rows0)
    for column in kept:
        assert list(result[f"{column}__group0"]) == list(valid[column].iloc[rows0])
        assert list(result[f"{column}__group1"]) == list(valid[column].iloc[rows1])
    np.testing.assert_allclose(result["pearsons_correlation"].to_numpy(), expected, atol=1e-9, equal_nan=True)
    # also when nothing is compared, the results have their columns
    assert list(result.columns) == ["pearsons_correlation"] + [f"{c}__group{g}" for c in kept for g in (0, 1)]


def test_the_helpers_give_answers_that_are_worked_out_by_hand():
    """The oracle of the pairs and the numpy correlations are checked on rows with answers that are worked out by hand."""
    # without `same` columns, rows 0 to 3 pair up when both a and b differ: (0, 1), (0, 3), (1, 2) and (2, 3).
    # The first row of a pair has the smaller (a, b), so (1, 2) is (2, 1), and the pairs are sorted by the `a` of both
    # rows before the `b` of either: sorting by the whole first row would put (0, 1) before (2, 3).
    df = pd.DataFrame({"a": [1, 3, 1, 2], "b": [2, 1, 3, 1], "well": ["W0", "W1", "W2", "W3"], "x": [1.0, 2, 3, 4], "y": [2.0, 1, 4, 3]})
    rows0, rows1 = expected_order(df, ["a", "b"])
    assert (rows0.tolist(), rows1.tolist()) == ([0, 2, 0, 2], [3, 3, 1, 1])

    result = compare_pairs(df, different=["a", "b"], features=["x", "y"])
    assert [result[f"{column}__group{group}"].tolist() for group in (0, 1) for column in "ab"] == [
        [1, 1, 1, 1], [2, 3, 2, 3], [2, 2, 3, 3], [1, 1, 1, 1],  # a and b of the first rows, then of the second rows
    ]

    # with `same` columns, each group g is ordered by the values of w, and the groups follow each other
    df = pd.DataFrame({"g": [1, 1, 1, 2, 2], "w": [3, 1, 2, 5, 4]})
    rows0, rows1 = expected_order(df, ["w"], ["g"])
    assert (rows0.tolist(), rows1.tolist()) == ([1, 1, 2, 4], [2, 0, 0, 3])

    # the wells are (1, 2, 3), (2, 4, 6) and (3, 2, 1) in their three features: W0 and W1 go up together (correlation 1),
    # and W2 goes the other way compared with both (correlation -1)
    wells = pd.DataFrame({"well": ["W0", "W1", "W2"], "f1": [1.0, 2, 3], "f2": [2.0, 4, 2], "f3": [3.0, 6, 1]})
    pairs = pd.DataFrame({"well__group0": ["W0", "W0", "W1"], "well__group1": ["W1", "W2", "W2"]})
    assert numpy_correlations(wells, ["f1", "f2", "f3"], pairs) == pytest.approx([1.0, -1.0, -1.0], abs=1e-12)
