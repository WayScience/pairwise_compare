"""Tests of which rows are compared, the values, and the columns of the results, on a small dataframe."""

import pandas as pd
import pytest
from helpers import FEATURES, correlation_of_wells, expected_pairs, result_pairs

from pairwise_compare import PearsonsCorrelation, compare_pairs
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare
from pairwise_compare.comparison_tools.PairwiseCompareManager import PairwiseCompareManager


def compare(df, **arguments):
    """Compare the wells of `df` with the defaults of these tests (different wells of the same plate)."""
    arguments = {"different": ["well"], "features": FEATURES, "same": ["plate"], **arguments}

    return compare_pairs(df, **arguments)


# the columns used to compare, and the number of pairs of wells that should be compared
@pytest.mark.parametrize(
    "different, same, n_pairs",
    [
        pytest.param(["well"], ["plate"], 12, id="wells of the same plate"),
        pytest.param(["line"], ["plate"], 8, id="lines of the same plate"),
        pytest.param(["line", "well"], ["plate"], 8, id="lines and wells of the same plate"),
        pytest.param(["line", "well"], None, 16, id="lines and wells"),
        pytest.param(["plate", "line"], None, 8, id="plates and lines"),
        pytest.param(["plate", "line", "well"], None, 8, id="plates, lines and wells"),
    ],
)
def test_compares_the_expected_pairs_once_with_the_right_values(small_df, different, same, n_pairs):
    """Compares the expected pairs once with the right values."""
    result = compare(small_df, different=different, same=same)

    assert len(result) == n_pairs == len(expected_pairs(small_df, different, same or []))

    # the wells in the results tell which pairs were compared
    if "well" in different:
        assert result_pairs(result) == expected_pairs(small_df, different, same or [])
        assert len(result_pairs(result)) == len(result), "a pair was compared more than once"

        for row in result.itertuples():
            expected = correlation_of_wells(small_df, FEATURES, row.well__group0, row.well__group1)
            assert row.pearsons_correlation == pytest.approx(expected, abs=1e-12)


def test_columns_of_the_results(small_df):
    """Columns of the results."""
    result = compare(small_df, different=["well", "line"])

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


def test_columns_follow_the_order_given(small_df):
    """Columns follow the order given."""
    first = compare(small_df, different=["line", "well"])
    second = compare(small_df, different=["well", "line"])

    assert list(first.columns)[3:] == ["line__group0", "line__group1", "well__group0", "well__group1"]
    assert list(second.columns)[3:] == ["well__group0", "well__group1", "line__group0", "line__group1"]
    assert result_pairs(first) == result_pairs(second)


@pytest.mark.parametrize(
    "same, drop, kept",
    [
        pytest.param(["plate"], ["well"], ["plate", "line"], id="one of two different columns"),
        pytest.param(["plate"], ["line"], ["plate", "well"], id="the other different column"),
        pytest.param(["plate"], ["plate"], ["well", "line"], id="the same column"),
        pytest.param(["plate"], ["line", "well"], ["plate"], id="all different columns"),
        pytest.param(["plate"], ["plate", "line", "well"], [], id="every group column"),
        pytest.param(None, ["line"], ["well"], id="without same columns, the first different column"),
        pytest.param(None, ["well"], ["line"], id="without same columns, the second different column"),
    ],
)
def test_dropped_columns_are_left_out_and_the_rest_is_unchanged(small_df, same, drop, kept):
    """Dropped columns are left out and the rest is unchanged."""
    everything = compare(small_df, different=["line", "well"], same=same)
    result = compare(small_df, different=["line", "well"], same=same, drop=drop)

    expected_columns = ["pearsons_correlation"] + [
        f"{column}__group{group}" for column in ["plate", "line", "well"] if column in kept for group in (0, 1)
    ]
    pd.testing.assert_frame_equal(result, everything[expected_columns])
    assert not any("hoc" in column for column in result.columns)


def test_integer_groups_work(small_df):
    """Integer groups work."""
    small_df["plate"] = small_df["plate"].map({"P1": 1, "P2": 2})
    result = compare(small_df)

    assert len(result) == 12
    assert set(result["plate__group0"]) == {1, 2}


def test_input_is_not_changed_and_the_order_of_the_rows_does_not_matter(small_df):
    """Input is not changed and the order of the rows does not matter."""
    before = small_df.copy()
    first = compare(small_df, different=["well", "line"])
    second = compare(small_df.sample(frac=1, random_state=0), different=["well", "line"])

    pd.testing.assert_frame_equal(small_df, before)

    key = ["well__group0", "well__group1"]
    pd.testing.assert_frame_equal(
        first.sort_values(key).reset_index(drop=True), second.sort_values(key).reset_index(drop=True)
    )


def test_a_new_comparator_is_used_unless_one_is_given(small_df):
    """A new comparator is used unless one is given."""
    pd.testing.assert_frame_equal(compare(small_df), compare(small_df))

    # a comparator that is used again keeps its earlier results
    comparator = PearsonsCorrelation()
    for _ in range(2):
        result = compare(small_df, comparator=comparator)

    assert len(result) == 2 * 12


def test_manager_gives_the_same_results(small_df):
    """Manager gives the same results."""
    manager = PairwiseCompareManager(small_df, different=["well"], features=FEATURES, same=["plate"])

    pd.testing.assert_frame_equal(manager(), compare(small_df))


def test_engine_compares_outer_groups_that_differ_in_every_outer_column(small_df):
    # used directly, the engine can have several outer columns: (P1, A) is compared with (P2, B) but not with (P2, A)
    """Engine compares outer groups that differ in every outer column."""
    comparator = PearsonsCorrelation()
    PairwiseCompare(
        small_df, comparator, outer_columns=["plate", "line"], inner_columns=["well"], features=FEATURES
    ).inter_comparisons()
    result = pd.DataFrame(comparator.comparisons)

    # 2 pairs of outer groups, with 2 x 2 wells each
    assert len(result) == 8
    assert (result["plate__group0"] != result["plate__group1"]).all()
    assert (result["line__group0"] != result["line__group1"]).all()
