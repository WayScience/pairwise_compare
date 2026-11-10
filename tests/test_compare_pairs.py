"""Tests of which rows are compared, the values, and the columns of the results, on a small dataframe."""

import numpy as np
import pandas as pd
import pytest
from helpers import correlation_of_wells, expected_pairs, result_pairs

from pairwise_compare import PearsonsCorrelation, compare_pairs
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare
from pairwise_compare.comparison_tools.PairwiseCompareManager import PairwiseCompareManager

# the columns used to compare, and the number of pairs of wells that should be compared
CASES = [
    pytest.param(["well"], ["plate"], 12, id="wells of the same plate"),
    pytest.param(["line"], ["plate"], 8, id="lines of the same plate"),
    pytest.param(["line", "well"], ["plate"], 8, id="lines and wells of the same plate"),
    pytest.param(["line", "well"], [], 16, id="lines and wells"),
    pytest.param(["plate", "line"], [], 8, id="plates and lines"),
    pytest.param(["plate", "line", "well"], [], 8, id="plates, lines and wells"),
]


# the cases that keep the wells in the results, to tell which wells were compared
CASES_WITH_WELLS = [case for case in CASES if "well" in case.values[0]]


@pytest.mark.parametrize("different, same, n_pairs", CASES)
def test_compares_the_expected_number_of_pairs(small_df, small_features, different, same, n_pairs):
    result = compare_pairs(small_df, different=different, features=small_features, same=same)

    assert len(expected_pairs(small_df, different, same)) == n_pairs
    assert len(result) == n_pairs


@pytest.mark.parametrize("different, same, n_pairs", CASES_WITH_WELLS)
def test_compares_the_expected_pairs_once(small_df, small_features, different, same, n_pairs):
    result = compare_pairs(small_df, different=different, features=small_features, same=same)

    # each pair is compared once, and not in both directions
    assert len(result) == len(result_pairs(result))
    assert result_pairs(result) == expected_pairs(small_df, different, same)


@pytest.mark.parametrize("different, same, n_pairs", CASES_WITH_WELLS)
def test_correlations_are_those_of_numpy(small_df, small_features, different, same, n_pairs):
    result = compare_pairs(small_df, different=different, features=small_features, same=same)

    for row in result.itertuples():
        expected = correlation_of_wells(small_df, small_features, row.well__group0, row.well__group1)
        assert row.pearsons_correlation == pytest.approx(expected, abs=1e-12)


def test_columns_of_the_results(small_df, small_features):
    result = compare_pairs(small_df, different=["well", "line"], features=small_features, same=["plate"])

    assert list(result.columns) == [
        "pearsons_correlation",
        "plate__group0",
        "plate__group1",
        "well__group0",
        "well__group1",
        "line__group0",
        "line__group1",
    ]
    # both rows of a pair have the same value in a `same` column
    assert (result["plate__group0"] == result["plate__group1"]).all()
    # and a different value in a `different` column
    assert (result["well__group0"] != result["well__group1"]).all()
    assert (result["line__group0"] != result["line__group1"]).all()


def test_columns_follow_the_order_given(small_df, small_features):
    first = compare_pairs(small_df, different=["line", "well"], features=small_features, same=["plate"])
    second = compare_pairs(small_df, different=["well", "line"], features=small_features, same=["plate"])

    assert list(first.columns)[3:] == ["line__group0", "line__group1", "well__group0", "well__group1"]
    assert list(second.columns)[3:] == ["well__group0", "well__group1", "line__group0", "line__group1"]
    # the same pairs are compared either way
    assert result_pairs(first) == result_pairs(second)


@pytest.mark.parametrize(
    "drop, columns_left",
    [
        pytest.param(["well"], ["plate", "line"], id="one of two different columns"),
        pytest.param(["line"], ["plate", "well"], id="the other different column"),
        pytest.param(["plate"], ["well", "line"], id="the same column"),
        pytest.param(["line", "well"], ["plate"], id="all different columns"),
        pytest.param(["plate", "line", "well"], [], id="every group column"),
    ],
)
def test_dropped_columns_are_left_out_and_the_others_kept(small_df, small_features, drop, columns_left):
    result = compare_pairs(
        small_df, different=["well", "line"], features=small_features, same=["plate"], drop=drop
    )

    expected_columns = ["pearsons_correlation"] + [
        f"{column}__group{group}" for column in ["plate", "well", "line"] if column in columns_left for group in (0, 1)
    ]
    assert list(result.columns) == expected_columns


def test_dropping_columns_does_not_change_the_comparisons(small_df, small_features):
    everything = compare_pairs(small_df, different=["well", "line"], features=small_features, same=["plate"])
    dropped = compare_pairs(
        small_df, different=["well", "line"], features=small_features, same=["plate"], drop=["well"]
    )

    pd.testing.assert_frame_equal(everything.drop(columns=["well__group0", "well__group1"]), dropped)


def test_dropping_columns_works_without_same_columns(small_df, small_features):
    # the first column of `different` defines the outer groups here, and the others the inner groups
    for drop, left in [(["line"], "well"), (["well"], "line")]:
        result = compare_pairs(small_df, different=["line", "well"], features=small_features, drop=drop)

        assert [c for c in result.columns if c != "pearsons_correlation"] == [f"{left}__group0", f"{left}__group1"]


def test_output_does_not_use_ante_hoc_and_post_hoc_names(small_df, small_features):
    result = compare_pairs(small_df, different=["well"], features=small_features, same=["plate"])

    assert not any("hoc" in column for column in result.columns)


def test_integer_groups_work(small_df, small_features):
    small_df["plate"] = small_df["plate"].map({"P1": 1, "P2": 2})
    result = compare_pairs(small_df, different=["well"], features=small_features, same=["plate"])

    assert len(result) == 12
    assert set(result["plate__group0"]) == {1, 2}


def test_input_is_not_changed(small_df, small_features):
    before = small_df.copy()
    compare_pairs(small_df, different=["well", "line"], features=small_features, same=["plate"])

    pd.testing.assert_frame_equal(small_df, before)


def test_result_does_not_depend_on_the_order_of_the_rows(small_df, small_features):
    shuffled = small_df.sample(frac=1, random_state=0)
    first = compare_pairs(small_df, different=["well", "line"], features=small_features, same=["plate"])
    second = compare_pairs(shuffled, different=["well", "line"], features=small_features, same=["plate"])

    key = ["well__group0", "well__group1"]
    pd.testing.assert_frame_equal(
        first.sort_values(key).reset_index(drop=True), second.sort_values(key).reset_index(drop=True)
    )


def test_unused_columns_are_ignored(small_df, small_features):
    small_df["unused"] = np.nan
    result = compare_pairs(small_df, different=["well"], features=small_features, same=["plate"])

    assert len(result) == 12


def test_each_call_starts_with_a_new_comparator(small_df, small_features):
    first = compare_pairs(small_df, different=["well"], features=small_features, same=["plate"])
    second = compare_pairs(small_df, different=["well"], features=small_features, same=["plate"])

    pd.testing.assert_frame_equal(first, second)


def test_a_comparator_that_is_used_again_keeps_its_earlier_results(small_df, small_features):
    comparator = PearsonsCorrelation()
    for _ in range(2):
        result = compare_pairs(
            small_df, different=["well"], features=small_features, same=["plate"], comparator=comparator
        )

    assert len(result) == 2 * 12


def test_manager_gives_the_same_results(small_df, small_features):
    manager = PairwiseCompareManager(small_df, different=["well"], features=small_features, same=["plate"])

    pd.testing.assert_frame_equal(
        manager(), compare_pairs(small_df, different=["well"], features=small_features, same=["plate"])
    )


def test_engine_compares_outer_groups_that_differ_in_every_outer_column(small_df, small_features):
    # used directly, the engine can have several outer columns: (P1, A) is compared with (P2, B) but not with (P2, A)
    comparator = PearsonsCorrelation()
    PairwiseCompare(
        small_df, comparator, outer_columns=["plate", "line"], inner_columns=["well"], features=small_features
    ).inter_comparisons()
    result = pd.DataFrame(comparator.comparisons)

    # 2 pairs of outer groups, with 2 x 2 wells each
    assert len(result) == 8
    assert (result["plate__group0"] != result["plate__group1"]).all()
    assert (result["line__group0"] != result["line__group1"]).all()
