"""Tests of rows with missing values and of the warnings about them."""

import warnings

import numpy as np
import pandas as pd
import pytest
from helpers import FEATURES, correlation_of_wells

from pairwise_compare import compare_pairs


def run(df, **arguments):
    """The results and the messages of the warnings that were raised."""

    arguments = {"different": ["well"], "features": FEATURES, "same": ["plate"], **arguments}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compare_pairs(df, **arguments)

    return result, [str(warning.message) for warning in caught]


def test_no_warning_unless_a_column_that_is_used_has_missing_values(small_df):
    _, clean = run(small_df)

    small_df["unused"] = np.nan
    _, with_unused_column = run(small_df)

    assert clean == with_unused_column == []


@pytest.mark.parametrize("column, arguments", [("line", {"different": ["line"]}), ("plate", {})])
def test_rows_with_a_missing_group_value_are_left_out_with_a_warning(small_df, column, arguments):
    small_df.loc[0, column] = np.nan
    result, messages = run(small_df, **arguments)
    without_the_row, _ = run(small_df.drop(index=0), **arguments)

    assert len(messages) == 1
    assert f"1 of 8 rows have a missing value in the group columns ['{column}']" in messages[0]
    assert "left out" in messages[0]
    pd.testing.assert_frame_equal(result, without_the_row)


def test_the_warning_names_every_group_column_with_missing_values(small_df):
    small_df.loc[0, "line"] = np.nan
    small_df.loc[1, "plate"] = np.nan
    _, messages = run(small_df, different=["well", "line"])

    assert "2 of 8 rows have a missing value in the group columns ['plate', 'line']" in messages[0]


def test_rows_with_a_missing_feature_value_are_kept_with_missing_results(small_df):
    small_df.loc[0, "f2"] = np.nan  # well W1
    result, messages = run(small_df)

    assert len(messages) == 1
    assert "1 of 8 rows have a missing value in the feature columns" in messages[0]
    assert len(result) == 12

    with_w1 = (result["well__group0"] == "W1") | (result["well__group1"] == "W1")
    assert with_w1.sum() == 3
    assert result.loc[with_w1, "pearsons_correlation"].isna().all()

    # the other comparisons are not affected
    for row in result[~with_w1].itertuples():
        expected = correlation_of_wells(small_df, FEATURES, row.well__group0, row.well__group1)
        assert row.pearsons_correlation == pytest.approx(expected, abs=1e-12)


def test_warnings_when_nothing_is_compared(small_df):
    # within a plate: one line per plate, so there is nothing to compare
    result, messages = run(small_df[small_df["line"] == "A"], different=["line"])

    assert result.empty
    assert sum("fewer than two inner groups" in message for message in messages) == 2  # one for each plate
    assert any("No rows were compared" in message for message in messages)

    # between groups: a single line
    small_df["line"] = "A"
    result, messages = run(small_df, different=["line", "well"], same=None)

    assert result.empty
    assert any("No rows were compared" in message for message in messages)


def test_warning_when_outer_groups_have_no_inner_groups_to_compare(small_df):
    # every row of line A has a missing well, so line A has no inner groups left
    small_df.loc[small_df["line"] == "A", "well"] = np.nan
    result, messages = run(small_df, different=["line", "well"], same=None)

    assert result.empty
    assert any("have no inner groups to compare" in message for message in messages)
