"""Tests of rows with missing values and of the warnings about them."""

import warnings

import numpy as np
import pandas as pd
import pytest
from helpers import numpy_correlations

from pairwise_compare import compare_pairs


def run(df, features, **arguments):
    """The results and the messages of the warnings that were raised."""

    arguments = {"different": ["well"], "features": features, "same": ["plate"], **arguments}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compare_pairs(df, **arguments)

    return result, [str(warning.message) for warning in caught]


def test_no_warning_unless_a_column_that_is_used_has_missing_values(small_data):
    """No warning unless a column that is used has missing values."""
    df, features = small_data
    _, clean = run(df, features)

    df["unused"] = np.nan
    _, with_unused_column = run(df, features)

    assert clean == with_unused_column == []


@pytest.mark.parametrize("column, arguments", [("line", {"different": ["line"]}), ("plate", {})])
def test_rows_with_a_missing_group_value_are_left_out_with_a_warning(small_data, column, arguments):
    """Rows with a missing group value are left out with a warning."""
    df, features = small_data
    df.loc[0, column] = np.nan
    result, messages = run(df, features, **arguments)
    without_the_row, _ = run(df.drop(index=0), features, **arguments)

    assert len(messages) == 1
    assert f"1 of 8 rows have a missing value in the group columns ['{column}']" in messages[0]
    assert "left out" in messages[0]
    pd.testing.assert_frame_equal(result, without_the_row)


def test_the_warning_names_every_group_column_with_missing_values(small_data):
    """The warning names every group column with missing values."""
    df, features = small_data
    df.loc[0, "line"] = np.nan
    df.loc[1, "plate"] = np.nan
    _, messages = run(df, features, different=["well", "line"])

    assert "2 of 8 rows have a missing value in the group columns ['plate', 'line']" in messages[0]


def test_rows_with_a_missing_feature_value_are_kept_with_missing_results(small_data):
    """Rows with a missing feature value are kept, and only their comparisons are missing."""
    df, features = small_data
    df.loc[0, "f2"] = np.nan  # well W1
    df.loc[4, "f1"] = np.nan  # well W5, on the other plate
    result, messages = run(df, features)

    assert len(messages) == 1
    assert "2 of 8 rows have a missing value in the feature columns" in messages[0]
    assert len(result) == 12

    with_missing = result[["well__group0", "well__group1"]].isin(["W1", "W5"]).any(axis=1)
    assert with_missing.sum() == 6
    assert result.loc[with_missing, "pearsons_correlation"].isna().all()

    # the other comparisons are not affected
    assert result["pearsons_correlation"].to_numpy() == pytest.approx(numpy_correlations(df, features, result), abs=1e-12, nan_ok=True)


def test_warnings_when_nothing_is_compared(small_data):
    """Warnings when nothing is compared: within a plate (one line per plate) and between groups (a single line)."""
    df, features = small_data
    result, messages = run(df[df["line"] == "A"], features, different=["line"])

    assert result.empty
    assert sum("fewer than two inner groups" in message for message in messages) == 2  # one for each plate
    assert any("The outer group (P1) has fewer" in message for message in messages)
    assert any("No rows were compared" in message for message in messages)

    # the groups of several columns are described by all their values
    _, messages = run(df.drop_duplicates(["plate", "line"]), features, same=["plate", "line"])
    assert any("The outer group (P1, A) has fewer" in message for message in messages)

    df["line"] = "A"
    result, messages = run(df, features, different=["line", "well"], same=None)

    assert result.empty
    assert any("No rows were compared" in message for message in messages)


def test_warning_when_outer_groups_have_no_inner_groups_to_compare(small_data):
    """Outer groups without inner groups are skipped with a warning, and the other pairs are still compared."""
    df, features = small_data
    df["line"] = ["A", "A", "B", "B", "C", "C", "C", "C"]
    df.loc[df["line"] == "A", "well"] = np.nan  # line A has no inner groups left
    result, messages = run(df, features, different=["line", "well"], same=None)

    assert any("The outer groups (A) and (B) have no inner groups to compare" in message for message in messages)
    assert len(result) == 8  # the 2 wells of line B with the 4 wells of line C
