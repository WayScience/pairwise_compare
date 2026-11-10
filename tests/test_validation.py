"""Tests of the checks of the arguments: each mistake should give a clear error."""

import pandas as pd
import pytest

from pairwise_compare import PearsonsCorrelation, compare_pairs
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare


def call(df, **changes):
    arguments = {"different": ["well"], "features": ["f1", "f2", "f3"], "same": ["plate"]}
    arguments.update(changes)

    return compare_pairs(df, **arguments)


def test_valid_call_works(small_df):
    assert len(call(small_df)) > 0


def test_dataframe_is_required():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        call([[1, 2]])


def test_empty_dataframe_is_refused(small_df):
    with pytest.raises(ValueError, match="empty"):
        call(small_df.iloc[0:0])


@pytest.mark.parametrize("comparator", ["pearson", PearsonsCorrelation])
def test_comparator_must_be_an_instance(small_df, comparator):
    with pytest.raises(TypeError, match="instance of Comparator"):
        call(small_df, comparator=comparator)


@pytest.mark.parametrize("name", ["different", "same", "features", "drop"])
def test_single_string_instead_of_a_list_is_refused(small_df, name):
    with pytest.raises(TypeError, match=f"`{name}` must be a list of column names, not a single string"):
        call(small_df, **{name: "well"})


@pytest.mark.parametrize("columns", [{"well"}, (c for c in ["well"]), 1])
def test_only_lists_and_tuples_are_accepted(small_df, columns):
    with pytest.raises(TypeError, match="list or tuple"):
        call(small_df, different=columns)


def test_tuples_are_accepted(small_df):
    assert len(call(small_df, different=("well",), same=("plate",))) > 0


def test_names_must_be_strings(small_df):
    with pytest.raises(TypeError, match="only contain column names"):
        call(small_df, different=["well", 3])


def test_different_columns_are_required(small_df):
    with pytest.raises(ValueError, match="`different` must have at least one column"):
        call(small_df, different=[])

    with pytest.raises(ValueError, match="`different` must have at least one column"):
        call(small_df, different=None)


def test_one_different_column_needs_a_same_column(small_df):
    with pytest.raises(ValueError, match="at least two columns"):
        call(small_df, same=None)

    with pytest.raises(ValueError, match="at least two columns"):
        call(small_df, same=[])


def test_two_different_columns_do_not_need_a_same_column(small_df):
    assert len(call(small_df, different=["well", "line"], same=None)) > 0


def test_a_column_cannot_be_same_and_different(small_df):
    with pytest.raises(ValueError, match=r"in common: \['plate'\]"):
        call(small_df, different=["well", "plate"], same=["plate"])


def test_features_are_required(small_df):
    with pytest.raises(ValueError, match="`features` must have at least one column"):
        call(small_df, features=[])


def test_features_must_be_numeric(small_df):
    with pytest.raises(TypeError, match=r"feature columns \['line'\] are not numeric"):
        call(small_df, features=["f1", "line"])


def test_columns_must_be_in_the_dataframe(small_df):
    with pytest.raises(ValueError, match=r"does not have the columns \['nope'\]"):
        call(small_df, features=["f1", "nope"])

    with pytest.raises(ValueError, match=r"does not have the columns \['nope'\]"):
        call(small_df, same=["nope"])


def test_dropped_columns_must_be_group_columns(small_df):
    with pytest.raises(ValueError, match=r"`drop` can only have columns of `same` or `different`, not \['f1'\]"):
        call(small_df, drop=["f1"])


def test_repeated_columns_are_used_once(small_df):
    once = call(small_df, different=["well"], features=["f1", "f2", "f3"])
    twice = call(small_df, different=["well", "well"], features=["f1", "f1", "f2", "f3"])

    pd.testing.assert_frame_equal(once, twice)


def test_the_engine_checks_its_arguments_too(small_df):
    with pytest.raises(TypeError, match="instance of Comparator"):
        PairwiseCompare(small_df, "pearson", ["plate"], ["well"], ["f1"])

    with pytest.raises(ValueError, match="does not have the columns"):
        PairwiseCompare(small_df, PearsonsCorrelation(), ["plate"], ["nope"], ["f1"])
