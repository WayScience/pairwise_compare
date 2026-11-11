"""Tests of the checks of the arguments: each mistake should give a clear error."""

import pandas as pd
import pytest
from pairwise_compare import PearsonsCorrelation, compare_pairs
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare


def call(data, **changes):
    """Call `compare_pairs` on the dataframe and features of `data` with valid arguments, changed by `changes`."""
    df, features = data

    return compare_pairs(df, **{"different": ["well"], "features": features, "same": ["plate"], **changes})


@pytest.mark.parametrize(
    "changes, error, message",
    [
        # the columns must be lists (or tuples) of column names
        *[
            pytest.param(
                {name: "well"}, TypeError, f"`{name}` must be a list of column names, not a single string", id=f"{name}: a string"
            )
            for name in ["different", "same", "features", "drop"]
        ],
        pytest.param({"different": {"well"}}, TypeError, "list or tuple", id="a set"),
        pytest.param({"different": (c for c in ["well"])}, TypeError, "list or tuple", id="a generator"),
        pytest.param({"different": 1}, TypeError, "list or tuple of column names, not int", id="a number"),
        pytest.param({"different": ["well", 3]}, TypeError, "only contain column names", id="a name that is not a string"),
        # which columns, and how many
        pytest.param({"different": []}, ValueError, "`different` must have at least one column", id="no different columns"),
        pytest.param({"different": None}, ValueError, "`different` must have at least one column", id="different is None"),
        pytest.param({"same": None}, ValueError, "at least two columns", id="one different column and no same column"),
        pytest.param({"same": []}, ValueError, "at least two columns", id="one different column and an empty same"),
        pytest.param({"different": ["well", "plate"]}, ValueError, r"in common: \['plate'\]", id="same and different overlap"),
        pytest.param({"features": []}, ValueError, "`features` must have at least one column", id="no features"),
        pytest.param({"features": ["f1", "line"]}, TypeError, r"feature columns \['line'\] are not numeric", id="a text feature"),
        pytest.param({"features": ["f1", "nope"]}, ValueError, r"does not have the columns \['nope'\]", id="unknown feature"),
        pytest.param({"same": ["nope"]}, ValueError, r"does not have the columns \['nope'\]", id="unknown same column"),
        pytest.param(
            {"drop": ["f1"]},
            ValueError,
            r"`drop` can only have columns of `same` or `different`, not \['f1'\]",
            id="dropping a feature",
        ),
        pytest.param({"comparator": "pearson"}, TypeError, "instance of Comparator", id="a comparator name"),
        pytest.param({"comparator": PearsonsCorrelation}, TypeError, "instance of Comparator", id="a comparator class"),
        pytest.param({"comparator": []}, ValueError, "empty list: give at least one comparator", id="an empty list of comparators"),
        pytest.param({"comparator": [PearsonsCorrelation(), "cosine"]}, TypeError, "list of instances of Comparator", id="a name in the list"),
        pytest.param(
            {"comparator": [PearsonsCorrelation(), PearsonsCorrelation()]},
            ValueError,
            r"need different names .* \['pearsons_correlation'\] are repeated",
            id="two comparators with the same name",
        ),
        pytest.param(
            {"comparator": [PearsonsCorrelation(name="plate__group0")]},
            ValueError,
            r"cannot be named like the columns of the group values: \['plate__group0'\]",
            id="a comparator named like a group column",
        ),
    ],
)
def test_mistakes_give_a_clear_error(small_data, changes, error, message):
    """Mistakes give a clear error."""
    with pytest.raises(error, match=message):
        call(small_data, **changes)


def test_the_dataframe_must_be_a_dataframe_with_rows(small_data):
    """The dataframe must be a dataframe with rows."""
    df, features = small_data
    with pytest.raises(TypeError, match="pandas DataFrame"):
        call(([[1, 2]], features))

    with pytest.raises(ValueError, match="empty"):
        call((df.iloc[0:0], features))


def test_tuples_and_repeated_columns_work_like_lists(small_data):
    """Tuples and repeated columns work like lists."""
    expected = call(small_data)

    pd.testing.assert_frame_equal(call(small_data, different=("well",), same=("plate",)), expected)
    pd.testing.assert_frame_equal(call(small_data, different=["well", "well"], features=small_data[1] * 2), expected)


def test_the_engine_checks_its_arguments_too(small_data):
    """The engine checks its arguments too."""
    df, features = small_data
    with pytest.raises(TypeError, match="instance of Comparator"):
        PairwiseCompare(df, "pearson", ["plate"], ["well"], ["f1"])

    with pytest.raises(ValueError, match="does not have the columns"):
        PairwiseCompare(df, PearsonsCorrelation(), ["plate"], ["nope"], ["f1"])

    with pytest.raises(ValueError, match="at least one column"):
        PairwiseCompare(df, PearsonsCorrelation(), [], ["well"], ["f1"])

    # a single string instead of a list is named after the argument of the engine
    for name in ["outer_columns", "inner_columns", "features", "drop"]:
        arguments = {"outer_columns": ["plate"], "inner_columns": ["well"], "features": ["f1"], "drop": None, name: "plate"}
        with pytest.raises(TypeError, match=f"`{name}` must be a list of column names"):
            PairwiseCompare(df, PearsonsCorrelation(), **arguments)
