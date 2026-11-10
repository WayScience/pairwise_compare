"""Tests of the Comparator template, with small comparators of our own."""

import numpy as np
import pandas as pd
import pytest

from pairwise_compare import Comparator, compare_pairs


class MeanDifference(Comparator):
    """Compares two rows with the difference of the mean of their values."""

    def __init__(self):
        """Name the column of the results."""
        super().__init__("mean_difference")

    def prepare(self, values):
        """The mean of each row."""
        return values.mean(axis=1)

    def compare(self, prepared, rows0, rows1):
        """The mean of the first row minus the mean of the second row, for each pair."""
        return prepared[rows0] - prepared[rows1]


class FirstFeature(Comparator):
    """Compares two rows with their first feature, and does not prepare the values."""

    def compare(self, prepared, rows0, rows1):
        """The first feature of the first row minus that of the second row, for each pair."""
        return prepared[rows0, 0] - prepared[rows1, 0]


class CorrelationOfPairs(Comparator):
    """Pearson's correlation, one pair at a time with numpy, as the slow check of the default comparator."""

    def compare(self, prepared, rows0, rows1):
        """The correlation of the two rows of each pair."""
        return np.array([np.corrcoef(prepared[first], prepared[second])[0, 1] for first, second in zip(rows0, rows1)])


def test_template_cannot_be_used_directly():
    """Template cannot be used directly."""
    with pytest.raises(TypeError, match="abstract"):
        Comparator("name")


def test_values_are_used_as_they_are_unless_the_comparator_prepares_them():
    """Values are used as they are unless the comparator prepares them."""
    values = np.arange(6.0).reshape(2, 3)

    assert FirstFeature("first").prepare(values) is values
    assert MeanDifference().prepare(values).tolist() == [1.0, 4.0]


@pytest.mark.parametrize(
    "comparator, expected",
    [
        pytest.param(MeanDifference(), lambda a, b: a.mean() - b.mean(), id="with prepared values"),
        pytest.param(FirstFeature("first_feature"), lambda a, b: a[0] - b[0], id="with the values as they are"),
    ],
)
def test_another_comparator_can_be_used(small_data, comparator, expected):
    """Another comparator can be used, and its name is the name of the column of the results."""
    df, features = small_data
    result = compare_pairs(df, different=["line"], features=features, same=["plate"], comparator=comparator)
    by_well = df.set_index("well")[features]

    # the lines A and B of each plate are compared row by row: 2 x 2 wells in each of the 2 plates
    assert len(result) == 8
    assert set(result.columns) == {comparator.name, "plate__group0", "plate__group1", "line__group0", "line__group1"}
    assert set(result["line__group0"]) == {"A"} and set(result["line__group1"]) == {"B"}

    # the wells that were compared come from the results of a comparison that keeps them
    wells = compare_pairs(df, different=["line", "well"], features=features, same=["plate"])[["well__group0", "well__group1"]]
    for value, well0, well1 in zip(result[comparator.name], wells["well__group0"], wells["well__group1"]):
        assert value == pytest.approx(expected(by_well.loc[well0].to_numpy(), by_well.loc[well1].to_numpy()))


@pytest.mark.parametrize("arguments", [{"different": ["well"], "same": ["plate"]}, {"different": ["line", "well"]}], ids=["within groups", "between groups"])
def test_a_comparator_that_takes_the_pairs_one_by_one_gives_the_results_of_the_default(small_data, arguments):
    """A comparator that gets the pairs of rows and nothing else gives the same table as the fast default comparator."""
    df, features = small_data

    default = compare_pairs(df, features=features, **arguments)
    one_by_one = compare_pairs(df, features=features, comparator=CorrelationOfPairs("pearsons_correlation"), **arguments)

    # the same pairs in the same order, with the same labels, and the same values
    pd.testing.assert_frame_equal(default, one_by_one)
