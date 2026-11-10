"""Tests of the Comparator template, with a small comparator of our own."""

import pandas as pd
import pytest

from pairwise_compare import Comparator, compare_pairs


class MeanDifference(Comparator):
    """Compares two groups with the difference of the mean of all their values (one row per pair of groups)."""

    def __call__(self, group0, group1):
        self._comparisons["mean_difference"].append(group0.to_numpy().mean() - group1.to_numpy().mean())

    def save_groups(self, group_columns, **groups):
        for position, column in enumerate(group_columns):
            for name, values in groups.items():
                self._comparisons[f"{column}__{name}"].append(values[position])


def test_template_cannot_be_used_directly():
    with pytest.raises(TypeError, match="abstract"):
        Comparator()


def test_comparisons_property_is_the_stored_results():
    comparator = MeanDifference()
    comparator(pd.DataFrame([[1.0, 3.0]]), pd.DataFrame([[1.0]]))

    assert dict(comparator.comparisons) == {"mean_difference": [1.0]}


def test_another_comparator_can_be_used(small_df, small_features):
    result = compare_pairs(
        small_df, different=["line"], features=small_features, same=["plate"], comparator=MeanDifference()
    )

    # the lines A and B of each plate are compared once, with the wells of a line taken together
    assert len(result) == 2
    assert set(result.columns) == {"mean_difference", "plate__group0", "plate__group1", "line__group0", "line__group1"}
    assert set(result["line__group0"]) == {"A"} and set(result["line__group1"]) == {"B"}
