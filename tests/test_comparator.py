"""Tests of the Comparator template, with a small comparator of our own."""

import pandas as pd
import pytest

from helpers import FEATURES

from pairwise_compare import Comparator, compare_pairs


class MeanDifference(Comparator):
    """Compares two groups with the difference of the mean of all their values (one row per pair of groups)."""

    def __call__(self, group0, group1):
        """Compare two groups with the difference of the mean of all their values."""
        self._comparisons["mean_difference"].append(group0.to_numpy().mean() - group1.to_numpy().mean())

    def save_groups(self, group_columns, **groups):
        """Save the group values in `<column>__<group name>` columns, one value for each pair of groups."""
        for position, column in enumerate(group_columns):
            for name, values in groups.items():
                self._comparisons[f"{column}__{name}"].append(values[position])


def test_template_cannot_be_used_directly():
    """Template cannot be used directly."""
    with pytest.raises(TypeError, match="abstract"):
        Comparator()


def test_comparisons_property_is_the_stored_results():
    """Comparisons property is the stored results."""
    comparator = MeanDifference()
    comparator(pd.DataFrame([[1.0, 3.0]]), pd.DataFrame([[1.0]]))

    assert dict(comparator.comparisons) == {"mean_difference": [1.0]}


def test_another_comparator_can_be_used(small_df):
    """Another comparator can be used."""
    result = compare_pairs(
        small_df, different=["line"], features=FEATURES, same=["plate"], comparator=MeanDifference()
    )

    # the lines A and B of each plate are compared once, with the wells of a line taken together
    assert len(result) == 2
    assert set(result.columns) == {"mean_difference", "plate__group0", "plate__group1", "line__group0", "line__group1"}
    assert set(result["line__group0"]) == {"A"} and set(result["line__group1"]) == {"B"}
