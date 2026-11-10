"""Tests on the real data of the repository: 60 wells of plate 4 with 1,182 columns."""

import numpy as np
import pytest
from helpers import expected_pairs

from pairwise_compare import compare_pairs


@pytest.mark.parametrize(
    "different, same, n_pairs",
    [
        pytest.param(["Metadata_siRNA", "Metadata_Well"], ["Metadata_Concentration"], 135, id="within a dose"),
        pytest.param(["Metadata_Concentration", "Metadata_Well"], [], 1485, id="between doses and wells"),
    ],
)
def test_pairs_and_values_of_plate4(plate4, plate4_features, different, same, n_pairs):
    # keep only the wells in the results (a dropped column still decides which rows are compared)
    """Pairs and values of plate4."""
    result = compare_pairs(
        plate4, different=different, features=plate4_features, same=same, drop=[c for c in different + same if c != "Metadata_Well"]
    )

    assert list(result.columns) == ["pearsons_correlation", "Metadata_Well__group0", "Metadata_Well__group1"]
    assert len(result) == n_pairs

    wells = list(plate4["well"])
    pairs = {frozenset(pair) for pair in zip(result["Metadata_Well__group0"], result["Metadata_Well__group1"])}
    # the slow check of the pairs only needs the group columns
    group_columns = plate4[["well"] + different + same].astype(str)
    assert pairs == expected_pairs(group_columns, different, same)

    # the correlation matrix of all the wells at once, from numpy
    correlations = np.corrcoef(plate4[plate4_features].to_numpy())
    expected_values = [
        correlations[wells.index(well0), wells.index(well1)]
        for well0, well1 in zip(result["Metadata_Well__group0"], result["Metadata_Well__group1"])
    ]
    assert result["pearsons_correlation"].to_numpy() == pytest.approx(expected_values, abs=1e-10)
