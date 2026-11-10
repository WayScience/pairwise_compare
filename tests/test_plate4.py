"""Tests on the real data of the repository: 60 wells of plate 4 with 1,182 columns, compared in four ways."""

import numpy as np
import pytest
from helpers import expected_pairs

from pairwise_compare import compare_pairs

# different columns, same columns, and the number of pairs of wells (checked by hand with expected_pairs)
CASES = [
    pytest.param(["Metadata_siRNA", "Metadata_Well"], ["Metadata_Concentration"], 135, id="siRNA and well, same dose"),
    pytest.param(
        ["Metadata_siRNA", "Metadata_Well", "Metadata_genotype"], [], 360, id="siRNA, well and genotype"
    ),
    pytest.param(["Metadata_Concentration", "Metadata_Well"], [], 1485, id="dose and well"),
]


@pytest.mark.parametrize("different, same, n_pairs", CASES)
def test_pairs_and_values_of_plate4(plate4, plate4_features, different, same, n_pairs):
    result = compare_pairs(
        plate4, different=different, features=plate4_features, same=same, drop=[c for c in different if c != "Metadata_Well"] + same
    )
    expected = expected_pairs(plate4.assign(**{c: plate4[c].astype(str) for c in different + same}), different, same)

    assert len(expected) == n_pairs
    assert len(result) == n_pairs
    assert list(result.columns) == ["pearsons_correlation", "Metadata_Well__group0", "Metadata_Well__group1"]

    pairs = {frozenset(pair) for pair in zip(result["Metadata_Well__group0"], result["Metadata_Well__group1"])}
    assert pairs == expected

    # the correlation matrix of all the wells at once, from numpy
    wells = list(plate4["well"])
    correlations = np.corrcoef(plate4[plate4_features].to_numpy())
    expected_values = [
        correlations[wells.index(well0), wells.index(well1)]
        for well0, well1 in zip(result["Metadata_Well__group0"], result["Metadata_Well__group1"])
    ]
    assert result["pearsons_correlation"].to_numpy() == pytest.approx(expected_values, abs=1e-10)


@pytest.mark.filterwarnings("ignore:The outer group")  # the control dose has one siRNA only
def test_one_different_column_within_a_dose(plate4, plate4_features):
    result = compare_pairs(
        plate4, different=["Metadata_siRNA"], features=plate4_features, same=["Metadata_Concentration"]
    )

    assert len(result) == 135
    assert list(result.columns) == [
        "pearsons_correlation",
        "Metadata_Concentration__group0",
        "Metadata_Concentration__group1",
        "Metadata_siRNA__group0",
        "Metadata_siRNA__group1",
    ]
    assert (result["Metadata_siRNA__group0"] != result["Metadata_siRNA__group1"]).all()
    assert (result["Metadata_Concentration__group0"] == result["Metadata_Concentration__group1"]).all()
