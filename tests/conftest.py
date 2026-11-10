"""Shared test data."""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

DATA_PATH = pathlib.Path(__file__).parents[1] / "data" / "Plate_4_bulk_feature_selected.parquet"


@pytest.fixture(autouse=True)
def fail_on_future_deprecations():
    """Make a pandas deprecation that will become an error in the next version fail the tests now."""

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*[Ii]n a future version.*")
        yield


@pytest.fixture
def small_df() -> pd.DataFrame:
    """
    Eight wells with three random features (a fixed seed).

    Wells W1 to W4 are on plate P1 and W5 to W8 on plate P2. Each plate has two cell lines (A and B), each with two wells.
    """

    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(8, 3)), columns=["f1", "f2", "f3"])
    df["well"] = [f"W{number}" for number in range(1, 9)]
    df["plate"] = ["P1"] * 4 + ["P2"] * 4
    df["line"] = ["A", "A", "B", "B"] * 2

    return df


@pytest.fixture(scope="session")
def plate4() -> pd.DataFrame:
    """The 60 wells of plate 4 (NF1 Schwann cells, 1,182 columns), with a missing siRNA filled in."""

    df = pd.read_parquet(DATA_PATH)
    df["Metadata_siRNA"] = df["Metadata_siRNA"].fillna("No siRNA")
    df["well"] = df["Metadata_Well"]

    return df


@pytest.fixture(scope="session")
def plate4_features(plate4) -> list[str]:
    """The feature columns of the plate 4 data (all columns that are not metadata)."""
    return [column for column in plate4.columns if not column.startswith("Metadata") and column != "well"]
