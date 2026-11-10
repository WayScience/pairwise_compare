"""Test data: each dataset is a fixture that gives its dataframe and its feature columns together."""

import pathlib
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

DATA_DIR = pathlib.Path(__file__).parents[1] / "data"


class Dataset(NamedTuple):
    """A real dataset: its dataframe, its feature columns, and the comparisons to make with it.

    The dataframe has a unique `well` column (an identifier of each row), and every comparison is a pair of lists
    `(same, different)` of group columns, with `well` among the different columns, so that the wells in the results
    tell which pairs of rows were compared.
    """

    df: pd.DataFrame
    features: list[str]
    comparisons: list[tuple[list[str], list[str]]]


@pytest.fixture(autouse=True)
def fail_on_future_deprecations():
    """Make a pandas deprecation that will become an error in the next version fail the tests now."""

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*[Ii]n a future version.*")
        yield


@pytest.fixture
def small_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Eight wells with three random features (a fixed seed), as the dataframe and the list of its features.

    Wells W1 to W4 are on plate P1 and W5 to W8 on plate P2. Each plate has two cell lines (A and B), each with two wells.
    """

    rng = np.random.default_rng(1)
    features = ["f1", "f2", "f3"]
    df = pd.DataFrame(rng.normal(size=(8, 3)), columns=features)
    df["well"] = [f"W{number}" for number in range(1, 9)]
    df["plate"] = ["P1"] * 4 + ["P2"] * 4
    df["line"] = ["A", "A", "B", "B"] * 2

    return df, features


def nf1_plate4() -> Dataset:
    """One plate of the NF1 Schwann cell painting experiment: 60 wells, with a missing siRNA filled in."""
    df = pd.read_parquet(DATA_DIR / "Plate_4_bulk_feature_selected.parquet")
    df["Metadata_siRNA"] = df["Metadata_siRNA"].fillna("No siRNA")
    df["well"] = df["Metadata_Well"]
    features = [column for column in df.columns if not column.startswith("Metadata") and column != "well"]

    return Dataset(df, features, [(["Metadata_Concentration"], ["Metadata_siRNA", "well"]), ([], ["Metadata_Concentration", "well"])])


def pccma_bulk() -> Dataset:
    """A trimmed plate of the PCCMA bulk profiles: 8 cell lines of 4 wells, and 200 features."""
    df = pd.read_parquet(DATA_DIR / "pccma_bulk_BR00150695_trimmed.parquet")
    df["well"] = df["Metadata_Well"]
    features = [column for column in df.columns if not column.startswith("Metadata") and column != "well"]

    return Dataset(df, features, [(["Metadata_cell_line"], ["well"]), ([], ["Metadata_cell_line", "well"])])


def scikit_learn_dataset(load) -> Dataset:
    """The first 20 rows of each class of a scikit-learn dataset (its measurements as features, its class as the group)."""
    data = load(as_frame=True)
    df = data.data.copy()
    df["class"] = data.target_names[data.target]
    df = df.groupby("class").head(20).reset_index(drop=True)  # the first 20 rows of each class keep the tests fast
    df["well"] = [f"row{number}" for number in range(len(df))]

    return Dataset(df, list(data.data.columns), [(["class"], ["well"]), ([], ["class", "well"])])


def iris() -> Dataset:
    """Irises of Fisher (4 measurements, 3 species)."""
    from sklearn.datasets import load_iris

    return scikit_learn_dataset(load_iris)


def wine() -> Dataset:
    """Wines of three cultivars (13 chemical measurements)."""
    from sklearn.datasets import load_wine

    return scikit_learn_dataset(load_wine)


DATASETS = {"nf1_plate4": nf1_plate4, "pccma_bulk": pccma_bulk, "iris": iris, "wine": wine}


@pytest.fixture(scope="session", params=list(DATASETS))
def real_dataset(request) -> Dataset:
    """Each real dataset in turn, with its comparisons."""
    return DATASETS[request.param]()
