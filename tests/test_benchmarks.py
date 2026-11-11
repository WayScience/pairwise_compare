"""Benchmarks of `compare_pairs`, which are skipped in the normal runs (see CONTRIBUTING.md)."""

import numpy as np
import pandas as pd
import pytest

from pairwise_compare import compare_pairs


def plates(n_wells: int, n_features: int) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic plates of 72 wells, each with 18 cell lines of 4 wells, as in `docs/speed_test.ipynb`."""

    position = np.arange(n_wells)
    features = [f"f{k}" for k in range(n_features)]
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(n_wells, n_features)), columns=features)
    df["plate"], df["line"], df["well"] = position // 72, (position % 72) // 4, position

    return df, features


@pytest.mark.parametrize(
    "different, same",
    [
        pytest.param(["well"], ["plate", "line"], id="replicates"),
        pytest.param(["line", "well"], ["plate"], id="cross-cell-line"),
    ],
)
def test_benchmark_compare_pairs(benchmark, different, same):
    """Time `compare_pairs` on 3,600 wells with 1,000 features."""
    df, features = plates(3600, 1000)

    result = benchmark(compare_pairs, df, different=different, features=features, same=same)

    assert len(result) > 0
