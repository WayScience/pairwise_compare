import pathlib
import sys

import pandas as pd
import pytest

from comparators.PearsonsCorrelation import PearsonsCorrelation
from comparison_tools.PairwiseCompareManager import PairwiseCompareManager

# Paths to original nuclear speckle data
data_dir = pathlib.Path(
    "data"
).resolve(strict=True)

plate4df = pd.read_parquet(data_dir / "Plate_4_bulk_feature_selected.parquet")

plate4df["Metadata_siRNA"] = plate4df["Metadata_siRNA"].fillna("No siRNA")

feat_cols = plate4df.columns[~plate4df.columns.str.contains("Metadata")].tolist()


# Specify tests cases
@pytest.mark.parametrize(
    "case",
    [
        {
            "_comparator": PearsonsCorrelation(),
            "_same_columns": ["Metadata_Concentration"],
            "_different_columns": None,
            "_feat_cols": feat_cols,
            "_drop_cols": ["Metadata_Concentration", "Metadata_Well"],
        },
        {
            "_comparator": PearsonsCorrelation(),
            "_same_columns": None,
            "_different_columns": ["Metadata_siRNA"],
            "_feat_cols": feat_cols,
            "_drop_cols": None,
        },
        {
            "_comparator": PearsonsCorrelation(),
            "_same_columns": ["Metadata_Concentration"],
            "_different_columns": ["Metadata_siRNA"],
            "_feat_cols": None,
            "_drop_cols": None,
        },
        {
            "_comparator": PearsonsCorrelation(),
            "_same_columns": ["Metadata_Concentration"],
            "_different_columns": ["Metadata_siRNA", "Metadata_Concentration"],
            "_feat_cols": feat_cols,
            "_drop_cols": None,
        },
    ],
)
def test_dataframe_shape_correct(case):
    # Pass the test if an exception is raised
    with pytest.raises(Exception):
        comparer = PairwiseCompareManager(
            _df=plate4df.copy(),
            _comparator=case["_comparator"],
            _same_columns=case["_same_columns"],
            _different_columns=case["_different_columns"],
            _feat_cols=case["_feat_cols"],
            _drop_cols=case["_drop_cols"],
        )
