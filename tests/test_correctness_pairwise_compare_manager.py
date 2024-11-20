import pathlib
import sys

import pandas as pd
import pytest

from comparators.PearsonsCorrelation import PearsonsCorrelation
from comparison_tools.PairwiseCompareManager import PairwiseCompareManager

# Paths to original nuclear speckle data
data_dir = pathlib.Path(
    "data/nf1_schwann_cell_painting_data/3.processing_features/data/bulk_profiles"
).resolve(strict=True)

plate4df = pd.read_parquet(data_dir / "Plate_4_bulk_feature_selected.parquet")

plate4df["Metadata_siRNA"] = plate4df["Metadata_siRNA"].fillna("No siRNA")

feat_cols = plate4df.columns[~plate4df.columns.str.contains("Metadata")].tolist()


# Compute comparison outputs
@pytest.fixture
def test_data(request):

    case = request.param

    comparer = PairwiseCompareManager(
        _df=plate4df.copy(),
        _comparator=case["_comparator"],
        _same_columns=case["_same_columns"],
        _different_columns=case["_different_columns"],
        _feat_cols=feat_cols,
        _drop_cols=case["_drop_cols"],
    )

    case["_comparer"] = comparer()

    return case


# Specify test cases
@pytest.mark.parametrize(
    "test_data",
    [
        (
            {
                "_comparator": PearsonsCorrelation(),
                "_same_columns": ["Metadata_Concentration"],
                "_different_columns": ["Metadata_siRNA", "Metadata_Well"],
                "_drop_cols": ["Metadata_Concentration", "Metadata_Well"],
            }
        ),
        (
            {
                "_comparator": PearsonsCorrelation(),
                "_same_columns": None,
                "_different_columns": ["Metadata_siRNA", "Metadata_Well"],
                "_drop_cols": None,
            }
        ),
        (
            {
                "_comparator": PearsonsCorrelation(),
                "_same_columns": ["Metadata_Concentration"],
                "_different_columns": ["Metadata_siRNA"],
                "_drop_cols": ["Metadata_Concentration"],
            }
        ),
        (
            {
                "_comparator": PearsonsCorrelation(),
                "_same_columns": ["Metadata_Concentration"],
                "_different_columns": ["Metadata_siRNA"],
                "_drop_cols": None,
            }
        ),
    ],
    indirect=True,
)
def test_dataframe_shape_correct(test_data: dict):
    """Tests if the output dataframe contains the correct number of rows and columns."""

    total_number_of_comparisons = 0

    def number_of_comparisons(_df: pd.DataFrame):
        """Calculate the number of expected comparisons."""

        number_of_comparisons = 0

        numberdf_samples = _df.shape[0]

        for first_row in range(numberdf_samples - 1):
            for second_row in range(first_row + 1, numberdf_samples):
                groupsdf = _df.iloc[[first_row, second_row]].copy()
                if (
                    not groupsdf[test_data["_different_columns"]]
                    .apply(lambda col: col.duplicated(keep=False))
                    .any()
                    .any()
                ):
                    number_of_comparisons += 1

        return number_of_comparisons

    # Calculate the expected number of comparisons if _same_columns is specified
    if test_data["_same_columns"]:
        for _, urow in plate4df.drop_duplicates(
            subset=test_data["_same_columns"]
        ).iterrows():
            same_df = plate4df[
                (
                    plate4df[test_data["_same_columns"]]
                    == urow[test_data["_same_columns"]]
                ).all(axis=1)
            ]
            total_number_of_comparisons += number_of_comparisons(same_df)

    else:
        total_number_of_comparisons += number_of_comparisons(plate4df)

    assert test_data["_comparer"].shape[0] == total_number_of_comparisons

    if not test_data["_drop_cols"]:
        number_drop_cols = 0

    else:
        number_drop_cols = len(test_data["_drop_cols"])

    if not test_data["_same_columns"]:
        number_same_columns = 0

    else:
        number_same_columns = len(test_data["_same_columns"])

    # There are 2 of each column, but only one comparison column
    assert (
        2
        * (
            number_same_columns
            + len(test_data["_different_columns"])
            - number_drop_cols
        )
        == test_data["_comparer"].shape[1] - 1
    )
