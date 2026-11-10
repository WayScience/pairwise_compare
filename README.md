[![Tests](https://github.com/WayScience/pairwise_compare/actions/workflows/python-app.yml/badge.svg)](https://github.com/WayScience/pairwise_compare/actions/workflows/python-app.yml)
# Pairwise Compare
Compare the rows of a [tidy](https://tidyr.tidyverse.org/articles/tidy-data.html) pandas dataframe in pairs, for example the correlation between the profiles of wells that differ in their cell line.
You choose which columns must be different and which must be the same between the two rows of a pair, and get one result per pair.

The [example notebook](docs/example_pairwise_comparisons.ipynb) uses a dataset from the [nf1_schwann_cell_painting_data](https://github.com/WayScience/nf1_schwann_cell_painting_data).
Most of the development is in the `src` folder.

## Quick start
Take four wells of one plate, with the metadata that identify them and four measured features:

```python
import pandas as pd
from pairwise_compare import compare_pairs

wells = pd.DataFrame(
    {
        "plate": ["P1", "P1", "P1", "P1"],
        "cell_line": ["U2OS", "U2OS", "HeLa", "HeLa"],
        "well": ["A1", "A2", "B1", "B2"],
        "area": [10.0, 11.0, 20.0, 19.0],
        "intensity": [5.0, 5.5, 2.0, 2.5],
        "texture": [0.1, 0.3, 0.9, 0.7],
        "eccentricity": [0.5, 0.4, 0.8, 0.9],
    }
)

compare_pairs(
    wells,
    different=["cell_line", "well"],  # the two wells of a pair are of different cell lines
    features=["area", "intensity", "texture", "eccentricity"],
    same=["plate"],  # and are on the same plate
)
```

The result has one row per pair of wells, with their Pearson correlation across the features:

| pearsons_correlation | plate__group0 | plate__group1 | cell_line__group0 | cell_line__group1 | well__group0 | well__group1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | P1 | P1 | HeLa | U2OS | B1 | A1 |
| 0.90 | P1 | P1 | HeLa | U2OS | B1 | A2 |
| 0.92 | P1 | P1 | HeLa | U2OS | B2 | A1 |
| 0.92 | P1 | P1 | HeLa | U2OS | B2 | A2 |

Each well of one cell line is compared with each well of the other, and wells of the same cell line are not compared with each other.

## Installation
You can install the package with either of the following commands:<br>
`pip install pairwise-compare`<br>
`uv add pairwise-compare`

## Data format
The dataframe is [tidy](https://tidyr.tidyverse.org/articles/tidy-data.html): one row per sample (for example, one well) and one column per variable.
- **Group columns** hold the metadata that define which rows are compared, such as the plate, the cell line, or the well.
  They are the columns listed in `different`, `same` and `drop`.
- **Feature columns** hold the numbers that are compared, such as measurements of the cell morphology.
  They are listed in `features` and must be numeric.

All other columns are ignored.

## Which rows are compared
- Two rows are compared when they have a different value in **every** `different` column and, if `same` is given, the same value in **every** `same` column.
- `different` needs at least two columns, or one column and at least one `same` column. A column cannot be in both.
- Each pair is compared once: the pair (A, B) is not repeated as (B, A), and a row is not compared with itself.
- `group0` and `group1` in the output are the two rows of a pair. For a `same` column both have the same value.
- `drop` leaves columns of `same` or `different` out of the results. They still decide which rows are compared. For example, `drop=["Metadata_Well"]` removes the well columns and keeps the others.

## Missing values
- A row with a missing value in a `same` or `different` column is **left out** of the comparisons, with a warning that says how many rows and which columns.
- A row with a missing value in a feature column is kept, and the results of its comparisons are missing (NaN), with a warning.
- Missing values in other columns are ignored.
- A row with the same value in every feature has no Pearson correlation (NaN), with a warning.

## Other comparisons
`compare_pairs` uses the Pearson correlation by default. To compare the rows another way, pass a comparator, which is a class that inherits from `Comparator` (see [Comparator.py](src/pairwise_compare/comparators/Comparator.py) and the example in [test_comparator.py](tests/test_comparator.py)):
- `__call__(group0, group1)` compares the feature values of two groups of rows and saves the results.
- `save_groups(group_columns, **groups)` saves the values of the group columns of the two groups.

A comparator keeps the results of every comparison it has made, so use a new one for each call to `compare_pairs`.

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run the tests and make a release.
