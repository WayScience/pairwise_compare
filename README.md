# Pairwise Compare
This tool allows the user to compare groups of data specified in a tidy-wide pandas dataframe with ease.
In this repo the capabilities of the PairwiseCompareManger are shown through an example using the nuclear speckle data (`example_pairwise_comparisons.ipynb`).
Although, most of the development efforts can be found in the `src` folder.
Users should almost exclusively interact with the PairwiseCompareManager, however, there may be rare exceptions.
If you choose to interact with another component of the tool, then there will be less input validation safeguards available.

## Data Formats
When passing arguments to the `PairwiseCompareManager` you can specify the columns that remain the same in each group-to-group comparison, and the columns that will be different in these comparisons.
These columns are parameterized by `_same_columns` and `_different_columns`, respectively.
The column values in these columns uniquely define each group.
During pairwise comparisons of groups, all of the column values of the columns specified in `_same_columns` will be the same between both groups compared for all paired combinations of groups.
Likewise, all of the column values of the columns specified in `_different_columns` will be different between both groups compared for all paired combinations of groups.

One of the following column arguments conditions must be satisified when using the `PairwiseCompareManager`:
1. `_same_columns` must include at least one list element if `_different_columns` has less than two list elements.
2. `_different_columns` must contain one or more list elements.
3. `_same_columns` and `_different_columns` should not contain any of the same columns.

## Limitations
1. Input validation is enforced in the `PairwiseCompareManager` and `PairwiseCompare` classes.
2. Additional column values are not tracked aside from columns used to compare groups (_same_columns, _different_columns).
3. Output and input python data structures are limited.
4. All of the data, in the supplied pandas dataframe, is used to compute comparisons.

## Extensibility
This tool compares features between any two groups of a tidy-wide pandas dataframe.
To incorporate additional comparators for making comparisons, you must introduce the functionality as a class and inherit from the `Comparator` class (See `Comparator.py` for details).
