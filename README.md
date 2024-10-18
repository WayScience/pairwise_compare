# pairwise_compare
This tool allows the user to compare instances (rows) of data specified in a tidy pandas dataframe with ease.
In this repo the capabilities of the PairwiseCompareManger are shown through an example using the nuclear speckle data (`example_pairwise_comparisons.ipynb`).
Although, most of the development efforts can be found in the `utils` folder.

## Limitations
Some input validation has be enforced, although not exclusively.
For example, you should not specify the same column name in both the _same_columns list and _different_columns list.

## Extensibility
This tool is designed to compare feature between any two different rows within a tidy pandas dataframe.
To include additional comparisons, you will need to introduce the functionality as a class and inherit from the Comparator class.
