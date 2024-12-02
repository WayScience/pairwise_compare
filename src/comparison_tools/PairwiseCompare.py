import warnings
from collections.abc import Iterable
from itertools import combinations, product
from typing import Any, Optional, Union

import pandas as pd

from comparators.Comparator import Comparator


class PairwiseCompare:
    """
    Compute inter or intra group comparisons between objects.
    """

    def __init__(
        self,
        _df: pd.DataFrame,
        _comparator: Comparator,
        _antehoc_group_cols: list[str],
        _posthoc_group_cols: list[str],
        _feat_cols: list[str],
        _one_different_comparison: bool = False,
        _drop_cols: Optional[list[str]] = None,
    ):
        """
        Perform Input validation and initializations.

        Parameters
        ----------
        _df:
            Contains the features and group columns to use for comparing paired data.

        _comparator:
            Defines how to perform and save comparisons.

        _antehoc_group_cols:
            Columns to groupby before organizing by _posthoc_group_cols.

        _feat_cols:
            Feature columns for creating comparisons.

        _posthoc_group_cols:
            Columns to groupby after organizing by _antehoc_group_cols.

        _one_different_comparison:
            Type of comparison. If True, only one column value needs to be different between two groups for both group to be considered different.
            Otherwise, all column values need to be different for both groups to be considered different.

        _drop_cols:
            Columns to not save in the final output.
        """

        if not isinstance(_df, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if not isinstance(_comparator, Comparator):
            raise TypeError("Expected type Comparator.")

        self.__is_iterable_with_strings(_antehoc_group_cols)
        self.__is_iterable_with_strings(_posthoc_group_cols)
        self.__is_iterable_with_strings(_feat_cols)

        if _df.empty:
            raise ValueError("DataFrame is empty")

        missing_cols = (
            set(_antehoc_group_cols) | set(_posthoc_group_cols) | set(_feat_cols)
        ) - set(_df.columns)

        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols}")

        if any(not pd.api.types.is_numeric_dtype(_df[col].dtype) for col in _feat_cols):
            raise TypeError(
                "At least one of the feature columns is not a numerical data type."
            )

        if _df.isna().any().any():
            warnings.warn("DataFrame contains NaNs")

        if not _drop_cols:
            self.__drop_cols = []

        else:
            self.__is_iterable_with_strings(_drop_cols)
            self.__drop_cols = _drop_cols

        self.__df = _df
        self.__comparator = _comparator
        self.__feat_cols = _feat_cols
        self.__one_different_comparison = _one_different_comparison

        group_suffixes = ("_group0", "_group1")

        self.__posthoc_group_names = [f"posthoc{suffix}" for suffix in group_suffixes]
        self.__antehoc_group_names = [f"antehoc{suffix}" for suffix in group_suffixes]

        self.__antehoc_group_cols = _antehoc_group_cols
        self.__posthoc_group_cols = _posthoc_group_cols

        self.__filtered_antehoc_col_idx = self.__get_group_column_idxs(
            _group_columns=self.__antehoc_group_cols,
        )

        self.__filtered_posthoc_col_idx = self.__get_group_column_idxs(
            _group_columns=self.__posthoc_group_cols,
        )

    def __warn_empty_comparisons(self, _comparison_type_name):

        warnings.warn(f"{_comparison_type_name} were empty", UserWarning)

    def __is_iterable_with_strings(self, _data_structure):

        prefix_msg = "Expected an Iterable of Strings."

        if isinstance(_data_structure, str):
            raise TypeError(f"{prefix_msg} Data is of type String.")

        elif not isinstance(_data_structure, Iterable):
            raise TypeError(f"{prefix_msg} Data is not an Iterable.")

        elif isinstance(_data_structure, Iterable):
            if any(not isinstance(element, str) for element in _data_structure):
                raise TypeError(f"{prefix_msg} Data in Iterable is not of type String.")

    def __get_group_column_idxs(self, _group_columns):
        """Get group fields after removing dropped columns."""

        return [
            col_idx
            for col_idx, group_col in enumerate(_group_columns)
            if group_col in self.__drop_cols
        ]

    def __get_group_column_element(
        self, _group_column_data: tuple[str], _group_column_idxs: list
    ):
        """Get the corresponding group column element from the index"""

        if _group_column_idxs:
            return tuple(_group_column_data[idx] for idx in _group_column_idxs)

        else:
            return _group_column_data

    def __contains_match(self, _groups):
        """Check if the same features between both groups are the same value."""

        if not self.__one_different_comparison:
            if len(self.__posthoc_group_cols) == 1:
                if _groups[0] == _groups[1]:
                    return True

            else:
                if any(_groups[0][i] == _groups[1][i] for i in range(len(_groups[0]))):
                    return True

        return False

    def inter_comparisons(self):
        """
        Computes comparisons between two pandas rows using both post hoc and ante groups.
        This is accomplished by computing comparisons between these rows in different ante groups (inter comparisons) and between all possible post hoc comparisons between those groups.
        """

        groupdf = self.__df.groupby(self.__antehoc_group_cols)

        # Retrieve keys for the ante group
        gkeys = groupdf.groups.keys()

        # Find all possible combinations of size 2 ante group keys
        apairs = list(combinations(gkeys, 2))

        # Iterate through each ante group combination
        for apair in apairs:

            if self.__contains_match(apair):
                continue

            apair = tuple(
                [(item,) if not isinstance(item, tuple) else item for item in apair[:2]]
            )

            apair0, apair1 = apair

            # Extract the keys for the first post hoc group
            group0df = groupdf.get_group(apair0).copy()
            group0df = group0df.groupby(self.__posthoc_group_cols)[self.__feat_cols]
            group0_keys = group0df.groups.keys()

            # Extract the keys for the second post hoc group
            group1df = groupdf.get_group(apair1).copy()
            group1df = group1df.groupby(self.__posthoc_group_cols)[self.__feat_cols]
            group1_keys = group1df.groups.keys()

            comparison_key_product = list(product(group0_keys, group1_keys))

            if not comparison_key_product:
                self.__warn_empty_comparisons(_comparison_type_name="Inter Comparisons")
                continue

            if len(self.__filtered_antehoc_col_idx) < len(self.__antehoc_group_cols):
                filtered_apair = tuple(
                    self.__get_group_column_element(
                        apair[group_idx], self.__filtered_antehoc_col_idx
                    )
                    for group_idx in range(2)
                )

            # Iterate through each well group cartesian product and save the data
            for ppair in comparison_key_product:

                if self.__contains_match(ppair):
                    continue

                ppair = tuple(
                    [
                        (item,) if not isinstance(item, tuple) else item
                        for item in ppair[:2]
                    ]
                )

                ppair0, ppair1 = ppair

                self.__comparator(
                    group0df.get_group(ppair0), group1df.get_group(ppair1)
                )

                if len(self.__filtered_antehoc_col_idx) < len(
                    self.__antehoc_group_cols
                ):
                    self.__comparator.save_groups(
                        self.__get_group_column_element(
                            self.__antehoc_group_cols, self.__filtered_antehoc_col_idx
                        ),
                        **dict(zip(self.__antehoc_group_names, filtered_apair)),
                    )

                if len(self.__filtered_posthoc_col_idx) < len(
                    self.__posthoc_group_cols
                ):
                    filtered_ppair = tuple(
                        self.__get_group_column_element(
                            ppair[group_idx], self.__filtered_posthoc_col_idx
                        )
                        for group_idx in range(2)
                    )

                    self.__comparator.save_groups(
                        self.__get_group_column_element(
                            self.__posthoc_group_cols, self.__filtered_posthoc_col_idx
                        ),
                        **dict(zip(self.__posthoc_group_names, filtered_ppair)),
                    )

    def intra_comparisons(self):
        """
        Computes comparisons between two pandas rows using both post hoc and ante groups.
        This is accomplished by computing comparisons between these rows only in the same ante group (intra comparisons), but between different post hoc groups.
        """

        groupdf = self.__df.groupby(self.__antehoc_group_cols)

        # Retrieve keys for the ante group
        akeys = groupdf.groups.keys()

        # Iterate through each ante group combination
        for agroup in akeys:

            # Avoids a future deprecation in the pandas get_group method
            if not isinstance(agroup, tuple):
                agroup = (agroup,)

            # Extract keys for poshoc group columns
            group = groupdf.get_group(agroup).copy()
            group = group.groupby(self.__posthoc_group_cols)[self.__feat_cols]
            group_keys = group.groups.keys()

            comparison_key_combinations = list(combinations(group_keys, 2))

            if not comparison_key_combinations:
                self.__warn_empty_comparisons(_comparison_type_name="Intra Comparisons")
                continue

            if len(self.__filtered_antehoc_col_idx) < len(self.__antehoc_group_cols):
                filtered_agroup = self.__get_group_column_element(
                    agroup, self.__filtered_antehoc_col_idx
                )

            # Iterate through the combinations pairs of the groups
            for ppair in comparison_key_combinations:

                if self.__contains_match(ppair):
                    continue

                ppair = tuple(
                    [
                        (item,) if not isinstance(item, tuple) else item
                        for item in ppair[:2]
                    ]
                )

                ppair0, ppair1 = ppair

                self.__comparator(group.get_group(ppair0), group.get_group(ppair1))

                if len(self.__filtered_antehoc_col_idx) < len(
                    self.__antehoc_group_cols
                ):
                    self.__comparator.save_groups(
                        self.__get_group_column_element(
                            self.__antehoc_group_cols, self.__filtered_antehoc_col_idx
                        ),
                        **dict(
                            zip(
                                self.__antehoc_group_names,
                                (filtered_agroup, filtered_agroup),
                            )
                        ),
                    )

                if len(self.__filtered_posthoc_col_idx) < len(
                    self.__posthoc_group_cols
                ):
                    filtered_ppair = tuple(
                        self.__get_group_column_element(
                            ppair[group_idx], self.__filtered_posthoc_col_idx
                        )
                        for group_idx in range(2)
                    )

                    self.__comparator.save_groups(
                        self.__get_group_column_element(
                            self.__posthoc_group_cols, self.__filtered_posthoc_col_idx
                        ),
                        **dict(zip(self.__posthoc_group_names, filtered_ppair)),
                    )
