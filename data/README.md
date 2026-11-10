# Data of the tests

These small datasets are real measurements, and the tests compare the package against numpy on them.

## `Plate_4_bulk_feature_selected.parquet`
One plate of the NF1 Schwann cell painting experiment (60 wells, 1,182 columns).
It comes from [nf1_schwann_cell_painting_data](https://github.com/WayScience/nf1_schwann_cell_painting_data), and it is also used by the example notebook in `docs/`.
The columns that start with `Metadata_` describe the wells (the plate, well, siRNA, genotype, and concentration), and the other columns are the features.

## `pccma_bulk_BR00150695_trimmed.parquet`
A trimmed part of one plate (`BR00150695`) of the pediatric cancer cell morphology atlas (PCCMA) of the Way lab.
The profiles are bulk profiles (the mean of the single cells of a well), normalized to the plate and feature selected.
It has 32 wells (8 of the 13 cell lines of the plate, with 4 wells each) and the first 200 features of the plate.
It is an unpublished subset, shared by the project owner for the tests.

The columns that start with `Metadata_` are the plate, well, cell line, row, column, seeding density, time point, plate barcode, plate coating, and PFA fixation of the well.
All the other columns are features, with the CellProfiler names of the measurements.
