# Contributing

Thanks for helping to improve Pairwise Compare!
This guide covers development and releases.
For using the package, see the [README](README.md).

## Set up
Install [uv](https://docs.astral.sh/uv/), clone the repository, and run the tests.
uv creates the environment from `uv.lock` when it runs a command:

```bash
uv run pytest
```

The continuous integration runs the tests on Python 3.10 to 3.13 for every pull request.

## Dependencies and the lock file
Dependencies are in `pyproject.toml`.
The runtime dependencies are in `dependencies`, and the tools for development are in the `dev` and `docs` groups of `[dependency-groups]`.
After changing them, update the lock file with `uv lock` and commit `uv.lock`.
The continuous integration installs exactly what the lock file lists (`--locked`), so it fails when the lock file is out of date.

## Layout
- `src/pairwise_compare/`: the package.
  - `compare_pairs` is the function that users call.
    It is in `comparison_tools/PairwiseCompareManager.py`, where it checks the arguments and starts the comparisons.
  - `comparison_tools/PairwiseCompare.py` finds the pairs of rows, puts them in order, and gives them to the comparator (the engine).
  - `comparators/` has the comparators (how the two rows of each pair are compared).
- `tests/`: the tests, which run with `pytest`.
  - `conftest.py` has the test data as fixtures (see the datasets below).
  - `helpers.py` has simple, slow ways to get the answers that the package computes, such as every pair of rows that should be compared, in the order of the results.
- `docs/`: an example notebook and a speed test notebook.
  Install their extra packages with `uv sync --group docs`.

## Tests
Test a change with the small dataframe, and compare results with numpy or with the helpers instead of with numbers copied from the output.
A new argument or message needs a case in `tests/test_validation.py` or `tests/test_missing_values.py` that checks the message.
The tests run with coverage (`uv run pytest --cov`), and the lines of the package should stay covered: the checks fail below 100% (`fail_under` in `pyproject.toml`).
After a merge into `main`, the workflow `tests.yml` uploads the coverage to [Codecov](https://codecov.io/gh/WayScience/pairwise_compare), which the badge of the README shows.
The upload needs the repository secret `CODECOV_TOKEN`, which an administrator of the organization gets from Codecov after enabling its GitHub app for this repository.

### Datasets
Each dataset is a fixture that gives the dataframe and the list of its feature columns together, as in `df, features = small_data`.
- `small_data` is eight wells with three random features, which are small enough to check by hand.
- `real_dataset` runs a test on four real datasets in turn: plate 4 of the NF1 Schwann cell painting data, a trimmed plate of the PCCMA bulk profiles, and the first 20 rows of each class of the iris and wine datasets of scikit-learn.
  Every dataset has a unique `well` column and a few comparisons to make.
- The files in `data/` are described in `data/README.md`.

To test another real dataset, add a function that returns a `Dataset` to `tests/conftest.py` and list it in `DATASETS`.

### Property-based test
One test uses [Hypothesis](https://hypothesis.readthedocs.io/) to make random dataframes (with missing values) and random valid `same`, `different` and `drop` columns.
It compares every result with the slow checks of `tests/helpers.py` and with numpy, including the order of the pairs and which row of a pair comes first.
The test runs 100 examples with a fixed seed (`derandomize=True`), so a failure is reproducible, and Hypothesis shows the smallest dataframe that fails.

### Mutation testing
Mutation testing changes the code of the package in small ways (for example, `!=` to `==`) and runs the tests for each change.
A change that no test notices (a surviving mutant) shows a gap in the tests.
Run it locally, which takes about a minute:

```bash
uv run --with mutmut mutmut run
```

Then list the survivors with `uv run --with mutmut mutmut results`, and see the change of one with `uv run --with mutmut mutmut show <name>`.
Many survivors are harmless (the wording of a message, or the default warning category), so look for the ones that change a result.
Mutation testing is not part of the checks of a pull request, because it is slow and has no pass or fail.
The workflow `.github/workflows/mutation.yml` runs it every week (Mondays), and on demand from the Actions tab.
It lists the surviving mutants in the summary of the run and in an artifact, and it does not fail when mutants survive.

### Benchmarks
`tests/test_benchmarks.py` times `compare_pairs` on 3,600 wells with 1,000 features (replicates, and wells of different cell lines) with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).
The normal test runs skip them (`--benchmark-skip` in `pyproject.toml`), so they do not slow down the checks.
Run them with:

```bash
uv run pytest -o addopts= --benchmark-only
```

After a merge into `main` (and on demand from the Actions tab), the workflow `.github/workflows/benchmark.yml` runs them and compares the times with the previous run using [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).
The times of a shared runner vary, so a run that is more than twice as slow only shows an alert in the summary of the run and does not fail it.
`docs/speed_test.ipynb` shows the speed for other sizes.

## Adding a comparator
A comparator is a class that inherits from `Comparator` (`comparators/Comparator.py`).
Its constructor gives the name of the column of the results, and it has two methods:
- `prepare(values)` is called once with the feature values of all the compared rows, and returns what `compare` needs.
  It is optional, and by default it returns the values as they are.
  Do the work that is the same for every pair here, such as normalizing each row.
- `compare(prepared, rows0, rows1)` returns one value for each pair of rows `(rows0[k], rows1[k])`.
  It should compare all the pairs with array operations, and not loop over them.

A comparator keeps no results, so one comparator can be used for any number of calls.
A comparator must be symmetric, because the two rows of a pair are in an arbitrary order.
Users pass an instance of it, or a list of instances, to the entry point, as in `compare_pairs(df, different=[...], features=[...], comparator=MyComparator())`.
`tests/test_comparator.py` has small examples.

### Adding a metric
The metrics are in `comparators/`, and each one is a comparator with a default name for its column.
- A metric that is the dot product of two prepared rows (`PearsonsCorrelation`, `SpearmanCorrelation` and `CosineSimilarity`) inherits from `DotProductComparator`, and only gives `prepare`.
  `DotProductComparator` takes the dot products with matrix products, or pair by pair when few pairs are needed.
- A distance (`EuclideanDistance` and `ManhattanDistance`) inherits from `PairwiseDistance` in `Distances.py`, and only gives `distance` of the differences of the features.
  It compares a chunk of pairs at a time, which keeps the memory use low.
- Export the comparator in `src/pairwise_compare/__init__.py`, and add it to the table of the README.
- Add a row to `METRICS` in `tests/test_metrics.py`, with the same metric from scipy as the oracle, the default name of the column, and the warning that it gives for a row that has no value for it.
  That row runs the metric against scipy for every pair, with ties, copies, rows of the same value, rows of zeros and a missing value.

## Versions and releases
The version is not written in any file.
It comes from the git tag of the release (with [setuptools-scm](https://setuptools-scm.readthedocs.io/)), and `pairwise_compare.__version__` reads it from the installed package.
A build from a commit after a tag has a development version, such as `0.2.1.dev3+g1a2b3c4`.

Releases are drafted automatically:
1. Each merge into `main` updates a draft GitHub release with the titles of the merged pull requests (Release Drafter, `.github/workflows/release-drafter.yml`).
2. The next version is a patch increase of the latest release, unless a merged pull request has a label that asks for more.
   Use `release-minor` for new features, and `release-major` for changes that break existing code.
   Add the label to the pull request before merging it.
3. To release, review the draft on GitHub (the notes and the version) and publish it.
   Publishing creates the tag.
   The workflow `.github/workflows/publish.yml` then builds the package and publishes it to PyPI with [trusted publishing](https://docs.pypi.org/trusted-publishers/) (environment `release`, no tokens or passwords).
