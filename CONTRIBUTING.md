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
  - `comparison_tools/PairwiseCompare.py` goes through the pairs of groups (the engine).
  - `comparators/` has the comparators (how two groups of rows are compared).
- `tests/`: the tests, which run with `pytest`.
  - `conftest.py` has the test data: a small dataframe with eight wells, and the plate 4 data in `data/`.
  - `helpers.py` has simple, slow ways to get the answers that the package computes, such as every pair of rows that should be compared.
- `docs/`: an example notebook.
  Install its extra packages with `uv sync --group docs`.

## Tests
Test a change with the small dataframe, and compare results with numpy or with the helpers instead of with numbers copied from the output.
A new argument or message needs a case in `tests/test_validation.py` or `tests/test_missing_values.py` that checks the message.
The tests run with coverage (`uv run pytest --cov`), and the lines of the package should stay covered.

## Adding a comparator
A comparator is a class that inherits from `Comparator` (`comparators/Comparator.py`) and has two methods:
- `__call__(group0, group1)` compares the feature values of two groups of rows and saves the results.
- `save_groups(group_columns, **groups)` saves the values of the group columns of the two groups.

Users pass an instance of it to the entry point, as in `compare_pairs(df, different=[...], features=[...], comparator=MyComparator())`.
`tests/test_comparator.py` has a small example.

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
