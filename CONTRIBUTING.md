# Contributing

Thanks for helping to improve Pairwise Compare! This guide covers development and releases. For using the package, see the [README](README.md).

## Set up
Install [uv](https://docs.astral.sh/uv/), clone the repository, and run the tests. uv creates the environment from `uv.lock` when it runs a command:

```bash
uv run pytest
```

The continuous integration runs the tests on Python 3.10 to 3.13 for every pull request.

## Dependencies and the lock file
Dependencies are in `pyproject.toml`: the runtime dependencies in `dependencies`, and the tools for development in the `dev` and `docs` groups of `[dependency-groups]`.
After changing them, update the lock file with `uv lock` and commit `uv.lock`.
The continuous integration installs exactly what the lock file lists (`--locked`), so it fails when the lock file is out of date.

## Layout
- `src/pairwise_compare/`: the package. `comparators/` has the comparators (how two groups of rows are compared) and `comparison_tools/` has the code that goes through the pairs of groups.
- `tests/`: the tests, which run with `pytest`.
- `docs/`: an example notebook. Install its extra packages with `uv sync --group docs`.

To add a comparator, inherit from `Comparator` (see the README).

## Versions and releases
The version is not written in any file. It comes from the git tag of the release (with [setuptools-scm](https://setuptools-scm.readthedocs.io/)), and `pairwise_compare.__version__` reads it from the installed package.
A build from a commit after a tag has a development version, such as `0.2.1.dev3+g1a2b3c4`.

Releases are drafted automatically:
1. Each merge into `main` updates a draft GitHub release with the titles of the merged pull requests (Release Drafter, `.github/workflows/release-drafter.yml`).
2. The next version is a patch increase of the latest release, unless a merged pull request has a label that asks for more: `release-minor` (for example, new features) or `release-major` (changes that break existing code).
   Add the label to the pull request before merging it.
3. To release, review the draft on GitHub (the notes and the version) and publish it. Publishing creates the tag, and the workflow `.github/workflows/publish.yml` builds the package and publishes it to PyPI with [trusted publishing](https://docs.pypi.org/trusted-publishers/) (environment `release`, no tokens or passwords).
