"""Compare groups of data specified in a tidy pandas dataframe."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pairwise-compare")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
