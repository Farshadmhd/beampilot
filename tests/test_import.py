"""Basic package sanity tests."""

import beampilot


def test_package_imports() -> None:
    """The package imports and exposes a version string."""
    assert isinstance(beampilot.__version__, str)
    assert beampilot.__version__
