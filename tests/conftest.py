"""Shared test fixtures."""

import pytest

from beampilot.session import MeasurementSession


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "allow_internal_error: the test provokes an internal_error"
    )


@pytest.fixture(autouse=True)
def no_internal_errors(request, monkeypatch):
    """Fail any test in which a tool call returned ``internal_error``.

    Tools catch every exception so a demo never crashes; this keeps such
    exceptions from hiding bugs in the test suite.
    """
    sessions: list[MeasurementSession] = []
    original_init = MeasurementSession.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        sessions.append(self)

    monkeypatch.setattr(MeasurementSession, "__init__", tracking_init)
    yield
    if request.node.get_closest_marker("allow_internal_error"):
        return
    errors = [
        entry
        for session in sessions
        for entry in session.log
        if "internal_error" in entry["result"]
    ]
    assert not errors, errors[0].get("traceback", errors[0])
