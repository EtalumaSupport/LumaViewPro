"""A crash while the microscope settings panel loads is recorded once.

The panel's load wrapped its whole body in three clauses. Two named failures
nothing in the body can raise -- the settings file was parsed before the
panel existed -- so a FileNotFoundError from anywhere else was misattributed
to the settings file and swallowed, and the build went on with a half-filled
panel. The third logged the traceback and re-raised into the app's build,
where the process's exception hook logs the same traceback again.
"""

import logging
import types

import pytest

import modules.app_context as app_context


def test_a_raise_in_the_load_reaches_the_caller_and_the_panel_logs_nothing(monkeypatch, caplog):
    from ui.microscope_settings import MicroscopeSettings

    # A store without the profiling section: the body's first read raises.
    monkeypatch.setattr(
        app_context, 'ctx', types.SimpleNamespace(lumaview=None, settings={}), raising=False
    )

    with (
        caplog.at_level(logging.DEBUG, logger='LVP.ui.microscope_settings'),
        pytest.raises(KeyError, match='profiling'),
    ):
        MicroscopeSettings.load_settings(types.SimpleNamespace())

    panel_lines = [
        r
        for r in caplog.records
        if r.name == 'LVP.ui.microscope_settings' and r.levelno >= logging.WARNING
    ]
    assert panel_lines == []
