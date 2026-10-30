# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What production logs during a test is in that test's captured log, at the level the test set.

The suite's ``lvp_logger`` stand-in once handed production a MagicMock for
its loggers, so a test that went red said nothing of why: a capture's
give-up line and a run's strike line were dropped. The stand-in now hands
out the real loggers under their production names, and the session's
process-wide log settings are put back after each test, so what a red's
captured log holds does not depend on what ran before it on the worker.
"""

import logging

import pytest

import lvp_logger
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# One worker, in file order: the second test reads what the first left.
pytestmark = pytest.mark.xdist_group('log_settings_restore')

_PRODUCTION_LOGGERS = {
    'logger': 'lvp_logger',
    'camera_logger': 'LVP.camera',
    'protocol_logger': 'LVP.protocol',
    'metrics_logger': 'LVP.metrics',
}


def test_a_line_through_each_production_logger_reaches_the_tests_captured_log(caplog):
    for attribute in _PRODUCTION_LOGGERS:
        getattr(lvp_logger, attribute).warning(f'the {attribute} line')

    captured = {(r.name, r.getMessage()) for r in caplog.records}
    missing = {
        name
        for attribute, name in _PRODUCTION_LOGGERS.items()
        if (name, f'the {attribute} line') not in captured
    }
    assert missing == set()


def test_a_session_at_debug_sets_the_main_log_to_debug_for_its_own_test():
    session = ScopeSession.create(complete_settings(debug_mode=True), simulate=True)
    try:
        assert logging.getLogger('lvp_logger').level == logging.DEBUG
    finally:
        session.shutdown()


def test_the_next_test_starts_at_the_default_level():
    # After the test above on this worker: its session's DEBUG is gone.
    assert (logging.getLogger('LVP').level, logging.getLogger('lvp_logger').level) == (
        logging.NOTSET,
        logging.NOTSET,
    )
