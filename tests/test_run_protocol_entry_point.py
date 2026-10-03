# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol run and a single scan say who asked for them and in which mode.

The protocol panel's Run and Scan start through these two members, the same
calls a script makes, so the members take what the panel knows and a script
does not: the trigger the panel's strings are named by, and the engineering
mode a plugin may have flipped after the session was built. A caller that
passes neither gets the API's own trigger and the session's mode.
"""

from unittest.mock import MagicMock

import pytest

from tests.test_run_zstack_entry_point import _prepared, _runner

_CONFIG = MagicMock(image_mode='8bit', capture_depth=8, save_encoding='8bit')


def _run(runner, member, **kwargs):
    protocol = MagicMock()
    getattr(runner, member)(protocol, image_capture_config=_CONFIG, **kwargs)
    return protocol


@pytest.mark.parametrize(
    'member, api_trigger',
    [('run_protocol', 'api_protocol'), ('run_single_scan', 'api_scan')],
)
class TestTheCallerNamesItsRun:
    def test_a_script_gets_the_apis_trigger_and_the_sessions_mode(self, member, api_trigger):
        runner = _runner()
        runner.session.engineering_mode = True

        _run(runner, member)

        assert _prepared(runner)['run_trigger_source'] == api_trigger
        assert _prepared(runner)['engineering_mode'] is True

    def test_the_panel_gets_its_own_trigger_and_its_live_mode(self, member, api_trigger):
        runner = _runner()
        runner.session.engineering_mode = True

        _run(runner, member, run_trigger_source='protocol', engineering_mode=False)

        assert _prepared(runner)['run_trigger_source'] == 'protocol'
        assert _prepared(runner)['engineering_mode'] is False

    def test_the_run_is_of_the_protocol_handed_in(self, member, api_trigger):
        runner = _runner()

        protocol = _run(runner, member)

        assert _prepared(runner)['protocol'] is protocol
