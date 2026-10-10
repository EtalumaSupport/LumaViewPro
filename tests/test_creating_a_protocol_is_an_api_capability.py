# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Creating a protocol is an API capability, and a click that builds nothing is refused there.

The GUI's New handler used to compose the build from two Session calls and
then decide, with its own channel check and its own popup, that an empty
result meant no channel was set to acquire. The Session now owns the
build: it refuses through the protocols API's one funnel, with the reason
Add Step already uses, and the panel displays what comes back. A labware
with no wells still gives an empty protocol, as it always did.
"""

from __future__ import annotations

import pytest

import modules.config_helpers as config_helpers
from modules.exceptions import ProtocolRunRefusedError
from modules.protocol import Protocol
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture
from tests.test_run_refusal_contract import _capture_notifications


def _set_acquire(session, **acquire_by_layer):
    for layer in config_helpers.get_layer_configs(session.settings):
        session.settings[layer]['acquire'] = acquire_by_layer.get(layer)


class TestTheSessionBuilds:
    def test_no_acquiring_layer_is_refused_once_and_builds_nothing(self, session, centre_posts):
        _set_acquire(session)
        captured = _capture_notifications(centre_posts)

        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            session.new_protocol()

        assert excinfo.value.reason == 'no_acquiring_layer'
        assert len(captured) == 1

    def test_an_acquiring_layer_gives_a_step_per_well(self, session):
        _set_acquire(session, BF='image')

        protocol = session.new_protocol()

        assert protocol.num_steps() > 0
        assert set(protocol.steps()['Color']) == {'BF'}

    def test_a_labware_with_no_wells_gives_an_empty_protocol(self, session):
        _set_acquire(session, BF='image')
        session.select_labware('Blank')

        protocol = session.new_protocol()

        assert protocol.num_steps() == 0


class TestOnePredicate:
    @pytest.mark.parametrize('acquire', [None, '', 'none', 'off'])
    def test_a_layer_not_set_to_image_or_video_does_not_acquire(self, acquire):
        assert Protocol.layer_acquires({'acquire': acquire}) is False

    @pytest.mark.parametrize('acquire', ['image', 'video'])
    def test_image_and_video_acquire(self, acquire):
        assert Protocol.layer_acquires({'acquire': acquire}) is True
