# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A ``ScopeSession.create`` that raises leaves nothing it started running.

``create``'s contract: every host builds its session here, and a factory
that raises hands its caller the fault and nothing else -- the scope it
built is disconnected (on a real scope that releases the instance lock,
so the next ``create`` is not refused ``scope_in_use``), every thread it
started has ended, and no settings file was written ("a factory that
raises writes no file"). It holds for every step ``create`` runs, so each
step below is made to raise in turn and the same three things are asked.

The executor bundle's own factory is held to the same contract: a start
that raises stops the starts before it.
"""

from __future__ import annotations

import threading

import pytest

from _lumascope.modules import executor_registry
from _lumascope.modules.exceptions import HardwareCommandRefusedError
from _lumascope.modules.frame_validity import FrameValidity
from _lumascope.modules.lumascope_api import Lumascope
from _lumascope.modules.lumascope_api._lumascope import scope_lane_closed
from _lumascope.modules.lumascope_api.imaging import ImagingAPI
from _lumascope.modules.lumascope_api.simulation import (
    SimulatedModelAnswer,
    SimulatedScope,
    SimulatedStall,
    SimulationAPI,
)
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.modules.sequential_io_executor import SequentialIOExecutor
from _lumascope.modules.settings_saver import SettingsSaver
from tests.settings_fixtures import complete_settings

# How long a stopped thread may take to end: each joins its owner's stop.
_ENDED_WITHIN_S = 5.0


class _StepFaultError(Exception):
    """The fault a step raises, so the caller's exception is known to be it."""


def _raise(*_args, **_kwargs):
    raise _StepFaultError()


# Each step of create, in its order, named by the member a raise from it
# comes out of, with the SimulatedScope that makes create run the step.
# A renamed member fails the setattr below rather than passing the test.
_STEPS = (
    ('the scope constructor', FrameValidity, 'set_settle_check', None),
    (
        'the simulated injections',
        SimulationAPI,
        'answer_model',
        SimulatedScope(model_answer=SimulatedModelAnswer('LS850T')),
    ),
    ('the executor bundle', executor_registry, 'create_default', None),
    ('the autofocus pair', ScopeSession, '_build_autofocus_pair', None),
    ('the session constructor', ScopeSession, '__init__', None),
    ('the old channel blocks', ScopeSession, '_place_old_channel_blocks', None),
    ('the scope configuration', ScopeSession, '_configure_scope', None),
    ('the absent layers', ScopeSession, '_stop_acquiring_absent_channels', None),
    ('the streaming check', ImagingAPI, 'is_streaming', None),
    (
        'the held file lane',
        ScopeSession,
        '_hold_the_file_lane',
        SimulatedScope(file_stall=SimulatedStall(after_s=60.0, for_s=1.0)),
    ),
    ('the settings saver', SettingsSaver, 'start', None),
)


def _what_was_left(tmp_path, monkeypatch, owner, member, simulated) -> list[str]:
    """Run one create whose ``owner.member`` raises; name what it left behind."""
    built: list[Lumascope] = []
    construct = Lumascope.__init__

    def recording_init(self, *args, **kwargs):
        built.append(self)
        construct(self, *args, **kwargs)

    settings_file = tmp_path / member / 'current.json'
    settings_file.parent.mkdir()
    before = set(threading.enumerate())
    with monkeypatch.context() as patch:
        patch.setattr(Lumascope, '__init__', recording_init)
        patch.setattr(owner, member, _raise)
        try:
            ScopeSession.create(
                complete_settings(live_folder=str(tmp_path / 'capture')),
                simulate=True,
                warn_pre_release=False,
                simulated=simulated,
                settings_file=settings_file,
            )
        except _StepFaultError:
            left = []
        except BaseException as other:
            return [f'raised {type(other).__name__}: {other}, not the step fault']
        else:
            return ['create returned']

    started = [t for t in threading.enumerate() if t not in before]
    for thread in started:
        thread.join(_ENDED_WITHIN_S)
    left += [f'thread {t.name} running' for t in started if t.is_alive()]
    if settings_file.exists():
        left.append('the settings file was written')
    # The constructor's own undo is _abandon_construction, not disconnect;
    # every later step hands back a scope that must refuse as disconnected.
    if member != 'set_settle_check':
        scope = built[0]
        try:
            scope.camera.set_gain_db(1.0)
            left.append('the scope still takes commands')
        except HardwareCommandRefusedError as refused:
            if refused.reason != 'scope_disconnected':
                left.append(f'the scope refused {refused.reason!r}, not scope_disconnected')
    return left


def test_a_create_that_raises_at_any_step_leaves_nothing_running(tmp_path, monkeypatch):
    left = {
        step: what
        for step, owner, member, simulated in _STEPS
        if (what := _what_was_left(tmp_path, monkeypatch, owner, member, simulated))
    }

    assert left == {}, 'a step that raised left this behind:\n' + '\n'.join(
        f'  {step}: {", ".join(what)}' for step, what in left.items()
    )


def test_a_bundle_start_that_raises_stops_the_starts_before_it(monkeypatch):
    io_lane = SequentialIOExecutor(name='IO', closed_answer=scope_lane_closed)
    camera_lane = SequentialIOExecutor(name='CAMERA', closed_answer=scope_lane_closed)
    starts = []
    start = SequentialIOExecutor.start

    def third_start_raises(self):
        starts.append(self)
        if len(starts) == 3:
            raise _StepFaultError()
        start(self)

    monkeypatch.setattr(SequentialIOExecutor, 'start', third_start_raises)
    before = set(threading.enumerate())

    with pytest.raises(_StepFaultError):
        executor_registry.create_default(io_lane, camera_lane)

    started = [t for t in threading.enumerate() if t not in before]
    for thread in started:
        thread.join(_ENDED_WITHIN_S)
    assert [t.name for t in started if t.is_alive()] == []
