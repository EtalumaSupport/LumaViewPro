# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A manual recording refuses, for everyone, what would falsify its file.

A recording fits every frame to the geometry, depth and pixel size it
started with, but refused nothing but a run start: a frame size, binning,
pixel format or turret change from REST, a script or the GUI landed in the
middle of one and the file went on claiming the start values. Those writes
now carry a mark the lane refuses while a recording holds -- at submit and
again when a queued one reaches the worker -- and everything else (X/Y/Z,
a single-axis home, LED, gain, exposure) stays open to anyone.
"""

import copy
import dataclasses
import threading
from unittest.mock import patch

import pytest

from modules.activity_claim import ActivityClaim
from modules.exceptions import HardwareCommandRefusedError
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

_WAIT_S = 2.0


@pytest.fixture
def claim():
    return ActivityClaim()


@pytest.fixture
def lane(claim):
    ex = SequentialIOExecutor(name='TEST_IO')
    ex.ask_claim(claim)
    ex.start()
    yield ex
    ex.shutdown()


def _submit_from_another_thread(lane, task):
    box = {}

    def _submit():
        box['fut'] = lane.put(task, return_future=True)

    t = threading.Thread(target=_submit)
    t.start()
    t.join(_WAIT_S)
    return box['fut']


class TestTheLane:
    def test_a_marked_task_is_refused_while_a_recording_holds(self, claim, lane):
        ran = threading.Event()
        held = claim.try_claim('recording')
        try:
            fut = _submit_from_another_thread(
                lane, IOTask(action=ran.set, silent_on_failure=True, falsifies_recording=True)
            )
            with pytest.raises(HardwareCommandRefusedError) as refused:
                fut.result(timeout=_WAIT_S)
            assert refused.value.holder == 'recording'
            assert not ran.wait(0.3)
        finally:
            held.release()

    def test_a_marked_task_queued_before_the_recording_is_refused_when_it_is_reached(
        self, claim, lane
    ):
        gate = threading.Event()
        ran = threading.Event()
        lane.put(IOTask(action=gate.wait, args=(_WAIT_S,)))
        queued = lane.put(
            IOTask(action=ran.set, silent_on_failure=True, falsifies_recording=True),
            return_future=True,
        )
        held = claim.try_claim('recording')
        try:
            gate.set()
            with pytest.raises(HardwareCommandRefusedError):
                queued.result(timeout=_WAIT_S)
            assert not ran.is_set()
        finally:
            held.release()

    def test_a_marked_task_runs_when_nothing_holds(self, lane):
        fut = _submit_from_another_thread(
            lane, IOTask(action=lambda: 'ran', falsifies_recording=True)
        )
        assert fut.result(timeout=_WAIT_S) == 'ran'


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
    )
    try:
        yield s
    finally:
        s.shutdown()


@pytest.fixture
def recording(sim_session):
    # Homed first, while nothing holds: a move needs a known position.
    assert sim_session.scope.motion.home('ALL') is True
    held = sim_session.activity_claim.try_claim('recording')
    try:
        yield held
    finally:
        held.release()


# (name, the API object, the body the member dispatches, the call)
_FALSIFIERS = [
    (
        'set_frame_size',
        'imaging',
        '_set_frame_size_impl',
        lambda sc: sc.imaging.set_frame_size(640, 480),
    ),
    (
        'set_binning_size',
        'imaging',
        '_set_binning_size_impl',
        lambda sc: sc.imaging.set_binning_size(2),
    ),
    (
        'set_pixel_format',
        'imaging',
        '_set_pixel_format_impl',
        lambda sc: sc.imaging.set_pixel_format('Mono12'),
    ),
    ('move_turret', 'motion', '_move_turret_impl', lambda sc: sc.motion.move_turret(2)),
    ('home T', 'motion', '_home_turret_impl', lambda sc: sc.motion.home('T')),
    ('home ALL', 'motion', '_home_impl', lambda sc: sc.motion.home('ALL')),
    (
        'move_home_and_wait ALL',
        'motion',
        '_home_impl',
        lambda sc: sc.motion.move_home_and_wait('ALL'),
    ),
    (
        'move_home_and_wait T',
        'motion',
        '_home_turret_impl',
        lambda sc: sc.motion.move_home_and_wait('T'),
    ),
]

_OPEN = [
    ('move_absolute Z', lambda sc: sc.motion.move_absolute('Z', 100.0, wait_until_complete=True)),
    ('move_relative X', lambda sc: sc.motion.move_relative('X', 10.0, wait_until_complete=True)),
    ('home Z', lambda sc: sc.motion.home('Z')),
    ('set_gain_db', lambda sc: sc.imaging.set_gain_db(2.0)),
    ('set_exposure_ms', lambda sc: sc.imaging.set_exposure_ms(20.0)),
]


class TestTheMembers:
    @pytest.mark.parametrize(
        ('api', 'impl', 'call'), [f[1:] for f in _FALSIFIERS], ids=[f[0] for f in _FALSIFIERS]
    )
    def test_a_falsifier_is_refused_and_its_body_never_runs(
        self, sim_session, recording, api, impl, call
    ):
        target = getattr(sim_session.scope, api)
        ran = []

        def _body(*args, **kwargs):
            ran.append(impl)

        with (
            patch.object(target, impl, _body),
            pytest.raises(HardwareCommandRefusedError) as refused,
        ):
            call(sim_session.scope)
        assert refused.value.holder == 'recording'
        assert ran == [], f'{impl} ran inside the recording'

    @pytest.mark.parametrize(('call',), [(o[1],) for o in _OPEN], ids=[o[0] for o in _OPEN])
    def test_an_open_write_is_admitted(self, sim_session, recording, call):
        call(sim_session.scope)

    def test_a_whole_scope_home_without_a_turret_is_admitted(self, sim_session, recording):
        scope = sim_session.scope
        motion = scope.motion
        no_turret = dataclasses.replace(scope.capabilities, has_turret=False)
        calls = []

        def _home_body():
            calls.append(1)
            return True

        with (
            patch.object(scope, 'capabilities', no_turret),
            patch.object(motion, '_home_impl', _home_body),
        ):
            assert motion.home('ALL') is True
        assert calls == [1]


class TestTheSessionStores:
    """The Session's camera writers reach the camera before they store, so a
    refused write leaves both the camera and the stored settings as they were."""

    @pytest.mark.parametrize(
        'call',
        [
            lambda s: s.set_binning_size(2),
            lambda s: s.set_frame_size(640, 480),
            lambda s: s.set_image_mode(
                '12bit_scientific' if s.settings['image_mode'] == '8bit' else '8bit'
            ),
        ],
        ids=['set_binning_size', 'set_frame_size', 'set_image_mode'],
    )
    def test_a_refused_camera_write_stores_nothing(self, sim_session, recording, call):
        before = copy.deepcopy(
            {k: sim_session.settings[k] for k in ('binning', 'frame', 'image_mode')}
        )
        with pytest.raises(HardwareCommandRefusedError):
            call(sim_session)
        after = {k: sim_session.settings[k] for k in ('binning', 'frame', 'image_mode')}
        assert after == before


class TestTheSessionConfiguration:
    """The objective and the plate are the Session's two copies each; a held
    scope refuses a change before either copy moves."""

    @pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording'])
    def test_a_labware_change_is_refused_under_any_holder(self, sim_session, kind):
        loader = sim_session.wellplate_loader
        current = sim_session.settings['protocol']['labware']
        other = next(name for name in loader.get_plate_list() if name != current)
        plate_before = sim_session.scope.runtime_state.get_labware()
        held = sim_session.activity_claim.try_claim(kind, run_trigger_source='test')
        try:
            with pytest.raises(HardwareCommandRefusedError) as refused:
                sim_session.select_labware(other)
        finally:
            held.release()
        assert refused.value.holder == kind
        assert sim_session.settings['protocol']['labware'] == current
        assert sim_session.scope.runtime_state.get_labware() is plate_before

    def test_an_objective_change_is_refused_under_a_recording(self, sim_session, recording):
        current = sim_session.scope.runtime_state.get_current_objective_id()
        other = next(o for o in sim_session.objective_helper.get_objectives_list() if o != current)
        turret_before = copy.deepcopy(sim_session.settings['turret_objectives'])
        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_session.select_objective(other)
        assert refused.value.holder == 'recording'
        assert sim_session.scope.runtime_state.get_current_objective_id() == current
        assert sim_session.settings['turret_objectives'] == turret_before


_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 10.0}


class TestTheLongestExposure:
    """Exposure and gain stay open during a recording, so its feed-death bound
    reads the longest exposure the camera may be using, not the start value."""

    def test_it_is_the_cached_exposure_with_no_auto_gain_armed(self, sim_session):
        imaging = sim_session.scope.imaging
        assert imaging.longest_exposure_ms == imaging.exposure_ms_cached

    def test_it_is_the_arms_ceiling_while_auto_gain_is_armed(self, sim_session):
        imaging = sim_session.scope.imaging
        imaging.set_auto_gain(True, {**_AUTO_GAIN, 'max_exposure_ms': 150.0})
        assert imaging.longest_exposure_ms == 150.0
        imaging.set_auto_gain(False, dict(_AUTO_GAIN))
        assert imaging.longest_exposure_ms == imaging.exposure_ms_cached


class TestReselectingThePlateInPlace:
    """The left panel's toggle re-selects the current plate on every press,
    and the toggle is live under every hold; that re-selection changes
    nothing and must not raise out of the GUI's handler."""

    @pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording'])
    def test_the_plate_in_place_is_admitted_under_any_holder(self, sim_session, kind):
        current = sim_session.settings['protocol']['labware']
        held = sim_session.activity_claim.try_claim(kind, run_trigger_source='test')
        try:
            assert sim_session.select_labware(current) is False
        finally:
            held.release()
        assert sim_session.settings['protocol']['labware'] == current


class TestTheLabwarePanelReportsARefusal:
    """A plate picked while a recording still holds the scope (its drain, when
    the GUI's controls are free again) is refused at the API; the panel shows
    that once, through the one reporter, and renders the plate in place."""

    def test_a_refused_pick_is_shown_once_and_the_panel_carries_on(self, monkeypatch):
        import types

        import modules.app_context as _app_ctx
        import ui.protocol_settings as protocol_settings_module
        from tests.shown_outcomes import capture_shown

        shown = capture_shown(monkeypatch)
        redrawn = []

        def _refuse(name):
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', 'select_labware', 'recording'
            )

        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            types.SimpleNamespace(
                wellplate_loader=types.SimpleNamespace(get_plate_list=lambda: ['A', 'B']),
                session=types.SimpleNamespace(select_labware=_refuse),
                stage=types.SimpleNamespace(full_redraw=lambda: redrawn.append('stage')),
            ),
        )
        monkeypatch.setattr(
            protocol_settings_module, 'get_selected_labware', lambda: ('A', object())
        )
        panel = types.SimpleNamespace(
            ids={'labware_spinner': types.SimpleNamespace(text='B', values=[])},
            _protocol=None,
        )

        protocol_settings_module.ProtocolSettings.select_labware(panel)

        assert [n.title for n in shown] == ['Microscope Busy']
        assert redrawn == ['stage'], 'the panel must still render the plate in place'
