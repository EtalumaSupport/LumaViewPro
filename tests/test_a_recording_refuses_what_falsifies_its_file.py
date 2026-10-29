# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A manual recording refuses, for everyone, what would falsify its file.

A recording fits every frame to the geometry, depth and pixel size it
started with, but refused nothing but a run start: a frame size, binning,
pixel format or turret change from REST, a script or the GUI landed in the
middle of one and the file went on claiming the start values. Those writes
now carry a mark the lane refuses while a recording holds -- at submit and
again when a queued one reaches the worker -- and everything else (X/Y/Z,
an LED off, gain, exposure) stays open to anyone. A home of any axis is
refused: it holds the whole scope, as a run does, and a recording already
holds it.

A recording is named for the shown layer and keeps the light it started
under. An LED lit mid-recording -- ``led_on('Ex405')`` over REST while the
recording was named Ex488 -- left frames lit by one channel and shot at
another's camera settings, under a manifest and an MP4 naming the first;
every write that would light now carries the same mark, and a start with
another LED lit, or with the camera on another layer, is refused.
"""

import copy
import dataclasses
import threading
from unittest.mock import patch

import pytest

from modules.activity_claim import ActivityClaim
from modules.exceptions import HardwareCommandRefusedError, RecordingRefusedError
from modules.lumascope_api.illumination import (
    LedTransition,
    LedTransitionCtx,
    snapshot_lit_pairs,
)
from modules.sequential_io_executor import IOTask, SequentialIOExecutor
from tests.protocol_drives import run_identity
from tests.case_walk import failing_cases
from tests.stock_channels import LED0

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
    sim_session.scope.motion.home('ALL')
    held = sim_session.activity_claim.try_claim('recording')
    try:
        yield held
    finally:
        held.release()


# (name, the API object, the body the member dispatches, the call)
_FALSIFIERS = [
    (
        'set_frame_size',
        'camera',
        '_set_frame_size_impl',
        lambda sc: sc.camera.set_frame_size(640, 480),
    ),
    (
        'set_binning',
        'camera',
        '_set_binning_impl',
        lambda sc: sc.camera.set_binning(2),
    ),
    (
        'set_image_mode',
        'camera',
        '_set_pixel_format_impl',
        lambda sc: sc.camera.set_image_mode('12bit_scientific'),
    ),
    ('move_turret', 'motion', '_move_turret_impl', lambda sc: sc.motion.move_turret(2)),
    ('home T', 'motion', '_home_turret_impl', lambda sc: sc.motion.home('T')),
    ('home ALL', 'motion', '_home_impl', lambda sc: sc.motion.home('ALL')),
]

_OPEN = [
    ('move_z', lambda sc: sc.motion.move_z(100.0)),
    ('move_xy_by X', lambda sc: sc.motion.move_xy_by(dx_um=10.0)),
    ('set_gain_db', lambda sc: sc.camera.set_gain_db(2.0)),
    ('set_exposure_ms', lambda sc: sc.camera.set_exposure_ms(20.0)),
]


class TestTheMembers:
    def test_a_falsifier_is_refused_and_its_body_never_runs(self, sim_session, recording):
        def check(api, impl, call):
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

        failed = failing_cases([f[1:] for f in _FALSIFIERS], check, label=lambda case: case[1])
        assert not failed, failed

    @pytest.mark.slow
    # parametrize-justified: each admitted write runs and changes the scope
    @pytest.mark.parametrize(('call',), [(o[1],) for o in _OPEN], ids=[o[0] for o in _OPEN])
    def test_an_open_write_is_admitted(self, sim_session, recording, call):
        call(sim_session.scope)

    @pytest.mark.parametrize('axis', ['Z', 'ALL'])
    def test_a_home_that_moves_no_turret_is_refused_naming_the_recording(
        self, sim_session, recording, axis
    ):
        scope = sim_session.scope
        motion = scope.motion
        no_turret = dataclasses.replace(scope.capabilities, has_turret=False)
        impl = '_zhome_impl' if axis == 'Z' else '_home_impl'
        calls = []

        def _home_body():
            calls.append(1)

        with (
            patch.object(scope, 'capabilities', no_turret),
            patch.object(motion, impl, _home_body),
            pytest.raises(HardwareCommandRefusedError) as refused,
        ):
            motion.home(axis)
        assert refused.value.holder == 'recording'
        assert str(refused.value) == 'A recording is using the microscope. Try again when it ends.'
        assert calls == []


class TestTheLights:
    """While a recording holds, a write that would light an LED is refused
    before anything changes, from any caller; an off is not."""

    @pytest.fixture
    def lit_bf(self, sim_session):
        """BF shown and lit, the LEDs as they are when a recording is held."""
        sim_session.scope.motion.home('ALL')
        sim_session.show_channel('BF')
        sim_session.light_shown_channel(True)
        return sim_session

    def test_every_write_that_would_light_is_refused_and_the_leds_stay(self, lit_bf):
        session = lit_bf
        illumination = session.scope.illumination
        led0 = illumination.color2ch(LED0)
        illumination.led_on(LED0, 80.0)
        lit_led0 = illumination.save_led_state('led0 and bf lit')
        illumination.led_off(LED0)
        before = illumination.get_led_states()
        lights = [
            ('led_on, the shown layer at its own current', lambda: illumination.led_on('BF', 50.0)),
            ('led_on, another layer', lambda: illumination.led_on(LED0, 100.0)),
            ('led_on_by_address', lambda: illumination.led_on_by_address(led0, 90.0)),
            ('restore_led_state', lambda: illumination.restore_led_state(lit_led0)),
            (
                'apply_transition',
                lambda: illumination.apply_transition(
                    LedTransition.MANUAL_STEP,
                    LedTransitionCtx(channel=led0, illumination_ma=60.0, preview_on=True),
                ),
            ),
            ('light_shown_channel', lambda: session.light_shown_channel(True)),
        ]
        held = session.activity_claim.try_claim('recording')
        try:

            def check(name, call):
                with pytest.raises(HardwareCommandRefusedError) as refused:
                    call()
                assert (refused.value.reason, refused.value.holder) == (
                    'exclusive_activity_running',
                    'recording',
                )
                assert illumination.get_led_states() == before

            failed = failing_cases(lights, check, label=lambda case: case[0])
        finally:
            held.release()
        assert not failed, failed

    def test_every_off_is_admitted(self, lit_bf):
        session = lit_bf
        illumination = session.scope.illumination
        offs = [
            ('led_off', lambda: illumination.led_off('BF')),
            (
                'led_off_by_address',
                lambda: illumination.led_off_by_address(illumination.color2ch('BF')),
            ),
            ('leds_off', illumination.leds_off),
            ('light_shown_channel(False)', lambda: session.light_shown_channel(False)),
        ]

        def check(name, call):
            session.light_shown_channel(True)
            held = session.activity_claim.try_claim('recording')
            try:
                call()
            finally:
                held.release()
            assert snapshot_lit_pairs(illumination.get_led_states()) == frozenset()

        failed = failing_cases(offs, check, label=lambda case: case[0])
        assert not failed, failed

    def test_a_shown_layers_illumination_is_stored_and_the_led_is_left(self, lit_bf):
        session = lit_bf
        illumination = session.scope.illumination
        before = illumination.get_led_states()
        stored = session.settings['BF']['illumination_ma']
        written = stored - 10.0 if stored > 10.0 else stored + 10.0
        held = session.activity_claim.try_claim('recording')
        try:
            session.update_settings('BF.illumination_ma', written)
        finally:
            held.release()
        assert session.settings['BF']['illumination_ma'] == written
        assert illumination.get_led_states() == before


class TestTheStartKeepsItsName:
    """A recording is named for the shown layer, and does not start when the
    name would be false: another LED lit, or the camera on another layer."""

    @pytest.fixture
    def sim_session(self, tmp_path):
        # A model with no turret: the still below needs a known objective.
        from modules.scope_session import ScopeSession
        from tests.scope_fakes import home_sim_scope
        from tests.settings_fixtures import complete_settings

        s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            home_sim_scope(s.scope)
            yield s
        finally:
            s.shutdown()

    def test_a_start_with_another_led_lit_is_refused(self, sim_session):
        sim_session.show_channel('BF')
        sim_session.scope.illumination.led_on(LED0, 50.0)
        with pytest.raises(RecordingRefusedError) as refused:
            sim_session.manual_recording.start(false_color_on=False)
        assert refused.value.reason == 'other_led_lit'
        assert LED0 in str(refused.value)
        assert not sim_session.manual_recording.is_recording
        assert sim_session.activity_claim.holder is None

    def test_a_start_with_the_camera_on_another_layer_is_refused(self, sim_session):
        sim_session.show_channel('BF')
        # A still under a raw light leaves the camera on that light's layer.
        sim_session.scope.illumination.led_on(LED0, 50.0)
        sim_session.manual_capture.capture(false_color_on=False).result(timeout=30)
        sim_session.scope.illumination.led_off(LED0)
        assert sim_session.scope.camera.get_held_layer() == LED0
        with pytest.raises(RecordingRefusedError) as refused:
            sim_session.manual_recording.start(false_color_on=False)
        assert refused.value.reason == 'camera_holds_another_layer'
        assert not sim_session.manual_recording.is_recording
        assert sim_session.activity_claim.holder is None


class TestTheSessionStores:
    """The Session's camera writers reach the camera before they store, so a
    refused write leaves both the camera and the stored settings as they were."""

    @pytest.mark.parametrize(
        'call',
        [
            lambda s: s.scope.camera.set_binning(2),
            lambda s: s.scope.camera.set_frame_size(640, 480),
            lambda s: s.scope.camera.set_image_mode(
                '12bit_scientific' if s.settings['image_mode'] == '8bit' else '8bit'
            ),
        ],
        ids=['set_binning', 'set_frame_size', 'set_image_mode'],
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

    @pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording', 'home'])
    def test_a_labware_change_is_refused_under_any_holder(self, sim_session, kind):
        loader = sim_session.scope.wellplate_loader
        current = sim_session.settings['protocol']['labware']
        other = next(name for name in loader.get_plate_list() if name != current)
        plate_before = sim_session.scope.runtime_state.get_labware()
        held = sim_session.activity_claim.try_claim(
            kind, run=run_identity() if kind == 'protocol' else None
        )
        try:
            with pytest.raises(HardwareCommandRefusedError) as refused:
                sim_session.select_labware(other)
        finally:
            held.release()
        assert refused.value.holder == kind
        assert sim_session.settings['protocol']['labware'] == current
        assert sim_session.scope.runtime_state.get_labware() == plate_before

    def test_an_objective_change_is_refused_under_a_recording(self, sim_session, recording):
        current = sim_session.scope.runtime_state.get_current_objective_id()
        other = next(
            o for o in sim_session.scope.objective_helper.get_objectives_list() if o != current
        )
        turret_before = copy.deepcopy(sim_session.settings['turret_objectives'])
        with pytest.raises(HardwareCommandRefusedError) as refused:
            sim_session.select_objective(other)
        assert refused.value.holder == 'recording'
        assert sim_session.scope.runtime_state.get_current_objective_id() == current
        assert sim_session.settings['turret_objectives'] == turret_before


class TestReselectingThePlateInPlace:
    """The left panel's toggle re-selects the current plate on every press,
    and the toggle is live under every hold; that re-selection changes
    nothing and must not raise out of the GUI's handler."""

    @pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording', 'home'])
    def test_the_plate_in_place_is_admitted_under_any_holder(self, sim_session, kind):
        current = sim_session.settings['protocol']['labware']
        held = sim_session.activity_claim.try_claim(
            kind, run=run_identity() if kind == 'protocol' else None
        )
        try:
            assert sim_session.select_labware(current) is False
        finally:
            held.release()
        assert sim_session.settings['protocol']['labware'] == current


class TestTheLabwarePanelReportsARefusal:
    """A plate picked while a recording still holds the scope (its drain, when
    the GUI's controls are free again) is refused at the API; the panel shows
    that once, through the one reporter, and renders the plate in place: the
    stage, and the spinner, which went on naming the refused plate."""

    def test_a_refused_pick_is_shown_once_and_the_panel_carries_on(self, monkeypatch, centre_posts):
        import types

        import modules.app_context as _app_ctx
        import ui.protocol_settings as protocol_settings_module

        redrawn = []
        protocol_plates = []

        def _refuse(name):
            raise HardwareCommandRefusedError(
                'exclusive_activity_running', 'select_labware', 'recording'
            )

        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            types.SimpleNamespace(
                scope=types.SimpleNamespace(labware={'A': None, 'B': None}),
                session=types.SimpleNamespace(
                    select_labware=_refuse,
                    set_protocol_labware=lambda protocol, key: protocol_plates.append(key),
                ),
                stage=types.SimpleNamespace(full_redraw=lambda: redrawn.append('stage')),
                settings={'protocol': {'labware': 'A'}},
            ),
        )
        panel = types.SimpleNamespace(
            ids={'labware_spinner': types.SimpleNamespace(text='B', values=[])},
            _protocol=object(),
        )
        panel.select_labware_ex = lambda selected: (
            protocol_settings_module.ProtocolSettings.select_labware_ex(panel, selected)
        )
        panel.show_labware = lambda: protocol_settings_module.ProtocolSettings.show_labware(panel)

        protocol_settings_module.ProtocolSettings.select_labware(panel)

        assert [n.title for n in centre_posts] == ['Microscope Busy']
        assert protocol_plates == [], 'a plate the scope refused was written into the protocol'
        assert redrawn == ['stage'], 'the panel must still render the plate in place'
        assert panel.ids['labware_spinner'].text == 'A', (
            'the spinner names the refused plate while the scope stays on the one in place'
        )
