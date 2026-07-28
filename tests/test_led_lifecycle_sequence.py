# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Lifecycle LED-sequence test -- the end-to-end guard the LED cluster lacked.

Every prior LED regression test (test_issue_612/_671/_695/_696,
test_led_idempotent_handoff, test_led_lease) is a per-issue lock pinning ONE
mechanism at ONE site. None drives the run/AF/step/manual lifecycle end to end
and asserts the LED command stream at every transition -- which is exactly why
each fix regressed the next transition.

This test builds that missing guard. It runs real protocol / autofocus /
manual-nav paths on a real ``Lumascope(simulate=True)`` with real
``SequentialIOExecutor`` workers and asserts the **LED-only command substream**
recorded by a driver listener. The substream is the sequence of
``(color, enabled, mA, owner)`` events that actually reached the LED driver, in
order. Because a no-op (a channel already at target) emits no driver command,
the listener is a direct measure of "did the LED blink / hold / switch".

Why the LED-only substream and not the whole api-log stream: the absolute
interleaving of LED-vs-capture-vs-file-save events crosses several executors and
is not deterministic. LED writes for one run serialize through a single worker,
so the LED substream IS deterministic in order; the test asserts on it with
relative anchors (this color on, then off, then that color on), not on absolute
positions in a whole-run stream.

The test is GREEN against today's shipped behavior. It is the safety net for the
LED State Authority migration (LED_STATE_AUTHORITY_PLAN.md): the emitted LED
stream is invariant as each decider moves into the authority, so a migration that
changes any transition's LED behavior fails here on the same commit.

Move-ordering note: the listener records LED ops only, not move order, so the
#671 member (leds_off must precede move_abs at a well boundary) is NOT covered
here -- it stays pinned by test_issue_671_added_location_led_ordering.py.
"""

from __future__ import annotations

import datetime
import logging
import sys
import threading
from unittest.mock import MagicMock

import pytest

# Mirror conftest pattern (heavy deps already mocked there): a settings_init
# stub so the protocol stack imports without a real settings file.
_mock_settings_init = MagicMock()
_mock_settings_init.settings = {
    'BF': {'autofocus': False},
    'PC': {'autofocus': False},
    'DF': {'autofocus': False},
    'Red': {'autofocus': False},
    'Green': {'autofocus': False},
    'Blue': {'autofocus': False},
    'Lumi': {'autofocus': False},
}
sys.modules.setdefault('modules.settings_init', _mock_settings_init)

from modules.image_mode import ImageCaptureConfig
from modules.lumascope_api import Lumascope
from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx
from modules.protocol import Protocol
from modules.sequenced_capture_runner import (
    SequencedCaptureRunner,
    SequencedCaptureRunMode,
)
from modules.sequential_io_executor import SequentialIOExecutor


# Plate coordinates in mm for distinct well positions.
WELL_COORDS = {
    'A1': (24.55, 24.0),
    'A2': (63.75, 24.0),
    'A3': (103.0, 24.0),
}
DEFAULT_Z = 6247.3684

CHANNEL_ILLUMINATION = {'BF': 5.0, 'Blue': 250.0, 'Green': 250.0, 'Red': 350.0}
CHANNEL_EXPOSURE = {'BF': 0.1, 'Blue': 1.0, 'Green': 6.76, 'Red': 600.0}
CHANNEL_GAIN = {'BF': 14.4, 'Blue': 20.0, 'Green': 20.0, 'Red': 20.0}


def _step_dict(name, x, y, z, color, idx, *, auto_focus=False, zstack_group=-1, z_slice=-1):
    return {
        'Name': name,
        'X': x,
        'Y': y,
        'Z': z,
        'Auto_Focus': auto_focus,
        'Color': color,
        'False_Color': color != 'BF',
        'Illumination': CHANNEL_ILLUMINATION[color],
        'Gain': CHANNEL_GAIN[color],
        'Auto_Gain': False,
        'Exposure': CHANNEL_EXPOSURE[color],
        'Sum': 1,
        'Objective': '4x Oly',
        'Well': name.split('_')[0] if '_' in name else 'A1',
        'Tile': '',
        'Z-Slice': z_slice,
        'Custom Step': False,
        'Tile Group ID': -1,
        'Z-Stack Group ID': zstack_group,
        'Acquire': 'image',
        'Video Config': {'duration': 5, 'fps': 30},
        'Stim_Config': {},
        'Step Index': idx,
        'Label': '',
    }


def _build_protocol(step_specs):
    """Build a Protocol from a list of (well, color, kwargs) specs.

    Each spec becomes one step at the well's X/Y. kwargs pass through to
    _step_dict (auto_focus, zstack_group).
    """
    import pandas as pd

    pytest.importorskip('modules.tiling_config')
    import pathlib

    tiling_path = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'

    rows = []
    for idx, (well, color, kwargs) in enumerate(step_specs):
        x, y = WELL_COORDS[well]
        rows.append(_step_dict(f'{well}_{color}', x, y, DEFAULT_Z, color, idx, **kwargs))
    df = pd.DataFrame(rows)
    config = {
        'version': Protocol.CURRENT_VERSION,
        'steps': df,
        'period': datetime.timedelta(minutes=20.0),
        'duration': datetime.timedelta(hours=48.0),
        'labware_id': '6 well microplate',
        'capture_root': '',
        'tiling': '1x1',
    }
    return Protocol(tiling_configs_file_loc=tiling_path, config=config)


class LedSubstream:
    """Thread-safe recorder of the LED-only command substream.

    Records (color, enabled, mA, owner) for every driver command, in order.
    The listener fires from whichever worker thread issued the command, so the
    append is locked.
    """

    def __init__(self):
        self._events: list[tuple] = []
        self._lock = threading.Lock()

    def __call__(self, color, enabled, mA, owner):
        with self._lock:
            self._events.append((color, bool(enabled), mA, owner))

    @property
    def events(self) -> list[tuple]:
        with self._lock:
            return list(self._events)

    def transitions(self, color: str) -> list[bool]:
        """The enabled (True/False) sequence for one color, with consecutive
        duplicates collapsed. This is the migration-invariant view: it ignores
        whether an off was emitted as a per-channel command or as part of a
        nuclear leds_off, and it ignores owner. A regression that fails to turn
        a color off (or blinks it) still changes this sequence."""
        out: list[bool] = []
        for c, e, _m, _o in self.events:
            if c != color:
                continue
            if not out or out[-1] != e:
                out.append(e)
        return out

    def lit_transitions(self, color: str) -> list[bool]:
        """transitions(color) with a leading False stripped.

        A leading False is an "off while already dark" -- a channel commanded
        off before it was ever lit (shipped leds_off nukes all channels). That
        no-op vanishes under the diff-based authority (it emits no command for a
        dark channel), so it must not be part of the asserted invariant. A
        channel's meaningful lifecycle starts at its first ON. A never-lit
        channel reduces to [].

        Caveat for future test authors: this view collapses an ON->ON
        re-assert at a DIFFERENT mA to a single True (no intervening OFF), so a
        brightness-drift regression is invisible here. Always pair it with an
        on_events() assertion (which preserves mA) -- every scenario below does."""
        trans = self.transitions(color)
        while trans and trans[0] is False:
            trans.pop(0)
        return trans

    def on_events(self) -> list[tuple]:
        """(color, mA) for every ON command, in order -- the lit sequence.
        Never-lit channels and offs do not appear. mA is preserved because the
        authority migration preserves the (op, channel, mA) target stream."""
        return [(c, m) for c, e, m, _o in self.events if e]

    def final_lit(self) -> set:
        """The set of colors lit at the end of the stream (the run end-state)."""
        lit: set[str] = set()
        for c, e, _m, _o in self.events:
            if e:
                lit.add(c)
            else:
                lit.discard(c)
        return lit

    def lit_at_most_one(self) -> bool:
        """Replay the stream; assert at most one color is ever lit at a time
        (the mutual-exclusion invariant -- no double illumination)."""
        lit: set[str] = set()
        for c, e, _m, _o in self.events:
            if e:
                lit.add(c)
            else:
                lit.discard(c)
            if len(lit) > 1:
                return False
        return True

    def render(self) -> str:
        lines = []
        for c, e, m, o in self.events:
            verb = 'ON ' if e else 'OFF'
            lines.append(f'  {verb} {c:<6} mA={m} owner={o!r}')
        return '\n'.join(lines) if lines else '  (no LED events)'


@pytest.fixture
def scope():
    s = Lumascope(simulate=True)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s._camera_driver.start_grabbing()
    yield s
    s._camera_driver.stop_grabbing()
    s.disconnect()


def _make_executors(file_queue_maxsize=0):
    """Set up + tear down the executor set (a generator: ``yield from`` it).

    No 'autofocus' executor: the AF-off protocol path uses a mocked AF runner
    (autofocus_thread / autofocus_runner below), so a real AF worker would be
    dead state. The AF-on lifecycle tests (s5-s7) build their own runner.

    file_queue_maxsize=0 keeps the file worker's protocol queue unbounded
    (the historical default); the wedged-writer test (s11) passes 1 so the
    queue can be filled to make the blocking write submit stall for real.
    """
    from modules.protocol_thread import ProtocolThread

    execs = {
        'io': SequentialIOExecutor(name='TEST_IO'),
        'file_io': SequentialIOExecutor(
            name='TEST_FILE', protocol_queue_maxsize=file_queue_maxsize
        ),
        'camera': SequentialIOExecutor(name='TEST_CAMERA'),
    }
    for e in execs.values():
        e.start()
    pt = ProtocolThread()
    pt.start()
    execs['protocol'] = pt
    try:
        yield execs
    finally:
        for name, e in execs.items():
            try:
                if name == 'protocol':
                    e.stop(timeout=2.0)
                else:
                    e.shutdown()
            except Exception:
                pass


@pytest.fixture
def executors():
    yield from _make_executors()


def _mock_af_runner():
    mock_af = MagicMock()
    mock_af.reset = MagicMock()
    mock_af.in_progress = MagicMock(return_value=False)
    mock_af.complete = MagicMock(return_value=False)
    mock_af.is_running = MagicMock(return_value=False)
    mock_af.result = MagicMock(return_value=None)
    mock_af.best_focus_position = MagicMock(return_value=DEFAULT_Z)
    mock_af.run_in_progress = MagicMock(return_value=False)
    return mock_af


def _make_runner(scope, execs):
    """A real SequencedCaptureRunner with real executors and a mocked AF
    runner -- faithful for AF-off scenarios (production does not invoke the AF
    runner when Auto_Focus is False). Takes the executor set as an argument so
    a test can substitute e.g. a bounded file-IO executor (s11)."""
    from modules.coord_transformations import CoordinateTransformer
    from modules.labware_loader import WellPlateLoader

    exc = SequencedCaptureRunner(
        scope=scope,
        stage_offset={'x': 0.0, 'y': 0.0},
        io_executor=execs['io'],
        protocol_thread=execs['protocol'],
        file_io_executor=execs['file_io'],
        camera_executor=execs['camera'],
        autofocus_thread=MagicMock(is_running=False),
        autofocus_runner=_mock_af_runner(),
    )
    exc._wellplate_loader = WellPlateLoader()
    exc._coordinate_transformer = CoordinateTransformer()
    return exc


@pytest.fixture
def runner(scope, executors):
    return _make_runner(scope, executors)


def _run_protocol(
    runner,
    protocol,
    tmp_path,
    *,
    leds_state_at_end='off',
    keep_led_between_steps=False,
    max_scans=1,
    timeout=30,
):
    """Run a protocol to completion (SINGLE_SCAN) and block on the done Event.

    max_scans > 1 runs a multi-scan (timelapse-shaped) session; pair it with
    _build_two_scan_protocol's near-zero period so it finishes in test time.
    """
    done = threading.Event()
    result_holder: dict = {}

    def on_complete(**kwargs):
        result_holder.update(kwargs)
        done.set()

    callbacks = {'run_complete': on_complete}

    plan = runner.prepare(
        keep_led_between_steps=keep_led_between_steps,
        protocol=protocol,
        run_trigger_source='test',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='led_lifecycle',
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        autogain_settings={
            'target_brightness': 0.3,
            'min_gain_db': 0.0,
            'max_gain_db': 20.0,
            'max_duration': datetime.timedelta(seconds=1),
        },
        parent_dir=tmp_path / 'output',
        max_scans=max_scans,
        callbacks=callbacks,
        leds_state_at_end=leds_state_at_end,
        initial_autofocus_states={
            'BF': False,
            'PC': False,
            'DF': False,
            'Red': False,
            'Green': False,
            'Blue': False,
            'Lumi': False,
        },
    )
    runner.start(plan)

    completed = done.wait(timeout=timeout)
    return completed, result_holder


# ---------------------------------------------------------------------------
# Protocol-path scenarios (AF off) -- a mocked AF runner is faithful here
# because production never invokes the AF runner when Auto_Focus is False.
# ---------------------------------------------------------------------------

ALL_COLORS = ['BF', 'Blue', 'Green', 'Red', 'PC', 'DF']


def _recorded_run(
    scope,
    runner,
    tmp_path,
    specs,
    *,
    keep_led_between_steps=False,
    leds_state_at_end='off',
    prelit=None,
):
    sub = LedSubstream()
    if prelit:
        # A pre-run Live LED so leds_state_at_end='return_to_original' has
        # something to restore (the snapshot is taken at lease acquire).
        scope.illumination.led_on(
            channel=scope.illumination.color2ch(prelit[0]), mA=prelit[1], owner='ui'
        )
    scope.illumination.add_led_listener(sub)
    protocol = _build_protocol(specs)
    completed, _ = _run_protocol(
        runner,
        protocol,
        tmp_path,
        leds_state_at_end=leds_state_at_end,
        keep_led_between_steps=keep_led_between_steps,
    )
    assert completed, f'protocol did not complete in time\n{sub.render()}'
    return sub


def _assert_only_lit(sub, *colors):
    """No color outside *colors* is ever meaningfully lit."""
    for color in ALL_COLORS:
        if color not in colors:
            assert sub.lit_transitions(color) == [], (
                f'{color} was lit but should never be\n{sub.render()}'
            )


def test_s1_single_channel_keep_off_offs_between_each_step(scope, runner, tmp_path):
    """3 same-color steps, keep_led_between_steps=False: the LED lights once per
    step and goes off at every step boundary -- one on + one off per step, no
    off->on blink within a step (pins #696 default-off and #697)."""
    sub = _recorded_run(
        scope,
        runner,
        tmp_path,
        [('A1', 'Green', {}), ('A2', 'Green', {}), ('A3', 'Green', {})],
        keep_led_between_steps=False,
    )
    assert sub.on_events() == [('Green', 250.0)] * 3, sub.render()
    assert sub.lit_transitions('Green') == [True, False, True, False, True, False], sub.render()
    _assert_only_lit(sub, 'Green')
    assert sub.lit_at_most_one()
    assert sub.final_lit() == set(), sub.render()


def test_s2_single_channel_keep_on_holds_across_boundaries(scope, runner, tmp_path):
    """3 same-color steps, keep_led_between_steps=True: the LED lights once and
    is HELD across every same-color boundary (zero commands at the boundary --
    the idempotent skip), off only at run end (pins #696 opt-in hold)."""
    sub = _recorded_run(
        scope,
        runner,
        tmp_path,
        [('A1', 'Green', {}), ('A2', 'Green', {}), ('A3', 'Green', {})],
        keep_led_between_steps=True,
    )
    assert sub.on_events() == [('Green', 250.0)], sub.render()
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    _assert_only_lit(sub, 'Green')
    assert sub.final_lit() == set(), sub.render()


def test_s3_zstack_holds_across_slices_regardless_of_flag(scope, runner, tmp_path):
    """3-slice z-stack, keep_led_between_steps=False: the LED stays lit across
    all slices REGARDLESS of the flag (z-stack is a single acquisition), off
    only after the last slice (pins the #696/304fedce z-stack regression)."""
    sub = _recorded_run(
        scope,
        runner,
        tmp_path,
        [
            # Distinct Z-Slice indices, as a real z-stack expansion assigns
            # them -- identical slices would render one capture filename and
            # the run would be refused at validation.
            ('A1', 'Green', {'zstack_group': 5, 'z_slice': 0}),
            ('A1', 'Green', {'zstack_group': 5, 'z_slice': 1}),
            ('A1', 'Green', {'zstack_group': 5, 'z_slice': 2}),
        ],
        keep_led_between_steps=False,
    )
    assert sub.on_events() == [('Green', 250.0)], sub.render()
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    _assert_only_lit(sub, 'Green')
    assert sub.final_lit() == set(), sub.render()


def test_s4_two_color_one_lit_at_a_time(scope, runner, tmp_path):
    """Green then Red: exactly one channel lit at a time; Green off + Red on at
    the color boundary, no double illumination."""
    sub = _recorded_run(
        scope,
        runner,
        tmp_path,
        [('A1', 'Green', {}), ('A1', 'Red', {})],
        keep_led_between_steps=False,
    )
    assert sub.on_events() == [('Green', 250.0), ('Red', 350.0)], sub.render()
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert sub.lit_transitions('Red') == [True, False], sub.render()
    _assert_only_lit(sub, 'Green', 'Red')
    assert sub.lit_at_most_one(), f'double illumination\n{sub.render()}'
    assert sub.final_lit() == set(), sub.render()


def test_s10_run_end_off_leaves_all_dark(scope, runner, tmp_path):
    """leds_state_at_end='off': every channel dark at run end."""
    sub = _recorded_run(
        scope,
        runner,
        tmp_path,
        [('A1', 'Green', {})],
        leds_state_at_end='off',
    )
    assert sub.on_events() == [('Green', 250.0)], sub.render()
    assert sub.final_lit() == set(), sub.render()


def test_s10_run_end_return_to_original_relights_prerun_channel(scope, runner, tmp_path):
    """leds_state_at_end='return_to_original': a pre-run Live channel is re-lit
    at the final boundary (no blink), distinct from the 'off' policy."""
    sub = _recorded_run(
        scope,
        runner,
        tmp_path,
        [('A1', 'Green', {})],
        leds_state_at_end='return_to_original',
        prelit=('Blue', 120.0),
    )
    assert sub.on_events() == [('Green', 250.0), ('Blue', 120.0)], sub.render()
    assert sub.final_lit() == {'Blue'}, sub.render()
    # The run's Green must be off before Blue is restored -- never two lit at once.
    assert sub.lit_at_most_one(), f'double illumination at the run-end restore\n{sub.render()}'


# ---------------------------------------------------------------------------
# Lease enforcement (transition 18) -- a live UI write is refused while a run
# owns the lease. Driven at the illumination layer: the lease IS the mechanism,
# and a paused mid-run protocol would add flakiness without testing anything the
# lease object doesn't already gate.
# ---------------------------------------------------------------------------


def test_s8_live_write_refused_while_run_holds_lease(scope):
    """While a 'protocol' lease is held, an out-of-turn live write (empty owner)
    is refused: no driver command, the run's channel unchanged (pins the lease
    enforcement, ffd5a83c). The refused write emits nothing to the listener."""
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)

    lease = ill.acquire_led_lease('protocol', alive=lambda: True)
    assert lease is not None
    ill.led_on(channel=ill.color2ch('Green'), mA=250.0, owner='protocol')

    # Out-of-turn live writes (empty owner) while the run holds the lease. Both
    # are refused by the LEASE check (_lease_violation), not by per-owner
    # ownership -- an empty owner skips the ownership gate, so the lease is the
    # only thing that can refuse them. The off is NOT a no-op skip: Green is lit,
    # so if the lease did not refuse it Green would go dark and the assertions
    # below would fail -- the refusal is what keeps Green lit.
    ill.led_on(channel=ill.color2ch('Red'), mA=350.0, owner='')  # refused by lease
    ill.led_off(channel=ill.color2ch('Green'), owner='')  # refused by lease

    assert ill.led_enabled('Green'), 'protocol channel was disturbed by a live write'
    assert not ill.led_enabled('Red'), 'live write lit a channel despite the lease'
    # Exactly one command reached the driver (the protocol's Green on): the two
    # refused writes emitted nothing -- no Red blink, no Green off.
    assert sub.on_events() == [('Green', 250.0)], sub.render()
    assert sub.lit_transitions('Red') == [], sub.render()
    assert sub.lit_transitions('Green') == [True], sub.render()  # lit, never offed

    lease.release(leave_on=False)


# ---------------------------------------------------------------------------
# Manual-nav preview (transitions 13/14). Driven via the production authority
# call apply_transition_async(MANUAL_STEP, ctx) -- the exact call
# ui/step_navigation.py makes when settings['protocol_led_on'] is True. The full
# go_to_step UI drive (the settings->preview gate) is UI-thread-bound and stays
# covered by the issue locks; here the LED-substream invariant is what matters.
# ---------------------------------------------------------------------------


@pytest.fixture
def scope_io(scope):
    ex = SequentialIOExecutor(name='TEST_LED_IO')
    ex.start()
    scope.register_executors(io_executor=ex)
    yield scope
    ex.shutdown(wait=True)


def _run_async(fn, *args, timeout=5, **kwargs):
    done = threading.Event()
    fn(*args, callback=lambda *a, **k: done.set(), **kwargs)
    assert done.wait(timeout), 'async LED task did not complete in time'


def test_s9_manual_nav_preview_lights_holds_and_switches(scope_io):
    """Manual-nav preview ON lights the step channel exclusively; re-navigating
    to the SAME color holds it with zero commands (no off->on blink -- the #697
    manual twin); navigating to a DIFFERENT color switches exclusively (never
    double-lit)."""
    ill = scope_io.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)

    def _preview(color, mA):
        return _run_async(
            ill.apply_transition_async,
            LedTransition.MANUAL_STEP,
            LedTransitionCtx(channel=ill.color2ch(color), mA=mA, preview_on=True),
        )

    # Preview to a Green step.
    _preview('Green', 250.0)
    assert ill.led_enabled('Green')
    assert sub.lit_transitions('Green') == [True], sub.render()

    # Re-navigate to the same color: idempotent hold, no blink.
    _preview('Green', 250.0)
    assert sub.lit_transitions('Green') == [True], (
        f'same-color re-nav blinked the channel\n{sub.render()}'
    )

    # Navigate to a different color: exclusive switch.
    _preview('Red', 350.0)
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert sub.lit_transitions('Red') == [True], sub.render()
    assert sub.lit_at_most_one(), f'double illumination during preview\n{sub.render()}'
    assert sub.final_lit() == {'Red'}, sub.render()


# ---------------------------------------------------------------------------
# Autofocus LED handoff (transitions 4/6/7). Driven on a real
# Lumascope(simulate=True) so illumination events fire, with a real
# AutofocusRunner whose focus loop (_iterate) is stubbed to converge on the
# first pass. The stub isolates the AF *LED lifecycle* (entry illuminate +
# the finally keep/restore) from the focus algorithm, which writes no LED.
# ---------------------------------------------------------------------------


def _af_runner(scope):
    from modules.autofocus_runner import AutofocusRunner

    r = AutofocusRunner(
        scope=scope,
        camera_executor=MagicMock(),
        io_executor=MagicMock(),
        file_io_executor=MagicMock(),
    )
    r._objective_loader = MagicMock()
    r._objective_loader.get_objective_info.return_value = {
        'AF_range': 50.0,
        'AF_max': 10.0,
        'AF_min': 5.0,
    }

    def _converge():
        # Signal success on the first pass: the run() loop sees
        # _is_focusing_event cleared and exits with completed_successfully=True,
        # so the real LED finally branch (keep vs restore) runs.
        r._best_focus_position = scope.motion.get_current_position('Z')
        r._is_complete_event.set()
        r._is_focusing_event.clear()

    r._iterate = _converge
    return r


def _drive_af(runner, **overrides):
    kwargs = {
        'objective_id': '4x Oly',
        'run_trigger_source': 'manual',
        'abort_event': threading.Event(),
    }
    kwargs.update(overrides)
    return runner.run(**kwargs)


def test_s5_protocol_af_same_channel_holds_to_capture(scope):
    """Protocol AF on the capture channel, keep_led_on=True: AF lights the
    channel once and HOLDS it (no off+on between AF end and capture) so the
    capture inherits a single lit segment spanning AF->capture (pins #612)."""
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)

    lease = ill.acquire_led_lease('protocol', alive=lambda: True)
    _drive_af(
        _af_runner(scope),
        led_color='Green',
        led_illumination=250.0,
        keep_led_on=True,
        led_lease=lease,
        run_trigger_source='protocol',
    )
    assert sub.on_events() == [('Green', 250.0)], sub.render()
    assert sub.lit_transitions('Green') == [True], sub.render()  # lit, never offed
    assert sub.final_lit() == {'Green'}, sub.render()
    assert sub.lit_at_most_one(), sub.render()
    _assert_only_lit(sub, 'Green')

    lease.release(leave_on=False)


def test_s6_protocol_af_then_different_color_no_stale_channel(scope):
    """Protocol AF on Green (kept for the same-channel capture), then the next
    step switches to Red: the AF/capture channel is switched off and Red lit at
    the boundary, no stale AF channel left lit."""
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)

    lease = ill.acquire_led_lease('protocol', alive=lambda: True)
    _drive_af(
        _af_runner(scope),
        led_color='Green',
        led_illumination=250.0,
        keep_led_on=True,
        led_lease=lease,
        run_trigger_source='protocol',
    )
    # Next-color step light (the protocol's STEP_LIGHT for a Red step): the
    # exclusive-Red diff the run drives on the held lease.
    ill._emit_led_diff(frozenset({(ill.color2ch('Red'), 350.0)}), owner='protocol', block=False)

    assert sub.on_events() == [('Green', 250.0), ('Red', 350.0)], sub.render()
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    assert sub.lit_transitions('Red') == [True], sub.render()
    assert sub.final_lit() == {'Red'}, f'stale AF channel left lit\n{sub.render()}'
    assert sub.lit_at_most_one(), f'double illumination at the boundary\n{sub.render()}'

    lease.release(leave_on=False)


def test_s7_interactive_af_restores_prerun_live_channel(scope):
    """Interactive AF (keep_led_on=False) with a pre-AF Live LED on a different
    channel: AF makes its channel exclusive (offing the Live channel), then
    restores the pre-AF Live channel on exit -- no stale AF channel, original
    Live state back (pins #695 restore path)."""
    ill = scope.illumination
    ill.led_on(channel=ill.color2ch('Blue'), mA=120.0, owner='ui')

    sub = LedSubstream()
    ill.add_led_listener(sub)
    _drive_af(
        _af_runner(scope),
        led_color='Green',
        led_illumination=250.0,
        keep_led_on=False,
        led_lease=None,
        run_trigger_source='manual',
    )
    assert sub.on_events() == [('Green', 250.0), ('Blue', 120.0)], sub.render()
    assert sub.lit_transitions('Green') == [True, False], sub.render()  # AF exclusive then off
    assert sub.lit_transitions('Blue') == [True], sub.render()  # restored
    assert sub.final_lit() == {'Blue'}, sub.render()
    # The restore must off Green before re-lighting Blue -- never two lit at once.
    assert sub.lit_at_most_one(), f'double illumination during AF restore\n{sub.render()}'


# ---------------------------------------------------------------------------
# Run-start LED-lease arbitration. Contention is decided on the resource by
# holder liveness: a lease stranded by a provably-dead owner is reclaimed with
# the evidence logged and the run proceeds; a LIVE holder refuses the run,
# which must fail itself instead of stealing illumination mid-operation.
# ---------------------------------------------------------------------------


def test_run_recovers_a_stranded_led_lease(scope, runner, tmp_path, caplog):
    """A lease whose owner is provably dead (its liveness probe answers False)
    must not lock out the next run: the run's acquire reclaims the stack,
    logging the dead owner and the evidence, and the run completes normally."""
    ill = runner._scope.illumination
    # Simulate a hard-killed prior run: a 'protocol' lease left on the stack
    # whose in-flight probe still answers True at acquire time...
    holder_alive = {'value': True}
    stranded = ill.acquire_led_lease('protocol', alive=lambda: holder_alive['value'])
    assert stranded is not None
    assert ill.acquire_led_lease('other', alive=lambda: True) is None, (
        'precondition: a live holder refuses a second acquire'
    )
    # ...and then the owning run dies without releasing.
    holder_alive['value'] = False

    with caplog.at_level(logging.WARNING, logger='LVP.api'):
        completed, result = _run_protocol(runner, _build_protocol([('A1', 'Green', {})]), tmp_path)

    assert completed, 'the run must complete after reclaiming the stranded lease'
    assert result.get('status') == 'completed', f'run must complete normally; got {result}'
    assert not stranded.held, 'the stranded lease must be dropped by the reclaim'
    assert ill.led_lease_owner is None, 'the completed run must have released its lease'
    reclaims = [
        r.getMessage() for r in caplog.records if 'reclaimed from stranded owner' in r.getMessage()
    ]
    assert reclaims, 'the reclaim must be logged as a warning'
    assert any("'protocol'" in m and 'liveness probe returned False' in m for m in reclaims), (
        f'the warning must name the dead owner and the evidence; got {reclaims}'
    )


def test_run_start_refused_by_live_lease_holder_fails_itself(scope, runner, tmp_path, monkeypatch):
    """A run started while a LIVE owner holds the LED lease must fail itself
    (run_complete fires exactly once with status 'failed_at_start', the user is
    notified) instead of stealing the lease: the holder keeps illumination
    authority and its applies still drive the LEDs."""
    import modules.notification_center as notification_center

    notified = []
    monkeypatch.setattr(
        notification_center.notifications,
        'error',
        lambda *args, **kwargs: notified.append(('error', args)),
    )
    monkeypatch.setattr(
        notification_center.notifications,
        'warning',
        lambda *args, **kwargs: notified.append(('warning', args)),
    )

    ill = scope.illumination
    af_lease = ill.acquire_led_lease('autofocus', alive=lambda: True)
    assert af_lease is not None

    completions = []
    done = threading.Event()

    def on_complete(**kwargs):
        completions.append(kwargs)
        done.set()

    plan = runner.prepare(
        keep_led_between_steps=False,
        protocol=_build_protocol([('A1', 'Green', {})]),
        run_trigger_source='test',
        run_mode=SequencedCaptureRunMode.SINGLE_SCAN,
        sequence_name='led_lease_live_holder',
        image_capture_config=ImageCaptureConfig.from_image_mode('8bit'),
        autogain_settings={
            'target_brightness': 0.3,
            'min_gain_db': 0.0,
            'max_gain_db': 20.0,
            'max_duration': datetime.timedelta(seconds=1),
        },
        parent_dir=tmp_path / 'output',
        max_scans=1,
        callbacks={'run_complete': on_complete},
        leds_state_at_end='off',
        initial_autofocus_states={
            'BF': False,
            'PC': False,
            'DF': False,
            'Red': False,
            'Green': False,
            'Blue': False,
            'Lumi': False,
        },
    )
    runner.start(plan)

    assert done.wait(timeout=30), 'the refused run must still terminate'
    assert len(completions) == 1, (
        f'run_complete must fire exactly once for the refused run; got {completions}'
    )
    assert completions[0].get('status') == 'failed_at_start', (
        f'the lease refusal must fail the run at start; got {completions[0]}'
    )
    assert notified, 'the failed start must notify the user'
    assert not runner.run_in_progress()

    # The live holder was not disturbed: its lease is held and still drives LEDs.
    assert af_lease.held, 'the live holder lease must survive the refused run'
    assert ill.led_lease_owner == 'autofocus'
    af_lease.apply(
        LedTransition.AF_ENTER,
        LedTransitionCtx(channel=ill.color2ch('Green'), mA=250.0),
    )
    assert ill.led_enabled('Green'), "the holder's apply must still drive the LEDs"
    af_lease.release(leave_on=False)


# ---------------------------------------------------------------------------
# Dark sample on write-path failure. A capture that cannot hand its frame to
# the file writer must never leave the sample lit: the run loop's inter-scan
# epilogue darkens after a mid-scan exception the loop classifies as
# transient, and a wedged file writer aborts the run -- whose cleanup owns
# the darkening. Without those, the channel stays lit on the sample for the
# whole inter-scan period, or indefinitely on an abort.
# ---------------------------------------------------------------------------


@pytest.fixture
def bounded_file_executors():
    """Executor set whose file-IO worker has a 1-slot bounded protocol queue.

    Production bounds the file queue too (the registry passes 32); maxsize=1
    makes the full-queue condition reachable with one wedge task plus one
    filler instead of 32 in-flight writes.
    """
    yield from _make_executors(file_queue_maxsize=1)


@pytest.fixture
def bounded_runner(scope, bounded_file_executors):
    return _make_runner(scope, bounded_file_executors)


def _build_two_scan_protocol(specs):
    """A protocol whose period is near zero, so the next scan starts as soon
    as the run loop's pacing check passes -- multi-scan runs finish in test
    time instead of waiting the builder's default 20-minute period."""
    protocol = _build_protocol(specs)
    protocol.modify_time_params(
        period=datetime.timedelta(milliseconds=10),
        duration=datetime.timedelta(hours=48.0),
    )
    return protocol


def test_s11_wedged_writer_aborts_run_and_goes_dark(scope, bounded_runner, tmp_path, monkeypatch):
    """A wedged file writer mid-run declares a stall: the capture fails, a
    fatal 'File Writer Stalled' notification fires, the run aborts, and the
    abort's cleanup leaves no LED lit on the sample. No capture is ever
    silently dropped -- the old queue-full drop path is unreachable by
    design now that the write submit blocks for a slot.

    Drives the REAL wedge path: a bounded file queue (maxsize=1) with the
    worker parked on a wedge task and the single slot occupied by a filler,
    so the write's blocking submit finds no slot and no task ever retires.
    The stall budget is shrunk so the wedge declares in test time."""
    import modules.protocol_image_writer as piw
    from modules.notification_center import Severity, notifications
    from modules.sequential_io_executor import IOTask, PROTOCOL_ENQUEUED

    monkeypatch.setattr(piw, '_WRITE_STALL_FATAL_S', 0.5)

    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)

    file_io = bounded_runner.file_io_executor
    wedge_started = threading.Event()
    wedge_release = threading.Event()
    installed = threading.Event()
    install_results = []

    def _wedge_task():
        wedge_started.set()
        wedge_release.wait(timeout=60)

    real_capture_and_wait = scope.imaging.capture_and_wait

    def _capture_and_wait_with_wedge(*args, **kwargs):
        # Runs on the protocol worker right before the grab -- i.e. before
        # this step's write is submitted, and only once the run is in session
        # (protocol_put drops tasks outside one). First call installs the
        # wedge: the worker parks on _wedge_task and a no-op filler occupies
        # the single queue slot, so the write's blocking submit can never get
        # a slot and the stall declares. Event-gated, no sleeps; results are
        # recorded (not asserted) here because a raise on this thread would
        # be classified as a transient scan failure, not a test failure.
        if not installed.is_set():
            install_results.append(file_io.protocol_put(IOTask(action=_wedge_task)))
            install_results.append(wedge_started.wait(timeout=10))
            install_results.append(file_io.protocol_put(IOTask(action=lambda: None)))
            installed.set()
        return real_capture_and_wait(*args, **kwargs)

    monkeypatch.setattr(scope.imaging, 'capture_and_wait', _capture_and_wait_with_wedge)

    fired = []
    # remove_listener unregisters by identity, so the exact same callable
    # object must be handed to both calls.
    listener = fired.append
    notifications.add_listener(listener, min_severity=Severity.CRITICAL)

    try:
        completed, result = _run_protocol(
            bounded_runner,
            _build_two_scan_protocol([('A1', 'Green', {})]),
            tmp_path,
            max_scans=2,
            timeout=60,
        )
    finally:
        # Unpark the file worker so fixture teardown can drain and shut down.
        wedge_release.set()
        notifications.remove_listener(listener)

    assert completed, f'run_complete never fired after the wedge abort\n{sub.render()}'
    assert result.get('status') == 'aborted', (
        f'a wedged writer must abort the run; status={result.get("status")!r}'
    )
    assert install_results == [PROTOCOL_ENQUEUED, True, PROTOCOL_ENQUEUED], (
        f'wedge install did not follow the expected sequence: {install_results}'
    )
    # The old contract dropped the capture silently; the new one never does.
    assert file_io.protocol_dropped_count() == 0, (
        'back-pressure must not silently drop a capture, even against a wedged writer'
    )
    stall_notes = [n for n in fired if n.title == 'File Writer Stalled']
    assert len(stall_notes) == 1, f'expected one fatal stall notification, saw {fired}'

    # Scan 1 ON, then the wedge abort's cleanup darkens; scan 2 never starts.
    assert sub.on_events() == [('Green', 250.0)], sub.render()
    assert sub.lit_transitions('Green') == [True, False], sub.render()
    _assert_only_lit(sub, 'Green')
    assert sub.final_lit() == set(), f'wedge abort left the sample lit\n{sub.render()}'


def test_s12_transient_scan_failure_goes_dark_before_retry(scope, runner, tmp_path, monkeypatch):
    """An exception from the grab mid-scan propagates to the run loop, which
    classifies it transient and retries a full period later; the failed scan
    died between the step's illuminate and its boundary decision, so only the
    SCAN_IDLE epilogue on the transient branch darkens the sample for that
    wait. Assert the channel goes OFF after the raise and BEFORE the retry's
    ON. Fails without the epilogue: the channel then rides the retry wait
    lit, and the retry's idempotent re-light emits nothing."""
    ill = scope.illumination
    sub = LedSubstream()
    ill.add_led_listener(sub)

    real_capture_and_wait = scope.imaging.capture_and_wait
    raised = threading.Event()

    def _raise_once_then_real(*args, **kwargs):
        if not raised.is_set():
            raised.set()
            raise RuntimeError('injected transient grab failure')
        return real_capture_and_wait(*args, **kwargs)

    monkeypatch.setattr(scope.imaging, 'capture_and_wait', _raise_once_then_real)

    completed, _ = _run_protocol(
        runner,
        _build_two_scan_protocol([('A1', 'Green', {})]),
        tmp_path,
        max_scans=2,
        timeout=60,
    )
    assert completed, f'protocol did not complete in time\n{sub.render()}'
    assert raised.is_set(), 'failure injection never fired'

    # Failed attempt ON -> SCAN_IDLE OFF (the transient-branch epilogue) ->
    # retry ON (a real re-light proving the off landed before the retry) ->
    # scan-boundary OFF -> scan 2 ON -> run-end OFF.
    assert sub.on_events() == [('Green', 250.0)] * 3, sub.render()
    assert sub.lit_transitions('Green') == [True, False, True, False, True, False], sub.render()
    _assert_only_lit(sub, 'Green')
    assert sub.lit_at_most_one(), f'double illumination\n{sub.render()}'
    assert sub.final_lit() == set(), sub.render()
