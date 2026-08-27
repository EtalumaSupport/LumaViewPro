"""Regression tests for the LED-state-leak cluster fix.

Bug shape: ``Lumascope.led_on(channel, mA, ...)`` is additive at every
layer (API, driver, firmware). Callers that need exclusive illumination
(only one channel lit at a time) must establish it explicitly. The
convention is documented at ``modules/step_navigation.py`` and
``modules/composite_capture.py``.

Two mode-entry sites were silently skipping the convention:

* ``modules/protocol_run_loop.py::_run_loop_inner`` -- at scan start
  with a Live-mode LED still on from before the user pressed Scan, the
  first protocol step's image would be lit by both the pre-scan LED and
  the step's own channel. The run loop used to fix this with a nuclear
  ``leds_off`` before step 0, but that cleared the LED-state cache so the
  following same-color ``led_on`` could not self-skip and blinked the LED
  off->on at every scan start. The clean slate now comes from the capture
  path making its channel exclusive (off other channels, leave an
  already-correct channel untouched) -- no leak into step 0, no blink.

* ``modules/autofocus_runner.py::run`` -- at AF start with a Live-mode
  LED on a different channel than the AF channel, additive illumination
  would leave both lit. AF's focus metric would see mixed illumination
  and converge to the wrong Z. AF now makes its channel the only one lit
  via the LED authority (the AF_ENTER transition diffs off other channels,
  and an already-lit AF channel is not blinked off->on), AFTER snapshotting
  prior state for the exit restore.

The tests drive the real run loop / AF run headlessly (via
tests/protocol_drives.py and tests/af_drives.py) and observe the LED
traffic, so a refactor that re-introduces the cache-clearing ``leds_off``
before step 0 or drops the exclusive illumination fails behaviorally.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from modules.lumascope_api.illumination import LedTransition
from tests.af_drives import af_runner_and_scope, drive_af
from tests.protocol_drives import (
    bare_capture_runner,
    protocol_step,
    run_loop_ready_runner,
    scr_run_kwargs,
)


# ---------------------------------------------------------------------------
# Site 1: protocol scan-start leds_off
# ---------------------------------------------------------------------------


class TestProtocolRunLoopNoCacheClearingLedsOffAtScanStart:
    """The run loop must not queue a nuclear leds_off before step 0; the
    clean slate comes from the capture path's exclusive wiring."""

    def test_no_nuclear_leds_off_before_go_to_step(self):
        runner = run_loop_ready_runner(protocol_step())
        events = []
        runner._callbacks.go_to_step.side_effect = lambda **kwargs: events.append('go_to_step')

        def recording_put(task, **kwargs):
            if getattr(task, 'action', None) is runner._scope.illumination.leds_off:
                events.append('leds_off')
            return MagicMock()

        runner._io_executor.protocol_put.side_effect = recording_put
        runner._run_loop_executor.run_loop()
        assert 'go_to_step' in events, 'the scan must reach go_to_step'
        before_step_zero = events[: events.index('go_to_step')]
        assert 'leds_off' not in before_step_zero, (
            'a nuclear leds_off before go_to_step clears the LED-state cache '
            'and re-introduces the scan-start off->on blink; LED traffic '
            f'observed: {events}'
        )

    def test_capture_leaf_does_not_drive_the_led(self):
        """The run-lifecycle illuminate lives on the authority, not the capture
        leaf: the runner applies STEP_LIGHT (offs other channels, self-skips an
        already-lit one) and confirms the channel on before the grab, so the
        leaf is a pure grab+save with no LED-on hook that could double-illuminate
        step 0 or blink a same-color step (exclusivity itself is pinned by the
        end-to-end lifecycle test's one-lit-at-a-time scenarios)."""
        runner = bare_capture_runner()
        runner.start(runner.prepare(**scr_run_kwargs()))
        assert not hasattr(runner._image_writer, '_led_on'), (
            'the capture leaf must not drive the LED; the STEP_LIGHT illuminate '
            'lives on the authority via the runner'
        )


# ---------------------------------------------------------------------------
# Site 2: autofocus run-start illumination
# ---------------------------------------------------------------------------


class TestAutofocusRunnerExclusiveIlluminationAtRunStart:
    """AF makes its channel the only lit one (or clears LEDs when no AF
    channel is configured) AFTER snapshotting pre-AF state, and restores
    the snapshot on exit -- so the focus metric never sees mixed
    illumination and the user's LED state survives the run."""

    def _af_enter_index(self, scope):
        """Index in scope.mock_calls of the AF_ENTER illuminate on the AF lease."""
        return next(
            i
            for i, (name, args, kwargs) in enumerate(scope.mock_calls)
            if name.endswith('.apply') and args and args[0] is LedTransition.AF_ENTER
        )

    def test_af_enter_illuminate_follows_save_led_state(self, monkeypatch):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        scope.led_connected = True
        drive_af(runner, led_color='Red', led_illumination=42.0)

        # AF makes its channel the only lit one via the authority's AF_ENTER
        # (the diff offs other channels) and restores the pre-AF snapshot on
        # exit via AF_TO_CAPTURE -- both transitions driven on the AF lease,
        # replacing the old direct per-channel LED calls AF made before the
        # authority.
        af_lease = scope.illumination.acquire_led_lease.return_value
        applied = [c.args[0] for c in af_lease.apply.call_args_list if c.args]
        assert LedTransition.AF_ENTER in applied, (
            f'AF must illuminate via the authority AF_ENTER; applied: {applied}'
        )
        assert LedTransition.AF_TO_CAPTURE in applied, (
            f'AF exit must restore via AF_TO_CAPTURE; applied: {applied}'
        )
        enter_ctx = next(
            c.args[1]
            for c in af_lease.apply.call_args_list
            if c.args and c.args[0] is LedTransition.AF_ENTER
        )
        assert enter_ctx.illumination_ma == 42.0
        # The pre-AF snapshot must precede the AF_ENTER illuminate, or the exit
        # restore would capture the already-changed (post-illuminate) state.
        ordered = [name for name, args, kwargs in scope.mock_calls]
        save_idx = next(i for i, n in enumerate(ordered) if n.endswith('save_led_state'))
        assert save_idx < self._af_enter_index(scope), (
            f'pre-AF snapshot must precede the AF_ENTER illuminate; calls: {ordered}'
        )

    def test_ambient_fallback_clears_leds(self, monkeypatch):
        """No AF channel configured: AF_ENTER's target is empty, so the
        authority diff clears every channel and ambient AF is not biased by a
        stray Live-mode LED."""
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        drive_af(runner)
        af_lease = scope.illumination.acquire_led_lease.return_value
        enter = [
            c
            for c in af_lease.apply.call_args_list
            if c.args and c.args[0] is LedTransition.AF_ENTER
        ]
        assert enter, 'ambient AF must still drive AF_ENTER (an empty target clears the LEDs)'
        assert enter[0].args[1].channel is None, (
            'no AF color -> empty AF_ENTER target -> every channel cleared'
        )
        ordered = [name for name, args, kwargs in scope.mock_calls]
        save_idx = next(i for i, n in enumerate(ordered) if n.endswith('save_led_state'))
        assert save_idx < self._af_enter_index(scope), (
            f'the snapshot must precede the clear; calls: {ordered}'
        )


class TestAutofocusAcquiresLeaseBeforeIllumination:
    """AF must hold its LED lease BEFORE it drives illumination.

    AF illuminates by calling apply(AF_ENTER) ON its lease, so holding the
    lease before illumination is now structural: a refused acquire leaves the
    field as-is rather than driving an out-of-turn write that a protocol's
    held lease would refuse (AF scanning an unlit field). These pin that the
    AF_ENTER illuminate runs on the correctly-acquired lease on both paths
    (interactive top-level lease and in-protocol child lease)."""

    def test_top_level_lease_precedes_illumination(self, monkeypatch):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        scope.led_connected = True
        drive_af(runner, led_color='Red', led_illumination=42.0)
        names = [name for name, args, kwargs in scope.illumination.method_calls]
        assert 'acquire_led_lease' in names, f'AF must acquire a top-level lease; calls: {names}'
        af_lease = scope.illumination.acquire_led_lease.return_value
        applied = [c.args[0] for c in af_lease.apply.call_args_list if c.args]
        assert LedTransition.AF_ENTER in applied, (
            f'AF must illuminate via AF_ENTER on its acquired lease; applied: {applied}'
        )

    def test_child_lease_precedes_illumination(self, monkeypatch):
        """In-protocol path: AF takes a child lease under the protocol's
        lease and illuminates through it. The child acquire must precede the
        AF_ENTER illuminate (it is a method on the child, so structurally so)."""
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        scope.led_connected = True
        # Attach the parent lease to the scope mock so its acquire_child call
        # is recorded in scope.mock_calls alongside the AF_ENTER apply on the
        # child -- one ordered record spanning both objects.
        parent_lease = scope.protocol_lease
        drive_af(runner, led_color='Red', led_illumination=42.0, led_lease=parent_lease)
        acquire = next(
            i
            for i, (name, args, kwargs) in enumerate(scope.mock_calls)
            if name.endswith('protocol_lease.acquire_child')
        )
        illuminate = next(
            i
            for i, (name, args, kwargs) in enumerate(scope.mock_calls)
            if name.endswith('.apply') and args and args[0] is LedTransition.AF_ENTER
        )
        assert acquire < illuminate, (
            'the child LED lease must be acquired before AF_ENTER illuminates; '
            f'call order: {[n for n, a, k in scope.mock_calls]}'
        )
