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
  and converge to the wrong Z. AF now uses ``leds_exclusive`` so its
  channel is the only one lit (and an already-lit AF channel is not
  blinked off->on), AFTER snapshotting prior state for the exit restore.

The tests drive the real run loop / AF run headlessly (via
tests/protocol_drives.py and tests/af_drives.py) and observe the LED
traffic, so a refactor that re-introduces the cache-clearing ``leds_off``
before step 0 or drops the exclusive illumination fails behaviorally.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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

    def test_capture_is_wired_to_leds_exclusive(self):
        """run() must hand the image writer the exclusive primitive (offs
        other channels, self-skips an already-lit one), not the additive
        led_on, so step 0 is not double-illuminated by a stray Live-mode
        LED and a same-color step is not blinked."""
        runner = bare_capture_runner()
        runner.run(**scr_run_kwargs())
        assert runner._image_writer._led_on == runner._step_executor.leds_exclusive, (
            "the writer's LED-on hook must be the step executor's "
            'leds_exclusive, not the additive led_on'
        )


# ---------------------------------------------------------------------------
# Site 2: autofocus run-start illumination
# ---------------------------------------------------------------------------


class TestAutofocusRunnerExclusiveIlluminationAtRunStart:
    """AF makes its channel the only lit one (or clears LEDs when no AF
    channel is configured) AFTER snapshotting pre-AF state, and restores
    the snapshot on exit -- so the focus metric never sees mixed
    illumination and the user's LED state survives the run."""

    def test_exclusive_illumination_follows_save_led_state(self, monkeypatch):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        scope.led_connected = True
        drive_af(runner, led_color='Red', led_illumination=42.0)

        names = [name for name, args, kwargs in scope.illumination.method_calls]
        assert 'leds_exclusive' in names, (
            f'AF must make its channel the only lit one; LED calls: {names}'
        )
        exclusive_kwargs = scope.illumination.leds_exclusive.call_args.kwargs
        assert exclusive_kwargs['mA'] == 42.0 and exclusive_kwargs['owner'] == 'autofocus'
        assert names.index('save_led_state') < names.index('leds_exclusive'), (
            'the pre-AF LED snapshot must precede the illumination change, '
            'or the exit restore would restore the wrong (already-changed) '
            f'state; LED calls: {names}'
        )
        assert 'restore_led_state' in names[names.index('leds_exclusive') :], (
            f'AF exit must restore the pre-AF snapshot; LED calls: {names}'
        )

    def test_ambient_fallback_clears_leds(self, monkeypatch):
        """No AF channel configured: any Live-mode LED must be cleared so
        it does not bias the focus metric."""
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        drive_af(runner)
        names = [name for name, args, kwargs in scope.illumination.method_calls]
        assert 'leds_off' in names, (
            f'ambient AF must clear the LEDs before scanning; LED calls: {names}'
        )
        assert names.index('save_led_state') < names.index('leds_off'), (
            f'the snapshot must precede the clear; LED calls: {names}'
        )


class TestAutofocusAcquiresLeaseBeforeIllumination:
    """AF must hold its LED lease BEFORE it drives illumination.

    The illumination write carries owner 'autofocus'. If it is issued
    before AF holds a lease, a protocol's already-held lease refuses the
    out-of-turn write and the AF channel never lights -- AF then scans an
    unlit field, the focus metric reads noise, and gain/exposure climb
    chasing nothing. The lease acquire must therefore precede the
    leds_exclusive call on both AF paths (interactive top-level lease and
    in-protocol child lease)."""

    def test_top_level_lease_precedes_illumination(self, monkeypatch):
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        scope.led_connected = True
        drive_af(runner, led_color='Red', led_illumination=42.0)
        names = [name for name, args, kwargs in scope.illumination.method_calls]
        assert 'acquire_led_lease' in names and 'leds_exclusive' in names, (
            f'expected a lease acquire and an illumination write; LED calls: {names}'
        )
        assert names.index('acquire_led_lease') < names.index('leds_exclusive'), (
            'the LED lease must be acquired before illumination is driven, or '
            "a protocol's held lease refuses the write and AF scans dark; "
            f'LED calls: {names}'
        )

    def test_child_lease_precedes_illumination(self, monkeypatch):
        """In-protocol path: AF takes a child lease under the protocol's
        lease. The child acquire must precede the illumination write."""
        monkeypatch.setattr('modules.autofocus_functions.focus_function', lambda image: 7.0)
        runner, scope = af_runner_and_scope()
        scope.led_connected = True
        # Attach the parent lease to the scope mock so its acquire_child
        # call is recorded in scope.mock_calls alongside the illumination
        # write -- one ordered record spanning both objects.
        parent_lease = scope.protocol_lease
        drive_af(runner, led_color='Red', led_illumination=42.0, led_lease=parent_lease)
        ordered = [call[0] for call in scope.mock_calls]
        acquire = next(
            i for i, name in enumerate(ordered) if name.endswith('protocol_lease.acquire_child')
        )
        illuminate = next(
            i for i, name in enumerate(ordered) if name.endswith('illumination.leds_exclusive')
        )
        assert acquire < illuminate, (
            'the child LED lease must be acquired before illumination is '
            f'driven; call order: {ordered}'
        )
