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

import threading

import pytest

from modules.exceptions import AutofocusFailedError
from modules.lumascope_api.illumination import LedState
from tests.af_drives import af_runner_on, sweep, z_asked
from tests.stock_channels import LED1, LED2


def _lit(scope):
    return [r.name for r in scope.illumination.get_led_states() if r.state is LedState.LIT]


# ---------------------------------------------------------------------------
# Site 2: autofocus run-start illumination
# ---------------------------------------------------------------------------


class TestAutofocusRunnerExclusiveIlluminationAtRunStart:
    """AF makes its channel the only lit one (or clears LEDs when no AF
    channel is configured) AFTER snapshotting pre-AF state, and restores
    the snapshot on exit -- so the focus metric never sees mixed
    illumination and the user's LED state survives the run."""

    def test_ambient_fallback_clears_leds(self, sim_scope, monkeypatch):
        """No AF channel configured: AF_ENTER's target is empty, so the
        authority diff clears every channel and ambient AF is not biased by a
        stray Live-mode LED; the snapshot taken before the clear puts the
        stray LED back when the sweep ends."""
        runner = af_runner_on(sim_scope)
        sim_scope.illumination.led_on(LED1, 5.0)
        lit_at_grabs = []
        grab = sim_scope.imaging.capture_still

        def watched(*args, **kwargs):
            lit_at_grabs.append(_lit(sim_scope))
            return grab(*args, **kwargs)

        monkeypatch.setattr(sim_scope.imaging, 'capture_still', watched)

        # Ambient on the simulator is a dark field: the sweep fails flat.
        with pytest.raises(AutofocusFailedError):
            sweep(runner, sim_scope, led_color=None, led_illumination=0.0)

        assert lit_at_grabs and all(lit == [] for lit in lit_at_grabs), (
            f'no AF color -> every channel cleared before the sweep grabs; lit at grabs: {lit_at_grabs}'
        )
        assert _lit(sim_scope) == [LED1], (
            f'the snapshot must precede the clear, so the stray LED comes back; lit: {_lit(sim_scope)}'
        )


class TestAutofocusAcquiresLeaseBeforeIllumination:
    """AF must hold its LED lease BEFORE it drives illumination.

    AF illuminates by calling apply(AF_ENTER) ON its lease, so holding the
    lease before illumination is now structural: a refused acquire leaves the
    field as-is rather than driving an out-of-turn write that a protocol's
    held lease would refuse (AF scanning an unlit field). This pins that the
    AF_ENTER illuminate runs on the child lease AF takes under its run's."""

    def test_autofocus_without_a_run_lease_is_refused_before_it_drives(
        self, sim_scope, monkeypatch
    ):
        """AF always runs inside a run, so a call with no run lease is
        refused at the call: no lease of its own, no LED or Z traffic."""
        runner = af_runner_on(sim_scope)
        sim_scope.illumination.led_on(LED1, 5.0)
        asked = z_asked(sim_scope, monkeypatch)
        with pytest.raises(TypeError, match='led_lease'):
            runner.run(
                objective_id='10x Oly',
                abort_event=threading.Event(),
                led_color=LED2,
                led_illumination=42.0,
            )
        assert _lit(sim_scope) == [LED1], (
            f'no illumination may precede the refusal; lit {_lit(sim_scope)}'
        )
        assert asked == [], f'no motion may precede the refusal; Z asked {asked}'
