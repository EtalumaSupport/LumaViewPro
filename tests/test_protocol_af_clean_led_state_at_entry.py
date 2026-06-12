"""Regression tests for the LED-state-leak cluster fix.

Bug shape: ``Lumascope.led_on(channel, mA, ...)`` is additive at every
layer (API, driver, firmware). Callers that need exclusive illumination
(only one channel lit at a time) must call ``leds_off()`` first. The
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
  blinked off->on).

Each fix establishes exclusive illumination: the protocol step path via
the capture wiring to ``leds_exclusive``, and AF via ``leds_exclusive``
after snapshotting prior state. The tests below are structural AST locks
-- they fail if a future refactor re-introduces a cache-clearing
``leds_off`` before step 0, or drops the exclusive illumination.
"""

from __future__ import annotations

import ast
import pathlib


def _module_source(rel_path: str) -> str:
    return (pathlib.Path(__file__).resolve().parent.parent / rel_path).read_text()


def _function_node(source: str, func_name: str) -> ast.FunctionDef:
    """Return the first FunctionDef matching func_name in source."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f'function {func_name!r} not found in source')


def _call_lineno(func_node: ast.FunctionDef, attr_chain: str) -> int | None:
    """Return the lineno of the FIRST call whose attribute chain ends
    with attr_chain (e.g. 'leds_off' matches any ``X.leds_off(...)``;
    'step_executor.leds_off' matches only that specific shape).

    Walks the function body recursively (handles nested ifs/trys/etc.).
    Returns None if no match.
    """
    target_parts = attr_chain.split('.')
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        # Unroll the attribute chain on the call's func: a.b.c() -> ['a','b','c']
        chain: list[str] = []
        cur: ast.AST = node.func
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain.append(cur.id)
        chain.reverse()
        # Match if the trailing N parts equal target_parts
        if len(chain) >= len(target_parts) and chain[-len(target_parts) :] == target_parts:
            return node.lineno
    return None


# ---------------------------------------------------------------------------
# Site 1: protocol scan-start leds_off
# ---------------------------------------------------------------------------


class TestProtocolRunLoopNoCacheClearingLedsOffAtScanStart:
    """Lock the ABSENCE of a cache-clearing leds_off before step 0.

    The run loop used to call a nuclear ``_step_executor.leds_off()`` before
    ``go_to_step(step_idx=0)`` to give step 0 a clean slate. That cleared the
    LED-state cache, so the following same-color ``led_on`` could not self-skip
    and blinked the LED off->on at every scan start. The clean-slate guarantee
    moved to the capture path (wired to ``leds_exclusive``), which turns off
    OTHER channels at capture time -- still killing a stray Live-mode LED so
    step 0 is not double-illuminated, but without clearing the target's cache.
    Behavioral coverage: tests/test_protocol_execution.py::TestProtocolLedNoFlash.
    """

    SRC = 'modules/protocol_run_loop.py'
    FUNC = '_run_loop_inner'
    WIRING_SRC = 'modules/sequenced_capture_runner.py'

    def test_no_nuclear_leds_off_before_go_to_step(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        leds_off_ln = _call_lineno(func, '_step_executor.leds_off')
        go_to_step_ln = _call_lineno(func, 'go_to_step')
        assert go_to_step_ln is not None, (
            f'{self.SRC}::{self.FUNC} must still call go_to_step.'
        )
        assert leds_off_ln is None or leds_off_ln > go_to_step_ln, (
            f'a nuclear _step_executor.leds_off() before go_to_step (line '
            f'{leds_off_ln}) clears the LED-state cache and re-introduces the '
            'scan-start off->on blink; step-0 exclusivity is established by the '
            'capture path instead.'
        )

    def test_capture_is_wired_to_leds_exclusive(self):
        nospace = _module_source(self.WIRING_SRC).replace(' ', '')
        assert 'led_on_fn=self._step_executor.leds_exclusive' in nospace, (
            'capture must light its channel via leds_exclusive (offs other '
            'channels, self-skips an already-lit one) so step 0 is not '
            'double-illuminated by a stray Live-mode LED and not blinked.'
        )
        assert 'led_on_fn=self._step_executor.led_on' not in nospace


# ---------------------------------------------------------------------------
# Site 2: autofocus run-start leds_off
# ---------------------------------------------------------------------------


class TestAutofocusRunnerExclusiveIlluminationAtRunStart:
    """Lock AF establishing exclusive illumination at run-start.

    Pre-AF save_led_state + post-AF restore_led_state(owner='autofocus')
    preserves user-visible LED state across an AF run, but DURING the run a
    pre-AF Live LED on another channel would leave mixed illumination on the
    focus metric and converge to the wrong Z. AF makes its own channel the
    only lit one via the idempotent exclusive primitive (leds_exclusive), or
    leds_off when no AF channel is configured, AFTER the snapshot. Using the
    exclusive primitive (rather than leds_off + led_on) also leaves an
    already-lit AF channel untouched, so AF does not blink it off->on at scan
    start; the snapshot/restore pair handles user-visible preservation.
    """

    SRC = 'modules/autofocus_runner.py'
    FUNC = 'run'

    def _clear_lineno(self, func):
        """Line of the call that establishes exclusive AF illumination --
        leds_exclusive (configured channel) or leds_off (ambient fallback)."""
        exclusive_ln = _call_lineno(func, 'leds_exclusive')
        return exclusive_ln if exclusive_ln is not None else _call_lineno(func, 'leds_off')

    def test_exclusive_illumination_call_exists_in_run(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        assert self._clear_lineno(func) is not None, (
            f'{self.SRC}::{self.FUNC} must establish exclusive AF '
            'illumination (leds_exclusive, or leds_off when no AF channel is '
            'configured) so the scan is not corrupted by a pre-AF Live LED.'
        )

    def test_exclusive_illumination_follows_save_led_state(self):
        """save_led_state must snapshot BEFORE the illumination is changed so
        the pre-AF state can be restored on exit. Order:
        save_led_state -> leds_exclusive/leds_off -> ... -> restore_led_state.
        """
        func = _function_node(_module_source(self.SRC), self.FUNC)
        save_ln = _call_lineno(func, 'save_led_state')
        clear_ln = self._clear_lineno(func)
        assert save_ln is not None and clear_ln is not None, (
            f'{self.SRC}::{self.FUNC} must call save_led_state and then '
            f'establish exclusive illumination (got save_led_state={save_ln}, '
            f'clear={clear_ln}).'
        )
        assert save_ln < clear_ln, (
            f'save_led_state (line {save_ln}) must precede the exclusive '
            f'illumination call (line {clear_ln}) so the pre-AF LED state is '
            'captured before being changed. Otherwise post-AF '
            'restore_led_state would restore the wrong snapshot.'
        )
