"""Regression tests for the LED-state-leak cluster fix.

Bug shape: ``Lumascope.led_on(channel, mA, ...)`` is additive at every
layer (API, driver, firmware). Callers that need exclusive illumination
(only one channel lit at a time) must call ``leds_off()`` first. The
convention is documented at ``modules/step_navigation.py`` and
``modules/composite_capture.py``.

Two mode-entry sites were silently skipping the convention:

* ``modules/protocol_run_loop.py::_run_loop_inner`` — at scan start
  with a Live-mode LED still on from before the user pressed Scan, the
  first protocol step's ``led_on`` would add its channel on top of the
  pre-scan LED. Both LEDs lit, first step's image blown out.

* ``modules/autofocus_runner.py::run`` — at AF start with a Live-mode
  LED on a different channel than the AF channel, AF's ``_led_on``
  would add its channel on top. AF's focus metric would see mixed
  illumination and converge to the wrong Z.

Each fix inserts ``leds_off`` at the mode-entry hook BEFORE the
operation's first ``led_on`` (or motion that precedes it). The tests
below are structural AST locks — they fail if a future refactor drops
the call, reorders it, or moves it past the first ``led_on``.
"""

from __future__ import annotations

import ast
import pathlib


def _module_source(rel_path: str) -> str:
    return (pathlib.Path(__file__).resolve().parent.parent
            / rel_path).read_text()


def _function_node(source: str, func_name: str) -> ast.FunctionDef:
    """Return the first FunctionDef matching func_name in source."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"function {func_name!r} not found in source")


def _call_lineno(func_node: ast.FunctionDef, attr_chain: str) -> int | None:
    """Return the lineno of the FIRST call whose attribute chain ends
    with attr_chain (e.g. 'leds_off' matches any ``X.leds_off(...)``;
    'step_executor.leds_off' matches only that specific shape).

    Walks the function body recursively (handles nested ifs/trys/etc.).
    Returns None if no match.
    """
    target_parts = attr_chain.split(".")
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
        if len(chain) >= len(target_parts) and chain[-len(target_parts):] == target_parts:
            return node.lineno
    return None


# ---------------------------------------------------------------------------
# Site 1: protocol scan-start leds_off
# ---------------------------------------------------------------------------

class TestProtocolRunLoopLedsOffAtScanStart:
    """Lock the scan-start leds_off in protocol_run_loop._run_loop_inner.

    The leds_off must be called BEFORE go_to_step(step_idx=0) so the
    stage moves with LEDs off and step 0's led_on fires into a clean
    state. The capture path's existing inter-step leds_off only covers
    transitions between steps; step 0 has no previous step to clean up
    after it, so the scan-start hook is the only place this happens.
    """

    SRC = "modules/protocol_run_loop.py"
    FUNC = "_run_loop_inner"

    def test_leds_off_call_exists_in_run_loop_inner(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        lineno = _call_lineno(func, "leds_off")
        assert lineno is not None, (
            f"{self.SRC}::{self.FUNC} must call leds_off() at scan start. "
            "Without this, a Live-mode LED enabled before Scan press leaks "
            "into step 0's illumination (issue #666 root cause)."
        )

    def test_leds_off_precedes_go_to_step(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        leds_off_ln = _call_lineno(func, "leds_off")
        go_to_step_ln = _call_lineno(func, "go_to_step")
        assert leds_off_ln is not None and go_to_step_ln is not None, (
            f"{self.SRC}::{self.FUNC} must contain both leds_off and "
            f"go_to_step calls (got leds_off={leds_off_ln}, "
            f"go_to_step={go_to_step_ln})."
        )
        assert leds_off_ln < go_to_step_ln, (
            f"leds_off (line {leds_off_ln}) must precede go_to_step "
            f"(line {go_to_step_ln}) in {self.FUNC}. Otherwise step 0's "
            "motion runs with the pre-scan LED still lit and the first "
            "captured image is blown out."
        )

    def test_leds_off_precedes_scan_loop(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        leds_off_ln = _call_lineno(func, "leds_off")
        scan_loop_ln = _call_lineno(func, "scan_loop")
        assert leds_off_ln is not None and scan_loop_ln is not None, (
            f"{self.SRC}::{self.FUNC} must contain both leds_off and "
            f"scan_loop calls (got leds_off={leds_off_ln}, "
            f"scan_loop={scan_loop_ln})."
        )
        assert leds_off_ln < scan_loop_ln, (
            f"leds_off (line {leds_off_ln}) must precede scan_loop "
            f"(line {scan_loop_ln}) in {self.FUNC}."
        )


# ---------------------------------------------------------------------------
# Site 2: autofocus run-start leds_off
# ---------------------------------------------------------------------------

class TestAutofocusRunnerLedsOffAtRunStart:
    """Lock the AF-run-start leds_off in autofocus_runner.run.

    Pre-AF save_led_state + post-AF restore_led_state(owner='autofocus')
    already preserves user-visible LED state across an AF run, but
    DURING the run, additive led_on would leave any pre-AF Live LED
    lit alongside the AF channel. The focus metric would integrate
    mixed illumination and converge to the wrong Z. Inserting leds_off
    after the save_led_state snapshot and before _led_on ensures AF
    scans with only its own channel lit; the snapshot/restore pair
    handles user-visible preservation independently.
    """

    SRC = "modules/autofocus_runner.py"
    FUNC = "run"

    def test_leds_off_call_exists_in_run(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        lineno = _call_lineno(func, "leds_off")
        assert lineno is not None, (
            f"{self.SRC}::{self.FUNC} must call leds_off before "
            "self._led_on() so the AF scan illuminates with only the "
            "AF channel (not pre-AF Live LED + AF LED combined)."
        )

    def test_leds_off_precedes_led_on(self):
        func = _function_node(_module_source(self.SRC), self.FUNC)
        leds_off_ln = _call_lineno(func, "leds_off")
        led_on_ln = _call_lineno(func, "_led_on")
        assert leds_off_ln is not None and led_on_ln is not None, (
            f"{self.SRC}::{self.FUNC} must contain both leds_off and "
            f"_led_on calls (got leds_off={leds_off_ln}, "
            f"_led_on={led_on_ln})."
        )
        assert leds_off_ln < led_on_ln, (
            f"leds_off (line {leds_off_ln}) must precede _led_on "
            f"(line {led_on_ln}) in {self.FUNC}. Otherwise AF's _led_on "
            "fires before the pre-AF Live LED is cleared and the focus "
            "metric integrates mixed illumination."
        )

    def test_leds_off_follows_save_led_state(self):
        """save_led_state must snapshot BEFORE leds_off so the
        pre-AF state can be restored on exit. The order must be:
        save_led_state -> leds_off -> _led_on -> ... -> restore_led_state.
        """
        func = _function_node(_module_source(self.SRC), self.FUNC)
        save_ln = _call_lineno(func, "save_led_state")
        leds_off_ln = _call_lineno(func, "leds_off")
        assert save_ln is not None and leds_off_ln is not None, (
            f"{self.SRC}::{self.FUNC} must call both save_led_state and "
            f"leds_off (got save_led_state={save_ln}, "
            f"leds_off={leds_off_ln})."
        )
        assert save_ln < leds_off_ln, (
            f"save_led_state (line {save_ln}) must precede leds_off "
            f"(line {leds_off_ln}) so the pre-AF LED state is captured "
            "before being cleared. Otherwise post-AF restore_led_state "
            "would restore the wrong snapshot."
        )
