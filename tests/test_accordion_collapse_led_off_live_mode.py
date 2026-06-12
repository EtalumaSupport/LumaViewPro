"""Regression test for #659 -- Live-mode LED stuck after channel switch.

Bug shape: ``ui/image_settings.py::_do_accordion_collapse`` skipped
``scope_leds_off()`` whenever the ``protocol_led_on`` setting was True,
regardless of whether a protocol was actively running. The setting
persists across protocol runs, so a user who had enabled the
"Protocol LEDs On" feature once would see Live-mode accordion-switch
behavior change permanently: previously-enabled channel LEDs would
stay lit until the user explicitly Enabled the new channel.

The original guard (commit 6760fe3 "fixes #605") was intended to
protect the LED that ``step_navigation`` had just turned on during a
protocol step. That intent only applies while a protocol is actively
running -- which is exactly what the second guard at line 538
already checks. Collapsing the two guards into one ``protocol_running``
check matches the original intent and fixes #659.

Test is structural: walk the source and assert that the
protocol_led_on-alone guard is gone.
"""

from __future__ import annotations

import ast
import pathlib


def _do_accordion_collapse_source() -> str:
    src_path = pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'image_settings.py'
    source = src_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_do_accordion_collapse':
            text = ast.get_source_segment(source, node)
            assert text is not None
            return text
    raise AssertionError('_do_accordion_collapse not found in ui/image_settings.py')


class TestAccordionCollapseLedOffInLiveMode:
    """Lock the fix shape for #659.

    The standalone ``protocol_led_on`` early-return guard is gone; only
    ``protocol_running.is_set()`` gates the leds_off skip. Otherwise
    Live-mode accordion-switching incorrectly preserves the previous
    channel's LED whenever the user had enabled Protocol LEDs On in a
    prior session.
    """

    def test_no_standalone_protocol_led_on_guard(self):
        body = _do_accordion_collapse_source()
        # The buggy shape was: if ctx.settings.get('protocol_led_on', False): return
        # The fixed shape is: only the protocol_running.is_set() guard remains.
        # Detect the buggy shape via AST so the test fails if it returns.
        tree = ast.parse(body)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # Look for `if ctx.settings.get('protocol_led_on', ...): return`
            # at any depth. AST walks all if statements.
            test = node.test
            # ctx.settings.get('protocol_led_on', ...) is a Call
            if isinstance(test, ast.Call):
                func = test.func
                args = test.args
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'get'
                    and args
                    and isinstance(args[0], ast.Constant)
                    and args[0].value == 'protocol_led_on'
                ):
                    # Found the buggy guard -- check it's not followed
                    # by a bare return.
                    body_returns = len(node.body) == 1 and isinstance(node.body[0], ast.Return)
                    assert not body_returns, (
                        '_do_accordion_collapse must not contain a '
                        'standalone `if protocol_led_on: return` guard. '
                        'This skips LED cleanup in Live mode whenever '
                        'the user had previously enabled Protocol LEDs '
                        'On, causing #659 (LED stuck after accordion '
                        'switch). Gate on ctx.protocol_running.is_set() '
                        'instead.'
                    )

    def test_still_guards_during_protocol_running(self):
        body = _do_accordion_collapse_source()
        # The protocol-active guard must remain so step_navigation's
        # LED-on for the current step survives accordion-collapse events
        # fired during the step transition.
        assert 'protocol_running.is_set()' in body, (
            '_do_accordion_collapse must still skip LED cleanup when a '
            "protocol is actively running. Without this, step_navigation's "
            'LED-on for the current step would be killed by the accordion-'
            'collapse event that fires when set_expanded_layer opens the '
            "step's channel (#605)."
        )

    def test_offs_collapsed_layers_individually(self):
        body = _do_accordion_collapse_source()
        # The LED cleanup must still be present below the guards: the
        # collapsed (non-open) layers' channels are switched off so a
        # previously-lit channel is cleared on a Live-mode drawer switch
        # (#659).
        assert 'led_off_async(' in body, (
            '_do_accordion_collapse must clear the collapsed layers LEDs '
            "(led_off_async) so the previous channel's LED is turned off "
            "before applying the new layer's settings (#659)."
        )

    def test_does_not_nuke_all_leds(self):
        body = _do_accordion_collapse_source()
        # The cache-clearing nuclear leds_off must not be used here: it blinked
        # the open layer's own channel (e.g. one a step just lit) off->on when
        # the drawer switched. Off the collapsed layers individually instead.
        assert 'scope_leds_off' not in body, (
            '_do_accordion_collapse must not call the nuclear scope_leds_off '
            '-- it clears the LED-state cache and blinks an already-correct '
            'channel off->on. Off the collapsed layers individually (#697).'
        )
