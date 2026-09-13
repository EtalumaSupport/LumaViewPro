# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: a layer's stored value is what both of its widgets show.

Bug
---
``slider.value`` WAS the stored setting's representation. The three handlers
``ill_slider`` / ``gain_slider`` / ``exp_slider`` read the slider and wrote it
into ``settings[layer][...]``, and the kv rendered each text box FROM the
slider (``text: format(ill_slider.value, '.4g')`` and its two siblings). Two
consequences, both measured in the simulator:

  - The display reverted. A typed 500 mA on BF pinned the slider at its 50 mA
    convenience ceiling; the box showed 500 only until the next thing that
    moved that slider or re-applied its max, after which the box read 50 while
    the store and the LED stayed at 500.
  - The store was overwritten. Any programmatic slider write that did not take
    the ``_initializing`` flag fired the handler, which committed the slider's
    value over the user's -- including the camera-cap reconciliation, which
    logged a phantom ``SLIDER GAIN_Red 20.0`` at startup with nobody touching
    the app.

The fix
-------
The store is the setting and the widgets are two representations of it. One
private primitive writes both: the text box takes the stored value as it is,
the slider takes the nearest position it can represent. A slider's range is a
convenience range that can legitimately be narrower than what the box accepts,
so a stored value above it pins the slider while the box keeps the real
number -- and the primitive suppresses the layer's handlers for both writes,
restoring the previous flag state so a caller already suppressing keeps its
own guard.

The camera's PHYSICAL caps are a different thing from a slider's convenience
range: a stored value above a physical cap is wrong in the store, not merely
un-renderable. So the cap reconciliation runs BEFORE anything renders, and
then renders and applies explicitly instead of leaning on a handler firing as
a side effect of a slider write.

Test approach
-------------
Source-level structural locks. ``LayerControl`` is a Kivy ``BoxLayout`` and is
MagicMock'd under the test mocks, so there is no real-widget harness; the
behavioural proof is the simulator run recorded with the change. These pin the
structure a future cleanup would have to break on purpose.
"""

from __future__ import annotations

import ast
import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_PATH = REPO / 'ui' / 'layer_control.py'
IMAGE_SETTINGS_PATH = REPO / 'ui' / 'image_settings.py'
MS_PATH = REPO / 'ui' / 'microscope_settings.py'
# pin-justified: the kv is declarative source with no headless seam, so the
# absence of a binding can only be read off the text.
KV_LINES = (REPO / 'ui' / 'lumaviewpro.kv').read_text(encoding='utf-8').splitlines()

RENDERER = 'render_layer_values_from_settings'
PRIMITIVE = '_show_value_on_widgets'
VALUE_SLIDERS = ('ill_slider', 'gain_slider', 'exp_slider')
UI_SOURCES = sorted(p for p in (REPO / 'ui').rglob('*.py'))


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'))


def _func(path: pathlib.Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == name)
            or (isinstance(n.func, ast.Name) and n.func.id == name)
        )
        for n in ast.walk(node)
    )


def _kv_sliders_binding_on_value() -> set[str]:
    """Every slider id in the kv whose own block binds ``on_value``."""

    def indent(line: str) -> int:
        expanded = line.expandtabs(4)
        return len(expanded) - len(expanded.lstrip(' '))

    found = set()
    for i, line in enumerate(KV_LINES):
        match = re.match(r'\s*id:\s*([A-Za-z_][A-Za-z0-9_]*)', line)
        if not match or 'slider' not in match.group(1):
            continue
        depth = indent(line)
        block = []
        for after in KV_LINES[i + 1 :]:
            if after.strip() and indent(after) < depth:
                break
            block.append(after.strip())
        for before in reversed(KV_LINES[max(0, i - 20) : i]):
            if before.strip() and indent(before) < depth:
                break
            block.append(before.strip())
        if any(b.startswith('on_value:') for b in block):
            found.add(match.group(1))
    return found


def _slider_value_writes(func: ast.FunctionDef) -> set[str]:
    """The widget ids whose ``.value`` *func* assigns, in any of the three
    forms a write can take."""
    aliases = {}
    for n in ast.walk(func):
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        target = n.targets[0]
        if not isinstance(target, ast.Name):
            continue
        widget_id = _resolve_widget_id(n.value)
        if widget_id:
            aliases[target.id] = widget_id

    written = set()
    for n in ast.walk(func):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if not (isinstance(t, ast.Attribute) and t.attr == 'value'):
                continue
            widget_id = _resolve_widget_id(t.value)
            if widget_id is None and isinstance(t.value, ast.Name):
                widget_id = aliases.get(t.value.id)
            if widget_id:
                written.add(widget_id)
    return written


def _resolve_widget_id(node: ast.AST) -> str | None:
    """``self.ids['x']`` or ``self.ids.x`` -> ``'x'``."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == 'ids'
    ):
        return node.slice.value
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr == 'ids':
            return node.attr
    return None


def _assigned_subscript_attrs(node: ast.AST, attr: str) -> set[str]:
    """The ``self.ids['x'].<attr> = ...`` keys assigned anywhere under *node*."""
    found = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if (
                isinstance(t, ast.Attribute)
                and t.attr == attr
                and isinstance(t.value, ast.Subscript)
                and isinstance(t.value.slice, ast.Constant)
            ):
                found.add(t.value.slice.value)
    return found


class TestThePrimitiveIsTheOneWriter:
    """One home for "show this value on its slider and its text box"."""

    def test_it_writes_both_widgets(self):
        fn = _func(LAYER_CONTROL_PATH, PRIMITIVE)
        slider_writes = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == 'value'
        ]
        text_writes = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == 'text'
        ]
        assert len(slider_writes) == 1, f'{PRIMITIVE} must write the slider exactly once.'
        assert len(text_writes) == 1, f'{PRIMITIVE} must write the text box exactly once.'

    def test_the_slider_is_clipped_and_the_text_is_not(self):
        fn = _func(LAYER_CONTROL_PATH, PRIMITIVE)
        body = ast.unparse(fn)
        assert 'clip(value, slider.min, slider.max)' in body, (
            'The slider must take the nearest position it can represent, clipped to '
            'its OWN range -- a stored value above a convenience ceiling pins it.'
        )
        written_to_text = {
            ast.unparse(n.value)
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == 'text'
        }
        assert written_to_text == {'str(value)'}, (
            'The text box must show the stored value itself, unclipped and uncast; '
            'anything else is the display reverting to the slider, which is the bug. '
            f'Found: {written_to_text}'
        )

    def test_it_restores_the_previous_guard_rather_than_clearing_it(self):
        fn = _func(LAYER_CONTROL_PATH, PRIMITIVE)
        tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
        assert tries, f'{PRIMITIVE} must restore the flag in a finally.'
        restores = [
            n
            for try_node in tries
            for stmt in try_node.finalbody
            for n in ast.walk(stmt)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == '_initializing'
        ]
        assert restores, 'The finally must restore _initializing.'
        assert all(not isinstance(n.value, ast.Constant) for n in restores), (
            'The finally must restore the SAVED flag state, not assign False -- a caller '
            'that is already suppressing (a layer still initializing, a capability sync) '
            'must not have its own guard lifted here.'
        )

    def test_nothing_else_writes_one_of_the_three_sliders(self):
        """The store-corruption vector: a programmatic write to a slider that
        binds on_value reaches the handler, which commits the written value
        over the user's and logs it as a drag.

        Scoped to the three the layer renders, and it reads all three write
        forms -- ``self.ids['x'].value``, ``self.ids.x.value``, and a local
        alias bound from either. The alias form is the one the predecessor
        lock could not see, and it is the form the writer itself uses.

        What this does NOT reach: a slider handed in from another widget,
        because the resolution below only follows names bound from
        ``self.ids``. That is a real gap, stated rather than implied.
        """
        offenders = {}
        for path in UI_SOURCES:
            for node in ast.walk(_tree(path)):
                if not isinstance(node, ast.FunctionDef) or node.name == PRIMITIVE:
                    continue
                written = _slider_value_writes(node) & set(VALUE_SLIDERS)
                if written:
                    offenders[f'{path.name}:{node.name}'] = sorted(written)
        assert not offenders, (
            f'These write a value slider directly instead of through {PRIMITIVE}: {offenders}'
        )

    def test_the_kv_still_names_exactly_the_sliders_this_locks(self):
        """A slider binding ``on_value`` is a store-commit vector: Kivy fires
        that handler for a PROGRAMMATIC write, not only a user drag, which is
        the whole shape of this bug.

        Pinned so that binding a new slider to on_value fails here and forces
        the question "who writes it, and is that write suppressed?" rather
        than shipping a fourth instance quietly.

        jpg_quality_slider is in the pin and is NOT covered by the lock above:
        it is a known open instance of this same shape in another widget --
        microscope_settings.load_settings writes it and update_jpg_quality
        has no guard, so a stored value other than the kv default emits a
        phantom SLIDER JPG_QUALITY at startup with nobody touching the app.
        Listing it keeps it visible instead of letting the pin imply it is
        clean.
        """
        assert _kv_sliders_binding_on_value() == {
            'ill_slider',
            'gain_slider',
            'exp_slider',
            'jpg_quality_slider',
        }


class TestTheRendererRendersTheStore:
    def test_it_covers_all_three_settings(self):
        table = _func(LAYER_CONTROL_PATH, RENDERER)
        rendered = {c.value for c in ast.walk(table) if isinstance(c, ast.Constant)}
        module_constants = {
            c.value
            for node in ast.walk(_tree(LAYER_CONTROL_PATH))
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
            for c in ast.walk(node)
            if isinstance(c, ast.Constant)
        }
        keys = {'illumination_ma', 'gain_db', 'exposure_ms'}
        assert keys <= (rendered | module_constants), (
            'The renderer must cover illumination, gain and exposure -- the three '
            'settings a layer shows on both a slider and a text box.'
        )

    def test_it_delegates_to_the_primitive(self):
        assert _calls(_func(LAYER_CONTROL_PATH, RENDERER), PRIMITIVE)

    def test_sync_widgets_from_settings_delegates_the_three(self):
        fn = _func(LAYER_CONTROL_PATH, 'sync_widgets_from_settings')
        assert _calls(fn, RENDERER), (
            'sync_widgets_from_settings must render the three through the one renderer, '
            'not raw-write their sliders beside it.'
        )

    def test_the_auto_gain_write_back_renders_instead_of_poking_widgets(self):
        fn = _func(LAYER_CONTROL_PATH, 'update_auto_gain_cb')
        assert _calls(fn, RENDERER), (
            'The toggle-off write-back must store the achieved values and then render '
            'them, not assign the sliders and let the handlers re-commit.'
        )


class TestTheClampReconcilesThenRenders:
    def test_it_still_reconciles_both_stored_values(self):
        fn = _func(IMAGE_SETTINGS_PATH, 'clamp_layer_settings_to_caps')
        clamped = {
            t.slice.value
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exposure_ms')
        }
        assert clamped == {'gain_db', 'exposure_ms'}

    def test_it_renders_and_applies_explicitly(self):
        fn = _func(IMAGE_SETTINGS_PATH, 'clamp_layer_settings_to_caps')
        assert _calls(fn, RENDERER), 'The clamp must re-render both widgets from the store.'
        assert _calls(fn, 'apply_settings'), (
            'The clamp must deliver the reconciled value to the camera itself. It used to '
            'arrive only as the debounced side effect of the slider write, which was the '
            'only apply on the reconnect path.'
        )

    def test_load_settings_reconciles_before_anything_renders(self):
        """Ordering invariant: a value the camera cannot honor must never be
        rendered as if it were a legitimate policy divergence."""
        fn = _func(MS_PATH, 'load_settings')
        clamp_lines = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'clamp_layer_settings_to_caps'
        ]
        render_lines = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ('sync_widgets_from_settings', RENDERER)
        ]
        assert clamp_lines, 'load_settings must still delegate to the clamp owner.'
        assert render_lines, 'load_settings must still fill the widgets.'
        assert max(clamp_lines) < min(render_lines), (
            'clamp_layer_settings_to_caps must run BEFORE the widgets are filled; '
            f'clamp at {clamp_lines}, render at {render_lines}.'
        )


class TestTextBoxesDoNotRenderFromSliders:
    def test_the_kv_bindings_are_gone(self):
        # Word-bounded: stim_ill_slider is a different control with its own
        # store and is not in scope here.
        bound = re.compile(r'\b(' + '|'.join(VALUE_SLIDERS) + r')\.value\b')
        offenders = [
            line.strip()
            for line in KV_LINES
            if line.strip().startswith('text:') and bound.search(line)
        ]
        assert not offenders, (
            'A text box bound to its slider cannot show anything the slider cannot '
            f'hold, which is how the display reverted: {offenders}'
        )
