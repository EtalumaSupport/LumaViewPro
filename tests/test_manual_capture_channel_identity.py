# Copyright Etaluma, Inc.
"""A manual still capture names the channel it imaged the way a manual recording does.

The still capture used to find its channel by scanning the layer accordions
for the open one, inside the camera-executor task. With no drawer open the scan
bound nothing: the wrong layer's folder was created and remembered, and the save
then raised on a name that was never assigned. With a LED lit on a different
channel than the open drawer, the file was named for the drawer while the light
was the truth -- the rule the manual recording already follows.

These tests pin the rule and the seam: the channel is resolved once, by the one
function both manual outputs share (a lit LED wins, else the open layer, else
brightfield); every widget read happens on the button, before the capture guard
is armed; and the open-layer scan has one implementation.
"""

import ast
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import tifffile as tf

from modules.labware_loader import WellPlateLoader
from tests.ast_seams import find_def, parse_module


# ui.composite_capture is a Kivy widget module; conftest mocks `kivy` but not
# the uix submodules, and CompositeCapture subclasses FloatLayout (a bare
# MagicMock cannot be subclassed).
class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in ('kivy.clock', 'kivy.uix'):
    sys.modules.setdefault(_name, MagicMock())

_floatlayout = types.ModuleType('kivy.uix.floatlayout')
_floatlayout.FloatLayout = _StubWidget
sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)

import modules.app_context as _app_ctx


PLATE = '24 well microplate'
ROOTS = ('modules', 'ui', 'lumaviewpro.py')


@pytest.fixture
def identity_scope(sim_scope):
    """A real Lumascope with the acquisition context the metadata writer reads."""
    loader = WellPlateLoader()
    sim_scope.runtime_state.set_objective('20x Oly')
    sim_scope.runtime_state.set_labware(loader.get_plate(PLATE))
    sim_scope.runtime_state.set_stage_offset({'x': 0.0, 'y': 0.0})
    return sim_scope


def _read_channel_name(path) -> str:
    with tf.TiffFile(str(path)) as handle:
        description = json.loads(handle.pages[0].description)
        return description['Channel']['Name'][0]


def _run_manual_capture(tmp_path, scope, *, layer, false_color_on, separate_folders):
    """Drive the real capture task with the values the button snapshots.

    The save underneath is real: the defect lived in what reached the disk,
    so a test that patches the save cannot see it.
    """
    from ui.composite_capture import CompositeCapture

    ctx = MagicMock()
    ctx.settings = {
        'live_folder': str(tmp_path),
        'separate_folder_per_channel': separate_folders,
        'image_output_format': {'live': 'TIFF'},
        'jpg_quality': 90,
    }
    ctx.scope = scope
    ctx.scope_display.add_crosshairs.side_effect = lambda img: img
    ctx.scope_display.transform_to_bullseye.side_effect = lambda img: img
    capture_config = SimpleNamespace(capture_depth=8, save_encoding='8bit')
    layer_rows = {
        name: {'exposure_ms': 10, 'sum': 1, 'illumination_ma': 0}
        for name in ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')
    }

    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    remembered = MagicMock()
    try:
        with (
            patch('ui.composite_capture.set_last_save_folder', remembered),
            patch(
                'modules.config_ui_getters.get_layer_configs',
                side_effect=lambda specific_layers=None: {
                    k: v for k, v in layer_rows.items() if k in specific_layers
                },
            ),
            patch(
                'modules.config_ui_getters.get_image_capture_config_from_ui',
                return_value=capture_config,
            ),
        ):
            CompositeCapture._live_capture_impl(
                object(),
                layer=layer,
                false_color_on=false_color_on,
                use_bullseye=False,
                use_crosshairs=False,
            )
    finally:
        _app_ctx.ctx = original

    return sorted((tmp_path / 'Manual').rglob('*.tiff')), remembered


# ---------------------------------------------------------------------------
# The three clauses, through the real capture path
# ---------------------------------------------------------------------------


def test_no_drawer_open_and_no_led_lit_saves_as_brightfield(identity_scope, tmp_path):
    """No open drawer and no lit LED is a brightfield capture: the file, its
    per-channel folder and the remembered save folder all say BF, and nothing
    raises. This is the state that used to create the wrong layer's folder and
    then raise on an unassigned name."""
    written, remembered = _run_manual_capture(
        tmp_path, identity_scope, layer=None, false_color_on=False, separate_folders=True
    )

    assert len(written) == 1, f'expected one file, got {[p.name for p in written]}'
    path = written[0]
    assert path.parent == tmp_path / 'Manual' / 'BF', f'saved under {path.parent}'
    assert 'BF' in path.name, f'{path.name} lacks the BF token'
    assert _read_channel_name(path) == 'BF'
    remembered.assert_called_once_with(dir=tmp_path / 'Manual' / 'BF')


def test_a_lit_led_outranks_the_open_drawer(identity_scope, tmp_path):
    """Green drawer open with false colour on, Red LED lit: the file is named,
    labelled and rendered as Red. The light is the truth; the drawer's toggle
    only says the frame is rendered in colour."""
    identity_scope.illumination.led_on(channel='Red', illumination_ma=100)

    written, _ = _run_manual_capture(
        tmp_path, identity_scope, layer='Green', false_color_on=True, separate_folders=False
    )

    assert len(written) == 1
    path = written[0]
    assert 'Red' in path.name and 'Green' not in path.name, path.name
    assert _read_channel_name(path) == 'Red'
    # An 8-bit still with false colour on is written PALETTE with a (3, 256)
    # colormap; the imaged channel decides which row carries the ramp.
    with tf.TiffFile(str(path)) as handle:
        page = handle.pages[0]
        assert int(page.photometric) == int(tf.PHOTOMETRIC.PALETTE), (
            'false colour on must render through a palette'
        )
        red, green, blue = page.colormap
    assert red.max() > 0, 'the red ramp is empty'
    assert green.max() == 0 and blue.max() == 0, (
        'the frame was rendered in a colour other than the imaged channel'
    )


# ---------------------------------------------------------------------------
# The resolver is one function, shared
# ---------------------------------------------------------------------------


def _illumination(states):
    return SimpleNamespace(get_led_states=lambda: states)


def _state(enabled):
    return {'enabled': enabled, 'illumination_ma': 100 if enabled else None, 'owner': ''}


def test_resolver_clauses():
    from modules.common_utils import resolve_channel_identity

    lit_red = {'BF': _state(False), 'Red': _state(True), 'Green': _state(False)}
    assert resolve_channel_identity(_illumination(lit_red), 'Green') == 'Red'
    assert resolve_channel_identity(_illumination(lit_red), None) == 'Red'
    none_lit = {'BF': _state(False), 'Red': _state(False)}
    assert resolve_channel_identity(_illumination(none_lit), 'Green') == 'Green'
    assert resolve_channel_identity(_illumination(none_lit), None) == 'BF'
    assert resolve_channel_identity(_illumination({}), None) == 'BF'
    two_lit = {'Blue': _state(True), 'Red': _state(True)}
    assert resolve_channel_identity(_illumination(two_lit), 'Green') == 'Blue'


def test_both_manual_outputs_share_the_resolver():
    """The recording controller no longer carries its own copy; both the
    still capture and the recording call the one function, and the default
    layer lives with it."""
    recording = parse_module('modules/manual_recording.py')
    private = [
        n.name
        for n in ast.walk(recording)
        if isinstance(n, ast.FunctionDef) and n.name == '_resolve_channel_identity'
    ]
    assert private == [], 'the recording controller still carries a private resolver'

    for rel in ('modules/manual_recording.py', 'ui/composite_capture.py'):
        calls = [
            n
            for n in ast.walk(parse_module(rel))
            if isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == 'resolve_channel_identity')
                or (isinstance(n.func, ast.Attribute) and n.func.attr == 'resolve_channel_identity')
            )
        ]
        assert calls, f'{rel} does not call resolve_channel_identity'

    defined_in = []
    dead = []
    for rel in _production_modules():
        for n in ast.walk(parse_module(rel)):
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'DEFAULT_LAYER' for t in n.targets
            ):
                defined_in.append(rel)
            if isinstance(n, ast.FunctionDef) and n.name == 'get_opened_layer_obj':
                dead.append(rel)
    assert defined_in == ['modules/common_utils.py'], defined_in
    assert dead == [], f'get_opened_layer_obj still defined in {dead}'


# ---------------------------------------------------------------------------
# The button snapshots; the task reads no widget
# ---------------------------------------------------------------------------


def test_capture_task_reads_no_widget_and_the_button_snapshots_before_arming():
    impl = find_def('ui/composite_capture.py', '_live_capture_impl', class_name='CompositeCapture')
    kwonly = [a.arg for a in impl.args.kwonlyargs]
    assert kwonly == ['layer', 'false_color_on', 'use_bullseye', 'use_crosshairs'], kwonly
    body = ast.unparse(impl)
    for forbidden in ('accordion_item_lookup', '.collapse', 'ids[', 'scope_display.use_'):
        assert forbidden not in body, (
            f'the capture task still reads widget state ({forbidden!r}) off the main thread'
        )

    button = ast.unparse(
        find_def('ui/composite_capture.py', 'live_capture', class_name='CompositeCapture')
    )
    assert 'get_opened_layer' in button, 'the button does not snapshot the open layer'
    assert button.index('get_opened_layer') < button.index('_capturing.set()'), (
        'the snapshot must precede arming the guard, or a raising read wedges the button'
    )


# ---------------------------------------------------------------------------
# The list-typed filter is handed a list
# ---------------------------------------------------------------------------


def _production_modules():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    files = (
        sorted(root.glob('modules/*.py')) + sorted(root.glob('ui/*.py')) + [root / 'lumaviewpro.py']
    )
    return [str(f.relative_to(root)) for f in files]


def test_specific_layers_is_always_handed_a_list():
    """`get_layer_configs(specific_layers=...)` filters with `in`; a bare string
    matches by substring. Every call site passes a list display or a name bound
    to a call in the same function."""
    offenders = []
    for rel in _production_modules():
        tree = parse_module(rel)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            bound_to_call = {
                t.id
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                for kw in call.keywords:
                    if kw.arg != 'specific_layers':
                        continue
                    ok = isinstance(kw.value, ast.List) or (
                        isinstance(kw.value, ast.Name) and kw.value.id in bound_to_call
                    )
                    if not ok:
                        offenders.append(f'{rel}:{call.lineno} {ast.unparse(kw.value)}')
    assert offenders == [], offenders
