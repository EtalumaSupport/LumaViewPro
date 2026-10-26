# Copyright Etaluma, Inc.
"""A manual still capture names the channel it imaged the way a manual recording does.

The still capture used to find its channel by scanning the layer accordions
for the open one, inside the camera-executor task. With no drawer open the scan
bound nothing: the wrong layer's folder was created and remembered, and the save
then raised on a name that was never assigned. With a LED lit on a different
channel than the open drawer, the file was named for the drawer while the light
was the truth -- the rule the manual recording already follows.

These tests pin the rule and the seam: the channel is resolved once, by the one
function both manual outputs share (a lit LED wins, else the Session's shown
channel); every widget read happens on the button, before the capture guard
is armed; and the open-layer scan has one implementation.
"""

import ast
import json
from types import SimpleNamespace

import pytest
import tifffile as tf

from tests.ast_seams import REPO_ROOT, parse_module
from tests.scope_fakes import bind_settings_like_a_session
from modules.layer_record import release_catalogue, release_colors
from tests.stock_channels import LED0, LED1, LED2


PLATE = '24 well microplate'
ROOTS = ('modules', 'ui', 'lumaviewpro.py')


@pytest.fixture
def identity_scope(sim_scope):
    """A real Lumascope with the acquisition context the metadata writer reads."""
    bind_settings_like_a_session(
        sim_scope,
        objective_id='20x Oly',
        protocol={'labware': PLATE},
        stage_offset={'x': 0.0, 'y': 0.0},
    )
    return sim_scope


def _read_channel_name(path) -> str:
    with tf.TiffFile(str(path)) as handle:
        description = json.loads(handle.pages[0].description)
        return description['Channel']['Name'][0]


def _run_manual_capture(tmp_path, scope, *, shown_layer, false_color_on, separate_folders):
    """Drive the one manual-capture path with the shown layer the Session hands it.

    The save underneath is real: the defect lived in what reached the disk,
    so a test that patches the save cannot see it. Returns the files on disk
    and the paths the capture reported, whose first parent is the folder the
    button remembers.
    """
    from modules.manual_capture import ManualCaptureController
    from tests.settings_fixtures import complete_settings

    settings = complete_settings(
        live_folder=str(tmp_path),
        separate_folder_per_channel=separate_folders,
        image_output_format={'live': 'TIFF', 'sequenced': 'TIFF'},
        image_mode='8bit',
    )
    for name in release_catalogue():
        settings[name].update({'exposure_ms': 10, 'sum': 1, 'illumination_ma': 0})
    capture = ManualCaptureController(
        scope=scope,
        settings_snapshot=lambda: settings,
        shown_layer=lambda: shown_layer,
        # No layer is put on this camera, so the still's hold writes nothing.
        layer_camera_request=lambda layer: None,
        engineering_mode=lambda: False,
        channel_colors=release_colors,
    )
    reported = capture.capture(false_color_on=false_color_on).result(timeout=30)

    return sorted((tmp_path / 'Manual').rglob('*.tiff')), reported


# ---------------------------------------------------------------------------
# The three clauses, through the real capture path
# ---------------------------------------------------------------------------


def test_a_lit_led_outranks_the_shown_channel(identity_scope, tmp_path):
    """LED1 shown with false colour on, LED 2 lit: the file is named,
    labelled and rendered as LED2. The light is the truth; the shown layer's
    toggle only says the frame is rendered in colour."""
    identity_scope.illumination.led_on(channel=LED2, illumination_ma=100)

    written, _ = _run_manual_capture(
        tmp_path, identity_scope, shown_layer=LED1, false_color_on=True, separate_folders=False
    )

    assert len(written) == 1
    path = written[0]
    assert LED2 in path.name and LED1 not in path.name, path.name
    assert _read_channel_name(path) == LED2
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


def _illumination(*readings):
    # a stand-in by design: the subject is the resolver, and these readings
    # are its interchangeable input, built to each clause
    return SimpleNamespace(get_led_states=lambda: readings)


def _lit(address, name):
    from modules.lumascope_api.illumination import LedReading, LedState

    return LedReading(address, name, LedState.LIT, 100.0)


def _dark(address, name):
    from modules.lumascope_api.illumination import LedReading, LedState

    return LedReading(address, name, LedState.DARK, None)


def test_resolver_clauses():
    """A lit, named LED wins over the shown layer; nothing lit falls to the
    shown layer; two lit, the first in address order wins; a lit address the
    identity names no layer for names no channel; no board (None) is nothing
    lit."""
    from modules.common_utils import read_light

    lit_red = _illumination(_dark(1, LED1), _lit(2, LED2), _dark(3, 'BF'))
    assert read_light(lit_red, LED1).channel == LED2
    none_lit = _illumination(_dark(2, LED2), _dark(3, 'BF'))
    assert read_light(none_lit, LED1).channel == LED1
    assert read_light(_illumination(), LED0).channel == LED0
    two_lit = _illumination(_lit(0, LED0), _lit(2, LED2))
    assert read_light(two_lit, LED1).channel == LED0
    unnamed_lit = _illumination(_lit(6, None), _dark(2, LED2))
    assert read_light(unnamed_lit, LED1).channel == LED1
    no_board = SimpleNamespace(get_led_states=lambda: None)
    assert read_light(no_board, LED1).channel == LED1


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

    for rel in ('modules/manual_recording.py', 'modules/manual_capture.py'):
        calls = [
            n
            for n in ast.walk(parse_module(rel))
            if isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == 'read_light')
                or (isinstance(n.func, ast.Attribute) and n.func.attr == 'read_light')
            )
        ]
        assert calls, f'{rel} does not call read_light'

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


def test_capture_task_reads_no_widget():
    """The capture lives below the GUI and reads nothing of it: the channel
    is the Session's shown channel, and the caller hands it only the
    false-colour toggle it read on the main thread."""
    body = ast.unparse(parse_module('modules/manual_capture.py'))
    for forbidden in ('_app_ctx', 'accordion_item_lookup', '.collapse', 'ids[', 'scope_display'):
        assert forbidden not in body, f'the capture reads GUI state ({forbidden!r})'


# ---------------------------------------------------------------------------
# The list-typed filter is handed a list
# ---------------------------------------------------------------------------


def _production_modules():
    files = (
        sorted(REPO_ROOT.glob('modules/*.py'))
        + sorted(REPO_ROOT.glob('ui/*.py'))
        + [REPO_ROOT / 'lumaviewpro.py']
    )
    return [str(f.relative_to(REPO_ROOT)) for f in files]


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


# ---------------------------------------------------------------------------
# No code reads the open drawer to say which channel is shown
# ---------------------------------------------------------------------------


def _first_open_scan_loops():
    """Every loop over the layer vocabulary that reads `.collapse` off an
    accordion lookup, deduplicated on (file, function): a nested function's
    loop is walked under its enclosure as well."""
    seen = set()
    for rel in _production_modules():
        tree = parse_module(rel)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for loop in ast.walk(fn):
                if not (isinstance(loop, ast.For) and 'get_layers(' in ast.unparse(loop.iter)):
                    continue
                body = ast.unparse(loop)
                if 'accordion_item_lookup' in body and '.collapse' in body:
                    seen.add((rel, fn.name))
    return seen


def test_no_code_scans_the_drawers_for_the_shown_channel():
    """The Session's shown channel says which channel is shown; the drawers
    display it. The one loop over the drawers' collapse is the one that
    draws them, and it writes them rather than reading them."""
    assert _first_open_scan_loops() == {('ui/image_settings.py', 'draw_shown_channel')}
