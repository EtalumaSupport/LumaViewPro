# Copyright Etaluma, Inc.
"""Every identity field a saved frame carries names the channel it was acquired on.

A saved image records WHAT was imaged. How that image is displayed -- the
per-layer false-color toggle, the per-step False_Color column -- is a separate
fact that must never reach a field describing the specimen.

The save seams used to carry two channel-valued arguments for one fact, and the
one that reached durable storage was the one nobody passed: it defaulted to
'BF', so every manual capture stamped itself brightfield regardless of the
channel its LED had just lit. Quick Enhance then read that back and declined to
false-color a green frame, because the file said brightfield. The filename beside
it said Green.

A second carrier hid underneath. Channel.Modality was derived from the render
color, which collapses to 'BF' whenever false color is off -- so one file could
carry Channel.Name = Green and Channel.Modality = BF, disagreeing with itself,
with the second field tracking a checkbox.

These tests assert the invariant directly rather than the symptom: identity is
independent of rendering, at every seam and in every identity field.
"""

import ast
import inspect
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile as tf

from modules import image_save
from modules.labware_loader import WellPlateLoader
from tests.ast_seams import find_def


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


LAYER = 'Green'
PLATE = '24 well microplate'


@pytest.fixture
def identity_scope(sim_scope):
    """A real Lumascope with the acquisition context generate_image_metadata reads.

    Production hands these seams a live Lumascope -- on a scope with no motor
    board, one composed with Null boards, never None. A None scope raises out of
    generate_image_metadata before a channel is ever written, so it cannot
    observe this invariant at all.
    """
    loader = WellPlateLoader()
    sim_scope.runtime_state.set_objective('20x Oly')
    sim_scope.runtime_state.set_labware(loader.get_plate(PLATE))
    sim_scope.runtime_state.set_stage_offset({'x': 0.0, 'y': 0.0})
    return sim_scope


def _read_identity(path) -> dict:
    """Every identity field on the file, from whichever container carries it.

    A 16-bit non-OME frame takes the ImageJ path, whose description holds the
    Channel block plus the Modality field; an 8-bit frame carries a JSON
    description with the same block and no Modality, which is ImageJ-only.
    Read as a third-party consumer (FIJI, MATLAB) would.
    """
    with tf.TiffFile(str(path)) as handle:
        photometric = int(handle.pages[0].photometric)
        imagej = handle.imagej_metadata
        if imagej and 'channel' in imagej:
            channel = ast.literal_eval(imagej['channel'])
            return {
                'Name': channel['Name'][0],
                'Modality': channel['Modality'][0],
                'mode': imagej.get('mode'),
                'photometric': photometric,
            }
        description = json.loads(handle.pages[0].description)
        return {
            'Name': description['Channel']['Name'][0],
            'Modality': None,
            'mode': None,
            'photometric': photometric,
            'plane': description.get('Plane', {}),
        }


def _read_channel(path) -> str:
    return _read_identity(path)['Name']


# ---------------------------------------------------------------------------
# T1 -- the invariant, driven through the real manual-capture entry point
# ---------------------------------------------------------------------------


def _capture_ctx(tmp_path, scope, false_color_active, use_crosshairs):
    ctx = MagicMock()
    # A bare MagicMock auto-answers source_path with a Mock, which the
    # data-root resolution would happily turn into a nonexistent path --
    # taking the layer vocabulary (and everything derived from it) down
    # with it. None means "no session override": files resolve from the
    # real source tree, as production does before a session exists.
    ctx.source_path = None
    ctx.settings = {
        'live_folder': str(tmp_path),
        'separate_folder_per_channel': False,
        'image_output_format': {'live': 'TIFF'},
        'jpg_quality': 90,
    }
    ctx.scope = scope
    # Only the layer under test is expanded: the capture sweep walks the
    # REAL layer vocabulary and asks each accordion, exactly as production
    # does -- narrowing the vocabulary itself would also narrow the layer
    # categories the save path derives from it.
    ctx.image_settings.accordion_item_lookup.side_effect = lambda layer: SimpleNamespace(
        collapse=(layer != LAYER)
    )
    ctx.image_settings.layer_lookup.return_value = SimpleNamespace(
        ids={'false_color': SimpleNamespace(active=false_color_active)}
    )
    ctx.scope_display.use_bullseye = False
    ctx.scope_display.use_crosshairs = use_crosshairs
    ctx.scope_display.add_crosshairs.side_effect = lambda img: img
    ctx.scope_display.transform_to_bullseye.side_effect = lambda img: img
    return ctx


def _run_manual_capture(tmp_path, scope, *, false_color_active, use_crosshairs):
    """Drive the real capture path with the real save underneath it.

    Deliberately does NOT patch save_image / save_live_image: the defect lives
    in what reaches the file, so a test that patches the save cannot see it.
    """
    from ui.composite_capture import CompositeCapture

    capture_config = SimpleNamespace(capture_depth=8, save_encoding='8bit')
    ctx = _capture_ctx(tmp_path, scope, false_color_active, use_crosshairs)

    original = _app_ctx.ctx
    _app_ctx.ctx = ctx
    try:
        with (
            patch('ui.composite_capture.set_last_save_folder'),
            patch(
                'modules.config_ui_getters.get_layer_configs',
                return_value={LAYER: {'exposure_ms': 10, 'sum': 1, 'illumination_ma': 100}},
            ),
            patch(
                'modules.config_ui_getters.get_image_capture_config_from_ui',
                return_value=capture_config,
            ),
        ):
            CompositeCapture._live_capture_impl(object())
    finally:
        _app_ctx.ctx = original

    return sorted((tmp_path / 'Manual').glob('*.tiff'))


@pytest.mark.parametrize('false_color_active', [True, False], ids=['fc_on', 'fc_off'])
@pytest.mark.parametrize('use_crosshairs', [False, True], ids=['plain', 'crosshairs'])
def test_manual_capture_stamps_the_acquiring_layer(
    identity_scope, tmp_path, false_color_active, use_crosshairs
):
    """A manual capture records the channel its LED lit, in all four
    combinations of the false-color toggle and the overlay.

    The toggle moves the photometric interpretation and never the name.
    """
    identity_scope.illumination.led_on(channel=LAYER, illumination_ma=100)

    written = _run_manual_capture(
        tmp_path,
        identity_scope,
        false_color_active=false_color_active,
        use_crosshairs=use_crosshairs,
    )

    assert written, 'the manual capture wrote no file'
    for path in written:
        assert _read_channel(path) == LAYER, (
            f'{path.name} records {_read_channel(path)!r}, not the acquiring '
            f'channel {LAYER!r} -- the false-color toggle '
            f'({false_color_active}) must not decide what the file says it is'
        )
        # Filenames are built from the layer, never from the render color, so
        # this fix must leave every one of them byte-identical.
        assert LAYER in path.name, f'{path.name} lost the layer token'


def test_manual_capture_records_the_real_drive_current(identity_scope, tmp_path):
    """Metadata looks up the LED current for the channel it is told. Asking for
    brightfield's current while a fluorescence LED was lit recorded 0 mA on
    every manual capture -- a zero that was never measured."""
    identity_scope.illumination.led_on(channel=LAYER, illumination_ma=100)

    written = _run_manual_capture(
        tmp_path, identity_scope, false_color_active=False, use_crosshairs=False
    )

    illumination = _read_identity(written[0])['plane'].get('Illumination')
    assert illumination, (
        f"recorded illumination is {illumination!r}: asking for brightfield's "
        'current while the green LED was lit recorded a zero that was never '
        'measured'
    )


# ---------------------------------------------------------------------------
# T2 -- the two acquisition paths agree
# ---------------------------------------------------------------------------


def test_manual_and_protocol_captures_of_one_frame_agree(identity_scope, tmp_path):
    """The same frame, saved through the manual call shape and the protocol
    call shape, must carry the same channel. This is the test that would have
    caught both halves of the defect as one."""
    array = np.zeros((8, 8), dtype=np.uint8)

    manual = image_save.save_image(
        identity_scope,
        array.copy(),
        save_folder=str(tmp_path),
        file_root='manual_',
        append='a',
        tail_id_mode=None,
        channel=LAYER,
        false_color_on=False,
        output_format='TIFF',
        save_encoding='8bit',
        significant_bits=8,
    )
    protocol = image_save.save_image(
        identity_scope,
        array.copy(),
        save_folder=str(tmp_path),
        file_root='protocol_',
        append='b',
        tail_id_mode=None,
        channel=LAYER,
        false_color_on=True,
        output_format='TIFF',
        save_encoding='8bit',
        significant_bits=8,
    )

    assert _read_channel(manual) == _read_channel(protocol) == LAYER, (
        f'manual says {_read_channel(manual)!r}, protocol says '
        f'{_read_channel(protocol)!r} -- one frame, one channel'
    )


# ---------------------------------------------------------------------------
# T3 -- the contract that keeps the default from creeping back
# ---------------------------------------------------------------------------


SEAMS = [
    image_save.prepare_image_for_saving,
    image_save.save_image,
    image_save.save_live_image,
]


@pytest.mark.parametrize('func', SEAMS, ids=lambda f: f.__qualname__)
def test_save_seams_require_a_channel(func):
    """A save that never states its channel must be unconstructible -- the
    guard is the absence of a default, not a reminder at each call site."""
    params = inspect.signature(func).parameters

    assert 'channel' in params, f'{func.__qualname__} must take a channel'
    param = params['channel']
    assert param.default is inspect.Parameter.empty, (
        f'{func.__qualname__} must require channel, not default it -- a '
        'defaulted channel silently asserts brightfield'
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f'{func.__qualname__}.channel must be keyword-only so no positional '
        'call can bind it by accident'
    )


@pytest.mark.parametrize('func', SEAMS, ids=lambda f: f.__qualname__)
def test_save_seams_no_longer_carry_two_channel_arguments(func):
    """Deleting the positional `color` slot is what makes the rename safe: a
    stale positional call would otherwise rebind silently to the next
    parameter."""
    params = inspect.signature(func).parameters
    for retired in ('color', 'true_color'):
        assert retired not in params, (
            f'{func.__qualname__} still takes {retired!r} -- two channel-valued '
            'arguments for one fact is the defect itself'
        )


def test_save_image_rejects_a_missing_channel(identity_scope):
    with pytest.raises(TypeError, match='channel'):
        image_save.save_image(
            identity_scope,
            np.zeros((4, 4), dtype=np.uint8),
            save_encoding='8bit',
            significant_bits=8,
            false_color_on=False,
        )


def test_metadata_rejects_a_channel_outside_the_vocabulary(identity_scope):
    """The seam is a public surface once channel is required; an index or a
    typo must not reach durable metadata as an identity."""
    with pytest.raises(ValueError, match='unknown channel'):
        image_save.generate_image_metadata(identity_scope, channel=3, x=0, y=0, z=0)


# ---------------------------------------------------------------------------
# T4 -- the composite exports
# ---------------------------------------------------------------------------


def test_composite_export_stamps_composite(identity_scope, tmp_path):
    """A multi-channel composite is what it is: Composite, an established
    vocabulary word the derived-output writers already emit."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    path = image_save.save_image(
        identity_scope,
        rgb,
        save_folder=str(tmp_path),
        file_root='composite_',
        append='A1',
        tail_id_mode='increment',
        channel='Composite',
        false_color_on=False,
        output_format='TIFF',
        save_encoding='8bit',
        significant_bits=8,
    )
    assert _read_identity(path)['Name'] == 'Composite', (
        'a multi-channel composite must record Composite, not brightfield'
    )


def test_single_channel_composite_export_stamps_that_channel(identity_scope, tmp_path):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    path = image_save.save_image(
        identity_scope,
        rgb,
        save_folder=str(tmp_path),
        file_root=f'{LAYER}_Image_',
        append='A1',
        tail_id_mode='increment',
        channel=LAYER,
        false_color_on=False,
        output_format='TIFF',
        save_encoding='8bit',
        significant_bits=8,
    )
    assert _read_identity(path)['Name'] == LAYER, (
        'a single-channel composite export records the channel it holds'
    )


# ---------------------------------------------------------------------------
# T5 -- rendering must not rewrite a recorded identity
# ---------------------------------------------------------------------------


def test_false_colour_off_does_not_rewrite_recorded_identity(identity_scope, tmp_path):
    """Both identity fields agree with each other and with the acquiring
    channel, whichever way the toggle is set -- while the rendering fields do
    differ, so the test cannot be satisfied by disabling false color outright.

    Pre-fix the toggle-off file carried Channel.Name = Green with
    Channel.Modality = BF: two identity fields in one file, disagreeing, the
    second tracking a checkbox.
    """
    array = np.zeros((8, 8), dtype=np.uint16)

    def _save(false_color_on, root):
        return image_save.save_image(
            identity_scope,
            array.copy(),
            save_folder=str(tmp_path),
            file_root=root,
            append='x',
            tail_id_mode=None,
            channel=LAYER,
            false_color_on=false_color_on,
            output_format='TIFF',
            save_encoding='right_aligned',
            significant_bits=12,
        )

    on = _read_identity(_save(True, 'on_'))
    off = _read_identity(_save(False, 'off_'))

    assert on['Name'] == off['Name'] == LAYER, (
        f'Channel.Name differs with the toggle: {on["Name"]!r} vs {off["Name"]!r}'
    )
    assert on['Modality'] == off['Modality'] == 'MIF', (
        f'Channel.Modality differs with the toggle: {on["Modality"]!r} vs '
        f'{off["Modality"]!r} -- it must describe the specimen, not the display'
    )
    assert on['mode'] != off['mode'], (
        f'the toggle must still change how the file is rendered (mode was '
        f'{on["mode"]!r} both ways), or this test passes vacuously by having '
        'disabled false color altogether'
    )


# ---------------------------------------------------------------------------
# T6 -- the video seam, where nothing structural stops the collapse
# ---------------------------------------------------------------------------


def test_video_frame_render_colour_cannot_reach_an_identity_field():
    """Video frames record no channel today, so there is no behaviour to
    observe here -- the absence of an observable is exactly the risk. The
    names carry the invariant instead: the acquiring channel arrives as
    `channel`, and the value collapsed from the toggle is named for the
    rendering job it does.

    Asserted over the AST rather than the source text, so it survives any
    reformatting of the seam it guards.
    """
    node = find_def('modules/image_save.py', 'write_video_frame')
    assert node is not None, 'write_video_frame not found'

    params = {a.arg for a in (*node.args.args, *node.args.kwonlyargs)}
    assert 'channel' in params and 'layer_color' not in params, (
        'write_video_frame must take `channel`: `layer_color` is the name that '
        'invites the next author to treat the collapsed render value as an '
        'identity, and only an unrelated early return stops it reaching one'
    )

    assigned = {
        target.id
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    assert 'save_color' not in assigned, (
        'save_color reads like the value a save records; the collapsed value '
        'must be named for the rendering job it does'
    )
    assert 'render_color' in assigned, (
        'the value collapsed from false_color_on must carry its role in its name'
    )

    collapse = next(
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'render_color' for t in stmt.targets)
    )
    assert isinstance(collapse.value, ast.IfExp), (
        'render_color must still be the toggle-conditional collapse'
    )
    assert isinstance(collapse.value.test, ast.Name), 'expected a bare flag test'
    assert collapse.value.test.id == 'false_color_on', (
        f'render_color must be derived from false_color_on, not {ast.dump(collapse.value.test)}'
    )
    assert isinstance(collapse.value.body, ast.Name), 'expected the channel name'
    assert collapse.value.body.id == 'channel', 'the false-color arm renders the acquiring channel'
