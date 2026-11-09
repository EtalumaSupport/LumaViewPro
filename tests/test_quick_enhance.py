from __future__ import annotations

import datetime
import json
import ast
import pathlib

import cv2
import numpy as np
import pytest
import tifffile as tf

from _lumascope.modules import image_utils
from _lumascope.modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from _lumascope.modules.quick_enhance import (
    QUANTITATIVE_USE_WARNING,
    QuickEnhanceSettings,
    QuickEnhancer,
)
from _lumascope.modules.channel_record import GREY, release_color
from _lumascope.modules.channel_record import release_colors
from tests.settings_fixtures import quick_enhance_settings, shipped_template
from tests.stock_channels import LED0, LED1, LED2


def _metadata(significant_bits: int, channel: str = LED1) -> dict:
    return {
        'pixel_size_um': 1.0,
        'channel': channel,
        'objective': {'id': '10x'},
        'exposure_time_ms': 1.0,
        'gain_db': 0.0,
        'illumination_ma': 1.0,
        'z_pos_um': 0.0,
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'datetime': datetime.datetime(2026, 7, 17, 0, 0, 0),
        'significant_bits': significant_bits,
    }


def _write_source(path, data: np.ndarray, significant_bits: int, channel: str = LED1) -> None:
    image_utils.write_tiff(
        data=data,
        file_loc=path,
        metadata=_metadata(significant_bits, channel),
        ome=False,
        # A composite source is RGB, and carries its own colour.
        color=GREY if data.ndim == 3 else release_color(channel),
        significant_bits=significant_bits,
        save_encoding='right_aligned',
        white_bits=None,
    )


@pytest.mark.parametrize(
    ('image', 'significant_bits', 'maximum'),
    [
        (np.arange(256, dtype=np.uint8).reshape(16, 16), 8, 255),
        (np.linspace(0, 4095, 256, dtype=np.uint16).reshape(16, 16), 12, 4095),
        (np.linspace(0, 65535, 256, dtype=np.uint16).reshape(16, 16), 16, 65535),
    ],
)
def test_apply_preserves_source_dtype_range_and_input(image, significant_bits, maximum):
    source = image.copy()
    result = QuickEnhancer().apply(image, quick_enhance_settings(), significant_bits)

    assert np.array_equal(image, source)
    assert result.dtype == image.dtype
    assert result.shape == image.shape
    assert int(result.min()) >= 0
    assert int(result.max()) <= maximum


def test_uniform_image_is_stable_and_does_not_divide_by_zero():
    image = np.full((12, 12), 1000, dtype=np.uint16)
    result = QuickEnhancer().apply(image, quick_enhance_settings(), 12)

    assert np.array_equal(result, image)


def test_rgb_images_use_one_shared_tone_curve_without_changing_shape_or_dtype():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8)[:, None] * 8
    image[..., 1] = 100
    image[..., 2] = 200
    source = image.copy()

    result = QuickEnhancer().apply(image, quick_enhance_settings(), 8)

    assert np.array_equal(image, source)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_representative_bf_fluorescence_and_composite_exports_preserve_sources(tmp_path):
    fixtures = {
        'bf': (np.arange(256, dtype=np.uint16).reshape(16, 16), 'BF', (16, 16)),
        'green': (np.arange(256, dtype=np.uint16).reshape(16, 16), LED1, (16, 16, 3)),
        'composite': (
            np.dstack([np.arange(256, dtype=np.uint16).reshape(16, 16)] * 3),
            'Composite',
            (16, 16, 3),
        ),
    }
    enhancer = QuickEnhancer()

    for name, (pixels, channel, expected_shape) in fixtures.items():
        source = tmp_path / f'{name}.tif'
        _write_source(source, pixels, 12, channel=channel)
        original = source.read_bytes()
        exported = enhancer.export_file(
            source, quick_enhance_settings(), channel_colors=release_colors()
        )
        output, bits, _ = image_utils.load_pixels(
            exported['output_path'], collapse_legacy_false_color=False
        )

        assert source.read_bytes() == original
        assert output.shape == expected_shape
        assert output.dtype == np.uint16
        assert bits == 12


@pytest.mark.parametrize('suffix', ('.png', '.jpg'))
def test_png_and_jpeg_export_as_derived_tiff(tmp_path, suffix):
    source = tmp_path / f'color{suffix}'
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[..., 0] = 30
    image[..., 1] = 120
    image[..., 2] = 220
    assert cv2.imwrite(str(source), image)

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )

    assert exported['output_path'].suffix == '.tiff'
    output, _bits, _white = image_utils.load_pixels(
        exported['output_path'], collapse_legacy_false_color=False
    )
    assert output.shape == image_utils.load_pixels(source)[0].shape


def test_invalid_settings_are_rejected_before_processing():
    settings = quick_enhance_settings(low_percentile=99, high_percentile=1, gamma=3.0)
    errors = QuickEnhancer().validate(settings, np.dtype(np.uint16), 12)

    assert any('percentile' in error.lower() for error in errors)
    assert any('gamma' in error.lower() for error in errors)
    with pytest.raises(ValueError, match='Invalid Quick Enhance settings'):
        QuickEnhancer().apply(np.ones((4, 4), dtype=np.uint16), settings, 12)


def test_each_fixed_step_number_out_of_its_range_is_refused():
    """Every number a step reads is judged by ``validate``, so a value its
    step cannot run with is refused before any pixel is touched; each row
    names the words its refusal states."""
    out_of_range = {
        'illumination_fit_excludes_upper_percentile': (100.0, 'illumination fit must exclude'),
        'illumination_fit_sample_target': (0, 'sample target'),
        'sharpen_sigma_px': (0.0, 'sharpen sigma'),
        'sharpen_gain': (-0.1, 'sharpen gain'),
        'sharpen_gate_background_percentile': (95.0, 'sharpen gate'),
        'sharpen_gate_top_percentile': (101.0, 'sharpen gate'),
    }
    unrefused = []
    for field, (value, words) in out_of_range.items():
        errors = QuickEnhancer().validate(
            quick_enhance_settings(**{field: value}), np.dtype(np.uint16), 12
        )
        if not any(words in error.lower() for error in errors):
            unrefused.append(f'{field}={value!r}: {errors}')

    assert not unrefused, unrefused


def test_quick_enhance_uses_one_fixed_non_ai_recipe():
    settings = quick_enhance_settings()

    assert settings.illumination_correction_enabled is True
    assert settings.denoise_enabled is False
    assert settings.gamma == 0.9


def test_broad_illumination_correction_improves_dim_feature_visibility_without_dark_ring():
    image = np.full((128, 128), 500, dtype=np.uint16)
    image[48:80, 48:80] = 560

    result = QuickEnhancer().apply(image, quick_enhance_settings(), 12)

    assert np.ptp(result) > np.ptp(image)
    assert int(result[32, 64]) >= int(result[8, 8])


def test_illumination_correction_never_wraps_unsigned_values():
    image = np.zeros((64, 64), dtype=np.uint16)
    image[:, 32:] = 100
    image[20:30, 20:30] = 1000
    settings = quick_enhance_settings(illumination_correction_enabled=True)

    result = QuickEnhancer().apply(image, settings, 12)

    assert result.dtype == np.uint16
    assert int(result.min()) >= 0
    assert int(result.max()) <= 4095


def test_illumination_plane_avoids_full_resolution_coordinate_grids(monkeypatch):
    class RejectFullResolutionGrid:
        def __getitem__(self, key):
            row_slice, column_slice = key
            if row_slice.step in (None, 1) and column_slice.step in (None, 1):
                raise AssertionError('full-resolution coordinate grids are not allowed')
            rows = np.arange(row_slice.start, row_slice.stop, row_slice.step)
            columns = np.arange(column_slice.start, column_slice.stop, column_slice.step)
            return np.meshgrid(rows, columns, indexing='ij')

    monkeypatch.setattr('_lumascope.modules.quick_enhance.np.mgrid', RejectFullResolutionGrid())
    plane = QuickEnhancer()._global_illumination_plane(
        np.ones((1000, 1000), dtype=np.float32), quick_enhance_settings()
    )

    assert plane.shape == (1000, 1000)
    assert plane.dtype == np.float32


def test_color_algorithm_sanitizes_non_finite_channels_before_reconstruction(monkeypatch):
    enhancer = QuickEnhancer()
    monkeypatch.setattr(enhancer, 'validate', lambda *_: [])
    image = np.full((8, 8, 3), 64.0, dtype=np.float32)
    image[0, 0, 0] = np.nan
    image[1, 1, 1] = np.inf

    result = enhancer.apply(image, quick_enhance_settings(), 8)

    assert np.isfinite(result).all()


def test_tiny_images_with_denoise_are_safe():
    image = np.array([[1]], dtype=np.uint8)
    settings = quick_enhance_settings(denoise_enabled=True, denoise_kernel_size=3)

    result = QuickEnhancer().apply(image, settings, 8)

    assert np.array_equal(result, image)


def test_denoise_runs_on_a_colour_image_at_every_kernel_its_recipe_can_state():
    """A colour image is denoised at kernel 5 and 7 as at 3: the size gate
    reads the image's height and width, not its channel axis, so the step
    the recipe states as enabled is the step that ran."""
    rng = np.random.default_rng(7)
    skipped = []
    for channels in (3, 4):
        image = (rng.random((64, 64, channels)) * 4095).astype(np.uint16)
        undenoised = QuickEnhancer().apply(image, quick_enhance_settings(), 12)
        for kernel in (3, 5, 7):
            denoised = QuickEnhancer().apply(
                image,
                quick_enhance_settings(denoise_enabled=True, denoise_kernel_size=kernel),
                12,
            )
            if np.array_equal(denoised, undenoised):
                skipped.append(f'{channels} channels, kernel {kernel}')

    assert not skipped, skipped


def _specimen_12_bit() -> np.ndarray:
    """A 480x640 12-bit field: a tilted background, noise, and bright
    blobs, so every step (the plane's fit, the levels, the gate, the blur)
    has something its number changes."""
    rng = np.random.default_rng(11)
    rows, columns = np.mgrid[0:480, 0:640]
    image = 300.0 + 0.4 * columns + 0.2 * rows + rng.normal(0.0, 25.0, (480, 640))
    for _ in range(40):
        y, x = rng.integers(20, 460), rng.integers(20, 620)
        image[(rows - y) ** 2 + (columns - x) ** 2 < rng.integers(16, 100)] += 1800.0
    return np.clip(image, 0, 4095).astype(np.uint16)


# Each fixed number a step reads, the value it is moved to, and where the
# recipe states it: (settings field, moved value, operation, recipe key).
_STEP_NUMBERS = (
    (
        'illumination_fit_excludes_upper_percentile',
        30.0,
        'illumination_correction',
        'fit_excludes_upper_percentile',
    ),
    ('illumination_fit_sample_target', 1000, 'illumination_correction', 'fit_sample_target'),
    ('sharpen_sigma_px', 2.0, 'sharpen', 'sigma_px'),
    ('sharpen_gain', 1.2, 'sharpen', 'gain'),
    ('sharpen_gate_background_percentile', 40.0, 'sharpen', 'gate_background_percentile'),
    ('sharpen_gate_top_percentile', 80.0, 'sharpen', 'gate_top_percentile'),
)


def test_each_number_a_step_reads_changes_the_copy_and_is_stated_in_its_recipe(tmp_path):
    """The recipe states every number the enhancement steps used, read from
    the one setting the step reads: moving one number alone changes the
    enhanced copy, and the copy's recipe states the moved value. A step
    that kept its own literal leaves the copy unchanged and fails here."""
    source = tmp_path / 'specimen.tif'
    _write_source(source, _specimen_12_bit(), 12)
    enhancer = QuickEnhancer()

    def enhance(settings):
        made = enhancer.export_file(source, settings, channel_colors=release_colors())
        pixels, _bits, _white = image_utils.load_pixels(
            made['output_path'], collapse_legacy_false_color=False
        )
        return pixels, json.loads(made['recipe_path'].read_text())

    default_pixels, _ = enhance(quick_enhance_settings())
    offenders = []
    for field, moved, operation, key in _STEP_NUMBERS:
        pixels, recipe = enhance(quick_enhance_settings(**{field: moved}))
        if np.array_equal(pixels, default_pixels):
            offenders.append(f'{field}={moved!r} left the copy unchanged')
        stated = recipe['operations'][operation].get(key)
        if stated != moved:
            offenders.append(f'{field}={moved!r}: the recipe states {operation}.{key}={stated!r}')

    assert not offenders, offenders


# Each step's switch and the recipe operation that states it.
_STEP_SWITCHES = {
    'auto_levels_enabled': 'auto_levels',
    'gamma_enabled': 'gamma',
    'illumination_correction_enabled': 'illumination_correction',
    'denoise_enabled': 'denoise',
    'sharpen_enabled': 'sharpen',
}


def test_each_step_switch_moved_from_its_shipped_value_changes_the_copy_and_is_stated(tmp_path):
    """Every step has one switch, the template's: flipping one alone from its
    shipped value changes the enhanced copy, and the copy's recipe states
    the step enabled as the switch says. A step that ignored its switch
    leaves the copy unchanged and fails here."""
    shipped = shipped_template()['quick_enhance']
    assert set(shipped) == set(_STEP_SWITCHES)
    source = tmp_path / 'specimen.tif'
    _write_source(source, _specimen_12_bit(), 12)
    enhancer = QuickEnhancer()

    def enhance(settings):
        made = enhancer.export_file(source, settings, channel_colors=release_colors())
        pixels, _bits, _white = image_utils.load_pixels(
            made['output_path'], collapse_legacy_false_color=False
        )
        return pixels, json.loads(made['recipe_path'].read_text())

    shipped_pixels, _ = enhance(quick_enhance_settings())
    offenders = []
    for switch, operation in _STEP_SWITCHES.items():
        flipped = not shipped[switch]
        pixels, recipe = enhance(quick_enhance_settings(**{switch: flipped}))
        if np.array_equal(pixels, shipped_pixels):
            offenders.append(f'{switch}={flipped} left the copy unchanged')
        stated = recipe['operations'][operation]['enabled']
        if stated is not flipped:
            offenders.append(
                f'{switch}={flipped}: the recipe states {operation} enabled={stated!r}'
            )

    assert not offenders, offenders


def test_with_every_step_off_the_steps_change_no_pixel_and_the_recipe_says_so(tmp_path):
    """All five switches off is allowed: no step changes a pixel, at any
    depth or in colour, and the recipe states every step off."""
    all_off = quick_enhance_settings(**dict.fromkeys(_STEP_SWITCHES, False))
    rng = np.random.default_rng(7)
    sources = {
        'mono 8-bit': (rng.integers(0, 256, (48, 64)).astype(np.uint8), 8),
        'mono 12-bit': (rng.integers(0, 4096, (48, 64)).astype(np.uint16), 12),
        'mono 16-bit': (rng.integers(0, 65536, (48, 64)).astype(np.uint16), 16),
        'RGB 8-bit': (rng.integers(0, 256, (48, 64, 3)).astype(np.uint8), 8),
    }
    changed = [
        name
        for name, (image, bits) in sources.items()
        if not np.array_equal(QuickEnhancer().apply(image, all_off, bits), image)
    ]
    assert not changed, f'every step off still changed: {changed}'

    recipe = QuickEnhancer().recipe_dict(
        source_path=tmp_path / 'source.tif',
        output_path=tmp_path / 'source_enhanced.tiff',
        settings=all_off,
        input_dtype=np.dtype(np.uint16),
        output_dtype=np.dtype(np.uint16),
        significant_bits=12,
    )
    stated_on = [op for op, step in recipe['operations'].items() if step['enabled'] is not False]
    assert not stated_on, f'the recipe states on: {stated_on}'


def test_gamma_runs_on_its_own_switch_and_follows_the_stretch_when_levels_run():
    """Gamma runs on its switch alone: with auto-levels off it runs on the
    image's own values. With auto-levels on it follows the stretch, so a
    flat image, which the stretch leaves as it is, gamma leaves too."""
    only = dict.fromkeys(_STEP_SWITCHES, False)
    image = _specimen_12_bit()
    gamma_alone = QuickEnhancer().apply(
        image, quick_enhance_settings(**{**only, 'gamma_enabled': True}), 12
    )
    assert not np.array_equal(gamma_alone, image), 'gamma on, auto-levels off, changed nothing'

    flat = np.full((64, 64), 1000, dtype=np.uint16)
    levels_and_gamma = quick_enhance_settings(
        **{**only, 'auto_levels_enabled': True, 'gamma_enabled': True}
    )
    assert np.array_equal(QuickEnhancer().apply(flat, levels_and_gamma, 12), flat)


def test_a_settings_without_its_switches_is_refused_at_construction():
    """The template is the switches' one home: Quick Enhance's settings
    have no default switch, so a construction that names none fails."""
    with pytest.raises(TypeError, match='sharpen_enabled'):
        QuickEnhanceSettings()


def test_recipe_has_required_provenance_and_quantitative_warning(tmp_path):
    source = tmp_path / 'source.tif'
    _write_source(source, np.arange(64, dtype=np.uint16).reshape(8, 8), 12)

    recipe = QuickEnhancer().recipe_dict(
        source_path=source,
        output_path=tmp_path / 'source_enhanced.tiff',
        settings=quick_enhance_settings(),
        input_dtype=np.dtype(np.uint16),
        output_dtype=np.dtype(np.uint16),
        significant_bits=12,
    )

    assert recipe['source_filename'] == 'source.tif'
    # The version pins the recipe GENERATION: the global-plane rework changed
    # the algorithm, the defaults, and the sidecar schema, so its outputs must
    # not be provenance-identical to generation-1 exports.
    assert recipe['pipeline_version'] == '3'
    assert recipe['quantitative_use_warning']
    assert recipe['input_dtype'] == 'uint16'
    assert recipe['operations']['auto_levels']['enabled'] is True
    # Every key a version-3 recipe has stated keeps its name and its value,
    # so a reader of an older recipe reads this one unchanged.
    assert recipe['operations']['illumination_correction'] == {
        'enabled': True,
        'method': 'global_plane_normalization',
        'fit_excludes_upper_percentile': 10.0,
        'fit_sample_target': 100_000,
    }
    assert recipe['operations']['sharpen'] == {
        'enabled': True,
        'method': 'signal_gated_unsharp_mask',
        'sigma_px': 1.0,
        'gain': 0.65,
        'gate_background_percentile': 65.0,
        'gate_top_percentile': 95.0,
    }


def test_quantitative_warning_routes_ai_workflows_to_lumaquant_pro():
    assert 'raw images' in QUANTITATIVE_USE_WARNING.lower()
    assert 'LumaQuant Pro' in QUANTITATIVE_USE_WARNING


def test_export_preserves_tiff_depth_writes_recipe_and_never_overwrites(tmp_path):
    source = tmp_path / 'original.tif'
    source_data = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    _write_source(source, source_data, 12)
    original_bytes = source.read_bytes()
    enhancer = QuickEnhancer()

    first = enhancer.export_file(source, quick_enhance_settings(), channel_colors=release_colors())
    second = enhancer.export_file(source, quick_enhance_settings(), channel_colors=release_colors())

    assert source.read_bytes() == original_bytes
    assert first['output_path'].name == 'original_enhanced.tiff'
    assert second['output_path'].name == 'original_enhanced_001.tiff'
    out, significant_bits, _ = image_utils.load_pixels(first['output_path'])
    assert out.dtype == np.uint16
    assert significant_bits == 12
    recipe_path = first['output_path'].with_suffix('.recipe.json')
    assert recipe_path.exists()
    assert json.loads(recipe_path.read_text())['source_filename'] == 'original.tif'
    assert (tmp_path / 'original_enhanced_001.recipe.json').exists()


def test_a_recipe_alone_at_a_name_still_takes_it(tmp_path):
    # An enhanced copy and its recipe are one name: a recipe left behind
    # by itself keeps the next enhance off its name, so that recipe never
    # describes a copy it was not written for.
    source = tmp_path / 'original.tif'
    _write_source(source, np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8), 12)
    (tmp_path / 'original_enhanced.recipe.json').write_text('{"left": "behind"}')

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )

    assert exported['output_path'].name == 'original_enhanced_001.tiff'
    assert (tmp_path / 'original_enhanced.recipe.json').read_text() == '{"left": "behind"}'


def test_brightfield_tiff_stays_monochrome_and_preserves_payload_depth(tmp_path):
    source = tmp_path / 'bf.tif'
    source_data = np.linspace(0, 4095, 256, dtype=np.uint16).reshape(16, 16)
    _write_source(source, source_data, 12, channel='BF')

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )
    output, significant_bits, _ = image_utils.load_pixels(exported['output_path'])

    assert output.ndim == 2
    assert output.dtype == np.uint16
    assert significant_bits == 12


def test_composite_tiff_stays_rgb_and_retains_composite_channel_metadata(tmp_path):
    source = tmp_path / 'composite.tif'
    source_data = np.zeros((16, 16, 3), dtype=np.uint16)
    source_data[..., 0] = 4095
    source_data[..., 1] = np.arange(16, dtype=np.uint16)[:, None] * 200
    source_data[..., 2] = 1000
    _write_source(source, source_data, 12, channel='Composite')

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )
    output, significant_bits, _ = image_utils.load_pixels(exported['output_path'])
    metadata = image_utils.read_file_record(exported['output_path'])

    assert output.shape == source_data.shape
    assert output.dtype == np.uint16
    assert significant_bits == 12
    assert metadata['channel'] == 'Composite'


def test_rgba_composite_exports_as_rgb_without_rejecting_composite_metadata(tmp_path):
    source = tmp_path / 'composite_rgba.tif'
    source_data = np.zeros((16, 16, 4), dtype=np.uint16)
    source_data[..., 0] = 4095
    source_data[..., 1] = 1200
    source_data[..., 3] = 4095
    tf.imwrite(source, source_data, photometric='rgb')

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )
    output, significant_bits, _ = image_utils.load_pixels(
        exported['output_path'], collapse_legacy_false_color=False
    )

    assert output.shape == (16, 16, 3)
    assert output.dtype == np.uint16
    assert significant_bits == 16


def test_rgba_composite_png_drops_alpha_before_rgb_tiff_export(tmp_path):
    source = tmp_path / 'composite_rgba.png'
    source_data = np.zeros((16, 16, 4), dtype=np.uint8)
    source_data[..., 1] = 180
    source_data[..., 3] = 255
    assert cv2.imwrite(str(source), source_data)

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )
    output, significant_bits, _ = image_utils.load_pixels(
        exported['output_path'], collapse_legacy_false_color=False
    )

    assert output.shape == (16, 16, 3)
    assert output.dtype == np.uint8
    assert significant_bits == 8


@pytest.mark.parametrize(
    ('channel', 'color_index'),
    [(LED2, 0), (LED1, 1), (LED0, 2)],
)
def test_fluorescence_mono_output_is_false_colored_without_changing_bf_rules(channel, color_index):
    enhancer = QuickEnhancer()
    mono = np.arange(64, dtype=np.uint16).reshape(8, 8)

    colored = enhancer.colorize_mono_for_visual_output(
        mono, channel, channel_colors=release_colors()
    )
    grayscale_bf = enhancer.colorize_mono_for_visual_output(
        mono, 'BF', channel_colors=release_colors()
    )

    assert colored.shape == (8, 8, 3)
    assert np.array_equal(colored[..., color_index], mono)
    assert all(not colored[..., index].any() for index in range(3) if index != color_index)
    assert grayscale_bf.ndim == 2


def test_green_export_is_rgb_and_a_second_run_keeps_green_metadata(tmp_path):
    source = tmp_path / 'green_sample.tiff'
    source_data = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    _write_source(source, source_data, 12, channel=LED1)
    enhancer = QuickEnhancer()

    first = enhancer.export_file(source, quick_enhance_settings(), channel_colors=release_colors())
    output, significant_bits, _ = image_utils.load_pixels(
        first['output_path'], collapse_legacy_false_color=False
    )
    metadata = image_utils.read_file_record(first['output_path'])

    assert output.shape == (8, 8, 3)
    assert output.dtype == np.uint16
    assert significant_bits == 12
    assert output[..., 1].any()
    assert not output[..., 0].any()
    assert not output[..., 2].any()
    assert metadata['channel'] == LED1


def test_metadata_less_legacy_green_rgb_with_numeric_prefix_exports_green(tmp_path):
    source = tmp_path / '0green_s.tiff'
    source_data = np.zeros((32, 32, 3), dtype=np.uint8)
    source_data[..., 1] = np.arange(32, dtype=np.uint8)[:, np.newaxis] * 8
    tf.imwrite(source, source_data, photometric='rgb')
    original = source.read_bytes()

    exported = QuickEnhancer().export_file(
        source, quick_enhance_settings(), channel_colors=release_colors()
    )
    output, significant_bits, _ = image_utils.load_pixels(
        exported['output_path'], collapse_legacy_false_color=False
    )
    metadata = image_utils.read_file_record(exported['output_path'])

    assert source.read_bytes() == original
    assert output.shape == source_data.shape
    assert output.dtype == np.uint8
    assert significant_bits == 8
    assert output[..., 1].any()
    assert not output[..., 0].any()
    assert not output[..., 2].any()
    assert metadata['channel'] == 'Green'


def test_fixed_recipe_deepens_a_close_bead_pair_saddle_without_lifting_background():
    yy, xx = np.mgrid[:256, :256]
    image = np.clip(
        16
        + 200 * np.exp(-((xx - 124) ** 2 + (yy - 128) ** 2) / (2 * 2.0**2))
        + 200 * np.exp(-((xx - 132) ** 2 + (yy - 128) ** 2) / (2 * 2.0**2)),
        0,
        255,
    ).astype(np.uint8)
    result = QuickEnhancer().apply(image, quick_enhance_settings(), 8)

    def saddle_to_peak(array):
        centerline = array[128, 124:133].astype(np.float32)
        return centerline[2:7].min() / min(centerline[:2].max(), centerline[-2:].max())

    assert saddle_to_peak(result) < saddle_to_peak(image) * 0.9
    assert result[0, 0] == image[0, 0]


def test_a_folder_of_only_undecodable_images_is_refused_as_the_count_refuses_it(tmp_path):
    (tmp_path / 'bad.tif').write_bytes(b'not a tiff')

    with pytest.raises(PostProcessingRefusedError) as refused:
        QuickEnhancer().export_folder(
            tmp_path, quick_enhance_settings(), channel_colors=release_colors()
        )

    assert refused.value.reason == 'no_data'
    assert 'None of the 1 image(s)' in str(refused.value)
    assert not list(tmp_path.glob('*_enhanced*'))


def test_folder_batch_skips_unreadable_file_and_continues(tmp_path):
    _write_source(
        tmp_path / 'valid.tif', np.arange(64, dtype=np.uint16).reshape(8, 8), significant_bits=12
    )
    (tmp_path / 'bad.tif').write_bytes(b'not a tiff')
    progress = []

    # Every image is still tried; the skipped one makes the pass incomplete,
    # and the outcome carries what was saved.
    with pytest.raises(PostProcessingFailedError) as raised:
        QuickEnhancer().export_folder(
            tmp_path,
            quick_enhance_settings(),
            progress_callback=lambda completed, total, path: progress.append(
                (completed, total, path.name)
            ),
            channel_colors=release_colors(),
        )

    assert raised.value.produced_paths == (str(tmp_path / 'valid_enhanced.tiff'),)
    assert len(raised.value.errors) == 1
    assert [(completed, total) for completed, total, _ in progress] == [(1, 2), (2, 2)]
    assert {name for _, _, name in progress} == {'valid.tif', 'bad.tif'}
    assert (tmp_path / 'valid_enhanced.tiff').exists()


def test_mixed_folder_exports_bf_composite_png_jpeg_and_skips_bad_or_derived_files(tmp_path):
    bf = tmp_path / 'bf.tif'
    composite = tmp_path / 'composite.tif'
    _write_source(bf, np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8), 12, channel='BF')
    composite_pixels = np.zeros((8, 8, 3), dtype=np.uint16)
    composite_pixels[..., 0] = 4095
    composite_pixels[..., 1] = 1200
    _write_source(composite, composite_pixels, 12, channel='Composite')
    png = tmp_path / 'share.png'
    jpeg = tmp_path / 'share.jpg'
    color = np.full((8, 8, 3), (20, 100, 220), dtype=np.uint8)
    assert cv2.imwrite(str(png), color)
    assert cv2.imwrite(str(jpeg), color)
    bad = tmp_path / 'bad.tif'
    bad.write_bytes(b'not an image')
    existing_derived = tmp_path / 'old_enhanced.tif'
    _write_source(existing_derived, np.ones((8, 8), dtype=np.uint16), 12, channel='BF')
    original_bytes = {
        path: path.read_bytes() for path in (bf, composite, png, jpeg, bad, existing_derived)
    }
    progress = []

    displayed = []
    with pytest.raises(PostProcessingFailedError) as raised:
        QuickEnhancer().export_folder(
            tmp_path,
            quick_enhance_settings(),
            progress_callback=lambda completed, total, path: progress.append(
                (completed, total, path.name)
            ),
            display_callback=lambda image, significant_bits: displayed.append(
                (image.shape, significant_bits)
            ),
            channel_colors=release_colors(),
        )

    assert len(raised.value.produced_paths) == 4
    assert [error.split(':')[0] for error in raised.value.errors] == ['bad.tif']
    assert [(completed, total) for completed, total, _ in progress] == [
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ]
    assert all(path.read_bytes() == original for path, original in original_bytes.items())
    assert (tmp_path / 'bf_enhanced.tiff').exists()
    assert (tmp_path / 'composite_enhanced.tiff').exists()
    assert (tmp_path / 'share_enhanced.tiff').exists()
    assert (tmp_path / 'share_enhanced_001.tiff').exists()
    assert len(displayed) == 4


def test_output_folder_is_reported_from_a_completed_export(tmp_path):
    completed = {
        'created': [
            {
                'output_path': tmp_path / 'source_enhanced.tiff',
            }
        ]
    }

    assert QuickEnhancer.output_folder(completed) == tmp_path
    assert QuickEnhancer.output_folder({'created': []}) is None


def test_ui_export_done_restores_controls_on_every_terminal_path():
    """Automatic Enhance owns ``busy`` for the complete export lifetime: the
    boundary calls ``_export_done`` once whatever the outcome, and it clears
    ``busy`` before anything else."""
    source = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'QuickEnhanceControls'
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'done' for target in node.targets)
        for node in cls.body
    )
    done = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == '_export_done'
    )
    first = done.body[0]
    assert (
        isinstance(first, ast.Assign)
        and ast.unparse(first.targets[0]) == 'self.busy'
        and isinstance(first.value, ast.Constant)
        and first.value.value is False
    ), 'self.busy = False must be the first statement of _export_done'


def test_a_selection_starts_enhance_through_one_member():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'QuickEnhanceControls'
    )
    members = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    assert 'set_source_file' not in members and 'set_source_folder' not in members
    assert 'self.export(' in ast.get_source_segment(source, members['set_source'])
