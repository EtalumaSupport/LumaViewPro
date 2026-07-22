from __future__ import annotations

import json
import ast
import pathlib

import cv2
import numpy as np
import pytest
import tifffile as tf

from modules import image_utils
from modules.quick_enhance import QuickEnhanceSettings, QuickEnhancer


def _metadata(significant_bits: int, channel: str = 'Green') -> dict:
    return {
        'pixel_size_um': 1.0,
        'channel': channel,
        'objective': '10x',
        'exposure_time_ms': 1.0,
        'gain_db': 0.0,
        'illumination_ma': 1.0,
        'z_pos_um': 0.0,
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'datetime': '2026:07:17 00:00:00',
        'significant_bits': significant_bits,
    }


def _write_source(path, data: np.ndarray, significant_bits: int, channel: str = 'Green') -> None:
    image_utils.write_tiff(
        data=data,
        file_loc=path,
        metadata=_metadata(significant_bits, channel),
        ome=False,
        color=channel,
        significant_bits=significant_bits,
        save_encoding='right_aligned',
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
    result = QuickEnhancer().apply(image, QuickEnhanceSettings(), significant_bits)

    assert np.array_equal(image, source)
    assert result.dtype == image.dtype
    assert result.shape == image.shape
    assert int(result.min()) >= 0
    assert int(result.max()) <= maximum


def test_uniform_image_is_stable_and_does_not_divide_by_zero():
    image = np.full((12, 12), 1000, dtype=np.uint16)
    result = QuickEnhancer().apply(
        image, QuickEnhanceSettings.for_preset('Brightfield / Phase'), 12
    )

    assert np.array_equal(result, image)


def test_rgb_images_use_one_shared_tone_curve_without_changing_shape_or_dtype():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8)[:, None] * 8
    image[..., 1] = 100
    image[..., 2] = 200
    source = image.copy()

    result = QuickEnhancer().apply(image, QuickEnhanceSettings(), 8)

    assert np.array_equal(image, source)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


@pytest.mark.parametrize('suffix', ('.png', '.jpg'))
def test_png_and_jpeg_load_preview_and_export_as_derived_tiff(tmp_path, suffix):
    source = tmp_path / f'color{suffix}'
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[..., 0] = 30
    image[..., 1] = 120
    image[..., 2] = 220
    assert cv2.imwrite(str(source), image)

    loaded, significant_bits = image_utils.load_pixels(source)
    before, after = QuickEnhancer().preview(loaded, QuickEnhanceSettings(), significant_bits)
    exported = QuickEnhancer().export_file(source, QuickEnhanceSettings())

    assert before.shape == loaded.shape
    assert after.shape == loaded.shape
    assert exported['output_path'].suffix == '.tif'
    assert exported['output_path'].exists()


def test_invalid_settings_are_rejected_before_processing():
    settings = QuickEnhanceSettings(low_percentile=99, high_percentile=1, gamma=3.0)
    errors = QuickEnhancer().validate(settings, np.dtype(np.uint16), 12)

    assert any('percentile' in error.lower() for error in errors)
    assert any('gamma' in error.lower() for error in errors)
    with pytest.raises(ValueError, match='Invalid Quick Enhance settings'):
        QuickEnhancer().apply(np.ones((4, 4), dtype=np.uint16), settings, 12)


def test_presets_are_stable_and_safe():
    automatic = QuickEnhanceSettings.for_preset('Auto (Recommended)')
    brightfield = QuickEnhanceSettings.for_preset('Brightfield / Phase')
    contrast_only = QuickEnhanceSettings.for_preset('Contrast Only')

    assert automatic.background_enabled is False
    assert automatic.denoise_enabled is False
    assert automatic.gamma == 1.0
    assert brightfield.gamma == 0.9
    assert contrast_only.gamma == 1.0
    with pytest.raises(ValueError, match='Unknown Quick Enhance preset'):
        QuickEnhanceSettings.for_preset('Custom')


def test_auto_detects_transmitted_bf_from_tiff_name_when_metadata_is_unavailable(tmp_path):
    source = tmp_path / 'scan_BF.tif'
    _write_source(source, np.arange(64, dtype=np.uint16).reshape(8, 8), 12, channel='BF')

    settings, description = QuickEnhancer().settings_for_source(
        source, QuickEnhanceSettings.for_preset('Auto (Recommended)')
    )

    assert settings.preset == 'Brightfield / Phase'
    assert settings.gamma == 0.9
    assert 'BF' in description


def test_auto_uses_neutral_levels_for_composite_and_unknown_images(tmp_path):
    composite = tmp_path / 'composite.tif'
    _write_source(composite, np.zeros((8, 8, 3), dtype=np.uint16), 12, channel='Composite')
    unknown = tmp_path / 'untagged.png'
    assert cv2.imwrite(str(unknown), np.zeros((8, 8, 3), dtype=np.uint8))
    auto = QuickEnhanceSettings.for_preset('Auto (Recommended)')

    composite_settings, composite_description = QuickEnhancer().settings_for_source(composite, auto)
    unknown_settings, unknown_description = QuickEnhancer().settings_for_source(unknown, auto)

    assert composite_settings.preset == 'Auto (Recommended)'
    assert 'Composite' in composite_description
    assert unknown_settings.preset == 'Auto (Recommended)'
    assert 'neutral' in unknown_description


def test_background_correction_never_wraps_unsigned_values():
    image = np.zeros((64, 64), dtype=np.uint16)
    image[:, 32:] = 100
    image[20:30, 20:30] = 1000
    settings = QuickEnhanceSettings(background_enabled=True, background_sigma=5.0)

    result = QuickEnhancer().apply(image, settings, 12)

    assert result.dtype == np.uint16
    assert int(result.min()) >= 0
    assert int(result.max()) <= 4095


def test_tiny_images_with_denoise_are_safe():
    image = np.array([[1]], dtype=np.uint8)
    settings = QuickEnhanceSettings(denoise_enabled=True, denoise_kernel_size=3)

    result = QuickEnhancer().apply(image, settings, 8)

    assert np.array_equal(result, image)


def test_recipe_has_required_provenance_and_quantitative_warning(tmp_path):
    source = tmp_path / 'source.tif'
    _write_source(source, np.arange(64, dtype=np.uint16).reshape(8, 8), 12)

    recipe = QuickEnhancer().recipe_dict(
        source_path=source,
        output_path=tmp_path / 'source_enhanced.tif',
        settings=QuickEnhanceSettings(),
        input_dtype=np.dtype(np.uint16),
        output_dtype=np.dtype(np.uint16),
        significant_bits=12,
    )

    assert recipe['source_filename'] == 'source.tif'
    assert recipe['pipeline_version']
    assert recipe['quantitative_use_warning']
    assert recipe['input_dtype'] == 'uint16'
    assert recipe['operations']['auto_levels']['enabled'] is True


def test_export_preserves_tiff_depth_writes_recipe_and_never_overwrites(tmp_path):
    source = tmp_path / 'original.tif'
    source_data = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    _write_source(source, source_data, 12)
    original_bytes = source.read_bytes()
    enhancer = QuickEnhancer()

    first = enhancer.export_file(source, QuickEnhanceSettings())
    second = enhancer.export_file(source, QuickEnhanceSettings())

    assert source.read_bytes() == original_bytes
    assert first['output_path'].name == 'original_enhanced.tif'
    assert second['output_path'].name == 'original_enhanced_2.tif'
    out, significant_bits = image_utils.load_pixels(first['output_path'])
    assert out.dtype == np.uint16
    assert significant_bits == 12
    recipe_path = first['output_path'].with_suffix('.recipe.json')
    assert recipe_path.exists()
    assert json.loads(recipe_path.read_text())['source_filename'] == 'original.tif'


def test_brightfield_tiff_stays_monochrome_and_preserves_payload_depth(tmp_path):
    source = tmp_path / 'bf.tif'
    source_data = np.linspace(0, 4095, 256, dtype=np.uint16).reshape(16, 16)
    _write_source(source, source_data, 12, channel='BF')

    exported = QuickEnhancer().export_file(
        source, QuickEnhanceSettings.for_preset('Brightfield / Phase')
    )
    output, significant_bits = image_utils.load_pixels(exported['output_path'])

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

    exported = QuickEnhancer().export_file(source, QuickEnhanceSettings())
    output, significant_bits = image_utils.load_pixels(exported['output_path'])
    metadata = image_utils.read_postproc_input_metadata(exported['output_path'])

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
        source, QuickEnhanceSettings.for_preset('Auto (Recommended)')
    )
    output, significant_bits = image_utils.load_pixels(
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
        source, QuickEnhanceSettings.for_preset('Auto (Recommended)')
    )
    output, significant_bits = image_utils.load_pixels(
        exported['output_path'], collapse_legacy_false_color=False
    )

    assert output.shape == (16, 16, 3)
    assert output.dtype == np.uint8
    assert significant_bits == 8


@pytest.mark.parametrize(
    ('channel', 'color_index'),
    [('Red', 0), ('Green', 1), ('Blue', 2)],
)
def test_fluorescence_mono_output_is_false_colored_without_changing_bf_rules(channel, color_index):
    enhancer = QuickEnhancer()
    mono = np.arange(64, dtype=np.uint16).reshape(8, 8)

    colored = enhancer.colorize_mono_for_visual_output(mono, channel)
    grayscale_bf = enhancer.colorize_mono_for_visual_output(mono, 'BF')

    assert colored.shape == (8, 8, 3)
    assert np.array_equal(colored[..., color_index], mono)
    assert all(not colored[..., index].any() for index in range(3) if index != color_index)
    assert grayscale_bf.ndim == 2


def test_green_auto_export_is_rgb_and_a_second_run_keeps_green_detection(tmp_path):
    source = tmp_path / 'green_sample.tiff'
    source_data = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    _write_source(source, source_data, 12, channel='Green')
    enhancer = QuickEnhancer()

    first = enhancer.export_file(source, QuickEnhanceSettings.for_preset('Auto (Recommended)'))
    output, significant_bits = image_utils.load_pixels(
        first['output_path'], collapse_legacy_false_color=False
    )
    settings, detection = enhancer.settings_for_source(
        first['output_path'], QuickEnhanceSettings.for_preset('Auto (Recommended)')
    )

    assert output.shape == (8, 8, 3)
    assert output.dtype == np.uint16
    assert significant_bits == 12
    assert output[..., 1].any()
    assert not output[..., 0].any()
    assert not output[..., 2].any()
    assert settings.preset == 'Auto (Recommended)'
    assert 'Green' in detection


def test_folder_batch_skips_unreadable_file_and_continues(tmp_path):
    _write_source(
        tmp_path / 'valid.tif', np.arange(64, dtype=np.uint16).reshape(8, 8), significant_bits=12
    )
    (tmp_path / 'bad.tif').write_bytes(b'not a tiff')
    progress = []

    result = QuickEnhancer().export_folder(
        tmp_path,
        QuickEnhanceSettings(),
        progress_callback=lambda completed, total, path: progress.append(
            (completed, total, path.name)
        ),
    )

    assert result['status'] is True
    assert result['created_count'] == 1
    assert len(result['skipped']) == 1
    assert [(completed, total) for completed, total, _ in progress] == [(1, 2), (2, 2)]
    assert {name for _, _, name in progress} == {'valid.tif', 'bad.tif'}
    assert (tmp_path / 'valid_enhanced.tif').exists()


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

    result = QuickEnhancer().export_folder(
        tmp_path,
        QuickEnhanceSettings(),
        progress_callback=lambda completed, total, path: progress.append(
            (completed, total, path.name)
        ),
    )

    assert result['total'] == 5
    assert result['created_count'] == 4
    assert [entry['source_path'].name for entry in result['skipped']] == ['bad.tif']
    assert [(completed, total) for completed, total, _ in progress] == [
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ]
    assert all(path.read_bytes() == original for path, original in original_bytes.items())
    assert (tmp_path / 'bf_enhanced.tif').exists()
    assert (tmp_path / 'composite_enhanced.tif').exists()
    assert (tmp_path / 'share_enhanced.tif').exists()
    assert (tmp_path / 'share_enhanced_2.tif').exists()


def test_output_folder_is_reported_from_a_completed_export(tmp_path):
    completed = {
        'created': [
            {
                'output_path': tmp_path / 'source_enhanced.tif',
            }
        ]
    }

    assert QuickEnhancer.output_folder(completed) == tmp_path
    assert QuickEnhancer.output_folder({'created': []}) is None


def test_ui_export_callback_restores_controls_on_every_terminal_path():
    """The popup wrapper binds ``done`` and failure must clear ``busy``."""
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
    callback = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == '_export_callback'
    )
    callback_source = ast.get_source_segment(source, callback)
    assert 'self.busy = False' in callback_source
    assert 'self._export_inflight = False' in callback_source


def test_refresh_preview_requests_the_rebuilding_status():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text()
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'QuickEnhanceControls'
    )
    refresh = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == 'refresh_preview'
    )

    assert 'refresh=True' in ast.get_source_segment(source, refresh)
    assert "'Updating preview…' if refresh" in source


def test_preview_worker_imports_the_fluorescence_channel_helper():
    source = (pathlib.Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text()

    assert 'from modules import common_utils' in source
