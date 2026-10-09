"""One-click Enhance picker and central-viewer handoff regression guards."""

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _class_method_source(path, class_name, method_name):
    source = (REPO / path).read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def _write_source(path):
    import numpy as np

    from modules import image_utils

    image_utils.write_tiff(
        data=np.full((8, 8), 100, dtype=np.uint8),
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'BF',
            'objective': '10x',
            'exposure_time_ms': 50.0,
            'gain_db': 0.0,
            'illumination_ma': 100.0,
            'z_pos_um': 1000.0,
            'plate_pos_mm': {'x': 10.0, 'y': 20.0},
            'datetime': '2026:06:18 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': 'A1',
            'significant_bits': 8,
        },
        ome=False,
        color='BF',
        significant_bits=8,
        save_encoding='right_aligned',
    )


def test_macos_enhance_picker_accepts_a_file_or_folder_in_one_native_dialog():
    source = (REPO / 'ui' / 'file_dialogs.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == '_macos_choose_file_or_folder'
    )
    assert 'choose file or folder' in ast.get_source_segment(source, method)


def test_non_macos_enhance_picker_offers_both_native_target_kinds():
    source = (REPO / 'ui' / 'file_dialogs.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_platform_native_choose_file_or_folder'
    )
    body = ast.get_source_segment(source, method)
    assert 'Toplevel' in body
    assert 'askopenfilename' in body
    assert 'askdirectory' in body


def test_non_macos_enhance_picker_uses_image_and_folder_labels():
    source = (REPO / 'ui' / 'file_dialogs.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_platform_native_choose_file_or_folder'
    )
    body = ast.get_source_segment(source, method)

    assert 'Choose image or folder' in body
    assert "text='Image'" in body
    assert "text='Folder'" in body


def test_enhance_picker_hands_the_path_on_after_the_protocol_guard():
    # Whether the path is a file, a folder or nothing is the API's question,
    # asked where the build reads it: the picker hands the path on as chosen.
    body = _class_method_source(
        'ui/file_dialogs.py', 'FileOrFolderChooseBTN', 'on_selection_function'
    )
    guard_idx = body.find('session.run_lockout')
    dispatch_idx = body.find('set_source(path)')
    assert guard_idx != -1 and guard_idx < dispatch_idx
    assert 'is_dir' not in body and 'is_file' not in body


def test_a_target_that_is_not_there_is_refused_on_the_lane(tmp_path):
    import threading

    import pytest

    from modules.exceptions import PostProcessingRefusedError
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    ran_on = []
    real = session.post_processing._enhance

    def recording(*args, **kwargs):
        ran_on.append(threading.current_thread().name)
        return real(*args, **kwargs)

    try:
        session.post_processing._enhance = recording
        with pytest.raises(PostProcessingRefusedError) as caught:
            session.post_processing.enhance(tmp_path / 'gone.tif')
    finally:
        session.shutdown()

    assert caught.value.reason == 'unreadable'
    assert 'gone.tif' in str(caught.value)
    assert ran_on and ran_on[0] != threading.main_thread().name, (
        'refused on the lane, not the caller'
    )


def test_enhance_progress_is_counted_and_each_saved_image_reaches_the_main_viewer(tmp_path):
    from modules.post_processing_api import PostProcessingAPI

    _write_source(tmp_path / 'a.tif')
    _write_source(tmp_path / 'b.tif')
    texts = []
    shown = []

    PostProcessingAPI._enhance(
        tmp_path,
        lambda percent, text: texts.append(text),
        lambda image, significant_bits: shown.append(significant_bits),
    )

    assert texts == ['Image 1 of 2', 'Image 2 of 2']
    assert len(shown) == 2
    panel = (REPO / 'ui' / 'post_processing.py').read_text(encoding='utf-8')
    assert 'hold_derived_image' in panel

    viewer_method = _class_method_source(
        'ui/scope_display.py', 'ScopeDisplay', 'hold_derived_image'
    )
    assert 'image_to_texture' in viewer_method
    assert 'bump_protocol_hold' in viewer_method


def test_enhance_completion_hides_the_derived_output_path(tmp_path):
    from modules.post_processing_api import PostProcessingAPI

    source = tmp_path / 'source.tif'
    _write_source(source)

    result = PostProcessingAPI._enhance(source, None, None)

    assert result['message'] == 'Enhance complete.'
    assert str(tmp_path) not in result['message']


def test_the_mode_router_keeps_a_stitcher_for_every_mode():
    # Previously also asserted docs/STITCHING.md described these modes. That
    # doc is not published in this repo, so the prose half of the check cannot
    # live here; what remains is that the router still routes to each stitcher.
    router = (REPO / 'modules' / 'stitching_core.py').read_text(encoding='utf-8')

    assert "mode == 'fast_preview'" in router
    assert 'fft_phase_stitcher' in router
    assert 'overlap_stitcher' in router
    assert 'stage_position_stitcher' in router


def test_application_registers_the_one_click_picker():
    app_source = (REPO / 'lumaviewpro.py').read_text(encoding='utf-8')
    assert 'FileOrFolderChooseBTN' in app_source

    router = (REPO / 'modules' / 'stitching_core.py').read_text(encoding='utf-8')
    assert "mode == 'fast_preview'" in router
    assert 'quality_local_ncc' in router
