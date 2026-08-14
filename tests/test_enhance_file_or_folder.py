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
    assert 'askyesnocancel' in body
    assert 'askopenfilename' in body
    assert 'askdirectory' in body


def test_enhance_picker_routes_files_and_folders_after_the_protocol_guard():
    body = _class_method_source(
        'ui/file_dialogs.py', 'FileOrFolderChooseBTN', 'on_selection_function'
    )
    guard_idx = body.find('protocol_running.is_set')
    folder_idx = body.find('path.is_dir')
    file_idx = body.find('path.is_file')
    assert guard_idx != -1 and guard_idx < folder_idx < file_idx
    assert 'set_source_folder' in body
    assert 'set_source_file' in body


def test_enhance_progress_is_counted_and_each_saved_image_reaches_the_main_viewer():
    post_processing = (REPO / 'ui' / 'post_processing.py').read_text(encoding='utf-8')
    assert "f'Image {completed} of {total}'" in post_processing
    assert 'hold_derived_image' in post_processing

    viewer_method = _class_method_source(
        'ui/scope_display.py', 'ScopeDisplay', 'hold_derived_image'
    )
    assert 'image_to_texture' in viewer_method
    assert 'bump_protocol_hold' in viewer_method


def test_enhance_completion_hides_the_derived_output_path():
    callback = _class_method_source(
        'ui/post_processing.py', 'QuickEnhanceControls', '_export_callback'
    )

    assert "summary = 'Enhance complete.'" in callback
    assert "f'Saved: {saved_path}'" not in callback


def test_stitching_manual_matches_the_stage_constrained_mode_router():
    manual = (REPO / 'docs' / 'STITCHING.md').read_text(encoding='utf-8')
    router = (REPO / 'modules' / 'stitching_core.py').read_text(encoding='utf-8')

    for phrase in ('FFT phase correlation', 'normalized cross-correlation', '0% overlap'):
        assert phrase in manual
    assert "mode == 'fast_preview'" in router
    assert 'fft_phase_stitcher' in router
    assert 'overlap_stitcher' in router
    assert 'stage_position_stitcher' in router


def test_application_registers_the_one_click_picker_and_manual_matches_stitch_router():
    app_source = (REPO / 'lumaviewpro.py').read_text(encoding='utf-8')
    assert 'FileOrFolderChooseBTN' in app_source

    manual = (REPO / 'docs' / 'STITCHING.md').read_text(encoding='utf-8')
    router = (REPO / 'modules' / 'stitching_core.py').read_text(encoding='utf-8')
    assert 'FFT phase correlation' in manual
    assert 'normalized cross-correlation' in manual
    assert "mode == 'fast_preview'" in router
    assert 'quality_local_ncc' in router
