"""Regression tests for the protocol-running UI lockout (issue #166, #704).

A `protocol_running` BooleanProperty on the App mirrors the
ctx.protocol_running Event so kv `disabled:` bindings grey out interactive
controls during a scan. The property must be published True at every run
start and False at every run-reset (the abort-safe convergence point), or
controls would either never lock or stay stuck disabled after a scan. These
assert on source structure (the suite mocks Kivy; widgets are not built),
ruff-format-agnostic.
"""

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (REPO / rel).read_text()


def _def_source(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _method_in_class(src, class_name, method_name):
    tree = ast.parse(src)
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    if cls is None:
        return None
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(src, node)
    return None


def test_app_defines_protocol_running_boolean_property():
    src = _read('lumaviewpro.py')
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'LumaViewProApp'
    )
    found = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'protocol_running' for t in node.targets)
        and 'BooleanProperty' in ast.unparse(node.value)
        for node in cls.body
    )
    assert found, 'LumaViewProApp must define protocol_running = BooleanProperty(...)'


def test_publish_helper_schedules_flip_on_main_thread():
    # The canonical publisher lives in ui_helpers (several widgets need
    # it); ProtocolSettings keeps a thin delegate for its call sites.
    body = _def_source(_read('ui/ui_helpers.py'), 'publish_protocol_running')
    assert body is not None, 'publish_protocol_running helper missing'
    assert 'Clock.schedule_once' in body, 'the property flip must run on the Kivy main thread'
    assert 'protocol_running' in body
    delegate = _def_source(_read('ui/protocol_settings.py'), '_publish_protocol_running')
    assert delegate is not None and 'publish_protocol_running(' in delegate


def test_every_run_start_publishes_true():
    # The scan/protocol starters commit run-start state through the shared
    # _commit_running_ui_state (called between prepare and start so a
    # REFUSED run never publishes True); the publish lives there. The
    # autofocus scan keeps its own commit set (it never owns
    # ctx.protocol_running) and still publishes directly.
    src = _read('ui/protocol_settings.py')
    commit_body = _def_source(src, '_commit_running_ui_state')
    assert commit_body is not None, '_commit_running_ui_state missing'
    assert re.search(r'_publish_protocol_running\(\s*True\s*\)', commit_body), (
        '_commit_running_ui_state must publish protocol_running True'
    )
    for method in ('_run_scan_from_ui_inner', '_run_protocol_from_ui_inner'):
        body = _def_source(src, method)
        assert body is not None, f'{method} missing'
        assert '_commit_running_ui_state' in body, (
            f'{method} must commit run-start state via _commit_running_ui_state '
            f'(which publishes protocol_running True)'
        )
    af_body = _def_source(src, 'run_autofocus_scan_from_ui')
    assert af_body is not None, 'run_autofocus_scan_from_ui missing'
    assert re.search(r'_publish_protocol_running\(\s*True\s*\)', af_body), (
        'run_autofocus_scan_from_ui must publish protocol_running True at run start'
    )


def test_every_run_reset_publishes_false():
    src = _read('ui/protocol_settings.py')
    for method in (
        '_reset_run_scan_button',
        '_reset_run_protocol_button',
        '_reset_run_autofocus_scan_button',
    ):
        body = _def_source(src, method)
        assert body is not None, f'{method} missing'
        assert re.search(r'_publish_protocol_running\(\s*False\s*\)', body), (
            f'{method} must publish protocol_running False on reset (abort-safe clear)'
        )


def test_session_exposes_is_protocol_running_accessor():
    body = _def_source(_read('modules/scope_session.py'), 'is_protocol_running')
    assert body is not None, 'ScopeSession.is_protocol_running accessor missing'
    assert 'protocol_running.is_set' in body.replace(' ', '')


def test_postprocessing_funnel_blocks_during_protocol():
    """The file-dialog funnel refuses the 5 post-processing actions while a
    protocol runs -- the backstop behind the disabled buttons, so the action
    cannot be silently dropped by the busy file executor (no do-nothing popup).
    """
    src = _read('ui/file_dialogs.py')
    tree = ast.parse(src)
    consts = {
        n.targets[0].id: n
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    assert '_POST_PROCESSING_CONTEXTS' in consts, 'post-processing context set missing'
    ctx_tuple = ast.unparse(consts['_POST_PROCESSING_CONTEXTS'].value)
    for c in (
        'apply_cell_count_method_to_folder',
        'apply_stitching_to_folder',
        'apply_composite_gen_to_folder',
        'apply_video_gen_to_folder',
        'apply_zprojection_to_folder',
    ):
        assert c in ctx_tuple, f'{c} missing from the post-processing contexts'

    body = _method_in_class(src, 'FolderChooseBTN', 'on_selection_function')
    assert body is not None, 'FolderChooseBTN.on_selection_function missing'
    assert '_POST_PROCESSING_CONTEXTS' in body
    assert 'protocol_running.is_set' in body.replace(' ', '')
    assert 'notifications.warning' in body
    guard_idx = body.find('_POST_PROCESSING_CONTEXTS')
    dispatch_idx = body.find('apply_composite_gen_to_folder')
    assert guard_idx != -1 and dispatch_idx != -1 and guard_idx < dispatch_idx, (
        'the protocol-running guard must run before the post-processing dispatch'
    )


def test_quick_enhance_image_funnel_blocks_during_protocol():
    """The FILE-choose funnel needs the same backstop as the folder funnel.

    Picking a Quick Enhance input image lands in set_source_file, which sets
    the panel's ``busy`` flag BEFORE an executor put() that the file executor
    silently drops while a protocol owns it -- the preview callback then never
    fires, ``busy`` sticks True, and the kv binding disables the whole panel
    until restart. The native dialog is async and outlives protocol start, so
    the disabled button alone cannot stop the selection arriving mid-run.
    """
    src = _read('ui/file_dialogs.py')
    body = _method_in_class(src, 'FileChooseBTN', 'on_selection_function')
    assert body is not None, 'FileChooseBTN.on_selection_function missing'
    assert 'protocol_running.is_set' in body.replace(' ', ''), (
        'the file-choose funnel must refuse post-processing selections mid-protocol'
    )
    assert 'notifications.warning' in body
    guard_idx = body.replace(' ', '').find('protocol_running.is_set')
    dispatch_idx = body.replace(' ', '').find('set_source_file')
    assert guard_idx != -1 and dispatch_idx != -1 and guard_idx < dispatch_idx, (
        'the protocol-running guard must run before the quick-enhance dispatch'
    )
