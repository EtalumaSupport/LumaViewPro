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


def test_mirror_is_listener_published_not_caller_pushed():
    """The kv mirror properties are written by the App's run-state
    publisher, fed by the session's transition listener -- no caller
    push helper remains. A reappearing per-caller publisher is a second
    writer for the mirror and the drift it caused."""
    helpers = _read('ui/ui_helpers.py')
    assert 'def publish_protocol_running' not in helpers
    app_src = _read('lumaviewpro.py')
    assert 'def publish_run_state' in app_src
    assert 'add_run_state_listener' in app_src


def test_session_exposes_is_protocol_running_accessor():
    body = _def_source(_read('modules/scope_session.py'), 'is_protocol_running')
    assert body is not None, 'ScopeSession.is_protocol_running accessor missing'
    assert "activity_claim.owner == 'protocol'" in body


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
    assert 'session.run_lockout' in body.replace(' ', '')
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
    assert 'session.run_lockout' in body.replace(' ', ''), (
        'the file-choose funnel must refuse post-processing selections mid-run-lockout'
    )
    assert 'notifications.warning' in body
    guard_idx = body.replace(' ', '').find('session.run_lockout')
    dispatch_idx = body.replace(' ', '').find('set_source_file')
    assert guard_idx != -1 and dispatch_idx != -1 and guard_idx < dispatch_idx, (
        'the protocol-running guard must run before the quick-enhance dispatch'
    )
