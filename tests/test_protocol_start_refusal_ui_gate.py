# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: UI starters must commit running-state only between
prepare() and start(), inside the shared refusal boundary.

Contract
--------
SequencedCaptureRunner.run() is retired. Its replacement splits the
start sequence in two:

- prepare(**kwargs) performs every refusal gate (hardware disconnected,
  files still writing, empty or invalid protocol) and RAISES
  ProtocolRunRefusedError instead of returning False. It commits no
  runner state and touches no disk.
- start(plan) is the commitment point; it can only be reached with a
  plan a successful prepare() produced.

Because a refusal raises before start(), UI callers commit their
"a run is now underway" state (ctx.protocol_running, the published
app property, motion lock, button text) BETWEEN prepare and start --
never before. Every UI starter routes its prepare/start sequence
through ui_helpers.run_with_refusal_boundary, the single catch site
for the typed refusal, whose on_refused callback undoes only the
pre-gate button cosmetics.

Test approach
-------------
The Kivy UI classes cannot be instantiated headlessly (ids, _app_ctx,
worker pool), so this locks the call-site structure via AST/source
order. The behavioral half of the contract (what prepare() raises and
what the getters answer) lives in
test_protocol_execution.py::TestRunReturnValueContract and
tests/test_run_refusal_contract.py.
"""

from __future__ import annotations

import ast
import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent

# The four UI starters that kick off a sequenced run.
UI_STARTERS = (
    ('ui/protocol_settings.py', 'ProtocolSettings', '_run_scan_from_ui_inner'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_run_protocol_from_ui_inner'),
    ('ui/protocol_settings.py', 'ProtocolSettings', 'run_autofocus_scan_from_ui'),
    ('ui/zstack.py', 'ZStack', 'run_zstack_acquire_from_ui'),
)

# Statements that commit "a run is now underway" state in the UI.
COMMIT_MARKERS = (
    'protocol_running.set()',
    '_publish_protocol_running(True)',
    'set_motion_capability(False)',
)


def _method_node(source_file: pathlib.Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {source_file}')


def _calls_named(node: ast.AST, func_name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == func_name)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == func_name)
        )
    ]


def test_run_sequenced_capture_orders_prepare_commit_start():
    """run_sequenced_capture must prepare, then commit UI running-state,
    then start -- so a refusal (raised by prepare) can never leave the
    UI committed to a run that does not exist."""
    method = _method_node(
        REPO / 'ui' / 'protocol_settings.py', 'ProtocolSettings', 'run_sequenced_capture'
    )

    prepare_calls = _calls_named(method, 'prepare')
    assert prepare_calls, (
        'run_sequenced_capture must build the run via sequenced_capture_runner.prepare()'
    )
    start_calls = _calls_named(method, 'start')
    assert start_calls, (
        'run_sequenced_capture must dispatch the prepared plan via '
        'sequenced_capture_runner.start(plan)'
    )

    src = ast.unparse(method)
    prepare_pos = src.index('.prepare(')
    # The commit rides the restoring boundary: run_committed_start
    # snapshots, commits, starts, and restores the snapshot if start()
    # still refuses -- so a post-commit refusal cannot strand the
    # committed lockout either.
    commit_pos = src.index('run_committed_start(commit_ui_state')
    start_pos = src.index('.start(')
    assert prepare_pos < commit_pos < start_pos, (
        'the commit_ui_state() invocation must sit BETWEEN prepare() and '
        'start(): committing before prepare re-opens the refused-run UI '
        'wedge; committing after start races the run loop'
    )

    # The retired bool-returning call must not creep back in.
    assert not _calls_named(method, 'run'), (
        'run_sequenced_capture must not call the retired sequenced_capture_runner.run() API'
    )

    # Started-run follow-ups still come after start(), so a refused run
    # can never point the save folder at the previous run.
    assert 'set_last_save_folder' in src
    assert start_pos < src.index('set_last_save_folder'), (
        'set_last_save_folder must run only after start() commits the run'
    )


def test_every_ui_starter_routes_through_refusal_boundary():
    """Each UI starter wraps its prepare/start sequence in
    run_with_refusal_boundary with an on_refused reset, the single UI
    catch site for the typed refusal -- no per-starter try/except
    drift."""
    for rel_path, class_name, method_name in UI_STARTERS:
        method = _method_node(REPO / rel_path, class_name, method_name)
        boundary_calls = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'run_with_refusal_boundary'
        ]
        assert boundary_calls, (
            f'{class_name}.{method_name} must route its start sequence '
            'through run_with_refusal_boundary'
        )
        for call in boundary_calls:
            has_on_refused = len(call.args) >= 2 or any(
                kw.arg == 'on_refused' for kw in call.keywords
            )
            assert has_on_refused, (
                f'{class_name}.{method_name} must pass an on_refused reset '
                'to run_with_refusal_boundary so a refusal undoes the '
                'pre-gate button state'
            )


def _commit_marker_sites(method: ast.FunctionDef) -> list[tuple[str, bool]]:
    """(marker, deferred) for every commit-marker call in the method.

    'deferred' means the call sits inside a nested commit_ui_state
    closure or inside a lambda -- either way it does not execute at
    starter entry, so it cannot commit running-state before the
    refusal gates run."""
    sites: list[tuple[str, bool]] = []

    def visit(node: ast.AST, inside: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_inside = (
                inside
                or (isinstance(child, ast.FunctionDef) and child.name == 'commit_ui_state')
                or isinstance(child, ast.Lambda)
            )
            if isinstance(child, ast.Call):
                snippet = ast.unparse(child)
                for marker in COMMIT_MARKERS:
                    if marker in snippet:
                        sites.append((marker, child_inside))
            visit(child, child_inside)

    visit(method, False)
    return sites


def test_ui_commit_blocks_live_inside_commit_closures():
    """The running-state commits (protocol_running.set,
    _publish_protocol_running(True), set_motion_capability(False)) in
    the scan/protocol/autofocus starters must not execute at the
    starter's top level, where they would run before the refusal
    gates: they live either in a nested commit_ui_state closure (the
    autofocus starter keeps its own set) or in the shared
    _commit_running_ui_state method handed to run_sequenced_capture
    as the commit_ui_state lambda."""
    for rel_path, class_name, method_name in UI_STARTERS:
        method = _method_node(REPO / rel_path, class_name, method_name)
        sites = _commit_marker_sites(method)
        for marker, inside in sites:
            assert inside, (
                f'{class_name}.{method_name}: "{marker}" sits outside the '
                'commit_ui_state closure, where it would commit '
                'running-state before the refusal gates'
            )
        # A starter that references the shared commit method may only do
        # so inside a deferred lambda, never as an inline top-level call.
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == '_commit_running_ui_state'
            ):
                parents = [
                    n
                    for n in ast.walk(method)
                    if isinstance(n, (ast.Lambda, ast.FunctionDef))
                    and any(c is node for c in ast.walk(n))
                    and n is not method
                ]
                assert parents, (
                    f'{class_name}.{method_name} calls _commit_running_ui_state '
                    'inline at starter top level -- it must be deferred via '
                    'the commit_ui_state lambda'
                )
    # The check is not vacuous: the shared commit method holds the full
    # running-state commit set, and the scan + protocol starters hand it
    # to run_sequenced_capture.
    commit_method = _method_node(
        REPO / 'ui' / 'protocol_settings.py', 'ProtocolSettings', '_commit_running_ui_state'
    )
    commit_src = ast.unparse(commit_method)
    for marker in COMMIT_MARKERS:
        assert marker in commit_src, (
            f'_commit_running_ui_state is missing the "{marker}" commit; '
            'the commit-closure gate is scanning the wrong method'
        )
    for method_name in ('_run_scan_from_ui_inner', '_run_protocol_from_ui_inner'):
        starter = _method_node(
            REPO / 'ui' / 'protocol_settings.py', 'ProtocolSettings', method_name
        )
        starter_src = ast.unparse(starter)
        assert '_commit_running_ui_state' in starter_src, (
            f'{method_name} must hand _commit_running_ui_state to '
            'run_sequenced_capture as its commit_ui_state'
        )


# ---------------------------------------------------------------------------
# Retired-API sweep: no production call site of the old bool-returning
# run() remains anywhere under modules/ or ui/.
# ---------------------------------------------------------------------------

_RUN_CALL = re.compile(r'\.run\(')


def _balanced_block(src: str, open_paren_idx: int) -> str:
    depth = 0
    for j in range(open_paren_idx, len(src)):
        if src[j] == '(':
            depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0:
                return src[open_paren_idx : j + 1]
    raise AssertionError('unbalanced parens while extracting call block')


def test_no_retired_runner_run_call_sites_remain():
    """The retired API is identified by its own required kwargs: any
    .run( whose argument block carries run_mode= or protocol= is a
    sequenced-capture run() call (subprocess.run and thread run()
    calls carry neither)."""
    offenders = []
    for sub in ('modules', 'ui'):
        for path in sorted((REPO / sub).rglob('*.py')):
            src = path.read_text()
            for m in _RUN_CALL.finditer(src):
                block = _balanced_block(src, m.end() - 1)
                if 'run_mode=' in block or 'protocol=' in block:
                    offenders.append(f'{path.relative_to(REPO)}: {block[:80]}')
    assert not offenders, (
        'Call sites of the retired SequencedCaptureRunner.run() remain; '
        'migrate them to prepare()/start():\n' + '\n'.join(offenders)
    )
