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

Because a refusal raises before start(), UI callers set their run
button cosmetics BETWEEN prepare and start -- never before. Run-state
truth itself is the session claim, committed inside start() and
mirrored to kv by the session's run-state listener, so no starter may
write running-state (Event, mirror, motion lock) at all. Every UI
starter routes its prepare/start sequence through
ui_helpers.run_with_refusal_boundary, the single catch site for the
typed refusal, whose on_refused callback undoes only the pre-gate
button cosmetics.

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

from tests.ast_seams import REPO_ROOT, parse_module


# The four UI starters that kick off a sequenced run.
UI_STARTERS = (
    ('ui/protocol_settings.py', 'ProtocolSettings', '_run_scan_from_ui_inner'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_run_protocol_from_ui_inner'),
    ('ui/protocol_settings.py', 'ProtocolSettings', 'run_autofocus_scan_from_ui'),
    ('ui/zstack.py', 'ZStack', 'run_zstack_acquire_from_ui'),
)

# Statements that would commit "a run is now underway" state in the
# UI -- all retired: the claim inside start() is the one commit, and
# the kv mirrors follow the session listener. Any reappearance is a
# second run-state store.
FORBIDDEN_COMMIT_MARKERS = (
    'protocol_running.set()',
    '_publish_protocol_running(True)',
    'set_motion_capability(False)',
    'publish_protocol_running(',
    'run_committed_start(',
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
        REPO_ROOT / 'ui' / 'protocol_settings.py', 'ProtocolSettings', 'run_sequenced_capture'
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
    # Cosmetics-only commit between prepare and start: a refusal from
    # prepare never shows a mid-run button, and start() itself owns the
    # run-state commit (the claim).
    commit_pos = src.index('commit_ui_state()')
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
        method = _method_node(REPO_ROOT / rel_path, class_name, method_name)
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


def test_no_starter_writes_running_state():
    """No UI starter may write running-state: the claim inside start()
    is the one commit, and the kv mirrors follow the session's
    run-state listener. A starter-side write re-creates the second
    store whose strand/mis-restore family this migration retired."""
    for rel_path, class_name, method_name in UI_STARTERS:
        method = _method_node(REPO_ROOT / rel_path, class_name, method_name)
        src_text = ast.unparse(method)
        for marker in FORBIDDEN_COMMIT_MARKERS:
            assert marker not in src_text, (
                f'{class_name}.{method_name} contains "{marker}" -- '
                'run-state truth lives on the session claim; starters '
                'own button cosmetics only'
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
        for path in sorted((REPO_ROOT / sub).rglob('*.py')):
            src = path.read_text()
            for m in _RUN_CALL.finditer(src):
                block = _balanced_block(src, m.end() - 1)
                if 'run_mode=' in block or 'protocol=' in block:
                    offenders.append(f'{path.relative_to(REPO_ROOT)}: {block[:80]}')
    assert not offenders, (
        'Call sites of the retired SequencedCaptureRunner.run() remain; '
        'migrate them to prepare()/start():\n' + '\n'.join(offenders)
    )


# ---------------------------------------------------------------------------
# Teardown authority
# ---------------------------------------------------------------------------


def _runner_reset_calls():
    """Every UI run teardown under ui/, derived, with the run it names.

    A teardown is a `<something>runner.reset(...)` call, or a call to the
    ui_helpers boundary that wraps one, direct or bound by a
    functools.partial. Derived rather than listed: the starter tuple above
    is hand-maintained and had already drifted -- it names four starters
    while the autofocus and composite buttons tear runs down too. A list
    that has to be updated by hand is the thing this test exists to
    prevent, so it must not depend on one.

    Yields (file, source, run argument or None when absent, keywords).
    """

    def _callee_and_args(node):
        """The callee this call reaches and the arguments it is handed.

        functools.partial(f, a, b) carries the arguments on the binding
        rather than the call, so the partial's own arguments are the ones
        that answer which run the teardown names.
        """
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'partial' and node.args:
            return node.args[0], node.args[1:], node.keywords
        return func, node.args, node.keywords

    def _run_argument(position, args, keywords):
        if len(args) > position:
            return args[position]
        return next((kw.value for kw in keywords if kw.arg == 'run'), None)

    for source_file in sorted((REPO_ROOT / 'ui').glob('*.py')):
        tree = ast.parse(source_file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee, args, keywords = _callee_and_args(node)
            if (
                isinstance(callee, ast.Attribute)
                and callee.attr == 'reset'
                and 'runner' in ast.unparse(callee.value).lower()
            ):
                run = _run_argument(0, args, keywords)
            elif isinstance(callee, ast.Name) and callee.id == 'reset_with_refusal_boundary':
                run = _run_argument(1, args, keywords)
            else:
                continue
            yield source_file.name, ast.unparse(node), run, keywords


def test_every_ui_run_teardown_names_its_run():
    """A UI teardown names the run it means to stop, by its handle.

    The defect this locks: a stale autofocus toggle reached reset() during
    someone else's scan and destroyed it, because nothing in the call said
    which run it meant. The engine now stops only the run a stop names and
    refuses a handle naming any other -- but only if the caller passes the
    handle its own start returned, so no ui/ call site may omit it, pass a
    literal in its place, or still say who is asking instead.
    """
    calls = list(_runner_reset_calls())
    assert calls, 'derivation found no teardown calls -- the AST shapes drifted'
    assert any(src.startswith('runner.reset') for _, src, _, _ in calls), (
        'derivation found no engine reset() call -- the AST shapes drifted'
    )
    assert any('reset_with_refusal_boundary' in src for _, src, _, _ in calls), (
        'derivation found no call through the teardown boundary -- the AST shapes drifted'
    )

    unnamed = [
        (where, src)
        for where, src, run, keywords in calls
        if run is None
        or isinstance(run, ast.Constant)
        or any(kw.arg == 'requester' for kw in keywords)
    ]
    assert not unnamed, (
        'run teardown that does not name a run by its handle -- the engine '
        f'cannot tell the run this control started from the live one: {unnamed}'
    )


def _teardown_iotasks():
    """Every IOTask under ui/ whose action is a bound `<runner>.reset`.

    Derived for the same reason as the list above: a hand-kept roster of
    "the tasks that can be refused" is one more mirror to forget.
    """
    for source_file in sorted((REPO_ROOT / 'ui').glob('*.py')):
        tree = ast.parse(source_file.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'IOTask'
            ):
                continue
            action = next((kw.value for kw in node.keywords if kw.arg == 'action'), None)
            if action is None or 'reset' not in ast.unparse(action):
                continue
            yield source_file.name, ast.unparse(action), node.keywords


def test_a_refusable_teardown_task_does_not_double_notify():
    """A refused teardown notifies once, not twice.

    reset() refuses a stop naming a run that is not the live one while
    another run is live, and that refusal has already logged once and
    notified once before it raises. The
    executor's generic failure popup would be a SECOND notification for
    one event -- and it titles that popup from the action, which for a
    functools.partial is the partial's repr, heap address included. The
    executor exposes silent_on_failure for exactly this: the caller that
    notifies for itself opts out of the generic one.
    """
    tasks = list(_teardown_iotasks())
    assert tasks, 'derivation found no teardown IOTasks -- the AST shapes drifted'

    loud = [
        (where, action)
        for where, action, keywords in tasks
        if not any(
            kw.arg == 'silent_on_failure' and getattr(kw.value, 'value', False) is True
            for kw in keywords
        )
    ]
    assert not loud, (
        'a teardown IOTask that can be refused must set silent_on_failure -- '
        f'otherwise one refusal raises two notifications: {loud}'
    )


def test_the_button_reset_funnel_never_aborts_autofocus():
    """A button-reset function resets the button. Nothing else.

    _reset_run_autofocus_button runs as the completion callback of the
    teardown task, and a completion callback fires whatever the outcome --
    including a teardown the engine REFUSED. An abort in there therefore
    killed the autofocus of a run the caller had just been told it did not
    own. Scoped to this funnel on purpose: the completion handler's own
    defensive abort is a different path (it runs when a run ENDS, never on
    a refusal) and has its own ordering test.
    """
    tree = parse_module('ui/vertical_control.py')
    funnel = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == '_reset_run_autofocus_button'
        ),
        None,
    )
    assert funnel is not None, 'ui/vertical_control.py: _reset_run_autofocus_button is gone'

    aborts = [
        ast.unparse(node)
        for node in ast.walk(funnel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'abort'
    ]
    assert not aborts, (
        'the button-reset funnel must not abort hardware -- it fires on a '
        f'refused teardown too: {aborts}'
    )
