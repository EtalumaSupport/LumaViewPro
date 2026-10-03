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

Run-state truth is the session claim, committed inside start() and
mirrored to kv by the session's run-state listener, so no starter may
write running-state (Event, mirror, motion lock) at all, and a run
button changes nothing ahead of the engine's answer: its redraw draws
what the engine then says. A starter hands its press to the boundary,
ui_helpers.submit_reported -- the one place a refusal or a fault from
the API is reported -- and a Stop goes at the Stop's priority, ahead of
queued work.

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


# The UI code that kicks off a sequenced run: the protocol panel's one
# press, the inputs it reads for each of its three runs, the z-stack and the
# autofocus.
UI_STARTERS = (
    ('ui/protocol_settings.py', 'ProtocolSettings', '_press_panel_run'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_scan_start'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_protocol_start'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_autofocus_scan_start'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_sequenced_capture_start'),
    ('ui/zstack.py', 'ZStack', 'run_zstack_acquire_from_ui'),
    ('ui/vertical_control.py', 'VerticalControl', 'run_autofocus_from_ui'),
)

# The presses that hand a start and a Stop to the boundary.
BOUNDARY_PRESSES = (
    ('ui/protocol_settings.py', 'ProtocolSettings', '_press_panel_run'),
    ('ui/composite_capture.py', 'CompositeCapture', 'composite_capture'),
    ('ui/zstack.py', 'ZStack', 'run_zstack_acquire_from_ui'),
    ('ui/vertical_control.py', 'VerticalControl', 'run_autofocus_from_ui'),
)

# A control's one submit, where it marks its request in flight before
# handing the call to submit_reported. A press may reach the boundary
# through its control's own.
SUBMIT_HELPERS = (
    ('ui/protocol_settings.py', 'ProtocolSettings', '_submit_panel_request'),
    ('ui/vertical_control.py', 'VerticalControl', '_submit_autofocus_request'),
)
_SUBMITTERS = {'submit_reported'} | {name for _, _, name in SUBMIT_HELPERS}

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


def test_the_panels_start_runs_through_the_api_then_names_the_folder():
    """The call the panel hands the pool is the runner member a script
    calls, then the save folder -- so a refusal (raised by the member)
    never points the save folder at the previous run, and the panel holds
    no second prepare-and-start of its own."""
    path = REPO_ROOT / 'ui' / 'protocol_settings.py'
    method = _method_node(path, 'ProtocolSettings', '_sequenced_capture_start')
    assert _calls_named(method, 'start_run'), (
        '_sequenced_capture_start must start the run through the runner member it is handed'
    )
    for name in ('prepare', 'start', 'run'):
        assert not _calls_named(method, name), (
            f"_sequenced_capture_start must not call the engine's {name}() itself: "
            'ProtocolRunner.run_protocol / run_single_scan are the one way to start a run'
        )

    src = ast.unparse(method)
    assert src.index('start_run(') < src.index('set_last_save_folder'), (
        'start, then set_last_save_folder: the folder must name only a run that started'
    )

    for starter, member in (
        ('_scan_start', 'run_single_scan'),
        ('_protocol_start', 'run_protocol'),
    ):
        handed = _calls_named(
            _method_node(path, 'ProtocolSettings', starter), '_sequenced_capture_start'
        )
        assert [
            ast.unparse(kw.value) for c in handed for kw in c.keywords if kw.arg == 'start_run'
        ] == [f'ctx.session.create_protocol_runner().{member}'], (
            f'{starter} must hand the panel start ProtocolRunner.{member}'
        )


def test_every_press_hands_its_start_and_its_stop_to_the_boundary():
    """A run button's start and Stop both reach submit_reported, the one
    place a refusal or a fault from the API is reported -- no per-button
    try/except drift."""
    for rel_path, class_name, method_name in BOUNDARY_PRESSES:
        method = _method_node(REPO_ROOT / rel_path, class_name, method_name)
        called = {
            n.func.id if isinstance(n.func, ast.Name) else n.func.attr
            for n in ast.walk(method)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
        }
        assert called & _SUBMITTERS, (
            f'{class_name}.{method_name} must hand its press to submit_reported'
        )
        assert not [n for n in ast.walk(method) if isinstance(n, ast.Try)], (
            f'{class_name}.{method_name} catches for itself; the boundary reports'
        )

    for rel_path, class_name, method_name in SUBMIT_HELPERS:
        helper = _method_node(REPO_ROOT / rel_path, class_name, method_name)
        assert _calls_named(helper, 'submit_reported'), (
            f'{class_name}.{method_name} must hand its request to submit_reported'
        )


def test_every_stop_goes_ahead_of_queued_work():
    """A Stop is submitted at the Stop's priority.

    The pool runs one worker, so a Stop queued behind ordinary work would
    not arrive until that work finished -- which is the thing the person
    is trying to interrupt. Derived: every submitted call that tears a run
    down must say stop=True.
    """
    stops = []
    for source_file in sorted((REPO_ROOT / 'ui').glob('*.py')):
        tree = ast.parse(source_file.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Name, ast.Attribute))
                and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
                in _SUBMITTERS
            ):
                continue
            if not any('.reset(' in ast.unparse(arg) for arg in node.args):
                continue
            stop = next((kw.value for kw in node.keywords if kw.arg == 'stop'), None)
            stops.append((source_file.name, ast.unparse(node), stop))

    assert len(stops) >= 3, 'derivation found too few Stops -- the AST shapes drifted'
    slow = [
        (where, src)
        for where, src, stop in stops
        if not (isinstance(stop, ast.Constant) and stop.value is True)
    ]
    assert not slow, f'a Stop submitted without stop=True waits behind queued work: {slow}'


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

    A teardown is a `<something>runner.reset(...)` call, direct or bound by
    a functools.partial. Derived rather than listed: the starter tuple above
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
            if action is None and node.args:
                action = node.args[0]
            if action is None or 'reset' not in ast.unparse(action):
                continue
            yield source_file.name, ast.unparse(action)


def test_a_refused_teardown_is_reported_once():
    """A refused teardown notifies once, not twice.

    reset() refuses a stop naming a run that is not the live one while
    another run is live. A Stop handed to the pool as a bare IOTask met the
    executor's generic failure popup as well as its own report -- two
    notifications for one event, the second titled from the action, which
    for a functools.partial is its repr, heap address included. Every Stop
    now goes through submit_reported, whose one reporter shows a refusal
    once; a teardown task built by hand is a second reporting path.
    """
    bare = list(_teardown_iotasks())
    assert not bare, (
        'a run teardown submitted as a bare IOTask bypasses the one reporter '
        f'submit_reported hands a refusal to: {bare}'
    )


def test_the_autofocus_redraw_never_aborts_autofocus():
    """The Autofocus button's redraw draws the button. Nothing else.

    draw_autofocus_button runs after each of the button's own requests,
    whatever the outcome -- including a Stop the engine REFUSED -- and on
    every run-state edge. An abort in there would kill the autofocus of a
    run the caller had just been told it did not own, as the old reset
    callback once did. Scoped to the redraw on purpose: the completion
    handler's own defensive abort is a different path (it runs when a run
    ENDS, never on a refusal) and has its own ordering test.
    """
    tree = parse_module('ui/vertical_control.py')
    redraws = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in ('draw_autofocus_button', '_autofocus_request_done')
    ]
    assert len(redraws) == 2, 'ui/vertical_control.py: the autofocus redraw path drifted'

    aborts = [
        ast.unparse(node)
        for redraw in redraws
        for node in ast.walk(redraw)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'abort'
    ]
    assert not aborts, (
        'the autofocus redraw must not abort hardware -- it runs after a '
        f'refused Stop too: {aborts}'
    )
