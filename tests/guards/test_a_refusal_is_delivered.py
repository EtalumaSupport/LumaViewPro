# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal is DELIVERED to the person who provoked it, not merely posted.

The notification centre suppresses on two rules, and each rests on a
premise that a refusal falsifies by construction:

- the unattended-run mute assumes nobody is watching, but a refusal
  answers a request that just arrived;
- dedup assumes "already shown recently", but a second press is a
  second question, not a repeat of one answer.

So a refusal carries ``solicited``, and both rules stand aside for it.
``fatal`` keeps its own meaning -- a fault that ends the operation must
reach a watching user, but one fault repeating is still one fault -- and
the tests below pin that the two axes stay separate.

Provenance, not severity, is what makes a notification solicited, and
only the boundary a request arrived through knows it: the GUI's marks a
person's press solicited, REST's answers the caller and shows no one. So
the refusal funnels raise and report nothing, and the boundary reports.
The AST test is what makes that unmissable: it discovers every function
that builds the typed refusal rather than reading a list.
"""

import ast

import pytest

from modules.notification_center import NotificationCenter, Severity
from tests.ast_seams import REPO_ROOT


@pytest.fixture
def centre_and_seen():
    """A centre with a listener low enough to see a warning-level refusal."""
    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener((lambda n: n.shown and seen.append(n)), min_severity=Severity.NOTICE)
    return centre, seen


def test_a_refusal_reaches_a_user_the_run_thinks_is_absent(centre_and_seen):
    """The unattended mute must not swallow an answer to a button press."""
    centre, seen = centre_and_seen
    centre.open_run_scope(attended=False)

    centre.warning(
        'Protocol', 'Already Running', 'A protocol run is already in progress.', solicited=True
    )

    assert len(seen) == 1, (
        'the refusal was suppressed as unattended; the user pressed a button and was told nothing'
    )


def test_asking_twice_gets_answered_twice(centre_and_seen):
    """Dedup collapses repeats of one fault; two presses are two questions.

    The window is ten seconds, which is longer than a user waits before
    pressing again when the first press appeared to do nothing.
    """
    centre, seen = centre_and_seen

    for _ in range(2):
        centre.warning(
            'Protocol', 'Already Running', 'A protocol run is already in progress.', solicited=True
        )

    assert len(seen) == 2, 'the second press was deduped, so the second answer never arrived'


def test_an_unsolicited_fault_is_still_muted_during_an_unattended_run(centre_and_seen):
    """The suppression this does NOT remove: nobody asked for this one."""
    centre, seen = centre_and_seen
    centre.open_run_scope(attended=False)

    centre.warning('Camera', 'Camera Setting Not Applied', 'the camera rejected the gain')

    assert seen == [], 'a machine-originated fault escaped the unattended mute'


def test_one_fault_repeating_is_still_one_popup(centre_and_seen):
    """``fatal`` must not acquire the dedup exemption along the way.

    ``critical()`` defaults ``fatal=True``, and the run-abort path behind
    it is documented as safe to re-enter -- which is true of the popup
    only because the dedup window holds. Five aborts must still be one
    dialog.
    """
    centre, seen = centre_and_seen

    for _ in range(5):
        centre.critical('Protocol', 'Run Aborted', 'the disk filled')

    assert len(seen) == 1, 'a repeating fatal fault now stacks a dialog per occurrence'


def test_shutdown_still_suppresses_a_refusal(centre_and_seen):
    """The one premise that stays true: during close there is no listener
    left to reach, so a refusal has nowhere to go."""
    centre, seen = centre_and_seen
    centre.set_shutting_down(True)

    centre.warning(
        'Protocol', 'Already Running', 'A protocol run is already in progress.', solicited=True
    )

    assert seen == [], 'a notification was dispatched after teardown began'


def test_the_gui_boundary_delivers_a_funnel_refusal_during_an_unattended_run(
    centre_posts, unattended_run
):
    """End to end: the production funnel raises, the GUI's boundary reports.

    A person's press during an unattended run is answered: the boundary
    marks it solicited, so the run's mute stands aside.
    """
    from modules.sequenced_capture_runner import SequencedCaptureRunner
    from ui.ui_helpers import run_reported

    runner = object.__new__(SequencedCaptureRunner)

    def press():
        runner._refuse(
            reason='already_running',
            title='Already Running',
            message='A protocol run is already in progress.',
        )

    run_reported(press, None, 'RUN_START')

    assert [(n.reason, n.shown) for n in centre_posts] == [('already_running', True)], (
        'the boundary posted a refusal that never reached the user'
    )
    assert centre_posts[0].solicited is True
    assert centre_posts[0].operation_key, 'without a key the bridge stacks a dialog per press'


def test_a_funnel_refusal_outside_any_boundary_is_raised_and_posts_nothing(centre_posts):
    """A Python caller, and REST beneath its own boundary, is answered by the raise.

    Posting at the funnel decided shown and solicited before any boundary
    saw the refusal, so a REST caller was answered AND every listener was
    shown it.
    """
    from modules.exceptions import ProtocolRunRefusedError
    from modules.sequenced_capture_runner import SequencedCaptureRunner

    runner = object.__new__(SequencedCaptureRunner)
    with pytest.raises(ProtocolRunRefusedError) as raised:
        runner._refuse(
            reason='already_running',
            title='Already Running',
            message='A protocol run is already in progress.',
        )

    assert raised.value.reason == 'already_running'
    assert list(centre_posts) == []


def _functions_raising_the_typed_refusal():
    """Every function in modules/ that builds a ProtocolRunRefusedError.

    Discovered rather than listed: a new refusal funnel added to a module
    nobody remembered to enumerate is exactly the drift this guards. Found
    by the construction, not by a ``raise`` of it, so a funnel that builds
    the refusal and hands it to the reporter before raising it is found.
    """
    found = []
    for path in sorted((REPO_ROOT / 'modules').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            builds = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == 'ProtocolRunRefusedError'
            ]
            if builds:
                found.append((path.relative_to(REPO_ROOT), node))
    return found


def test_no_refusal_funnel_reports_what_it_raises():
    """The report belongs to the boundary, not to the funnel."""
    funnels = _functions_raising_the_typed_refusal()
    assert funnels, 'the scan found no refusal funnel at all -- the AST shape drifted'

    reporting = [
        f'{rel_path}::{node.name} (line {node.lineno})'
        for rel_path, node in funnels
        if any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'report_outcome'
            for n in ast.walk(node)
        )
    ]

    assert not reporting, (
        'these build the typed refusal and report it, deciding shown and solicited '
        'before the boundary the caller sits behind sees it: ' + ', '.join(reporting)
    )
