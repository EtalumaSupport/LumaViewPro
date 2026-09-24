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

Provenance, not severity, is what makes a notification solicited, so the
axis is set at the refusal funnels rather than remembered per gate. The
AST test is what makes that unmissable: it discovers every module that
raises the typed refusal rather than reading a list.
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
    centre.add_listener(seen.append, min_severity=Severity.NOTICE)
    return centre, seen


def test_a_refusal_reaches_a_user_the_run_thinks_is_absent(centre_and_seen):
    """The unattended mute must not swallow an answer to a button press."""
    centre, seen = centre_and_seen
    centre.set_unattended_run(True)

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
    centre.set_unattended_run(True)

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


def test_the_runner_funnel_delivers_its_refusal_during_an_unattended_run(monkeypatch):
    """End to end through the production funnel, not a reconstruction of it."""
    from modules.exceptions import ProtocolRunRefusedError
    from modules.sequenced_capture_runner import SequencedCaptureRunner
    import modules.notification_center as nc

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.NOTICE)
    centre.set_unattended_run(True)
    monkeypatch.setattr(nc, 'notifications', centre)

    runner = object.__new__(SequencedCaptureRunner)
    with pytest.raises(ProtocolRunRefusedError):
        runner._refuse(
            reason='already_running',
            title='Already Running',
            message='A protocol run is already in progress.',
        )

    assert len(seen) == 1, 'the runner funnel posted a refusal that never reached the user'
    assert seen[0].solicited is True
    assert seen[0].operation_key, 'without a key the bridge stacks a dialog per press'


def test_the_composite_funnel_delivers_its_refusal_during_an_unattended_run(monkeypatch):
    """The second funnel. It fires from the same Composite press as the first."""
    from modules.config_helpers import _refuse_composite
    from modules.exceptions import ProtocolRunRefusedError
    import modules.notification_center as nc

    centre = NotificationCenter(dedup_window_s=10.0)
    seen = []
    centre.add_listener(seen.append, min_severity=Severity.NOTICE)
    centre.set_unattended_run(True)
    monkeypatch.setattr(nc, 'notifications', centre)

    with pytest.raises(ProtocolRunRefusedError):
        _refuse_composite(
            reason='composite_needs_two_channels',
            title='Not Enough Channels',
            message='A composite combines at least 2 channels.',
        )

    assert len(seen) == 1, 'the composite funnel posted a refusal that never reached the user'
    assert seen[0].solicited is True


def _functions_raising_the_typed_refusal():
    """Every function in modules/ that raises ProtocolRunRefusedError.

    Discovered rather than listed: a new refusal funnel added to a module
    nobody remembered to enumerate is exactly the drift this guards.
    """
    found = []
    for path in sorted((REPO_ROOT / 'modules').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            raises = [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Raise)
                and isinstance(n.exc, ast.Call)
                and isinstance(n.exc.func, ast.Name)
                and n.exc.func.id == 'ProtocolRunRefusedError'
            ]
            if raises:
                found.append((path.relative_to(REPO_ROOT), node))
    return found


def test_every_refusal_funnel_marks_its_notification_solicited():
    """The contract lives at the funnel, not in each gate's memory."""
    funnels = _functions_raising_the_typed_refusal()
    assert funnels, 'the scan found no refusal funnel at all -- the AST shape drifted'

    silent = []
    for rel_path, node in funnels:
        posts_solicited = any(
            isinstance(n, ast.Call)
            and any(
                kw.arg == 'solicited'
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in n.keywords
            )
            for n in ast.walk(node)
        )
        if not posts_solicited:
            silent.append(f'{rel_path}::{node.name} (line {node.lineno})')

    assert not silent, (
        'these raise the typed refusal but post no solicited notification, so the '
        'refusal is dropped whenever a run is in flight: ' + ', '.join(silent)
    )
