# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: a run-starting click that gets refused must tell the user WHY.

Contract
--------
A refused starter resets its button cosmetics, which is visible but mute:
the button pops back up, saying the click did not take but never what is
holding the scope. Two gates in these same ladders (the file-drain gate
and the protocol-validity gate) already raise a popup, so before this
guard existed the explanation a user got depended on which gate happened
to fire first -- an implementation accident deciding what the user is
told.

So: any branch in a run starter that resets button cosmetics and returns
must also raise a user-facing notification. A branch that resets nothing
is not a refusal -- it is a rapid-double-click re-entry guard, where
silence is correct and a popup would be noise -- and that distinction is
what this guard keys on, so the exclusions fall out of the rule instead
of being listed by hand.

The refusal popup deliberately bypasses the notification centre. The
centre drops every non-fatal notification for the whole of a run nobody
is watching, which is every run kind but a standalone autofocus; a
refusal answers a button press, so a human is present by construction
and must be told regardless.

Test approach
-------------
The Kivy UI classes cannot be instantiated headlessly (ids, _app_ctx,
worker pool), so the call-site half of the contract is locked by AST over
the starter roster, the same way the sibling guards in
test_protocol_start_refusal_ui_gate.py and test_controls_lockout.py do
it. The message half IS directly testable, because the two helpers are
plain functions, so those get real behavioural tests.
"""

from __future__ import annotations

import ast

from tests.ast_seams import parse_module


# Every starter that can refuse a run-or-capture-starting click. Wider
# than test_protocol_start_refusal_ui_gate.py's UI_STARTERS, which lists
# only the four that drive a sequenced run: a refusal can also come from
# the standalone autofocus button and from composite capture.
REFUSAL_STARTERS = (
    ('ui/protocol_settings.py', 'ProtocolSettings', '_run_scan_from_ui_inner'),
    ('ui/protocol_settings.py', 'ProtocolSettings', '_run_protocol_from_ui_inner'),
    ('ui/protocol_settings.py', 'ProtocolSettings', 'run_autofocus_scan_from_ui'),
    ('ui/zstack.py', 'ZStack', 'run_zstack_acquire_from_ui'),
    ('ui/vertical_control.py', 'VerticalControl', 'run_autofocus_from_ui'),
    ('ui/composite_capture.py', 'CompositeCapture', 'composite_capture'),
)

# Calls that reset a starter's pre-gate button cosmetics. The inline form
# (`btn.state = 'normal'`) is handled separately -- keying on the helper
# NAMES alone is what let a silent site hide during this fix's own census.
COSMETICS_RESETS = frozenset(
    {
        'run_refused_func',
        'run_not_started_func',
        '_reset_run_button_cosmetics',
        '_reset_run_autofocus_button_cosmetics',
    }
)

# The canonical ways a starter tells the user. Closed by Rule 35: one
# capability, four spellings, not a growing list of exempt sites.
NOTIFIERS = frozenset(
    {
        'show_notification_popup',
        'show_run_refused_popup',
        'show_autofocus_busy_popup',
        'require_file_writes_idle',
        '_is_protocol_valid',
        '_offer_wedged_writer_recovery',
    }
)


def _method_node(rel_path: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = parse_module(rel_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    return sub
    raise AssertionError(f'{rel_path}: {class_name}.{method_name} is gone -- roster drifted')


def _called_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                out.add(func.id)
            elif isinstance(func, ast.Attribute):
                out.add(func.attr)
                if isinstance(func.value, ast.Name):
                    out.add(f'{func.value.id}.{func.attr}')
    return out


def _resets_cosmetics(statements: list[ast.stmt]) -> bool:
    for stmt in statements:
        if _called_names(stmt) & COSMETICS_RESETS:
            return True
        for sub in ast.walk(stmt):
            # The inline form: `some_btn.state = 'normal'`.
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == 'state'
                        and isinstance(sub.value, ast.Constant)
                        and sub.value.value == 'normal'
                    ):
                        return True
    return False


def _refusal_branches(method: ast.FunctionDef):
    """Yield (lineno, branch) for each branch that refuses and returns."""
    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue
        for branch in (node.body, node.orelse):
            if not branch:
                continue
            returns = any(isinstance(stmt, ast.Return) for stmt in branch)
            if returns and _resets_cosmetics(branch):
                yield node, branch


def test_every_refusing_branch_in_a_starter_explains_itself():
    """The guard. A ninth silent gate fails the build.

    Eight hand-written per-site tests would pin eight sites and answer
    nothing about a ninth; this is the Rule 50 shape -- the illegal state
    is caught by construction rather than by remembering to add a test.
    """
    silent = []
    checked = 0
    for rel_path, class_name, method_name in REFUSAL_STARTERS:
        method = _method_node(rel_path, class_name, method_name)
        for node, branch in _refusal_branches(method):
            checked += 1
            body_names: set[str] = set()
            for stmt in branch:
                body_names |= _called_names(stmt)
            notifies = bool(body_names & NOTIFIERS) or any(
                name.startswith('notifications.') for name in body_names
            )
            # The drain and validity gates notify inside the helper called
            # in the branch's TEST, not in its body.
            notifies = notifies or bool(_called_names(node.test) & NOTIFIERS)
            if not notifies:
                silent.append(f'{rel_path}:{node.lineno} in {class_name}.{method_name}')

    assert checked, 'derivation found no refusing branches -- the AST shapes drifted'
    assert not silent, (
        'a refused click resets the button and says nothing, so the user is '
        'left to guess and clicks again: ' + ', '.join(silent)
    )


def test_the_guard_would_catch_a_silent_branch():
    """The guard's own falsifier.

    A guard that cannot fail is not a guard. This feeds the classifier a
    branch that resets cosmetics and returns without notifying, and
    requires it to be reported.
    """
    module = ast.parse(
        'class X:\n'
        '    def starter(self):\n'
        '        if blocked:\n'
        '            run_refused_func()\n'
        '            logger.warning("nope")\n'
        '            return\n'
    )
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef))
    found = list(_refusal_branches(method))
    assert len(found) == 1, 'the classifier no longer recognises a refusing branch'
    _node, branch = found[0]
    names: set[str] = set()
    for stmt in branch:
        names |= _called_names(stmt)
    assert not (names & NOTIFIERS), 'the fixture branch is supposed to be silent'


def test_a_double_click_guard_is_not_treated_as_a_refusal():
    """Silence is correct where nothing was refused.

    A re-entry guard swallows the second half of a rapid double-click. It
    resets no cosmetics, and a popup there would be noise -- so the rule
    must not drag it in. This is the distinction that keeps the exclusion
    derivable instead of a hand-maintained list.
    """
    module = ast.parse(
        'class X:\n'
        '    def starter(self):\n'
        '        if already_starting:\n'
        '            logger.warning("ignored -- already starting")\n'
        '            return\n'
    )
    method = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef))
    assert not list(_refusal_branches(method)), (
        'a re-entry guard that resets nothing was classified as a refusal'
    )


class TestTheRefusalNamesWhatIsHoldingTheScope:
    """The message half, tested for real -- these helpers are plain functions."""

    def _capture(self, monkeypatch):
        import ui.notification_popup as popup_mod

        shown = []
        monkeypatch.setattr(
            popup_mod,
            'show_notification_popup',
            lambda title, message: shown.append((title, message)),
        )
        return shown

    def test_a_rival_run_is_named_by_kind(self, monkeypatch):
        from ui.ui_helpers import show_run_refused_popup

        shown = self._capture(monkeypatch)
        show_run_refused_popup('start a scan', 'protocol')

        assert len(shown) == 1, 'exactly one popup per refused click'
        title, message = shown[0]
        assert 'protocol' in message, (
            'the refusal must name WHICH run holds the scope; a message that '
            'only says "busy" is what sent the user back to click again'
        )
        assert 'start a scan' in message, 'the refusal must name what it refused'
        assert title

    def test_an_unknown_holder_does_not_render_a_blank(self, monkeypatch):
        """Reachable: the windows around the activity claim are not zero."""
        from ui.ui_helpers import show_run_refused_popup

        shown = self._capture(monkeypatch)
        show_run_refused_popup('start a scan', '')

        assert len(shown) == 1
        _title, message = shown[0]
        assert 'Another run' in message
        assert '  ' not in message, 'an empty holder was interpolated into the sentence'

    def test_an_autofocus_sweep_names_the_run_that_owns_it(self, monkeypatch):
        """A protocol's own AF steps drive the same thread.

        Naming only the autofocus would report a sweep when a full
        protocol is the thing the user actually has to stop.
        """
        from ui.ui_helpers import show_autofocus_busy_popup

        shown = self._capture(monkeypatch)
        show_autofocus_busy_popup('start a scan', 'protocol')

        assert len(shown) == 1
        _title, message = shown[0]
        assert 'autofocus' in message.lower()
        assert 'protocol' in message, 'the sweep belongs to a run; the message must name that run'


def test_the_composite_guard_does_not_blame_the_wrong_subsystem():
    """A composite's own second click is taken by the stop branch above it.

    So the only way to reach the already-capturing branch is a LIVE
    capture still holding the guard, and calling that "composite capture
    already in progress" told the user about the wrong subsystem.

    Read through the AST seam rather than as source text: a text pin on a
    production file is what the fragile-pin ratchet forbids, and it would
    also match the phrase in a comment or a docstring, which is not what
    the user is shown.
    """
    tree = parse_module('ui/composite_capture.py')
    blaming = sorted(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and 'Composite capture already in progress' in node.value
    )
    assert not blaming, (
        'the guard names the composite again, but a composite cannot be what '
        f'holds the flag on that path: {blaming}'
    )
