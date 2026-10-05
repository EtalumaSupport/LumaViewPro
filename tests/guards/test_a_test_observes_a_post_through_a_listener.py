# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A test sees a post the way a client does: through a listener on the real centre.

Tests learned to spy on the notification centre's posting methods because the
reporter posted through them. A spy pins one internal path, not what a client
receives: when the reporter posts through another, every "nothing was posted"
assertion behind the spy passes with nothing watching, and a spy that returns
None tells the reporter its post was never delivered, a path production never
takes. A module whose centre is swapped for a fake or a MagicMock hides every
post from a listener on the real one. The ``centre_posts`` fixture in
``tests/conftest.py`` is the one way to observe a post.

This walk over every test module refuses three forms:

* replacing a posting method on a centre (``debug`` ... ``critical``, or
  ``notify``), by object, by dotted string or in a loop;
* assigning a posting method on a centre;
* swapping a module's ``notifications`` for anything but the real singleton.

Replacing ``report_outcome`` is not refused: it watches what the code under
test reported, not what was posted. The allowlist names each remaining site
by file and enclosing def, with why it stays.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import iter_package_modules

POST_METHODS = frozenset({'debug', 'info', 'notice', 'warning', 'error', 'critical', 'notify'})

_SETATTR_TAILS = ('setattr', 'patch.object')
# A dotted path that ends at a module's centre, or one level below it.
_CENTRE_PATH = re.compile(r'\.notifications(?:\.(?P<attr>\w+|\{[^}]*\}))?$')

_D2 = 'swaps a fresh centre in; moves to centre_posts on the D2 follow-up row'
_REPORTER_MOCK = (
    'a stand-in centre asserted only on report_outcome; moves to a report_outcome '
    'spy on the real singleton on the D2 follow-up row'
)

# (file, enclosing def) -> why the site stays.
ALLOWED = {
    # A direct post from ui/file_dialogs.py, which the guard's subject is not.
    ('tests/test_issue_750_native_dialog_async.py', 'test_stuck_dialog_reclick_notifies_user'): (
        'spies the direct warning post in ui/file_dialogs.py'
    ),
    # The D2 class.
    (
        'tests/guards/test_a_refusal_is_delivered.py',
        'test_the_runner_funnel_delivers_its_refusal_during_an_unattended_run',
    ): _D2,
    ('tests/shown_outcomes.py', 'capture_shown'): _D2,
    ('tests/test_a_failed_motor_stop_is_raised.py', 'centre'): _D2,
    ('tests/test_a_failed_move_is_shown_once.py', 'centre'): _D2,
    ('tests/test_a_failed_startup_home_is_reported_once.py', 'shown'): _D2,
    ('tests/test_a_missing_motor_is_not_homed_at_startup.py', 'heard'): _D2,
    ('tests/test_a_plugin_failure_is_reported_once.py', 'shown'): _D2,
    ('tests/test_a_refusal_in_a_task_is_a_refusal.py', '_run_task'): _D2,
    ('tests/test_a_run_refusal_is_reported_once.py', 'centre'): _D2,
    (
        'tests/test_a_simulated_file_stall_holds_the_file_lane.py',
        'TestTheLaneIsHeld.test_a_drain_behind_the_hold_is_reported_as_a_stalled_writer',
    ): _D2,
    ('tests/test_a_stalled_writer_is_reported_by_the_api.py', 'heard'): _D2,
    ('tests/test_an_unknown_objective_is_a_refusal.py', '_run_on_the_lane'): _D2,
    ('tests/test_bring_up_is_a_record.py', 'heard'): _D2,
    ('tests/test_one_lane_reporter.py', 'shown'): _D2,
    (
        'tests/test_one_lane_reporter.py',
        'test_a_listener_that_submits_from_inside_a_report_does_not_deadlock',
    ): _D2,
    ('tests/test_outcome_subscription.py', 'centre'): _D2,
    (
        'tests/test_outcome_subscription.py',
        'TestABrokenListenerIsLoud.test_a_raising_scope_listener_is_reported_once',
    ): _D2,
    (
        'tests/test_protocol_execution.py',
        'TestMotionTimeoutEndsRunInsteadOfWedging.test_a_failed_stop_is_folded_into_the_one_fatal_popup',
    ): _D2,
    (
        'tests/test_refusal_reaches_the_user.py',
        'test_both_safety_refusals_reach_the_user_in_their_own_words',
    ): _D2,
    (
        'tests/test_refusal_reaches_the_user.py',
        'test_each_failure_has_its_own_identity_and_none_of_it_is_a_symbol',
    ): _D2,
    (
        'tests/test_settings_question_failure_parity.py',
        'TestTheQuestionIsAskable.test_the_answer_path_survives_a_locked_file',
    ): _D2,
    ('tests/test_the_raw_posts_are_typed_outcomes.py', 'centre'): _D2,
    (
        'tests/test_the_raw_posts_are_typed_outcomes.py',
        'TestTheAutoGainLimitsAreShown.session',
    ): _D2,
    (
        'tests/test_waited_move_truth.py',
        'test_the_executor_shows_the_failure_in_its_own_words',
    ): _D2,
    # Reporter-input mocks, on the D2 row.
    (
        'tests/test_auto_gain_lock.py',
        'test_live_view_lock_tells_the_user_and_a_protocol_lock_does_not',
    ): _REPORTER_MOCK,
    (
        'tests/test_auto_gain_lock.py',
        'test_failed_lock_under_a_live_view_arm_is_an_error_to_the_user',
    ): _REPORTER_MOCK,
    (
        'tests/test_imaging_frame_listener.py',
        'test_drop_at_K_consecutive_over_budget',
    ): _REPORTER_MOCK,
    (
        'tests/test_a_refused_frame_listener_is_raised.py',
        'TestAFailingHandlerIsBounded.test_a_handler_raising_k_frames_running_is_removed_with_one_traceback',
    ): _REPORTER_MOCK,
    (
        'tests/test_a_refused_frame_listener_is_raised.py',
        'TestAFailingHandlerIsBounded.test_a_handler_that_recovers_before_k_is_kept',
    ): _REPORTER_MOCK,
    (
        'tests/test_a_refused_frame_listener_is_raised.py',
        'TestTheUnwindAndRemovalEdges.test_an_auto_remove_stops_calls_even_when_the_driver_unregister_raises',
    ): _REPORTER_MOCK,
    (
        'tests/test_hyperstack_run_trigger.py',
        'TestRunnerHyperstackTrigger.test_a_held_batch_builds_nothing_and_reports_the_timeout',
    ): _REPORTER_MOCK,
    (
        'tests/test_protocol_modules.py',
        'TestRunCleanupCancelledHandoff.test_real_led_restore_failure_still_surfaces',
    ): _REPORTER_MOCK,
    (
        'tests/test_protocol_overwrite_guard.py',
        'test_load_warns_on_duplicate_filename_keys_and_loads',
    ): _REPORTER_MOCK,
    (
        'tests/test_protocol_overwrite_guard.py',
        'test_load_warns_on_cross_tgid_filename_collision',
    ): _REPORTER_MOCK,
    (
        'tests/test_capture_collision_policy.py',
        'test_load_warns_same_base_in_same_tile_group_and_still_loads',
    ): _REPORTER_MOCK,
    (
        'tests/test_capture_collision_policy.py',
        'test_load_soft_warns_same_base_across_tile_groups',
    ): _REPORTER_MOCK,
    (
        'tests/test_capture_collision_policy.py',
        'test_labels_differing_only_in_stripped_chars_collide',
    ): _REPORTER_MOCK,
}


def _text(node) -> str:
    return ast.unparse(node)


def _str_path(node) -> str | None:
    """A string or f-string argument as its source text, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return _text(node)[2:-1]
    return None


def _is_the_singleton(node) -> bool:
    """The real centre, read off a module that holds it."""
    return isinstance(node, ast.Attribute) and node.attr == 'notifications'


def _is_a_centre(node) -> bool:
    return _text(node).endswith('notifications')


def _swap_value(call: ast.Call, at: int):
    if len(call.args) > at:
        return call.args[at]
    for kw in call.keywords:
        if kw.arg in ('value', 'new'):
            return kw.value
    return None


def _call_violations(call: ast.Call):
    name = _text(call.func)
    if name.endswith(_SETATTR_TAILS) and len(call.args) >= 2:
        target, attr = call.args[0], call.args[1]
        const = attr.value if isinstance(attr, ast.Constant) else None
        if isinstance(const, str):
            if const in POST_METHODS and _is_a_centre(target):
                yield f'replaces {_text(target)}.{const}'
            elif const == 'notifications' and not _is_the_singleton(_swap_value(call, 2)):
                yield f"swaps {_text(target)}'s centre"
        elif _str_path(target) is None and _is_a_centre(target):
            yield f'replaces a method of {_text(target)} chosen at run time'
    if name.endswith('setattr') or name.endswith('patch'):
        path = _str_path(call.args[0]) if call.args else None
        match = _CENTRE_PATH.search(path) if path else None
        if match is not None:
            attr = match.group('attr')
            if attr is None:
                at = 1 if name.endswith('setattr') else None
                value = _swap_value(call, at) if at is not None else _swap_value(call, 99)
                if not _is_the_singleton(value):
                    yield f"swaps {path.rsplit('.', 1)[0]}'s centre"
            elif attr in POST_METHODS or attr.startswith('{'):
                yield f'replaces {path}'


def _assign_violations(targets):
    for target in targets:
        if not isinstance(target, ast.Attribute):
            continue
        if (target.attr in POST_METHODS and _is_a_centre(target.value)) or (
            target.attr == 'notifications'
        ):
            yield f'assigns {_text(target)}'


class _Finder(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.found = []

    def _scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def _record(self, node, what):
        self.found.append(('.'.join(self.scope) or '<module>', node.lineno, what))

    def visit_Call(self, node):
        for what in _call_violations(node):
            self._record(node, what)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for what in _assign_violations(node.targets):
            self._record(node, what)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        for what in _assign_violations([node.target]):
            self._record(node, what)
        self.generic_visit(node)


def violations(tree: ast.AST):
    """Yield ``(enclosing def, line, what)`` for every forbidden form."""
    finder = _Finder()
    finder.visit(tree)
    yield from finder.found


def _found_in_tests():
    for rel_path, tree in iter_package_modules(('tests',)):
        if rel_path == 'tests/guards/test_a_test_observes_a_post_through_a_listener.py':
            continue
        for qualname, lineno, what in violations(tree):
            yield rel_path, qualname, lineno, what


def test_no_test_replaces_a_posting_method_or_swaps_a_centre():
    found = [
        f'{rel}:{lineno} {qualname} {what}'
        for rel, qualname, lineno, what in _found_in_tests()
        if (rel, qualname) not in ALLOWED
    ]
    assert found == [], (
        'observe a post through the centre_posts fixture (tests/conftest.py), '
        f'never by replacing the centre or its posting methods: {found}'
    )


def test_every_allowed_site_still_exists():
    # An entry whose site is gone would exempt the next test written there.
    sites = {(rel, qualname) for rel, qualname, _lineno, _what in _found_in_tests()}
    assert sorted(set(ALLOWED) - sites) == []


def _forms(source: str) -> list[str]:
    return [what for _q, _l, what in violations(ast.parse(source))]


def test_the_guard_sees_every_form_it_refuses():
    refused = [
        'monkeypatch.setattr(nc.notifications, "warning", spy)',
        'monkeypatch.setattr(notifications, "notify", spy)',
        'patch.object(notification_center.notifications, "error", spy)',
        'setattr(notifications, "critical", spy)',
        'for level in ("error", "warning"):\n    monkeypatch.setattr(notifications, level, spy)',
        "monkeypatch.setattr('modules.notification_center.notifications.notify', spy)",
        "monkeypatch.setattr(f'modules.lumascope_api.imaging.notifications.{level}', spy)",
        "monkeypatch.setattr(manual_recording_module, 'notifications', recorder)",
        "monkeypatch.setattr('modules.lumascope_api.imaging.notifications', fake)",
        "with patch('modules.notification_center.notifications') as mock:\n    pass",
        "with patch.object(imaging_mod, 'notifications') as mock:\n    pass",
        'notifications.warning = spy',
        'sio.notifications = centre',
    ]
    assert [len(_forms(source)) for source in refused] == [1] * len(refused)


def test_the_guard_passes_what_observes_rightly():
    allowed = [
        'monkeypatch.setattr(notifications, "report_outcome", spy)',
        "monkeypatch.setattr('modules.notification_center.notifications.report_outcome', spy)",
        "patch('modules.protocol.notifications.report_outcome')",
        'monkeypatch.setattr(sio, "notifications", notification_center.notifications)',
        'monkeypatch.setattr(module.logger, "warning", spy)',
        "monkeypatch.setattr('ui.notification_popup.show_notification_popup', spy)",
        'monkeypatch.setattr(notification_popup, name, _Widget)',
        'notifications.add_listener(posts.append)',
    ]
    assert [_forms(source) for source in allowed] == [[]] * len(allowed)
