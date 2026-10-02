# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Below the GUI, an outcome reaches the notification centre only through its reporter.

``report_outcome`` reads what an outcome is -- refusal, fault, notice -- and
its words, title, reason code and fatality from the exception's type, so
every client subscribed to the Session hears the same record. A direct
``notifications.warning(...)`` from a module carries none of that: its kind
is unclassified, its reason empty, and whether it is fatal is the poster's
choice. So nothing outside ``ui/`` and the centre itself posts directly.

The posts still made directly are listed below by function and count, each
with the open work that will move it; an entry goes when that work lands, and
the guard fails on an entry whose posts are gone, so the list cannot outlive
its reasons.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules

_POSTS = frozenset({'notify', 'debug', 'info', 'notice', 'warning', 'error', 'critical'})
_EXEMPT_FILES = frozenset({'modules/notification_center.py'})

# (file, enclosing function) -> (direct posts in it, the open work that moves
# them). The count, so a new post in a listed function is not waved through
# with the one already there.
_BRING_UP = 'step 6 slice C, bring-up'
_NOT_YET_TYPED = {
    ('modules/lumascope_api/_lumascope.py', '_notify_board_failure'): (1, _BRING_UP),
    ('modules/lumascope_api/_lumascope.py', 'Lumascope.__init__'): (1, _BRING_UP),
    ('modules/lumascope_api/_lumascope.py', 'Lumascope.initialize'): (1, _BRING_UP),
    ('modules/lumascope_api/_lumascope.py', 'Lumascope._notify_partial_hardware'): (1, _BRING_UP),
    ('modules/lumascope_api/imaging.py', 'ImagingAPI._notify_camera_absent'): (
        1,
        "primary's ASK-1, the camera-absent setters",
    ),
    ('modules/lumascope_api/imaging.py', 'ImagingAPI._get_image_impl'): (
        1,
        "primary's ASK-1, a capture with the camera gone",
    ),
    ('modules/lumascope_api/imaging.py', 'ImagingAPI._set_conversion_gain_mode_impl'): (
        1,
        'the camera toggles',
    ),
    ('modules/lumascope_api/imaging.py', 'ImagingAPI._set_line_noise_reduction_impl'): (
        1,
        'the camera toggles',
    ),
    ('modules/lumascope_api/illumination.py', 'IlluminationAPI._notify_if_led_command_failed'): (
        1,
        'the LED confirmation-before-cache work',
    ),
    ('modules/autofocus_runner.py', 'AutofocusRunner.run'): (
        1,
        'the attendedness of the autofocus lease refusal',
    ),
}


def _below_the_gui():
    for rel_path, tree in production_modules():
        if not rel_path.startswith('ui/'):
            yield rel_path, tree
    yield from iter_package_modules(('drivers', 'lib'))


def _is_centre(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == 'notifications'
    return isinstance(node, ast.Attribute) and node.attr == 'notifications'


def _direct_posts(tree: ast.Module):
    """Yield (enclosing qualname, lineno) for each direct post to the centre."""

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield from visit(child, [*scope, child.name])
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in _POSTS
                and _is_centre(child.func.value)
            ):
                yield '.'.join(scope), child.lineno
            yield from visit(child, scope)

    yield from visit(tree, [])


def _all_direct_posts():
    found = {}
    for rel_path, tree in _below_the_gui():
        if rel_path in _EXEMPT_FILES:
            continue
        for qualname, lineno in _direct_posts(tree):
            found.setdefault((rel_path, qualname), []).append(lineno)
    return found


def test_nothing_below_the_gui_posts_to_the_centre_directly():
    found = _all_direct_posts()
    unlisted = [
        f'{path}:{lines} in {qualname}'
        for (path, qualname), lines in sorted(found.items())
        if len(lines) > _NOT_YET_TYPED.get((path, qualname), (0, ''))[0]
    ]
    assert unlisted == [], (
        'report the outcome through notifications.report_outcome with a typed '
        f'exception (Refusal, Notice, or a fault) instead of posting it: {unlisted}'
    )


def test_every_listed_post_is_still_there():
    found = _all_direct_posts()
    gone = [
        f'{path} {qualname} ({row})'
        for (path, qualname), (count, row) in _NOT_YET_TYPED.items()
        if len(found.get((path, qualname), [])) < count
    ]
    assert gone == [], f'these posts were moved; delete their entries: {gone}'


def test_the_guard_sees_a_direct_post():
    tree = ast.parse(
        'class A:\n'
        '    def f(self):\n'
        "        notifications.warning('C', 'T', 'm')\n"
        "        notification_center.notifications.notice('C', 'T', 'm')\n"
        "        notifications.report_outcome(ex, solicited=False, category='C')\n"
    )
    assert list(_direct_posts(tree)) == [('A.f', 3), ('A.f', 4)]
