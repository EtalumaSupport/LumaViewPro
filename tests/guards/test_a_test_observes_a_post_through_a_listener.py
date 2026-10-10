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
test reported, not what was posted.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import iter_package_modules

POST_METHODS = frozenset({'debug', 'info', 'notice', 'warning', 'error', 'critical', 'notify'})

_SETATTR_TAILS = ('setattr', 'patch.object')
# A dotted path that ends at a module's centre, or one level below it.
_CENTRE_PATH = re.compile(r'\.notifications(?:\.(?P<attr>\w+|\{[^}]*\}))?$')


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
        f'{rel}:{lineno} {qualname} {what}' for rel, qualname, lineno, what in _found_in_tests()
    ]
    assert found == [], (
        'observe a post through the centre_posts fixture (tests/conftest.py), '
        f'never by replacing the centre or its posting methods: {found}'
    )


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
