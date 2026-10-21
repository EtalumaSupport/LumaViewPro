# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every public member that writes a setting or a file is a guarded write, or says why not.

While a control session is held, a write that is not its holder's is refused
before it writes, and a write running is counted so no take lands mid-way
(``activity_claim.guarded_write``). A member that reached a write without the
guard would write past the session: its setting writes are refused by the
store's own check but not counted, and its file writes not refused at all.

So the walk follows each public member of the three owners of such writes,
through the same-class members it names (``self._x(...)``, and
``IOTask(action=self._x)`` alike), to the writes themselves -- the settings
stores, a protocol file, a recording's pending frames, the file writer's
recovery, the driver install, the support artefacts -- and every member that
reaches one is guarded, by the decorator or by ``a_guarded_write`` in its
body, or named below with why not. A guarded member covers what it reaches.
"""

from __future__ import annotations

import ast

from tests.ast_seams import parse_module

# The calls that write, by the attribute they are named by.
_WRITES = {
    '_store_setting': "the Session's settings store, one leaf",
    '_store_settings': "the Session's settings store, several leaves",
    'store_settings': "the camera's writes into the Session's store",
    'to_file': 'a protocol file',
    'discard_pending': "a recording's pending frames given up",
    'replace_stuck_worker': "the file writer's recovery, which gives up its queued files",
    'run_as_administrator': 'the camera driver install',
    'generate': 'a support report',
    'generate_logs_only': 'a logs zip',
}

_OWNERS = (
    ('modules/scope_session.py', 'ScopeSession'),
    ('modules/lumascope_api/camera.py', 'CameraAPI'),
    ('modules/manual_recording.py', 'ManualRecordingController'),
)

# Public members that reach a write unguarded, each with why that is right.
_NOT_GUARDED = {
    'ScopeSession.make_support_report': (
        'writes a support artefact and changes no scope state; made by anyone'
    ),
    'ScopeSession.make_logs_zip': (
        'writes a support artefact and changes no scope state; made by anyone'
    ),
}


def _is_guarded(fn: ast.FunctionDef) -> bool:
    if any(isinstance(d, ast.Name) and d.id == '_guarded_write' for d in fn.decorator_list):
        return True
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'a_guarded_write'
        for node in ast.walk(fn)
    )


def _walk(rel_path: str, class_name: str):
    """Per public member, the writes it reaches unguarded; and every write reached at all."""
    tree = parse_module(rel_path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    members: dict[str, list[ast.FunctionDef]] = {}
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A property's getter and setter share a name; both are the member.
            members.setdefault(node.name, []).append(node)

    guarded, writes, names = {}, {}, {}
    for name, defs in members.items():
        guarded[name] = any(_is_guarded(fn) for fn in defs)
        writes[name], names[name] = set(), set()
        for fn in defs:
            for node in ast.walk(fn):
                if not isinstance(node, ast.Attribute):
                    continue
                if node.attr in _WRITES:
                    writes[name].add(node.attr)
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in ('self', 'cls')
                    and node.attr in members
                    and node.attr != name
                ):
                    names[name].add(node.attr)

    def reached(name: str, through: frozenset[str]) -> set[str]:
        found = set(writes[name])
        for other in names[name] - through:
            if not guarded[other]:
                found |= reached(other, through | {name})
        return found

    unguarded = {
        f'{class_name}.{name}': found
        for name in members
        if not name.startswith('_') and not guarded[name]
        for found in [reached(name, frozenset())]
        if found
    }
    return unguarded, set().union(*writes.values())


def test_every_member_that_writes_is_guarded_or_says_why_not():
    unguarded, reached_at_all = {}, set()
    for rel_path, class_name in _OWNERS:
        found, writes = _walk(rel_path, class_name)
        unguarded |= found
        reached_at_all |= writes

    # The walk sees each write it looks for, or a rename has blinded it.
    assert reached_at_all == set(_WRITES), (
        f'writes no member calls: {set(_WRITES) - reached_at_all}'
    )
    past_the_session = {
        member: sorted(_WRITES[w] for w in found)
        for member, found in unguarded.items()
        if member not in _NOT_GUARDED
    }
    assert not past_the_session, (
        f'writes past a control session, each to be guarded or named with why: {past_the_session}'
    )
    stale = set(_NOT_GUARDED) - set(unguarded)
    assert not stale, f'named as unguarded writers, and no longer are: {stale}'
