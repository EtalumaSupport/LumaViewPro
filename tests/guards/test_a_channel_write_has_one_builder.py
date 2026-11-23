# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a channel write stores is built by ``ScopeSession._channel_writing`` alone.

The one store of a channel's fields, ``ScopeSession._write_channel_fields``,
takes a ``_ChannelWriting``: each value by dotted settings path, the lock's
gain and exposure rounded as the settings carry them, a channel set to
acquire stopped stimulating. A mapping of each channel's values by key,
handed to that store, put each channel's name at the top of the settings and
replaced the channel's whole block (Set All Bookmarks left every channel its
focus alone). The record is that store's contract only while nothing else
builds one: every ``_ChannelWriting(...)`` call in LVP is in
``_channel_writing``.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, tracked_files

_RECORD = '_ChannelWriting'
_BUILDER = ('src/_lumascope/modules/scope_session.py', '_channel_writing')


def _constructions(tree: ast.AST) -> list[tuple[int, str | None]]:
    """(line, enclosing function) of every call that builds the record."""
    found = []

    def walk(node, function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
            if name == _RECORD:
                found.append((node.lineno, function))
        for child in ast.iter_child_nodes(node):
            walk(child, function)

    walk(tree, None)
    return found


def _offenders(files) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Every construction found, and those outside the one builder."""
    built = [
        (rel, line, function)
        for rel, source in files
        for line, function in _constructions(ast.parse(source))
    ]
    offenders = [
        f'{rel}:{line} in {function}'
        for rel, line, function in built
        if (rel, function) != _BUILDER
    ]
    return [(rel, function) for rel, _line, function in built], offenders


def test_only_channel_writing_builds_what_a_channel_write_stores():
    rels = ['lumaviewpro.py', *tracked_files(('src', 'ui', 'tools'))]
    files = [(rel, (REPO_ROOT / rel).read_text()) for rel in rels]
    built, offenders = _offenders(files)
    assert _BUILDER in built, 'the walk did not find the one builder: it is not reading it'
    assert offenders == []


def test_a_record_built_anywhere_else_is_named():
    elsewhere = (
        'def save_all_bookmarks(self):\n'
        '    self._write_channel_fields(_ChannelWriting(paths={}, locks={}))\n'
        'def other(self):\n'
        '    return scope_session._ChannelWriting({}, {})\n'
    )
    _built, offenders = _offenders([('src/_lumascope/modules/scope_session.py', elsewhere)])
    assert offenders == [
        'src/_lumascope/modules/scope_session.py:2 in save_all_bookmarks',
        'src/_lumascope/modules/scope_session.py:4 in other',
    ]
