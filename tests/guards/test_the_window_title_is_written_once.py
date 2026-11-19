# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The window's title is the product name and version, written once, at startup.

What LumaViewPro is doing is on the status line, read from the session's
live work. The title was a second place for it, written by the status
tick and by every piece that started some work, each setting and clearing
one shared slot; and the tick overwrote the startup title, so its build
stamp showed only until the first tick. The App sets the title once and
nothing writes it again: no ``Window.set_title`` and no ``.title =`` on the
App or the ``Window``. A popup or a drawer setting its own ``.title`` is
its own widget's, and not the window's.
"""

import ast

from tests.ast_seams import REPO_ROOT, tracked_files

_APP_MODULE = 'lumaviewpro.py'


def _app_classes(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == 'App' for base in node.bases)
    }


def _title_writes(rel_path: str) -> list[str]:
    """Each write of the window's title in *rel_path*, as ``path:line what``."""
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding='utf-8'))
    apps = _app_classes(tree)
    writes = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)] + [tree]:
        in_app = isinstance(cls, ast.ClassDef) and cls.name in apps
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'set_title'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'Window'
            ):
                writes.append(f'{rel_path}:{node.lineno} Window.set_title')
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not (isinstance(target, ast.Attribute) and target.attr == 'title'):
                        continue
                    owner = target.value
                    if isinstance(owner, ast.Name) and owner.id == 'Window':
                        writes.append(f'{rel_path}:{node.lineno} Window.title =')
                    elif in_app and isinstance(owner, ast.Name) and owner.id == 'self':
                        writes.append(f'{rel_path}:{node.lineno} App.title =')
    return sorted(set(writes))


def test_the_window_title_is_written_once_by_the_app_at_startup():
    writes = [w for path in [_APP_MODULE, *tracked_files(['ui'])] for w in _title_writes(path)]
    assert len(writes) == 1 and writes[0].startswith(f'{_APP_MODULE}:'), (
        f'the window title is written once, by the App at startup, and nowhere else: {writes}'
    )
