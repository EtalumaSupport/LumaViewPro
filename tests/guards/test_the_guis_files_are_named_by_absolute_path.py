# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The GUI names its shipped files by absolute path, through ``build_file``.

The kv file and every icon reached Kivy as a path relative to nothing, and
Kivy's ``resource_find`` returns ``abspath(filename)`` when that exists
before it searches its own paths: started from inside another checkout, the
GUI ran that checkout's kv file and icons. Every shipped file the GUI names
is now ``_lumascope.lib.build_files.build_file('<path under the build root>')``, in
Python and in kv (each kv source with its own ``#:import``, since an import
is process-wide and a source leaning on another's would load only by
order). This scan refuses a relative path to a shipped file (``data/``,
``./data/``, ``ui/*.kv``) in the kv file, in every string in ``ui/`` and
the GUI entry (the ``Builder.load_string`` blocks among them), and in an
f-string's literal parts; kv comment lines and Kivy's own ``atlas://``
theme names are not ours.
"""

from __future__ import annotations

import ast
import re

from tests.ast_seams import REPO_ROOT, iter_package_modules, parse_module, tracked_files

_HELPER = 'build_file'
# A whole string that is a relative path to one of our shipped files.
_SHIPPED = re.compile(r'^(?:\./)?(?:data/.+|ui/[^/]+\.kv)$')
# The same path quoted inside kv text, unless it is the helper's argument.
_QUOTED_IN_KV = re.compile(rf"""(?<!{_HELPER}\()(['"])(?:\./)?(?:data/[^'"]+|ui/[^/'"]+\.kv)\1""")


def _kv_offences(text: str) -> list[tuple[int, str]]:
    """(line within ``text``, the quoted path) for each relative shipped path in kv text."""
    return [
        (number, match.group(0))
        for number, line in enumerate(text.splitlines(), 1)
        if not line.lstrip().startswith('#')
        for match in _QUOTED_IN_KV.finditer(line)
    ]


def _helper_arguments(tree: ast.AST) -> set[int]:
    """The ids of the string nodes handed to the helper."""
    return {
        id(arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, 'id', getattr(node.func, 'attr', None)) == _HELPER
        for arg in node.args
    }


def _python_offences(tree: ast.Module) -> list[tuple[int, str]]:
    """(line, the path) for each relative shipped path in a module's strings."""
    handed = _helper_arguments(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in handed:
            continue
        if _SHIPPED.match(node.value):
            found.append((node.lineno, node.value))
        else:
            found.extend(
                (node.lineno + offset - 1, path) for offset, path in _kv_offences(node.value)
            )
    return found


def _gui_sources():
    """(path, offences) for the kv files, ``ui/``'s modules and the GUI entry."""
    for rel_path in tracked_files(['ui'], pattern='*.kv'):
        yield rel_path, _kv_offences((REPO_ROOT / rel_path).read_text(encoding='utf-8'))
    for rel_path, tree in iter_package_modules(('ui',)):
        yield rel_path, _python_offences(tree)
    yield 'lumaviewpro.py', _python_offences(parse_module('lumaviewpro.py'))


def test_the_gui_names_no_shipped_file_by_a_relative_path():
    found = [
        f'{rel_path}:{line} {path}'
        for rel_path, offences in _gui_sources()
        for line, path in offences
    ]
    assert found == [], f"name a shipped file as {_HELPER}('<path under the build root>'): {found}"


def test_each_kv_source_that_calls_the_helper_imports_it():
    import_line = f'#:import {_HELPER} _lumascope.lib.build_files.{_HELPER}'
    kv_texts = [
        (rel_path, (REPO_ROOT / rel_path).read_text(encoding='utf-8'))
        for rel_path in tracked_files(['ui'], pattern='*.kv')
    ]
    modules = [*iter_package_modules(('ui',)), ('lumaviewpro.py', parse_module('lumaviewpro.py'))]
    for rel_path, tree in modules:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, 'attr', None) == 'load_string'
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                kv_texts.append((f'{rel_path}:{node.lineno}', node.args[0].value))
    missing = [
        source for source, text in kv_texts if f'{_HELPER}(' in text and import_line not in text
    ]
    assert missing == [], f'each kv source calling {_HELPER} carries {import_line!r}: {missing}'


def test_the_scan_refuses_planted_paths_and_passes_the_helper():
    planted = ast.parse(
        "ICON = './data/icons/a.png'\n"
        "KV = 'ui/lumaviewpro.kv'\n"
        'NAME = f"data/icons/{name}.png"\n'
        "OK = build_file('data/icons/a.png')\n"
        'Builder.load_string("""\n'
        '<A>:\n'
        "    source: 'data/icons/b.png'\n"
        "    # source: './data/icons/c.png'\n"
        "    other: build_file('data/icons/d.png')\n"
        "    theme: 'atlas://data/images/defaulttheme/button'\n"
        '""")\n'
    )
    assert sorted(path for _line, path in _python_offences(planted)) == [
        "'data/icons/b.png'",
        './data/icons/a.png',
        'data/icons/',
        'ui/lumaviewpro.kv',
    ]
