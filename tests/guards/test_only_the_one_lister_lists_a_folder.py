# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No guard lists a folder itself: every scan reads ``tracked_files``.

``pathlib``'s ``glob`` and ``rglob``, ``os.walk``, and a prefix filter of
``git ls-files`` all answer a missing folder with nothing, so a guard that
lists ``src/_lumascope/drivers/`` itself passes over no files the day ``src/_lumascope/drivers/`` moves.
``tests/ast_seams.py`` ``tracked_files`` refuses such a folder by name;
this holds every guard to it. Keyed on the listing call, not on where its
path came from, which an AST cannot follow across the spellings of the
repository root. A call listing a folder a test made (``tmp_path``, the
build output) is exempt, each by file and call with its reason.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, Words, tracked_files

# Path methods that list a folder; ``walk`` only on a receiver other than
# ``ast``, since ``ast.walk`` walks a tree, not a folder.
_PATH_LISTERS = Words({'glob', 'rglob', 'iglob', 'iterdir', 'walk'})
_OS_LISTERS = Words({'listdir', 'scandir'})

# (guard file, the call's source): why it lists a folder the test itself made.
_OUTPUT_FOLDERS = {
    (
        'tests/guards/test_manual_capture_channel_identity.py',
        "(tmp_path / 'Manual').rglob('*.tiff')",
    ): "the test's own capture folder",
    (
        'tests/guards/test_memory_cap.py',
        "(REPO_ROOT / 'build').glob('memcap_*.txt')",
    ): 'the measurement files a run writes under the ignored build/',
    (
        'tests/guards/test_profile_trace.py',
        'Path(base).iterdir()',
    ): "the trace folder the test's own run wrote",
}


def listing_calls(tree: ast.AST) -> list[ast.Call]:
    """Every call in ``tree`` that lists a folder or asks git for its files.

    A module is known by what the tree imports it as (``import ast as _ast``;
    ``from subprocess import run``).
    """
    modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module == 'subprocess':
            for alias in node.names:
                modules[alias.asname or alias.name] = 'subprocess'
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            receiver = modules.get(func.value.id) if isinstance(func.value, ast.Name) else None
            lists = (func.attr in _PATH_LISTERS and receiver != 'ast') or (
                func.attr in _OS_LISTERS and receiver == 'os'
            )
            runs = receiver == 'subprocess'
        else:
            lists = False
            runs = isinstance(func, ast.Name) and modules.get(func.id) == 'subprocess'
        if lists or (runs and _asks_git_for_files(node)):
            found.append(node)
    return found


def _asks_git_for_files(call: ast.Call) -> bool:
    return any(
        isinstance(arg, (ast.List, ast.Tuple))
        and any(isinstance(e, ast.Constant) and e.value == 'ls-files' for e in arg.elts)
        for arg in call.args
    )


def _offenders() -> tuple[list[str], set[tuple[str, str]]]:
    offenders, exempt_seen = [], set()
    for rel in tracked_files(('tests/guards',)):
        source = (REPO_ROOT / rel).read_text(encoding='utf-8')
        for call in listing_calls(ast.parse(source, filename=rel)):
            key = (rel, ast.get_source_segment(source, call))
            if key in _OUTPUT_FOLDERS:
                exempt_seen.add(key)
            else:
                offenders.append(f'{rel}:{call.lineno} {key[1]}')
    return offenders, exempt_seen


def test_the_instrument_finds_every_listing_call_and_no_tree_walk():
    source = (
        'import ast as _ast, glob, os, subprocess\n'
        'from subprocess import run\n'
        "(REPO_ROOT / 'ui').rglob('*.py')\n"
        "REPO_ROOT.glob('src/_lumascope/modules/*.py')\n"
        'os.walk(root)\n'
        'os.listdir(root)\n'
        "glob.iglob('x/*')\n"
        "subprocess.run(['git', 'ls-files', '-z'], cwd=REPO_ROOT)\n"
        "run(['git', 'ls-files'])\n"
        '_ast.walk(tree)\n'
        'walk(tree, None)\n'
        "any(c == 'ls-files' for c in names)\n"
    )
    found = sorted(call.lineno for call in listing_calls(ast.parse(source)))
    assert found == [3, 4, 5, 6, 7, 8, 9]


def test_no_guard_lists_a_folder_itself():
    offenders, exempt_seen = _offenders()
    assert offenders == [], (
        'these list a folder themselves, which answers a missing folder with '
        'nothing; read it through tests/ast_seams.py tracked_files, which '
        f'refuses one by name: {offenders}'
    )
    assert exempt_seen == set(_OUTPUT_FOLDERS), (
        f'exempt calls no longer found: {set(_OUTPUT_FOLDERS) - exempt_seen}'
    )
