# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Free disk space is read in one place and judged by two.

Whether a run, a manual recording or a post-processing build may start
where it saves is ``common_utils.room_to_start``'s answer; whether a write
about to land while one runs still has room is ``common_utils.room_to_write``'s,
a recording's frames through ``DiskCheckInterval``. Both read free space
through ``common_utils.check_disk_space_ok``, the one reading. Each site that
read the free space and decided for itself had its own threshold, its own
answer to a probe that failed, its own reason and its own words, so a
recording was refused one way and a run another for the same fact.

This walk over every production module refuses a call of
``check_disk_space_ok`` or of ``disk_usage`` (psutil's or shutil's) outside
the owners named below, each with why it reads, and refuses an owner named
here that no longer reads, so the walk is seen finding what it allows.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, parse_module, walk_defs

_READS = ('check_disk_space_ok', 'disk_usage')
_COMMON = 'src/_lumascope/modules/common_utils.py'

# (module, function) -> what it reads, and why it may.
_OWNERS = {
    (_COMMON, 'check_disk_space_ok'): ('disk_usage', 'the one reading'),
    (_COMMON, 'room_to_start'): ('check_disk_space_ok', 'the start judge'),
    (_COMMON, 'room_to_write'): ('check_disk_space_ok', 'the running judge'),
    ('src/_lumascope/modules/host_resources.py', '_free_space'): (
        'check_disk_space_ok',
        "the startup notice's reader; its line is the host reading's",
    ),
    (_COMMON, 'get_extra_disks_info'): ('disk_usage', 'the support line, judging nothing'),
    ('src/_lumascope/modules/config_helpers.py', '_disk_line'): (
        'disk_usage',
        'the metrics line, judging nothing',
    ),
    ('src/_lumascope/modules/tech_support_report.py', 'TechSupportReport._step_disk_speed'): (
        'disk_usage',
        "the support report's disk speed test, judging nothing",
    ),
}


def _calls_in(node):
    """The names called in *node*'s own body, its nested defs left to themselves."""
    names = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Call):
            func = child.func
            name = getattr(func, 'attr', None) or getattr(func, 'id', None)
            if name in _READS:
                names.append((name, child.lineno))
        stack.extend(ast.iter_child_nodes(child))
    return names


def _modules():
    yield from iter_package_modules(('src/_lumascope', 'ui'))
    yield 'lumaviewpro.py', parse_module('lumaviewpro.py')


def test_free_space_is_read_only_by_its_owners():
    seen = set()
    strays = []
    for rel_path, tree in _modules():
        for name, line in _calls_in(tree):
            strays.append(f'{rel_path}:{line} calls {name} at module level')
        for qualname, node in walk_defs(tree.body):
            for name, line in _calls_in(node):
                owner = _OWNERS.get((rel_path, qualname))
                if owner is not None and owner[0] == name:
                    seen.add((rel_path, qualname))
                else:
                    strays.append(f'{rel_path}:{line} {qualname} calls {name}')

    assert strays == [], (
        'free disk space is read outside its owners; a start asks '
        'common_utils.room_to_start, a write common_utils.room_to_write:\n' + '\n'.join(strays)
    )
    gone = sorted(f'{path} {name}' for path, name in set(_OWNERS) - seen)
    assert gone == [], 'owners named here that no longer read free space:\n' + '\n'.join(gone)
