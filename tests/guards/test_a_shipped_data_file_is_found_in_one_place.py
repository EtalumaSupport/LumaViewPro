# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Where a file under ``data/`` is read from has one answer, ``path_utils.resolve_data_file``.

An installed build's data folder is in Documents, named for the version,
and a later build of that version finds the earlier one's folder. The
template and catalogues were read from copies seeded there, so a later
build read the earlier build's: a setting it added was missing, and every
Home was refused. ``resolve_data_file`` now answers the build for every
shipped file but the user's (``USER_DATA_FILES``), and the seeding copies
only those.

A reader that joins ``'data'`` onto a data root itself skips that answer
and reads the copy again. This walk refuses such a join outside
``path_utils`` but for the named owners below; a join onto ``build_root()``
reads the build and is not one.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules, walk_defs

# (module, qualname) -> what its join of 'data' onto a data root reaches.
_JOINS = {
    ('src/_lumascope/modules/settings_init.py', 'current_json_path'): (
        "the user's current.json, which no build ships"
    ),
    ('src/_lumascope/modules/tech_support_report.py', 'TechSupportReport._data_folder'): (
        'the data folder as a whole, which the report copies'
    ),
    ('src/_lumascope/modules/tech_support_report.py', 'TechSupportReport._step_data_folder'): (
        "the copy's own folder in the report"
    ),
    ('src/_lumascope/modules/exceptions.py', 'InstallationFileError.__init__'): (
        'the installed data folder, to tell which advice fixes a file'
    ),
    ('tools/firmware_tools.py', 'cmd_push_ini'): (
        "the board INIs of the checkout the tool runs in, the build's own"
    ),
    ('tools/pylon_probe_sweep.py', '_session_on_this_scope'): "the user's current.json",
}


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('src/_lumascope/drivers', 'tools'))


def _is_build_root(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and getattr(node.func, 'id', getattr(node.func, 'attr', None)) == 'build_root'
    )


def _joins_of_data(node: ast.AST):
    """Yield the line of every join of ``'data'`` onto something but ``build_root()``."""
    for child in ast.walk(node):
        divides = (
            isinstance(child, ast.BinOp)
            and isinstance(child.op, ast.Div)
            and isinstance(child.right, ast.Constant)
            and child.right.value == 'data'
            and not _is_build_root(child.left)
        )
        joins = (
            isinstance(child, ast.Call)
            and getattr(child.func, 'attr', None) in ('join', 'joinpath')
            and any(isinstance(a, ast.Constant) and a.value == 'data' for a in child.args)
            and not _is_build_root(getattr(child.func, 'value', None))
        )
        if divides or joins:
            yield child.lineno


def test_no_reader_joins_data_onto_a_data_root_outside_path_utils():
    found = []
    for rel_path, tree in _all_production_modules():
        if rel_path == 'src/_lumascope/modules/path_utils.py':
            continue
        # Each join at its innermost def: walk_defs yields an outer def before
        # the defs inside it, so the inner one's name is written last.
        joins = dict.fromkeys(_joins_of_data(tree), '<module>')
        for qualname, node in walk_defs(tree.body):
            joins.update(dict.fromkeys(_joins_of_data(node), qualname))
        found += [
            f'{rel_path}:{lineno} {qualname}'
            for lineno, qualname in sorted(joins.items())
            if (rel_path, qualname) not in _JOINS
        ]
    assert found == [], (
        'a file under data/ reached by a join of its own; take '
        f'path_utils.resolve_data_file, which knows the build from the user: {found}'
    )
