# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Only ``lvp_logger.git_revision`` asks git which commit the running code is.

The banner, the capability harness, a profile trace and every plugin record
name the build through it, so they cannot disagree. A second reader drifted
before: the window title ran its own ``git rev-parse`` in the launch folder,
and a profile trace ran its own beside it, missing the commit a GitHub ZIP
carries in ``.git_archival.txt``. ``tools/`` is outside the sweep: its git
calls are about the repository, not the running build.
"""

import ast

from tests.ast_seams import iter_package_modules, parse_module

_PACKAGES = ('modules', 'ui', 'drivers', 'lib')
_ROOT_MODULES = ('lumaviewpro.py', 'lvp_logger.py')


def _rev_parse_sites():
    """Each ``'rev-parse'`` string in production code, as ``(path, function, line)``.

    A constant, so both shapes the code has used are seen: a command list
    literal, and an argument handed to a helper that builds the list.
    """
    modules = list(iter_package_modules(_PACKAGES))
    modules += [(path, parse_module(path)) for path in _ROOT_MODULES]
    sites = []
    for path, tree in modules:
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(function):
                if isinstance(node, ast.Constant) and node.value == 'rev-parse':
                    sites.append((path, function.name, node.lineno))
    return sorted(set(sites))


def test_the_sweep_sees_the_one_reader():
    """The instrument reports presence: the reader's own call is found."""
    assert ('lvp_logger.py', 'git_revision') in {(p, f) for p, f, _ in _rev_parse_sites()}


def test_no_other_production_module_asks_git_for_the_build():
    others = [s for s in _rev_parse_sites() if (s[0], s[1]) != ('lvp_logger.py', 'git_revision')]
    assert others == [], f'ask lvp_logger.git_revision() instead: {others}'
