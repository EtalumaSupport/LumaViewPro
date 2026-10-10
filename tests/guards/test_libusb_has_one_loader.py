# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Only the FX2 driver touches libusb, so its bundled-library load comes first.

pyusb keeps one libusb backend per process, bound by whichever call loads
first, and python-libusb1 loads its library once. The FX2 driver binds both
to libusb-package's copy at import and refuses when something else got
there first -- but that refusal turns FX2 off; it cannot put the right
library back. The load is first only while no other production module can
reach either binding: a bare ``usb.core.find()`` anywhere else, run before
the driver imports, would bind the whole process to whatever libusb the
host has. An AST walk sees every import at any nesting depth.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, production_modules

_LOADER = 'drivers/fx2driver.py'
_BINDINGS = ('usb', 'usb1', 'libusb_package')


def _imported_names(tree: ast.Module):
    """Yield (lineno, dotted_name) for every absolute import in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools'))


def test_no_module_but_the_fx2_driver_imports_a_libusb_binding():
    found = []
    for rel_path, tree in _all_production_modules():
        if rel_path == _LOADER:
            continue
        for lineno, name in _imported_names(tree):
            if any(name == root or name.startswith(f'{root}.') for root in _BINDINGS):
                found.append(f'{rel_path}:{lineno} imports {name}')
    assert found == [], (
        'only drivers/fx2driver.py may import pyusb, python-libusb1 or '
        'libusb_package, so its load of the bundled libusb is the first in '
        f'the process: {found}'
    )


def test_the_guard_sees_the_loader_it_exempts():
    # The exemption must name a real importer, or a rename of the driver
    # would leave the guard exempting nothing while the check still passed.
    trees = dict(_all_production_modules())
    names = {name for _lineno, name in _imported_names(trees[_LOADER])}
    assert {'usb.core', 'usb1', 'libusb_package'} <= names
