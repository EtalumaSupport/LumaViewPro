# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A test never installs a stand-in for a production module.

A module stand-in written into ``sys.modules`` by a test file is a
process-level swap: a production module first imported while the stand-in is
installed binds it through its own ``from ... import`` and keeps it for the
rest of the process, so the failure lands in another file, by collection
order. Seventeen test files installed a ``MagicMock`` as
``modules.settings_init`` this way (2026-10-09, skillspass session 37);
``test_a_stored_value_the_writer_refuses_is_replaced_at_load.py`` then got a
``MagicMock`` back from ``prepare_settings`` whenever one of them was collected
first, 11 red for a file that was green alone.

``tests/conftest.py`` is the one place a module is stood in for, and only for
the layers that never run in the test process (Kivy, the camera SDKs, the USB
bus, requests, platformdirs). This walk over every other test module refuses a
``sys.modules`` write whose key is a production name, ``modules.*``, ``ui.*``,
``drivers.*`` or ``lumaviewpro``, in each of its forms: an assignment, a
``setdefault`` and a ``monkeypatch.setitem``. A key that reaches the write
through a ``for`` over literal names is read through the loop. A write of
``None`` is not refused: it installs nothing, it makes the import raise
``ImportError``, which is how a test produces an SDK the host does not have.

A test that needs the real module to behave differently patches the one
attribute the code under test reads, on the real module, through
``monkeypatch``.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules

PRODUCTION_ROOTS = frozenset({'modules', 'ui', 'drivers', 'lumaviewpro'})
EXEMPT = frozenset(
    {
        'tests/conftest.py',
        'tests/guards/test_no_test_installs_a_module_stand_in.py',
    }
)


def _is_production_name(name: str) -> bool:
    return name.split('.')[0] in PRODUCTION_ROOTS


def _is_sys_modules(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'modules'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'sys'
    )


def _loop_bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Every name a ``for`` binds to literal strings, with those strings."""
    bound: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        targets = (
            [node.target]
            if isinstance(node.target, ast.Name)
            else list(node.target.elts)
            if isinstance(node.target, ast.Tuple)
            else []
        )
        for elt in node.iter.elts:
            values = (
                [elt] if len(targets) == 1 else (elt.elts if isinstance(elt, ast.Tuple) else [])
            )
            for target, value in zip(targets, values, strict=False):
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    bound.setdefault(target.id, set()).add(value.value)
    return bound


def _key_names(key: ast.AST, bound: dict[str, set[str]]) -> set[str]:
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return {key.value}
    if isinstance(key, ast.Name):
        return set(bound.get(key.id, ()))
    return set()


def _is_none(value: ast.AST | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is None


def violations(tree: ast.AST):
    """Yield ``(line, form, key)`` for every production stand-in installed."""
    bound = _loop_bindings(tree)

    def refused(key, value, form, line):
        if _is_none(value):
            return
        for name in sorted(_key_names(key, bound)):
            if _is_production_name(name):
                yield line, form, name

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and _is_sys_modules(target.value):
                    yield from refused(target.slice, node.value, 'sys.modules[...] =', node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if func.attr == 'setdefault' and _is_sys_modules(func.value) and node.args:
                value = node.args[1] if len(node.args) > 1 else None
                yield from refused(node.args[0], value, 'sys.modules.setdefault', node.lineno)
            elif func.attr == 'setitem' and len(node.args) >= 2 and _is_sys_modules(node.args[0]):
                value = node.args[2] if len(node.args) > 2 else None
                yield from refused(
                    node.args[1], value, 'monkeypatch.setitem(sys.modules', node.lineno
                )


def _found_in_tests():
    for rel_path, tree in iter_package_modules(('tests',)):
        if rel_path in EXEMPT:
            continue
        for lineno, form, key in violations(tree):
            yield f'{rel_path}:{lineno} {form} {key!r}'


def test_no_test_installs_a_stand_in_for_a_production_module():
    found = list(_found_in_tests())
    assert found == [], (
        'a production module is never stood in for by a test; patch the attribute '
        f'the code under test reads, on the real module: {found}'
    )


def _forms(source: str) -> list[tuple[str, str]]:
    return [(form, key) for _line, form, key in violations(ast.parse(source))]


def test_the_guard_sees_every_form_it_refuses():
    refused = [
        "sys.modules['modules.settings_init'] = MagicMock()",
        "sys.modules.setdefault('modules.settings_init', _mock)",
        "monkeypatch.setitem(sys.modules, 'ui.ui_helpers', ui_helpers)",
        "mp.setitem(sys.modules, 'drivers.registry', registry_mod)",
        "sys.modules['lumaviewpro'] = fake",
        "for _name in ('kivy.uix', 'modules.settings_init'):\n    sys.modules.setdefault(_name, MagicMock())",
        "for _name, _attr in (('ui.layer_control', 'LayerControl'),):\n    sys.modules[_name] = _module",
    ]
    assert [len(_forms(source)) for source in refused] == [1] * len(refused)


def test_an_import_failure_injection_is_not_a_stand_in():
    allowed = [
        "monkeypatch.setitem(sys.modules, 'drivers.idscamera', None)",
        "sys.modules['modules.optional'] = None",
    ]
    assert [_forms(source) for source in allowed] == [[], []]


def test_a_third_party_name_is_not_refused():
    allowed = [
        "sys.modules.setdefault('kivy.uix.floatlayout', _floatlayout)",
        "monkeypatch.setitem(sys.modules, 'lvp_logger', lvp_logger_mod)",
        "sys.modules['libusb_package'] = None",
    ]
    assert [_forms(source) for source in allowed] == [[], [], []]
