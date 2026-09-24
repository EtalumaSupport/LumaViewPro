# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Typed text has one emitter, and no declaration is left without a consumer.

Two shapes rotted invisibly in the GUI log and this pins both shut.

A deferring wrapper around the typed-text emitter meant the apply record
reached the file BEFORE the line saying what was typed, and a freeze inside
the deferral window lost the last entry entirely. Every adopter now calls
``gui_logger.text_input`` directly, so the record lands before the entry does
anything, and there is one public name for one capability.

A write-back declaration marks the app's own write so the record it provokes
is dropped. That only works where something CONSUMES it: ``gui_logger.select``
does, for spinners, which really do dispatch on a programmatic assignment.
``text_input`` does not, because a text box does not -- assigning ``.text``
rebuilds the lines and the cursor and never touches ``focus``, so the echo the
declaration was guarding against cannot arrive. Twelve declarations sat at text
sites doing nothing, and nothing in the tree could tell. A declaration whose
name no ``select`` call can consume is dead on arrival, and this says so at
build time rather than leaving the next reader to re-derive it.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules

_RETIRED = 'text_input_debounced'
_PRODUCTION = ('ui', 'modules', 'drivers')


def _gui_logger_calls(tree, attr):
    """First argument of every ``gui_logger.<attr>(...)`` call, as source text."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == attr
            and isinstance(func.value, ast.Name)
            and func.value.id == 'gui_logger'
        ):
            found.append((node.lineno, ast.unparse(node.args[0])))
    return found


def test_no_production_module_defines_or_imports_the_deferring_wrapper():
    """One public name for typed text: the emitter in the logging module."""
    offenders = []
    for rel, tree in iter_package_modules(_PRODUCTION):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _RETIRED:
                offenders.append(f'{rel}:{node.lineno} defines {_RETIRED}')
            elif isinstance(node, ast.ImportFrom) and any(a.name == _RETIRED for a in node.names):
                offenders.append(f'{rel}:{node.lineno} imports {_RETIRED}')
    assert not offenders, (
        'a second public name for typed-text logging is back; call '
        'gui_logger.text_input directly so the record is written before the '
        'entry acts on itself:\n  ' + '\n  '.join(offenders)
    )


def test_every_write_back_declaration_has_something_that_can_consume_it():
    """A declaration no emitter consumes is dead, and reads as protection."""
    declared, consumable = [], set()
    for rel, tree in iter_package_modules(('ui',)):
        for lineno, name in _gui_logger_calls(tree, 'note_write_back'):
            declared.append((rel, lineno, name))
        consumable.update(name for _, name in _gui_logger_calls(tree, 'select'))

    orphans = [
        f'{rel}:{lineno} declares {name} -- no gui_logger.select({name}) anywhere in ui/'
        for rel, lineno, name in declared
        if name not in consumable
    ]
    assert not orphans, (
        'a write-back declaration is only consumed by gui_logger.select. At a '
        'TEXT site nothing consumes it and nothing needs to: assigning .text '
        'dispatches no focus event, so there is no echo to absorb. Delete the '
        'declaration rather than leaving a line that looks like a guard:\n  ' + '\n  '.join(orphans)
    )


def test_the_scan_sees_the_declarations_that_remain():
    """Guards the test above against passing because it found nothing."""
    declared = [
        name
        for _, tree in iter_package_modules(('ui',))
        for _, name in _gui_logger_calls(tree, 'note_write_back')
    ]
    assert len(declared) >= 9, (
        f'expected at least the nine spinner declarations, saw {len(declared)}; '
        'the scan stopped matching and the guard above went vacuous'
    )
