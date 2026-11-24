# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every fault a person can be shown says, in their words, what failed and what they can do.

A fault's message is a developer's unless its type writes it for the person
(``words_for_person``). Twenty-seven types wrote neither, so a camera read
that failed, a board that did not answer and a missing firmware file all
showed one generic sentence that sent the person to the log, which is ours.
Each production fault now either writes its message for the person or
declares ``person_words``, the one sentence its every raise is shown as;
a fault type added without either fails here, by name, before a person
reads it.
"""

import ast
import importlib

from tests import ast_seams
from tests.guards.test_every_refusal_states_its_reason import _base_name


def _production_faults() -> dict[str, type]:
    """Every production class below ``Fault``, imported, so an inherited declaration counts."""
    classes = {}
    modules = list(ast_seams.iter_package_modules(('src/_lumascope', 'ui')))
    modules.append(('lumaviewpro.py', ast_seams.parse_module('lumaviewpro.py')))
    for rel, tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (rel, node)
    faults = {'Fault'}
    grew = True
    while grew:
        grew = False
        for name, (_, node) in classes.items():
            if name not in faults and any(_base_name(b) in faults for b in node.bases):
                faults.add(name)
                grew = True
    faults.discard('Fault')
    return {
        name: getattr(importlib.import_module(ast_seams.module_name(classes[name][0])), name)
        for name in faults
    }


def _says_it_for_a_person(cls: type) -> bool:
    return bool(getattr(cls, 'words_for_person', False)) or bool(getattr(cls, 'person_words', ''))


def test_every_production_fault_has_words_for_a_person():
    faults = _production_faults()
    # The instrument on known positives: a driver's fault, one below
    # another fault, a module's, and the settings loader's.
    for name in (
        'BoardNoReplyError',
        '_PropertyTypeUnreadError',
        'RunFailedError',
        'SettingsFileError',
    ):
        assert name in faults, name
    unworded = sorted(name for name, cls in faults.items() if not _says_it_for_a_person(cls))

    assert not unworded, (
        'these faults show a person nothing of what failed: declare person_words, '
        'or write the message for the person and declare words_for_person = True:\n  '
        + '\n  '.join(unworded)
    )
