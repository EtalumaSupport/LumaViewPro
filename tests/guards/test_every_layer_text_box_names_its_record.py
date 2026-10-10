# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every caller of the layer text helper names its box with a literal id ending in '_text'.

``LayerControl._validate_and_apply_text_input`` derives the record name from
the box's id (``exp_text`` on the Blue panel records ``EXP_Blue``) and raises
``ValueError`` on an id that does not end in ``_text``, before it writes
anything. That raise is the runtime half of the contract, and it fires under
a person's cursor, on the commit of the one box whose caller got it wrong.

This is the build-time half: a walk over every call site in ``ui/`` that the
first argument is a string literal ending in ``_text``. A computed id is
refused too, because nothing short of running it says what it derives. It is
a guard rather than a behavioural test because the fact is about every call
site at once: a behavioural test drives the callers it names, and a new box
wired with a bad id would pass all of them.
"""

import ast

from tests.ast_seams import iter_package_modules

_HELPER = '_validate_and_apply_text_input'


def _call_sites():
    for rel_path, tree in iter_package_modules(('ui',)):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == _HELPER
            ):
                yield rel_path, node


def test_every_caller_supplies_a_derivable_text_id():
    sites = list(_call_sites())
    assert sites, f'no call to {_HELPER} found under ui/; the helper or its callers moved'

    bad = [
        f'{rel_path}:{node.lineno} {ast.unparse(node.args[0]) if node.args else "<no id>"}'
        for rel_path, node in sites
        if not (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.endswith('_text')
        )
    ]
    assert not bad, f'call sites whose text id is not a literal ending in _text: {bad}'
