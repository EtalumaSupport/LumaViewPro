# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every refusal states its own reason, so a caller can act on it by code.

A ``Refusal`` reaches a REST caller, the SDK and the GUI as a ``reason`` it
branches on. Ten refusal types carried none, so over the wire their reason
was ``''`` and a client could tell a position out of range from a missing
step only by parsing the sentence. Each now states its own, and a new one
fails here until it does.

A class states its reason in one of three forms: a ``reason`` in its own
class body, ``self.reason`` assigned in its own ``__init__``, or a
``reason`` argument passed to its parent's ``__init__``. A reason inherited
from a parent does not count: a subclass is a different refusal, and one
that took its parent's code would be indistinguishable from it.
"""

import ast

from tests.ast_seams import production_modules


def _base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _classes():
    """``{name: (rel_path, ClassDef)}`` for every class in production code."""
    found = {}
    for rel, tree in production_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found[node.name] = (rel, node)
    return found


def _refusal_classes():
    classes = _classes()
    refusals = {'Refusal'}
    grew = True
    while grew:
        grew = False
        for name, (_, node) in classes.items():
            if name not in refusals and any(_base_name(b) in refusals for b in node.bases):
                refusals.add(name)
                grew = True
    refusals.discard('Refusal')
    return {name: classes[name] for name in refusals}


def _states_in_body(cls):
    for node in cls.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign) and node.value is not None
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == 'reason' for t in targets):
            return True
    return False


def _own_init(cls):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            return node
    return None


def _assigns_self_reason(init):
    for node in ast.walk(init):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == 'reason'
                    and isinstance(t.value, ast.Name)
                    and t.value.id == 'self'
                ):
                    return True
    return False


def _passes_reason_to_parent(init):
    for node in ast.walk(init):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '__init__'
        ):
            if any(kw.arg == 'reason' for kw in node.keywords):
                return True
            if any(isinstance(a, ast.Name) and a.id == 'reason' for a in node.args):
                return True
    return False


def _states_its_reason(cls):
    if _states_in_body(cls):
        return True
    init = _own_init(cls)
    return init is not None and (_assigns_self_reason(init) or _passes_reason_to_parent(init))


def test_the_walk_finds_the_refusals():
    # The instrument on known positives, one per form: a reason passed to
    # the parent, one assigned in __init__, one stated in the class body.
    found = _refusal_classes()
    for name in ('PostProcessingRefusedError', 'HardwareCommandRefusedError', 'StepNotFoundError'):
        assert name in found, name
        assert _states_its_reason(found[name][1]), name


def test_every_refusal_states_its_reason():
    offenders = sorted(
        f'{rel}::{name}'
        for name, (rel, cls) in _refusal_classes().items()
        if not _states_its_reason(cls)
    )

    assert offenders == [], (
        'these refusals carry no reason of their own, so a caller branching on '
        f'reason reads an empty one: {offenders}. State it in the class body '
        "(reason = '...'), assign self.reason in __init__, or pass reason to the "
        "parent's __init__."
    )
