# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every outcome class states its own reasons, so a caller can act on it by code.

An outcome reaches a REST caller, the SDK and the GUI as a ``reason`` it
branches on. Ten refusal types once carried none, so over the wire their
reason was ``''`` and a client could tell a position out of range from a
missing step only by parsing the sentence.

Every outcome class -- a refusal, a notice, a quiet outcome or a fault --
declares its reasons as ``causes`` in its own class body, and the root
checks each declaration where the class is defined
(``lib.lumascope_error``). Reasons inherited from a parent do not count
here: a subclass is a different outcome, and one that took its parent's
codes would be indistinguishable from it. An umbrella names a subject and
is never raised, so it declares none.

Each reason of a class with several also has its sentence, in the class's
docstring as ``'reason'`` -- what it means, so the reference tells a
caller what each code it branches on says. A class with one reason is
explained by its docstring; a reason several classes carry as one fact
(a base's, or ``same_fact_as``) is explained once, by any of them.
"""

import ast
import re

from tests.api_reference import own_doc, outcome_classes
from tests.ast_seams import iter_package_modules, production_modules


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


def _outcome_classes():
    """Every class below a kind in the layers that raise, umbrellas aside.

    The layers of ``tests/guards/test_every_outcome_joins_the_root.py``, with
    ``lib/``, where the fault kind lives: a driver's fault is an outcome a
    caller branches on as much as a module's.
    """
    classes = {}
    for rel, tree in iter_package_modules(('lib', 'drivers', 'modules', 'rest')):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = (rel, node)
    kinds = {'Refusal', 'Notice', 'Quiet', 'Fault'}
    outcomes = set(kinds)
    grew = True
    while grew:
        grew = False
        for name, (_, node) in classes.items():
            if name not in outcomes and any(_base_name(b) in outcomes for b in node.bases):
                outcomes.add(name)
                grew = True
    return {
        name: classes[name]
        for name in outcomes - kinds
        if '_Umbrella' not in {_base_name(b) for b in classes[name][1].bases}
    }


def _states_its_reasons(cls):
    for node in cls.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign) and node.value is not None
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == 'causes' for t in targets):
            return True
    return False


def test_the_walk_finds_the_outcomes():
    # The instrument on known positives, one per kind and one below an
    # umbrella, and an umbrella left out.
    found = _outcome_classes()
    for name in (
        'StepNotFoundError',
        'NoHardwareDetectedNotice',
        'RunWaitExpiredError',
        'BoardNoReplyError',
        'HomingFaultError',
    ):
        assert name in found, name
        assert _states_its_reasons(found[name][1]), name
    assert 'HomingFailedError' not in found


def test_every_outcome_states_its_reasons():
    offenders = sorted(
        f'{rel}::{name}'
        for name, (rel, cls) in _outcome_classes().items()
        if not _states_its_reasons(cls)
    )

    assert offenders == [], (
        'these outcomes declare no reasons of their own, so a caller branching on '
        f"reason reads its parent's: {offenders}. State causes in the class body, "
        'each reason mapped to its cause.'
    )


def _same_fact(cls, classes):
    """*cls* and every class carrying its reasons as the same fact: its bases, and ``same_fact_as`` either way."""
    linked = {c: {*c.__mro__, *c.same_fact_as} for c in classes}
    for c, others in list(linked.items()):
        for other in others:
            linked.setdefault(other, set()).add(c)
    found, queue = set(), [cls]
    while queue:
        c = queue.pop()
        if c not in found:
            found.add(c)
            queue += linked.get(c, ())
    return found


def _explains(cls, reason):
    """*cls*'s own docstring gives *reason* its sentence: ``'reason'`` -- the sentence."""
    doc = ' '.join(own_doc(cls).split())
    return re.search(rf"``'{re.escape(reason)}'`` -- \S", doc) is not None


def test_every_reason_is_explained():
    # A class with one reason is explained by its own docstring. A reason
    # several classes carry as one fact is explained once, by any of them.
    classes = list(outcome_classes().values())
    unexplained = {}
    for cls in sorted(classes, key=lambda c: c.__name__):
        causes = vars(cls).get('causes') or {}
        if len(causes) < 2:
            continue
        carriers = _same_fact(cls, classes)
        missing = [
            reason
            for reason in causes
            if not any(
                reason in (getattr(c, 'causes', None) or {}) and _explains(c, reason)
                for c in carriers
            )
        ]
        if missing:
            unexplained[cls.__name__] = missing
    assert not unexplained, (
        f'{sum(map(len, unexplained.values()))} reasons a caller branches on are explained '
        "nowhere: give each its sentence in its class's docstring, ``'reason'`` -- what it "
        'means:\n' + ''.join(f'  {name}: {reasons}\n' for name, reasons in unexplained.items())
    )
