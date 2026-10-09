# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every refusal declares whether retrying it can help, and a reason no cause answers is never built.

A REST server answers a refusal 422 when the request as sent cannot succeed
on this scope and 409 when the scope's state refused it, and a polling
client retries only the second. Nothing on a refusal said which, so a
server would have decided per exception, and a client retrying on a 409
would retry a bad argument for ever. Each type now states its ``cause``,
or ``causes`` keyed by reason where its reasons differ; a type stating
neither is refused at import, and a mixed type refuses to be built with a
reason its table lacks.

The tables are held to the raise sites both ways: for
``ProtocolRunRefusedError`` by ``test_every_runner_refusal_reason_is_covered``
(``tests/test_run_refusal_contract.py``), for the other two mixed types here.
``HardwareCommandRefusedError``'s reasons arrive from four modules and a
driver, so a reason handed on from a part or from the stage's interlock is
resolved to the set its source declares, and the drivers' interlock
literals are held to ``INTERLOCK_REASONS``, since a driver cannot import the
table.
"""

import ast
import importlib

import pytest

from modules.exceptions import (
    INTERLOCK_REASONS,
    HardwareCommandRefusedError,
    LiveFolderPathRefusedError,
    MissingPart,
    ProtocolRunRefusedError,
    Quiet,
    Refusal,
    RefusalCause,
)
from tests import ast_seams
from tests.guards.test_every_refusal_states_its_reason import _base_name, _classes


def _types_mixing_in(mixin):
    """Every production class below ``mixin``, imported, by the AST walk the reason guard uses."""
    classes = _classes()
    names = {mixin.__name__}
    grew = True
    while grew:
        grew = False
        for name, (_, node) in classes.items():
            if name not in names and any(_base_name(b) in names for b in node.bases):
                names.add(name)
                grew = True
    names.discard(mixin.__name__)
    found = {}
    for name in names:
        rel = classes[name][0]
        module = importlib.import_module(rel.removesuffix('.py').replace('/', '.'))
        found[name] = getattr(module, name)
    return found


def _mixed(types):
    return {name: cls for name, cls in types.items() if 'causes' in vars(cls)}


# How to build each mixed type with a given reason. A new mixed type fails
# test_the_factories_cover_every_mixed_type until it has a row here.
_FACTORIES = {
    'LiveFolderPathRefusedError': lambda reason: LiveFolderPathRefusedError(reason, 'n', 'm'),
    'ProtocolRunRefusedError': lambda reason: ProtocolRunRefusedError(reason, 't', 'm'),
    'HardwareCommandRefusedError': lambda reason: HardwareCommandRefusedError(reason, 'member'),
}


def test_the_walk_finds_both_kinds():
    # The instrument on known positives: one single-cause refusal, the
    # three mixed ones, one inheriting its cause, and the quiet type.
    refusals = _types_mixing_in(Refusal)
    for name in ('StepNotFoundError', 'ProtocolScheduleRefusedError', *_FACTORIES):
        assert name in refusals, name
    assert set(_mixed(refusals)) == set(_FACTORIES)
    assert 'RunAlreadyEndedError' in _types_mixing_in(Quiet)


def test_every_refusal_and_quiet_type_declares_a_cause():
    offenders = []
    for name, cls in {**_types_mixing_in(Refusal), **_types_mixing_in(Quiet)}.items():
        table = cls.causes if 'causes' in vars(cls) else None
        if table is not None:
            if not table or not all(isinstance(c, RefusalCause) for c in table.values()):
                offenders.append(name)
        elif not isinstance(cls.cause, RefusalCause):
            offenders.append(name)
    assert offenders == [], (
        f'these outcome types declare no RefusalCause: {sorted(offenders)}. State '
        'cause = RefusalCause.REQUEST or .STATE, or causes keyed by every reason.'
    )


def test_a_refusal_that_declares_no_cause_is_refused_at_import():
    with pytest.raises(TypeError, match='declares no cause'):

        class _NoCauseError(Refusal, Exception):
            title = 'Refused'

    with pytest.raises(TypeError, match='declares no cause'):

        class _NoCauseQuietError(Quiet, Exception):
            pass


def test_a_cause_inherited_from_a_declaring_type_counts():
    class _ParentError(Refusal, Exception):
        cause = RefusalCause.REQUEST

    class _ChildError(_ParentError):
        pass

    assert _ChildError().cause == RefusalCause.REQUEST


def test_the_factories_cover_every_mixed_type():
    assert set(_mixed(_types_mixing_in(Refusal))) == set(_FACTORIES)


@pytest.mark.parametrize('name', sorted(_FACTORIES))
def test_a_mixed_type_is_not_built_with_a_reason_it_declares_no_cause_for(name):
    with pytest.raises(TypeError, match="declares no cause for the reason 'no_such_reason'"):
        _FACTORIES[name]('no_such_reason')


def test_a_mixed_types_cause_follows_its_reason():
    assert HardwareCommandRefusedError('exclusive_activity_running', 'm').cause == (
        RefusalCause.STATE
    )
    absent = HardwareCommandRefusedError('axis_absent', 'm', missing=MissingPart.X)
    assert absent.cause == RefusalCause.REQUEST
    assert LiveFolderPathRefusedError('capture_location_unusable', 'n', 'm').cause == (
        RefusalCause.STATE
    )
    assert LiveFolderPathRefusedError('outside_live_folder', 'n', 'm').cause == (
        RefusalCause.REQUEST
    )


def _missing_part_reasons():
    parts = [p for p in vars(MissingPart).values() if isinstance(p, MissingPart)]
    parts += [MissingPart.led('Red'), MissingPart.led(1)]
    return {p.reason for p in parts}


class _RaisedReasons(ast.NodeVisitor):
    """The reasons a module builds ``type_name`` with, a handed-on reason resolved to its source's set.

    A reason is the ``reason`` keyword or the first argument. A name's
    ``.reason`` resolves to the missing parts' reasons when the call passes
    ``missing=``, and to ``INTERLOCK_REASONS`` when the name is bound to a
    ``MotionInterlockError`` (an ``except ... as`` or an annotated
    parameter); anything else is unread, and fails the census.
    """

    def __init__(self, rel, type_name, raised, unread):
        self.rel = rel
        self.type_name = type_name
        self.raised = raised
        self.unread = unread
        self.interlocks = [set()]

    def _visit_def(self, node):
        args = node.args
        bound = {
            a.arg
            for a in args.posonlyargs + args.args + args.kwonlyargs
            if a.annotation is not None and _base_name(a.annotation) == 'MotionInterlockError'
        }
        self.interlocks.append(bound)
        self.generic_visit(node)
        self.interlocks.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _visit_def

    def visit_ExceptHandler(self, node):
        if node.name and node.type is not None and _base_name(node.type) == 'MotionInterlockError':
            self.interlocks[-1].add(node.name)
            self.generic_visit(node)
            self.interlocks[-1].discard(node.name)
        else:
            self.generic_visit(node)

    def visit_Call(self, node):
        callee = getattr(node.func, 'attr', None) or getattr(node.func, 'id', None)
        if callee == self.type_name:
            reason = next((kw.value for kw in node.keywords if kw.arg == 'reason'), None)
            if reason is None and node.args:
                reason = node.args[0]
            self._resolve(node, reason)
        self.generic_visit(node)

    def _resolve(self, node, reason):
        if isinstance(reason, ast.Constant) and isinstance(reason.value, str):
            self.raised.add(reason.value)
            return
        handed_on = (
            isinstance(reason, ast.Attribute)
            and reason.attr == 'reason'
            and isinstance(reason.value, ast.Name)
        )
        if handed_on and any(kw.arg == 'missing' for kw in node.keywords):
            self.raised.update(_missing_part_reasons())
        elif handed_on and reason.value.id in self.interlocks[-1]:
            self.raised.update(INTERLOCK_REASONS)
        else:
            text = ast.unparse(reason) if reason is not None else '(none)'
            self.unread.append(f'{self.rel}:{node.lineno} reason={text}')


def _raised(type_name):
    raised, unread = set(), []
    for rel, tree in ast_seams.iter_package_modules(('modules',)):
        _RaisedReasons(rel, type_name, raised, unread).visit(tree)
    return raised, unread


@pytest.mark.parametrize('cls', [LiveFolderPathRefusedError, HardwareCommandRefusedError])
def test_the_raise_sites_and_the_table_agree(cls):
    raised, unread = _raised(cls.__name__)
    assert raised, f'found no {cls.__name__} raise sites; the census is broken'
    assert unread == [], (
        f'{cls.__name__} reasons this census cannot read: {unread}. Pass the reason as a '
        "literal, a part's (missing=part), or a MotionInterlockError's."
    )
    assert raised - set(cls.causes) == set(), 'raised with no cause declared'
    assert set(cls.causes) - raised == set(), 'declared, and nothing raises it'


def test_the_drivers_interlock_reasons_are_the_tables():
    raised, unread = set(), []
    for rel, tree in ast_seams.iter_package_modules(('drivers',)):
        _RaisedReasons(rel, 'MotionInterlockError', raised, unread).visit(tree)
    assert unread == [], unread
    assert raised == INTERLOCK_REASONS
    assert set(HardwareCommandRefusedError.causes) >= INTERLOCK_REASONS


def test_a_stale_stop_carries_a_reason_a_title_and_a_cause():
    from modules.exceptions import RunAlreadyEndedError

    error = RunAlreadyEndedError('That run has already ended; no run is live.')
    assert error.reason == 'run_already_ended'
    assert error.title == 'Run Already Ended'
    assert error.cause == RefusalCause.REQUEST
