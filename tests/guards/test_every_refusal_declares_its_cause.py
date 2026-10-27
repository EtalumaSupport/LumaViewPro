# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every refusal declares whether retrying it can help, and a reason no cause answers is never built.

A REST server answers a refusal 422 when the request as sent cannot succeed
on this scope and 409 when the scope's state refused it, and a polling
client retries only the second. Nothing on a refusal said which, so a
server would have decided per exception, and a client retrying on a 409
would retry a bad argument for ever. Each type now maps every reason it
carries to its cause in ``causes``, checked where the type is defined
(``tests/guards/test_a_reason_means_one_thing.py``), and refuses to be
built with a reason its table lacks.

The tables are held to the raise sites both ways: for
``ProtocolRunRefusedError`` by ``test_every_runner_refusal_reason_is_covered``
(``tests/test_run_refusal_contract.py``), for the other two mixed types here.
``HardwareCommandRefusedError``'s reasons arrive from four modules and a
driver, so a reason handed on from a part or from the stage's interlock is
resolved to the set its source declares: a part's own, and for an
interlock the refusal before motion its ``REFUSED_BEFORE_MOTION`` names.
The drivers' interlock literals are held to the interlocks' one list
(``INTERLOCK_TITLES``), whose keys the driver's fault carries.
"""

import ast
import importlib

import pytest

from lib.stage_interlock import INTERLOCK_TITLES
from modules.exceptions import (
    INTERLOCK_REASONS,
    AlreadyStartedError,
    BlockingCallRefusedError,
    CameraDriverRepairRefusedError,
    ControlSessionRefusedError,
    HardwareCommandRefusedError,
    LiveFolderPathRefusedError,
    MissingPart,
    PostProcessingRefusedError,
    ProtocolRunRefusedError,
    Quiet,
    Refusal,
    RefusalCause,
)
from tests import ast_seams
from tests.guards.test_every_refusal_states_its_reason import _base_name, _classes
from tests.case_walk import failing_cases
from tests.stock_channels import LED2


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
        module = importlib.import_module(ast_seams.module_name(rel))
        found[name] = getattr(module, name)
    return found


# How to build each type whose reasons differ in cause with a given reason.
_FACTORIES = {
    'AlreadyStartedError': lambda reason: AlreadyStartedError(reason, 'm'),
    'CameraDriverRepairRefusedError': CameraDriverRepairRefusedError,
    'ControlSessionRefusedError': ControlSessionRefusedError,
    'LiveFolderPathRefusedError': lambda reason: LiveFolderPathRefusedError(reason, 'n', 'm'),
    'ProtocolRunRefusedError': lambda reason: ProtocolRunRefusedError(reason, 'm'),
    'HardwareCommandRefusedError': lambda reason: HardwareCommandRefusedError(reason, 'member'),
    'BlockingCallRefusedError': lambda reason: BlockingCallRefusedError(reason, 'member'),
    'PostProcessingRefusedError': lambda reason: PostProcessingRefusedError(
        operation='Stitch', reason=reason, message='m'
    ),
}


def test_the_walk_finds_both_kinds():
    # The instrument on known positives: one refusal of one reason, one
    # built through a factory, and the quiet type.
    refusals = _types_mixing_in(Refusal)
    for name in ('StepNotFoundError', 'ProtocolScheduleRefusedError', *_FACTORIES):
        assert name in refusals, name
    assert 'RunWaitExpiredError' in _types_mixing_in(Quiet)


def test_a_refusal_is_not_built_with_a_reason_it_declares_no_cause_for():
    def check(name):
        with pytest.raises(TypeError, match="declares no reason 'no_such_reason'"):
            _FACTORIES[name]('no_such_reason')

    failed = failing_cases(sorted(_FACTORIES), check)
    assert not failed, failed


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
    parts += [MissingPart.led(LED2), MissingPart.led_address(1)]
    return {p.reason for p in parts}


class _RaisedReasons(ast.NodeVisitor):
    """The reasons a module builds ``type_name`` with, a handed-on reason resolved to its source's set.

    A reason is the ``reason`` keyword or the first argument. A name's
    ``.reason`` resolves to the missing parts' reasons when the call passes
    ``missing=``. When the name is bound to a ``MotionInterlockError`` (an
    ``except ... as`` or an annotated parameter), its ``.reason`` resolves to
    the interlocks' own list, and ``REFUSED_BEFORE_MOTION[name.reason]`` to
    ``INTERLOCK_REASONS``. Anything else is unread, and fails the census.
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
        before_motion = (
            isinstance(reason, ast.Subscript)
            and _base_name(reason.value) == 'REFUSED_BEFORE_MOTION'
            and isinstance(reason.slice, ast.Attribute)
            and reason.slice.attr == 'reason'
            and isinstance(reason.slice.value, ast.Name)
            and reason.slice.value.id in self.interlocks[-1]
        )
        if handed_on and any(kw.arg == 'missing' for kw in node.keywords):
            self.raised.update(_missing_part_reasons())
        elif handed_on and reason.value.id in self.interlocks[-1]:
            self.raised.update(INTERLOCK_TITLES)
        elif before_motion:
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
    assert raised == set(INTERLOCK_TITLES)
    assert set(HardwareCommandRefusedError.causes) >= INTERLOCK_REASONS
