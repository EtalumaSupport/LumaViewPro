# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A reason means one thing wherever it is carried, checked where each outcome class is defined.

A client branches on the reason alone: a REST problem's type is the bare
string, so two classes carrying one string under two kinds or two causes
answered one URN with two statuses, and nothing that read one class at a
time could see it. Every outcome class declares its reasons as ``causes``,
each mapped to its cause, and the root (``lib.lumascope_error``) checks the
declaration against every outcome class Python holds when the class is
defined: one kind and one cause per string, and a string another class
carries is the same fact only when the class names that class in
``same_fact_as``. An outcome built with a reason its class does not declare
fails where it is built, whichever class builds it.

Each class a test defines lives until it is collected, and a live one is a
carrier to every later definition; the fixture collects them after each test.
"""

import gc
from typing import ClassVar

import pytest

from lib.lumascope_error import Fault, LumascopeError, RefusalCause, _Umbrella
from modules.exceptions import (
    CameraSettingOutOfRangeError,
    CompositeFailedError,
    FrameHandlerRemovedError,
    HardwareCommandRefusedError,
    HomingFaultError,
    MoveStoppedError,
    NoHardwareDetectedNotice,
    Notice,
    ProtocolRunRefusedError,
    Quiet,
    Refusal,
    SessionClosingError,
    SettingRefusedError,
)
from modules.path_utils import CaptureLocationError
from modules.plugins import PluginRegistrationError
from rest.problems import ServerRefusedError, outcome_classes
from tests.case_walk import failing_cases
from tests.guards.test_every_outcome_joins_the_root import RAISABLE


@pytest.fixture(autouse=True)
def _collected():
    yield
    gc.collect()


def test_each_kind_takes_only_its_causes():
    with pytest.raises(TypeError, match='does not take'):

        class _RefusedWithNoCauseError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_refused': None}

    with pytest.raises(TypeError, match='does not take'):

        class _QuietWithNoCauseError(Quiet, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_quiet': None}

    with pytest.raises(TypeError, match='does not take'):

        class _FaultWithACauseError(Fault, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_fault': RefusalCause.STATE}

    class _RaisedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_raised_notice': RefusalCause.STATE}

    class _ReportedNotice(Notice, Exception):  # noqa: N818 -- a notice, not an error
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_reported_notice': None}

    assert _RaisedNotice().cause == RefusalCause.STATE
    assert _ReportedNotice().cause is None


def test_an_outcome_class_that_declares_no_reasons_is_refused():
    with pytest.raises(TypeError, match='declares no reasons'):

        class _NoReasonsError(Fault, Exception):
            pass


def test_a_reason_or_cause_stated_in_the_class_body_is_refused():
    with pytest.raises(TypeError, match='states reason'):

        class _StatedReasonError(Fault, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_stated': None}
            reason = 'zz_stated'

    with pytest.raises(TypeError, match='states cause'):

        class _StatedCauseError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'zz_stated_cause': RefusalCause.STATE
            }
            cause = RefusalCause.STATE


def test_a_string_carried_with_a_second_kind_or_cause_is_refused():
    # 'exclusive_activity_running' is a refusal for the scope's state.
    with pytest.raises(TypeError, match='one string, one meaning'):

        class _BusyFaultError(Fault, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'exclusive_activity_running': None}
            same_fact_as = (HardwareCommandRefusedError,)

    with pytest.raises(TypeError, match='one string, one meaning'):

        class _BusyRequestError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'exclusive_activity_running': RefusalCause.REQUEST
            }
            same_fact_as = (HardwareCommandRefusedError,)


def test_a_string_another_class_carries_is_taken_only_as_the_same_fact():
    with pytest.raises(TypeError, match='same_fact_as'):

        class _UnmarkedBusyError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'exclusive_activity_running': RefusalCause.STATE
            }

    class _MarkedBusyError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {
            'exclusive_activity_running': RefusalCause.STATE
        }
        same_fact_as = (HardwareCommandRefusedError,)

    assert _MarkedBusyError().cause == RefusalCause.STATE


def test_a_subclass_carries_its_parents_reasons_as_the_same_fact():
    class _ParentError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_parent': RefusalCause.REQUEST}

    class _ChildError(_ParentError):
        causes: ClassVar[dict[str, RefusalCause | None]] = {
            'zz_parent': RefusalCause.REQUEST,
            'zz_child': RefusalCause.STATE,
        }

        def __init__(self, reason):
            super().__init__()
            self.reason = reason

    class _InheritingError(_ParentError):
        pass

    assert _ChildError('zz_child').cause == RefusalCause.STATE
    assert _InheritingError().reason == 'zz_parent'
    assert _InheritingError().cause == RefusalCause.REQUEST


def test_a_subclass_that_restates_a_parents_reason_with_another_cause_is_refused():
    class _ParentError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_restated': RefusalCause.STATE}

    with pytest.raises(TypeError, match='zz_restated'):

        class _RestatingError(_ParentError):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_restated': RefusalCause.REQUEST}


def test_a_subclass_extends_its_parents_reasons_whoever_else_carries_them():
    # ProtocolRunRefusedError carries reasons other classes carry as the same fact.
    class _ExtendedRunRefusedError(ProtocolRunRefusedError):
        causes: ClassVar[dict[str, RefusalCause | None]] = {
            **ProtocolRunRefusedError.causes,
            'zz_extended': RefusalCause.STATE,
        }

    assert _ExtendedRunRefusedError.causes['exclusive_activity_running'] == RefusalCause.STATE


def test_an_outcome_class_of_two_kinds_is_refused():
    with pytest.raises(TypeError, match='two kinds'):

        class _RefusedFaultError(Refusal, Fault, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_two_kinds': RefusalCause.STATE}


def test_a_class_on_the_root_that_is_no_kind_is_refused():
    with pytest.raises(TypeError, match='has no kind'):

        class _KindlessError(LumascopeError, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_kindless': None}


def test_a_subject_carries_no_reasons():
    with pytest.raises(TypeError, match='subject'):

        class _SubjectWithReasonsError(_Umbrella, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_subject': None}


def test_a_class_nothing_holds_leaves_no_trace_once_collected():
    def define():
        class _GoneError(Fault, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_gone': None}

        return _GoneError

    gone = define()
    del gone
    gc.collect()

    class _AfterError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_gone': RefusalCause.STATE}

    assert _AfterError().reason == 'zz_gone'


def test_a_class_refused_where_it_is_defined_is_no_carrier():
    # Uncollected: a refused class lives in its base's subclasses until a
    # collection, which is what the definition check must not depend on.
    gc.disable()
    try:
        with pytest.raises(TypeError, match='one string, one meaning'):

            class _RefusedBusyError(Fault, Exception):
                causes: ClassVar[dict[str, RefusalCause | None]] = {
                    'exclusive_activity_running': None
                }
                same_fact_as = (HardwareCommandRefusedError,)

        class _BusyAfterError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'exclusive_activity_running': RefusalCause.STATE
            }
            same_fact_as = (HardwareCommandRefusedError,)

        assert _BusyAfterError().cause == RefusalCause.STATE
        assert '_RefusedBusyError' not in outcome_classes()
    finally:
        gc.enable()


# One builder per class that accepted any reason before it declared its own.
_BUILT_WITH = {
    'SettingRefusedError': lambda reason: SettingRefusedError(reason, 'a.b', 'd'),
    'CaptureLocationError': lambda reason: CaptureLocationError(reason, 'm'),
    'PluginRegistrationError': lambda reason: PluginRegistrationError(reason, 'm'),
    'HomingFaultError': lambda reason: HomingFaultError('ALL', reason, ()),
    'FrameHandlerRemovedError': lambda reason: FrameHandlerRemovedError(
        'n', reason, budget_ms=1, drop_k=1
    ),
    'CompositeFailedError': lambda reason: CompositeFailedError('m', reason),
    'CameraSettingOutOfRangeError': lambda reason: CameraSettingOutOfRangeError(
        reason, 1.0, 0.0, 2.0, message='m'
    ),
}


def test_a_reason_the_class_does_not_declare_is_refused_where_it_is_built():
    def check(name):
        with pytest.raises(TypeError, match='declares no reason'):
            _BUILT_WITH[name]('zz_undeclared')

    failed = failing_cases(sorted(_BUILT_WITH), check)
    assert not failed, failed


def test_a_class_with_one_reason_answers_it_from_the_class_itself():
    assert NoHardwareDetectedNotice.reason == 'no_hardware'
    assert NoHardwareDetectedNotice.cause is None
    assert SessionClosingError.cause == RefusalCause.RESTART
    # Which of several reasons is the instance's.
    with pytest.raises(AttributeError):
        MoveStoppedError.reason  # noqa: B018 -- the read is the test


def test_only_the_server_s_own_refusal_answers_a_status_per_reason():
    """A member's refusal answers the status its cause says; a status chosen per
    reason belongs to the refusals HTTP defines (``ServerRefusedError``)."""
    assert ServerRefusedError in RAISABLE
    own = [
        cls.__qualname__
        for cls in RAISABLE
        if cls is not ServerRefusedError and {'_STATUSES', 'status'} & set(vars(cls))
    ]
    assert own == [], f'a status per reason outside ServerRefusedError: {own}'
