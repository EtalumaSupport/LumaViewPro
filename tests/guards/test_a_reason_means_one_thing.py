# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A reason means one thing wherever it is carried, checked where each outcome class is defined.

A client branches on the reason alone: a REST problem's type is the bare
string, so two classes carrying one string under two kinds or two causes
answered one URN with two statuses, and nothing that read one class at a
time could see it. Every outcome class declares its reasons as ``causes``,
each mapped to its cause, and the root (``_lumascope.lib.lumascope_error``) checks the
declaration against every outcome class Python holds when the class is
defined: one kind, one cause and one title per string, and a string
another class carries is the same fact only when the class names that class
in ``same_fact_as``; a class that states no title has none to compare. An
outcome built with a reason its class does not declare fails where it is
built, whichever class builds it.

Two record vocabularies carry reasons outside any class: the run's endings
(``run_outcome.ENDING_KINDS``, each with its kind) and the bring-up record's
causes (``bring_up.CAUSE_PHRASES``, why a part is not up). A string in
either that a class also carries is carried with the record's kind: a
failed run's ending and a part's cause by a fault, a stopped run's ending by
a notice or a quiet outcome, and a completed run's by no class; and a class
carrying a run's ending states the ending's heading as its title.

Each class a test defines lives until it is collected, and a live one is a
carrier to every later definition; the fixture collects them after each test.
"""

import gc
from typing import ClassVar

import pytest

from _lumascope.lib.lumascope_error import (
    Fault,
    LumascopeError,
    RefusalCause,
    _title_of,
    _Umbrella,
)
from _lumascope.modules.exceptions import (
    CellCountScaleDroppedNotice,
    CompositeFailedError,
    FrameHandlerRemovedError,
    HardwareCommandRefusedError,
    HomingFaultError,
    MoveStoppedError,
    Notice,
    ProtocolRunRefusedError,
    Quiet,
    Refusal,
    SessionClosingError,
    SettingRefusedError,
)
from _lumascope.modules.lumascope_api.bring_up import CAUSE_PHRASES
from _lumascope.modules.path_utils import CaptureLocationError
from _lumascope.modules.plugins import PluginRegistrationError
from _lumascope.modules.run_outcome import ENDING_KINDS, ENDING_TITLES, EndingKind
from _lumascope.rest.problems import outcome_classes
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
        titles: ClassVar[dict[str, str]] = {
            **ProtocolRunRefusedError.titles,
            'zz_extended': 'Extended',
        }

    assert _ExtendedRunRefusedError.causes['exclusive_activity_running'] == RefusalCause.STATE


def test_a_reason_another_class_carries_under_another_title_is_refused():
    with pytest.raises(TypeError, match='one title'):

        class _OtherTitleBusyError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'exclusive_activity_running': RefusalCause.STATE
            }
            same_fact_as = (HardwareCommandRefusedError,)
            title = 'Another Activity Running'


def test_a_class_reading_the_carriers_title_is_defined():
    class _SameTitleBusyError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {
            'exclusive_activity_running': RefusalCause.STATE
        }
        same_fact_as = (HardwareCommandRefusedError,)
        title = ProtocolRunRefusedError.titles['exclusive_activity_running']

    assert (
        _SameTitleBusyError().title
        == HardwareCommandRefusedError.titles['exclusive_activity_running']
    )


def test_a_subclass_that_inherits_its_parents_reason_under_another_title_is_refused():
    class _TitledParentError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_inherited': RefusalCause.STATE}
        title = 'Parent Title'

    with pytest.raises(TypeError, match='one title'):

        class _RetitledChildError(_TitledParentError):
            title = 'Child Title'


def test_a_subclass_that_restates_its_parents_reason_under_another_title_is_refused():
    class _TitledParentError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_restated_title': RefusalCause.STATE}
        title = 'Parent Title'

    with pytest.raises(TypeError, match='one title'):

        class _RestatingChildError(_TitledParentError):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'zz_restated_title': RefusalCause.STATE
            }
            title = 'Child Title'


def test_a_title_read_from_the_instance_is_refused():
    with pytest.raises(TypeError, match='property'):

        class _PropertyTitleError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_property': RefusalCause.STATE}

            @property
            def title(self):
                return 'Chosen Per Raise'


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


def test_a_subject_states_no_title():
    """A subject is never raised, so a title on it is read only by a child
    that lost its own, in front of the child's ``titles``."""
    with pytest.raises(TypeError, match='states title'):

        class _TitledSubjectError(_Umbrella, Exception):
            title = 'Subject'


def test_a_title_comes_from_the_class_s_own_titles_for_every_reason():
    class _TitledError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {
            'zz_titled_a': RefusalCause.STATE,
            'zz_titled_b': RefusalCause.REQUEST,
        }
        titles: ClassVar[dict[str, str]] = {'zz_titled_a': 'A', 'zz_titled_b': 'B'}

        def __init__(self, reason):
            super().__init__()
            self.reason = reason

    assert (_TitledError('zz_titled_a').title, _TitledError('zz_titled_b').title) == ('A', 'B')


def test_titles_that_miss_a_reason_are_refused_where_the_class_is_defined():
    with pytest.raises(TypeError, match='no row for zz_untitled'):

        class _ShortTitlesError(Refusal, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {
                'zz_titled_c': RefusalCause.STATE,
                'zz_untitled': RefusalCause.STATE,
            }
            titles: ClassVar[dict[str, str]] = {'zz_titled_c': 'C'}


def test_a_class_with_titles_reads_no_other_title():
    """A body title, or one inherited from a parent, stands in front of the
    root's read of ``titles``, and every reason would show it."""
    with pytest.raises(TypeError, match=r'its title is _BodyTitleError\.title'):

        class _BodyTitleError(Fault, Exception):
            causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_body_title': None}
            titles: ClassVar[dict[str, str]] = {'zz_body_title': 'Row'}
            title = 'Body'

    class _HeadedError(Fault, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'zz_headed': None}
        title = 'Parent'

    with pytest.raises(TypeError, match=r'its title is _HeadedError\.title'):

        class _HeadedChildError(_HeadedError):
            titles: ClassVar[dict[str, str]] = {'zz_headed': 'Row'}


def test_a_title_is_never_set_on_an_outcome():
    """The title is the type's, so the place that raises cannot choose another."""
    refusal = ProtocolRunRefusedError('already_running', 'A run is live.')
    with pytest.raises(AttributeError, match="title is its type's"):
        refusal.title = 'Chosen at the raise'
    assert refusal.title == ProtocolRunRefusedError.titles['already_running']


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
        'n',
        reason,
        budget_ms=1,
        drop_k=1,
        first_error='ValueError: e' if reason == 'raised' else None,
    ),
    'CompositeFailedError': lambda reason: CompositeFailedError('m', reason),
}


def test_a_reason_the_class_does_not_declare_is_refused_where_it_is_built():
    def check(name):
        with pytest.raises(TypeError, match='declares no reason'):
            _BUILT_WITH[name]('zz_undeclared')

    failed = failing_cases(sorted(_BUILT_WITH), check)
    assert not failed, failed


def test_a_class_with_one_reason_answers_it_from_the_class_itself():
    assert CellCountScaleDroppedNotice.reason == 'cell_count_scale_dropped'
    assert CellCountScaleDroppedNotice.cause is None
    assert SessionClosingError.cause == RefusalCause.RESTART
    # Which of several reasons is the instance's.
    with pytest.raises(AttributeError):
        MoveStoppedError.reason  # noqa: B018 -- the read is the test


def test_no_outcome_class_carries_a_status():
    """An outcome is what happened; how HTTP sends it is ``_lumascope.rest.problems``'s. A
    member's refusal answers the status its cause says, and the server's own
    refusals the status ``_lumascope.rest.problems`` gives each reason."""
    own = [cls.__qualname__ for cls in RAISABLE if {'_STATUSES', 'status'} & set(vars(cls))]
    assert own == [], f'an outcome class carrying a status: {own}'


# The kinds that may carry a record's string, by what the record says it is.
_KINDS_OF_ENDING = {
    EndingKind.FAILURE: (Fault,),
    EndingKind.STOP: (Notice, Quiet),
    EndingKind.COMPLETION: (),
}
_KINDS_OF_PART_CAUSE = (Fault,)


def _carried_with_another_kind(kinds_of: dict[str, tuple[type, ...]], classes) -> list[str]:
    """Each class that carries a record's string with a kind the record does not allow."""
    return [
        f'{cls.__name__} carries {reason!r}, which the record allows only on '
        f'{[kind.__name__ for kind in kinds] or "no class"}'
        for reason, kinds in kinds_of.items()
        for cls in classes
        if reason in (getattr(cls, 'causes', None) or {}) and not issubclass(cls, kinds or ())
    ]


def _endings_kinds():
    return {str(reason): _KINDS_OF_ENDING[kind] for reason, kind in ENDING_KINDS.items()}


def _part_causes_kinds():
    return dict.fromkeys(map(str, CAUSE_PHRASES), _KINDS_OF_PART_CAUSE)


def test_a_run_ending_a_class_carries_is_carried_with_the_ending_s_kind():
    assert _carried_with_another_kind(_endings_kinds(), RAISABLE) == []


def test_a_class_carrying_a_run_ending_states_the_ending_s_heading():
    """The record is no class, so the definition check cannot compare with it."""
    wrong = [
        f'{cls.__name__} titles {reason!r} {_title_of(cls, reason)!r}, '
        f'the ending {ENDING_TITLES[reason]!r}'
        for cls in RAISABLE
        for reason in getattr(cls, 'causes', None) or {}
        if reason in ENDING_TITLES and _title_of(cls, reason) not in (None, ENDING_TITLES[reason])
    ]
    assert wrong == []


def test_a_bring_up_cause_a_class_carries_is_carried_as_a_fault():
    assert _carried_with_another_kind(_part_causes_kinds(), RAISABLE) == []


def test_a_record_string_carried_with_another_kind_is_named():
    # Strings no class carries, so the definition check has nothing to refuse.
    class _ShutdownFaultError(Fault, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {'shutdown': None}

    class _StreamRefusedError(Refusal, Exception):
        causes: ClassVar[dict[str, RefusalCause | None]] = {
            'stream_not_started': RefusalCause.STATE
        }

    planted = (_ShutdownFaultError, _StreamRefusedError)
    assert _carried_with_another_kind(_endings_kinds(), planted) == [
        "_ShutdownFaultError carries 'shutdown', which the record allows only on "
        "['Notice', 'Quiet']"
    ]
    assert _carried_with_another_kind(_part_causes_kinds(), planted) == [
        "_StreamRefusedError carries 'stream_not_started', which the record allows only on "
        "['Fault']"
    ]
