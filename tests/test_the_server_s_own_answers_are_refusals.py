# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the REST server answers itself is a refusal, with the cause of each reason.

A request the server declines before it reaches a member -- a handle it does
not hold, a body that does not fit, a job still running, more held than it
keeps -- is a refusal: nothing broke. Its cause says whether the same request
may succeed later, read where every refusal's is read (``outcome_of``), while
the server keeps answering each with its own status.
"""

import pytest

from modules.exceptions import RefusalCause
from modules.notification_center import OutcomeKind, outcome_of
from rest import problems

REQUEST, STATE = RefusalCause.REQUEST, RefusalCause.STATE


@pytest.mark.parametrize(
    ('refusal', 'status', 'cause'),
    [
        (problems.not_found('x'), 404, REQUEST),
        (problems.invalid_request('x'), 422, REQUEST),
        (problems.method_not_allowed('x'), 405, REQUEST),
        (problems.unsupported_media_type('x'), 415, REQUEST),
        (problems.overloaded('x', 1), 503, STATE),
        (problems.server_closing(), 503, STATE),
        (problems.no_frame_yet(1), 503, STATE),
    ],
    ids=lambda v: v.reason if isinstance(v, problems.ServerRefusedError) else None,
)
def test_each_answer_is_a_refusal_with_its_cause_and_keeps_its_status(refusal, status, cause):
    outcome = outcome_of(refusal)
    assert (outcome.kind, outcome.cause) == (OutcomeKind.REFUSAL, cause)
    assert problems.refused_by_server(refusal, 'id').status == status


def test_each_reason_s_status_and_title_are_the_type_s():
    """A raise names the reason; its status and heading are the class's, one per reason."""
    expected = {
        'not_found': (404, 'Not Found'),
        'invalid_request': (422, 'Invalid Request'),
        'method_not_allowed': (405, 'Method Not Allowed'),
        'unsupported_media_type': (415, 'Unsupported Media Type'),
        'handle_shared': (409, 'Handle Shared'),
        'overloaded': (503, 'Overloaded'),
        'server_closing': (503, 'Server Closing'),
        'no_frame_yet': (503, 'No Frame Yet'),
        'not_cancellable': (409, 'Not Cancellable'),
    }
    assert set(expected) == set(problems.ServerRefusedError.causes)
    answered = {}
    for reason in expected:
        refusal = problems.ServerRefusedError(reason, 'detail')
        answered[reason] = (problems.refused_by_server(refusal, 'id').status, refusal.title)
    assert answered == expected


def test_a_reason_with_no_declared_cause_fails_where_it_is_raised():
    with pytest.raises(TypeError, match='no_such_reason'):
        problems.ServerRefusedError('no_such_reason', 'detail')
