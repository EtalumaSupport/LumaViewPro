# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every answer that is not a result is an RFC 9457 problem.

``application/problem+json``, with ``type`` (``urn:lumascope:problem:<reason>``,
or ``about:blank`` for an outcome that declares no reason), ``title``,
``detail``, ``status``, ``instance`` (``urn:uuid:<request id>``), and the
members a client branches on: ``kind`` (``OutcomeKind``'s value),
``reason`` and ``remedy``. Beside them, a member's outcome carries each
field its type publishes (``@api_fields``), such as the name of a refused
argument, so a client corrects its request without reading the words.

A member's outcome is read as every host reads it
(``notification_center.outcome_of``), so a wire client is refused in the
words and with the reason a Python caller is. A refusal or quiet outcome is
422 when the request as sent cannot succeed on this scope and 409 when the
scope's state refused it, as its type declares (``RefusalCause``), so a
client retries only a 409; one that declares no cause (a by-contract
cancel) is 409. A fault is 500.

The server's own answers carry their own reasons, kind ``refusal``:
``invalid_request`` 422, ``not_found`` 404, ``method_not_allowed`` 405,
``unsupported_media_type`` 415, ``overloaded`` 503, ``server_closing`` 503
for a member asked while the server closes, ``no_frame_yet`` 503 for a
live view the camera has sent no frame to, and ``handle_shared`` 409 for
an id every client shares.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import ClassVar

from fastapi.responses import JSONResponse

from modules import notification_center
from modules.api_surface import fields_of
from modules.exceptions import Refusal, RefusalCause
from modules.notification_center import OutcomeKind, outcome_of
from rest.routes import wire_form

MEDIA_TYPE = 'application/problem+json'

# What every route declares in the OpenAPI description for an answer that is
# not a result: each is a problem, whatever its status. Ranges, since which
# reasons a route can answer is the member's, not the route's.
_SCHEMA = {
    'type': 'object',
    'required': ['type', 'title', 'detail', 'status', 'instance', 'kind', 'reason', 'remedy'],
    'properties': {
        'type': {'type': 'string', 'description': 'urn:lumascope:problem:<reason>, or about:blank'},
        'title': {'type': 'string'},
        'detail': {'type': 'string'},
        'status': {'type': 'integer'},
        'instance': {'type': 'string', 'description': 'urn:uuid:<request id>'},
        'kind': {'type': 'string', 'enum': [kind.value for kind in OutcomeKind]},
        'reason': {'type': ['string', 'null'], 'description': 'What a client branches on'},
        'remedy': {'type': ['object', 'null']},
    },
    'additionalProperties': True,
}
OPENAPI_RESPONSES: dict[int | str, dict] = {
    code: {'description': words, 'content': {MEDIA_TYPE: {'schema': _SCHEMA}}}
    for code, words in (
        (
            '4XX',
            'Refused: 422 when the request cannot succeed on this scope, 409 when its state refused it or a stop ended it; 404, 405, 415 for the request itself. Beside the members above, a problem carries the fields its type publishes, such as `argument`, and `errors` for a body that does not fit.',
        ),
        (
            '5XX',
            'A fault (500), or the server overloaded or closing (503, `Retry-After` when asking again may succeed).',
        ),
    )
}
TYPE_PREFIX = 'urn:lumascope:problem:'

# What each cause of a refusal, quiet outcome or raised notice answers: a
# notice ends a request that nothing failed, as a stopped move does.
_CAUSE_STATUS = {RefusalCause.REQUEST: 422, RefusalCause.STATE: 409, None: 409}
# What a fault answers.
_FAULT_STATUS = 500


def now() -> datetime.datetime:
    """The local time with its offset, as answers and jobs are stamped."""
    return datetime.datetime.now().astimezone()


@dataclasses.dataclass(frozen=True)
class Answer:
    """What a request is answered: a result, or a problem.

    Made where the call ends, on the call's own thread, so a call that
    outlives the client's wait is held as a job with its answer already
    decided, and read the same way whenever it is read.

    Attributes:
        status: The HTTP status.
        body: The JSON-ready body: the encoded result, or the problem.
        headers: Headers the answer carries, such as ``Retry-After``.
        ended: When the call ended: when its answer was made. A job's end
            time is this, so a job read as finished always has one.
    """

    status: int
    body: object
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    ended: datetime.datetime = dataclasses.field(default_factory=now)

    @property
    def is_problem(self) -> bool:
        """Whether the answer is a problem rather than a result."""
        return self.status >= 400

    def response(self) -> JSONResponse:
        """The answer as the HTTP response."""
        media_type = MEDIA_TYPE if self.is_problem else 'application/json'
        return JSONResponse(
            self.body, status_code=self.status, media_type=media_type, headers=self.headers
        )


def result(value: object) -> Answer:
    """A call's encoded result."""
    return Answer(200, value)


class ServerRefusedError(Refusal, Exception):
    """A request the server itself answers: it never reached a member.

    A refusal, answered with its own status rather than its cause's: a
    server too busy says 503 and when to ask again, where a member's refusal
    for the scope's state says 409.

    Attributes:
        status: The HTTP status, the reason's.
        reason: The machine-readable code.
        title: The heading, the reason's.
        headers: Headers the answer carries, such as ``Retry-After``.
        extra: Further members of the problem, such as ``errors``.
    """

    causes: ClassVar[dict[str, RefusalCause]] = {
        'not_found': RefusalCause.REQUEST,
        'invalid_request': RefusalCause.REQUEST,
        'method_not_allowed': RefusalCause.REQUEST,
        'unsupported_media_type': RefusalCause.REQUEST,
        'handle_shared': RefusalCause.REQUEST,
        'overloaded': RefusalCause.STATE,
        'server_closing': RefusalCause.STATE,
        'no_frame_yet': RefusalCause.STATE,
        'job_running': RefusalCause.STATE,
    }

    _STATUSES: ClassVar[dict[str, int]] = {
        'not_found': 404,
        'invalid_request': 422,
        'method_not_allowed': 405,
        'unsupported_media_type': 415,
        'handle_shared': 409,
        'overloaded': 503,
        'server_closing': 503,
        'no_frame_yet': 503,
        'job_running': 409,
    }

    _TITLES: ClassVar[dict[str, str]] = {
        'not_found': 'Not Found',
        'invalid_request': 'Invalid Request',
        'method_not_allowed': 'Method Not Allowed',
        'unsupported_media_type': 'Unsupported Media Type',
        'handle_shared': 'Handle Shared',
        'overloaded': 'Overloaded',
        'server_closing': 'Server Closing',
        'no_frame_yet': 'No Frame Yet',
        'job_running': 'Job Running',
    }

    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = self._declared(reason)
        self.status = self._STATUSES[reason]
        self.title = self._TITLES[reason]
        self.headers = headers or {}
        self.extra = extra or {}


def not_found(detail: str) -> ServerRefusedError:
    """A route, a handle or a job that is not there."""
    return ServerRefusedError('not_found', detail)


def invalid_request(detail: str, **extra: object) -> ServerRefusedError:
    """A request whose arguments do not fit the member."""
    return ServerRefusedError('invalid_request', detail, extra=extra)


def method_not_allowed(detail: str) -> ServerRefusedError:
    """A route asked with a method it does not answer."""
    return ServerRefusedError('method_not_allowed', detail)


def unsupported_media_type(detail: str) -> ServerRefusedError:
    """A body that is not JSON."""
    return ServerRefusedError('unsupported_media_type', detail)


def overloaded(detail: str, retry_after_s: int) -> ServerRefusedError:
    """More is held than the server keeps; asking again later may succeed."""
    return ServerRefusedError('overloaded', detail, headers={'Retry-After': str(retry_after_s)})


def server_closing() -> ServerRefusedError:
    """A member asked while the server closes: it will not be answered again."""
    return ServerRefusedError(
        'server_closing',
        'The server is closing: it finishes what is under way, then exits.',
    )


def no_frame_yet(retry_after_s: int) -> ServerRefusedError:
    """A live view with no frame to show: the camera sent none in time; asking again later may succeed."""
    return ServerRefusedError(
        'no_frame_yet',
        'The camera has sent no frame to show yet; it may not be streaming.',
        headers={'Retry-After': str(retry_after_s)},
    )


def refused_by_server(refusal: ServerRefusedError, request_id: str) -> Answer:
    """The problem for a request the server answered itself."""
    body = _problem(
        type_=TYPE_PREFIX + refusal.reason,
        title=refusal.title,
        detail=str(refusal),
        status=refusal.status,
        request_id=request_id,
        kind=OutcomeKind.REFUSAL,
        reason=refusal.reason,
        remedy=None,
    )
    return Answer(refusal.status, {**body, **refusal.extra}, dict(refusal.headers))


def answered_by_member(exception: Exception, request_id: str) -> Answer:
    """The problem for a member's outcome, reported once as every answered outcome is.

    Logged and shown to nobody: the problem is the answer, given to the
    client that asked.
    """
    notification_center.notifications.report_outcome(
        exception, solicited=True, category='REST', log_only=True
    )
    outcome = outcome_of(exception)
    if outcome.kind in (OutcomeKind.REFUSAL, OutcomeKind.QUIET, OutcomeKind.NOTICE):
        status = _CAUSE_STATUS[outcome.cause]
    else:
        status = _FAULT_STATUS
    body = _problem(
        type_=TYPE_PREFIX + outcome.reason if outcome.reason else 'about:blank',
        title=outcome.title or type(exception).__name__,
        detail=outcome.words,
        status=status,
        request_id=request_id,
        kind=outcome.kind,
        reason=outcome.reason or None,
        remedy=wire_form(outcome.remedy),
    )
    return Answer(status, {**_published(exception), **body})


def _published(exception: Exception) -> dict[str, object]:
    """Each field *exception*'s type publishes, in its wire form."""
    found: dict[str, object] = {}
    for klass in type(exception).__mro__:
        for name in fields_of(klass):
            if name not in found:
                found[name] = wire_form(getattr(exception, name))
    return found


def _problem(
    *,
    type_: str,
    title: str,
    detail: str,
    status: int,
    request_id: str,
    kind: OutcomeKind,
    reason: str | None,
    remedy: dict | None,
) -> dict[str, object]:
    return {
        'type': type_,
        'title': title,
        'detail': detail,
        'status': status,
        'instance': f'urn:uuid:{request_id}',
        'kind': kind.value,
        'reason': reason,
        'remedy': remedy,
    }
