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
409 when the same request can succeed later, 422 when the request must
change, and 503 with no ``Retry-After`` when this server cannot do it again
until the application restarts, as its type declares (``RefusalCause``), so
a client retries only a 409; one that declares no cause (a by-contract
cancel) is 409. A fault is 500.

The server's own answers carry their own reasons, kind ``refusal``:
``invalid_request`` 422, ``not_found`` 404, ``method_not_allowed`` 405,
``unsupported_media_type`` 415, ``overloaded`` 503, ``server_closing`` 503
for a member asked while the server closes, ``no_frame_yet`` 503 for a
live view the camera has sent no frame to, ``handle_shared`` 422 for an
id every client shares, and ``not_cancellable`` 409 for a stop asked of a
job that holds no operation.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import ClassVar, NamedTuple

from fastapi.responses import JSONResponse

from modules import notification_center, wire_encoding
from lib.lumascope_error import Fault, LumascopeError, defined_classes
from modules.api_surface import api_fields, raises_of
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
            "Refused: 409 when the same request can succeed later (the scope's state refused it, or a stop ended it), 422 when the request must change; 404, 405, 415 for the request itself. Beside the members above, a problem carries the fields its type publishes, such as `argument`, and `errors` for a body that does not fit.",
        ),
        (
            '5XX',
            'A fault (500); or 503: with `Retry-After` when asking again later may succeed (the server overloaded, no frame yet), without it when this server cannot do it again until the application restarts (the scope disconnected, the motion monitor ended, the session or the server closing).',
        ),
    )
}
TYPE_PREFIX = 'urn:lumascope:problem:'

# What each cause of a refusal, quiet outcome or raised notice answers: a
# notice ends a request that nothing failed, as a stopped move does. A
# restart's 503 carries no Retry-After, since no wait makes it succeed.
_CAUSE_STATUS = {
    RefusalCause.STATE: 409,
    RefusalCause.REQUEST: 422,
    RefusalCause.RESTART: 503,
    None: 409,
}
# What a fault answers.
_FAULT_STATUS = 500


def outcome_classes() -> dict[str, type[LumascopeError]]:
    """Every outcome class defined so far, by name: what a ``Raises:`` entry names.

    Read when the server is built, once the API's modules are imported.

    Raises:
        TypeError: Two classes share a name, so an entry naming it is ambiguous.
    """
    found: dict[str, type[LumascopeError]] = {}
    for cls in defined_classes():
        if found.setdefault(cls.__name__, cls) is not cls:
            raise TypeError(f'two outcome classes are named {cls.__name__}')
    return found


def declared(doc: str, classes: dict[str, type[LumascopeError]]) -> dict[int, dict]:
    """The responses a member's ``Raises:`` section declares, for its route's description.

    Each class an entry names is described under the status each of its
    reasons answers -- a refusal's cause per reason, a fault 500 -- with
    the reasons the entry quotes, or every reason the class can carry when
    it quotes none. Each status carries ``x-refusals``, class to reasons.

    Raises:
        KeyError: An entry names a class that is not an outcome class.
    """
    by_status: dict[int, dict[str, list[str]]] = {}
    for name, quoted in raises_of(doc):
        cls = classes[name]
        for reason in quoted or _reasons_of(cls) or ('',):
            said = by_status.setdefault(_status_of(cls, reason), {}).setdefault(name, [])
            if reason and reason not in said:
                said.append(reason)
    return {
        status: {
            'description': '; '.join(
                f'`{name}`' + (f': {", ".join(f"`{r}`" for r in reasons)}' if reasons else '')
                for name, reasons in refusals.items()
            ),
            'content': {MEDIA_TYPE: {'schema': _SCHEMA}},
            'x-refusals': refusals,
        }
        for status, refusals in sorted(by_status.items())
    }


def _reasons_of(cls: type) -> tuple[str, ...]:
    """Every reason *cls* can carry: its ``causes``; none for an umbrella."""
    return tuple(getattr(cls, 'causes', ()))


def _status_of(cls: type, reason: str) -> int:
    """What *cls* raised with *reason* answers, as ``answered_by_member`` decides it."""
    if issubclass(cls, Fault):
        return _FAULT_STATUS
    return _CAUSE_STATUS[getattr(cls, 'causes', {}).get(reason)]


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


@api_fields('loc', 'msg', 'type')
class UnfitArgument(NamedTuple):
    """One argument of a request that does not fit its member, as the request reader found it."""

    # Where the argument is: ``body``, ``query`` or ``path``, then its name
    # and, inside it, the key or index.
    loc: tuple[str | int, ...]
    # Why it does not fit, in the reader's words.
    msg: str
    # The reader's code for why.
    type: str


@api_fields('errors')
class ServerRefusedError(Refusal, Exception):
    """A request the server itself answers: it never reached a member.

    A refusal, answered with its own status rather than its cause's: a
    server too busy says 503 and when to ask again, where a member's refusal
    for the scope's state says 409. The status and the headers are how the
    answer is sent, so they are this module's to read, not the problem's.

    Attributes:
        reason: The machine-readable code.
        title: The heading, the reason's.
        errors: For ``'invalid_request'`` from the request reader, each
            argument that does not fit; None otherwise.
    """

    errors: tuple[UnfitArgument, ...] | None

    causes: ClassVar[dict[str, RefusalCause]] = {
        'not_found': RefusalCause.REQUEST,
        'invalid_request': RefusalCause.REQUEST,
        'method_not_allowed': RefusalCause.REQUEST,
        'unsupported_media_type': RefusalCause.REQUEST,
        'handle_shared': RefusalCause.REQUEST,
        'overloaded': RefusalCause.STATE,
        'server_closing': RefusalCause.RESTART,
        'no_frame_yet': RefusalCause.STATE,
        'not_cancellable': RefusalCause.STATE,
        'idempotency_key_reused': RefusalCause.REQUEST,
    }

    _STATUSES: ClassVar[dict[str, int]] = {
        'not_found': 404,
        'invalid_request': 422,
        'method_not_allowed': 405,
        'unsupported_media_type': 415,
        'handle_shared': 422,
        'overloaded': 503,
        'server_closing': 503,
        'no_frame_yet': 503,
        'not_cancellable': 409,
        'idempotency_key_reused': 422,
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
        'not_cancellable': 'Not Cancellable',
        'idempotency_key_reused': 'Idempotency Key Reused',
    }

    def __init__(
        self,
        reason: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
        errors: tuple[UnfitArgument, ...] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.title = self._TITLES[reason]
        self.errors = errors
        self._status = self._STATUSES[reason]
        self._headers = headers or {}


def not_found(detail: str) -> ServerRefusedError:
    """A route, a handle or a job that is not there."""
    return ServerRefusedError('not_found', detail)


def invalid_request(
    detail: str, errors: tuple[UnfitArgument, ...] | None = None
) -> ServerRefusedError:
    """A request whose arguments do not fit the member."""
    return ServerRefusedError('invalid_request', detail, errors=errors)


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
    body = {
        'type': TYPE_PREFIX + refusal.reason,
        **wire_encoding.outcome_fields(refusal, wire_form),
        'status': refusal._status,
        'instance': f'urn:uuid:{request_id}',
    }
    return Answer(refusal._status, body, dict(refusal._headers))


def answered_by_member(exception: Exception, request_id: str) -> Answer:
    """The problem for a member's outcome, reported once as every answered outcome is.

    Logged and shown to nobody: the problem is the answer, given to the
    client that asked. The log line carries the request's id, the one the
    problem's ``instance`` names, so a fault's developer words, which the
    client is not sent, are found from the client's answer.
    """
    notification_center.notifications.report_outcome(
        exception, solicited=True, category=f'REST {request_id}', log_only=True
    )
    outcome = outcome_of(exception)
    if outcome.kind in (OutcomeKind.REFUSAL, OutcomeKind.QUIET, OutcomeKind.NOTICE):
        status = _CAUSE_STATUS[outcome.cause]
    else:
        status = _FAULT_STATUS
    body = {
        'type': TYPE_PREFIX + outcome.reason if outcome.reason else 'about:blank',
        **wire_encoding.outcome_fields(exception, wire_form),
        'status': status,
        'instance': f'urn:uuid:{request_id}',
    }
    return Answer(status, body)
