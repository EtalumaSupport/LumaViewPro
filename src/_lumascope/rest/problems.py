# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every answer that is not a result is an RFC 9457 problem.

``application/problem+json``, with ``type`` (``urn:lumascope:problem:<reason>``,
or ``about:blank`` for an outcome that declares no reason), ``title``,
``detail``, ``status``, ``instance`` (``urn:uuid:<request id>``), and the
members a client branches on: ``kind`` (``OutcomeKind``'s value),
``reason``, ``cause`` (``RefusalCause``'s value) and ``remedy``. Beside
them, a member's outcome carries each field its type publishes
(``@api_fields``), such as the name of a refused argument, so a client
corrects its request without reading the words.

A member's outcome is read as every host reads it
(``notification_center.outcome_of``), so a wire client is refused in the
words and with the reason a Python caller is. A refusal or quiet outcome is
409 when the same request can succeed later, 422 when the request must
change, and 503 with no ``Retry-After`` when this server cannot do it again
until the application restarts, as its type declares (``RefusalCause``), so
a client retries only a 409; one that declares no cause (a by-contract
cancel) is 409. A fault is 500.

The server's own answers carry their own reasons, kind ``refusal``:
``ServerRefusedError``'s, each explained on that class, each sent with the
status ``_SERVER_STATUS`` gives it rather than its cause's.
"""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import math

from fastapi.responses import JSONResponse

from _lumascope.modules import notification_center, wall_time, wire_encoding
from _lumascope.lib.lumascope_error import Fault, LumascopeError, defined_classes
from _lumascope.modules.activity_claim import ActivityClaim
from _lumascope.modules.api_surface import Admission, raises_of
from _lumascope.modules.exceptions import (
    BlockingCallRefusedError,
    RefusalCause,
    ServerRefusedError,
    UnfitArgument,
)
from _lumascope.modules.notification_center import OutcomeKind, outcome_of
from _lumascope.modules.post_processing_api import PostProcessingAPI
from _lumascope.modules.sequential_io_executor import SequentialIOExecutor
from _lumascope.rest.routes import wire_form

MEDIA_TYPE = 'application/problem+json'

# What every route declares in the OpenAPI description for an answer that is
# not a result: each is a problem, whatever its status, published once.
# Ranges, since which reasons a route can answer is the member's, not the
# route's.
_SCHEMA = {'$ref': '#/components/schemas/Problem'}


def openapi_responses(
    json_ceiling_bytes: int,
    upload_ceiling_bytes: int,
    body_floor_s: float,
    body_floor_bytes_per_s: int,
) -> dict[int | str, dict]:
    """The problem ranges every route declares, naming the body ceilings and the body deadline a server refuses past."""
    return {
        code: {'description': words, 'content': {MEDIA_TYPE: {'schema': _SCHEMA}}}
        for code, words in (
            (
                '4XX',
                "Refused: 409 when the same request can succeed later (the scope's state refused it, or a stop ended it), 422 when the request must change; 404, 405, 408, 413, 415 for the request itself "
                f'(413: a body larger than its route takes, {json_ceiling_bytes} bytes for JSON, {upload_ceiling_bytes} for a file upload; '
                f'408: a body that did not arrive within {body_floor_s:g} s plus its length at {body_floor_bytes_per_s} bytes a second, its connection closed). '
                'Beside the members above, a problem carries the fields its type publishes, such as `argument`, and `errors` for a body that does not fit.',
            ),
            (
                '5XX',
                'A fault (500); or 503: with `Retry-After` when asking again later may succeed (the server overloaded, no frame yet, every upload place held), without it when this server cannot do it again until the application restarts (the scope disconnected, the motion monitor ended, the session or the server closing).',
            ),
        )
    }


TYPE_PREFIX = 'urn:lumascope:problem:'

# The function each door is, whose ``Raises:`` states the door's refusals.
DOORS: dict[Admission, object] = {
    Admission.LANE_CALL: SequentialIOExecutor.call,
    Admission.LANE_OVERRIDE: SequentialIOExecutor.call_overriding,
    Admission.LANE_SUBMIT: SequentialIOExecutor.submit,
    Admission.GUARDED_WRITE: ActivityClaim.enter_guarded_write,
    Admission.POST_PROCESSING: PostProcessingAPI._refuse_while_busy,
}

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
# What each of the server's own refusals answers: its own status rather than
# its cause's, since a server too busy says 503 and when to ask again where a
# member's refusal for the scope's state says 409.
_SERVER_STATUS = {
    'not_found': 404,
    'invalid_request': 422,
    'method_not_allowed': 405,
    'unsupported_media_type': 415,
    'body_too_large': 413,
    'body_too_slow': 408,
    'handle_shared': 422,
    'overloaded': 503,
    'session_closing': 503,
    'no_frame_yet': 503,
    'not_cancelable': 409,
    'idempotency_key_reused': 422,
    'client_not_allowed': 403,
    'wrong_host': 421,
    'foreign_origin': 403,
}
# The server's own refusals whose answer ends its connection: a body the
# server stopped reading, and a client it does not serve, which has no use
# for a connection. ``Connection`` is the connection's, not the answer's:
# neither reason is ever a job's stored answer.
_ENDS_CONNECTION = frozenset({'body_too_slow', 'client_not_allowed'})


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


# The outcome classes no route answers. Every REST call runs its member on
# a thread of its own (``JobRegistry.run``): never the UI thread, an
# executor's worker or the GUI's inline boundary, the only threads a
# blocking call is refused on.
NOT_ON_THE_WIRE = frozenset({BlockingCallRefusedError.__name__})


def declared(
    doc: str,
    classes: dict[str, type[LumascopeError]],
    admitted_by: tuple[Admission, ...] = (),
    *,
    on_the_wire: bool,
) -> dict[int, dict]:
    """The responses a member's ``Raises:`` section and the doors it enters declare, for its route's description.

    The entries are the member's own and those of each door in
    ``admitted_by`` (``DOORS``). Each class an entry names is described
    under the status each of its reasons answers -- a refusal's cause per
    reason, a fault 500 -- with the reasons the entry quotes, or every
    reason the class can carry when it quotes none. Each status carries
    ``x-outcomes``, class to reasons. ``on_the_wire``: what a route answers,
    which leaves out ``NOT_ON_THE_WIRE``; False is what an in-process
    caller can meet.

    Raises:
        KeyError: An entry names a class that is not an outcome class.
    """
    entries = raises_of(doc)
    for door in admitted_by:
        entries += raises_of(inspect.getdoc(DOORS[door]) or '')
    by_status: dict[int, dict[str, list[str]]] = {}
    for name, quoted in entries:
        if on_the_wire and name in NOT_ON_THE_WIRE:
            continue
        cls = classes[name]
        for reason in quoted or _reasons_of(cls) or ('',):
            said = by_status.setdefault(_status_of(cls, reason), {}).setdefault(name, [])
            if reason and reason not in said:
                said.append(reason)
    return {
        status: {
            'description': '; '.join(
                f'`{name}`' + (f': {", ".join(f"`{r}`" for r in reasons)}' if reasons else '')
                for name, reasons in outcomes.items()
            ),
            'content': {MEDIA_TYPE: {'schema': _SCHEMA}},
            'x-outcomes': outcomes,
        }
        for status, outcomes in sorted(by_status.items())
    }


def _reasons_of(cls: type) -> tuple[str, ...]:
    """Every reason *cls* can carry: its ``causes``; none for an umbrella."""
    return tuple(getattr(cls, 'causes', ()))


def _status_of(cls: type, reason: str) -> int:
    """What *cls* raised with *reason* answers, as ``refused_by_server`` or ``answered_by_member`` decides it."""
    if issubclass(cls, ServerRefusedError):
        return _SERVER_STATUS[reason]
    if issubclass(cls, Fault):
        return _FAULT_STATUS
    return _CAUSE_STATUS[getattr(cls, 'causes', {}).get(reason)]


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
    ended: datetime.datetime = dataclasses.field(default_factory=wall_time.now)

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


def body_too_large(ceiling_bytes: int) -> ServerRefusedError:
    """A body larger than its route takes, refused before more of it than *ceiling_bytes* is held."""
    return ServerRefusedError(
        'body_too_large', f'This route takes a body of at most {ceiling_bytes} bytes.'
    )


def body_too_slow(deadline_s: float, arrived_bytes: int) -> ServerRefusedError:
    """A body that did not arrive within its deadline: the server stopped reading it, and closes its connection."""
    return ServerRefusedError(
        'body_too_slow',
        f'The body did not arrive within {deadline_s:g} s; {arrived_bytes} bytes of it had.',
    )


def overloaded(detail: str, retry_after_s: float) -> ServerRefusedError:
    """More is held than the server keeps -- calls, jobs, handles or uploads; asking again later may succeed."""
    return ServerRefusedError('overloaded', detail, retry_after_s=retry_after_s)


def session_closing() -> ServerRefusedError:
    """A member asked while the server closes, its session's close's first step: it will not be answered again."""
    return ServerRefusedError(
        'session_closing',
        'The server is closing: it finishes what is under way, then exits.',
    )


def client_not_allowed(peer: str | None) -> ServerRefusedError:
    """A client at an address the server's allowlist does not name."""
    return ServerRefusedError(
        'client_not_allowed',
        f'This scope serves only the clients its rest_api.allowed_clients names, and {peer} '
        'is not one of them. Add its address at the scope PC, with LumaViewPro closed.',
    )


def wrong_host(host: str | None, served: str) -> ServerRefusedError:
    """A request whose ``Host`` names another server than this one."""
    return ServerRefusedError(
        'wrong_host',
        f'This server answers only requests addressed to {served}, and this one was addressed '
        f'to {host!r}. Address the scope by its IP address and port, not by a name.',
    )


def foreign_origin(origin: str) -> ServerRefusedError:
    """A request a page from another site sent."""
    return ServerRefusedError(
        'foreign_origin',
        f'This server answers no request a page from another site sends, and this one came '
        f'from {origin!r}. Call the API from a program, or from a page this server serves.',
    )


def no_frame_yet(retry_after_s: float) -> ServerRefusedError:
    """A live view with no frame to show: the camera sent none in time; asking again later may succeed."""
    return ServerRefusedError(
        'no_frame_yet',
        'The camera has sent no frame to show yet; it may not be streaming.',
        retry_after_s=retry_after_s,
    )


def refused_by_server(refusal: ServerRefusedError, request_id: str) -> Answer:
    """The problem for a request the server answered itself.

    Its wait, when it has one, is sent as ``Retry-After`` in whole seconds
    (RFC 9110's form), rounded up so a client never asks before it; a
    reason that ends its connection is sent with ``Connection: close``.
    """
    status = _SERVER_STATUS[refusal.reason]
    body = {
        'type': TYPE_PREFIX + refusal.reason,
        **wire_encoding.outcome_fields(refusal, wire_form),
        'status': status,
        'instance': f'urn:uuid:{request_id}',
    }
    headers = {}
    if refusal.retry_after_s is not None:
        headers['Retry-After'] = str(math.ceil(refusal.retry_after_s))
    if refusal.reason in _ENDS_CONNECTION:
        headers['Connection'] = 'close'
    return Answer(status, body, headers)


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
