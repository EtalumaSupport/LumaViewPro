# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every answer that is not a result is an RFC 9457 problem.

``application/problem+json``, with ``type`` (``urn:lumascope:problem:<reason>``,
or ``about:blank`` for an outcome that declares no reason), ``title``,
``detail``, ``status``, ``instance`` (``urn:uuid:<request id>``), and the
members a client branches on: ``kind`` (``OutcomeKind``'s value),
``reason`` and ``remedy``.

A member's outcome is read as every host reads it
(``notification_center.outcome_of``), so a wire client is refused in the
words and with the reason a Python caller is. Every refusal is 409 and a
fault 500 until each refusal declares whether the request or the scope's
state refused it.

The server's own answers carry their own reasons, kind ``refusal``:
``invalid_request`` 422, ``not_found`` 404, ``method_not_allowed`` 405,
``unsupported_media_type`` 415, ``overloaded`` 503, and ``handle_shared``
409 for an id every client shares.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from modules import notification_center
from modules.notification_center import OutcomeKind, outcome_of
from rest.routes import wire_form

MEDIA_TYPE = 'application/problem+json'
TYPE_PREFIX = 'urn:lumascope:problem:'

# What each kind of outcome a member raises answers. A notice is reported,
# never raised, so one raised is a defect and answers as a fault does.
_STATUS = {
    OutcomeKind.REFUSAL: 409,
    OutcomeKind.QUIET: 409,
    OutcomeKind.FAULT: 500,
    OutcomeKind.NOTICE: 500,
}


class ServerRefusedError(Exception):
    """A request the server itself answers: it never reached a member.

    Attributes:
        status: The HTTP status.
        reason: The machine-readable code.
        title: The heading.
        headers: Headers the answer carries, such as ``Retry-After``.
        extra: Further members of the problem, such as ``errors``.
    """

    def __init__(
        self,
        status: int,
        reason: str,
        title: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.title = title
        self.headers = headers or {}
        self.extra = extra or {}


def not_found(detail: str) -> ServerRefusedError:
    """A route, a handle or a job that is not there."""
    return ServerRefusedError(404, 'not_found', 'Not Found', detail)


def invalid_request(detail: str, **extra: object) -> ServerRefusedError:
    """A request whose arguments do not fit the member."""
    return ServerRefusedError(422, 'invalid_request', 'Invalid Request', detail, extra=extra)


def method_not_allowed(detail: str) -> ServerRefusedError:
    """A route asked with a method it does not answer."""
    return ServerRefusedError(405, 'method_not_allowed', 'Method Not Allowed', detail)


def unsupported_media_type(detail: str) -> ServerRefusedError:
    """A body that is not JSON."""
    return ServerRefusedError(415, 'unsupported_media_type', 'Unsupported Media Type', detail)


def overloaded(detail: str, retry_after_s: int) -> ServerRefusedError:
    """More is held than the server keeps; asking again later may succeed."""
    return ServerRefusedError(
        503, 'overloaded', 'Overloaded', detail, headers={'Retry-After': str(retry_after_s)}
    )


def refused_by_server(refusal: ServerRefusedError, request_id: str) -> JSONResponse:
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
    return JSONResponse(
        {**body, **refusal.extra},
        status_code=refusal.status,
        media_type=MEDIA_TYPE,
        headers=refusal.headers,
    )


def answered_by_member(exception: Exception, request_id: str) -> JSONResponse:
    """The problem for a member's outcome, reported once as every answered outcome is.

    Logged and shown to nobody: the problem is the answer, given to the
    client that asked.
    """
    notification_center.notifications.report_outcome(
        exception, solicited=True, category='REST', log_only=True
    )
    outcome = outcome_of(exception)
    status = _STATUS[outcome.kind]
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
    return JSONResponse(body, status_code=status, media_type=MEDIA_TYPE)


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
