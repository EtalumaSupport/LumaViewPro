# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal's status says what a client does next, read from its type's cause.

409 when the same request can succeed later, 422 when the request must
change, 503 with no ``Retry-After`` when this server cannot do it again until
the application restarts (``RefusalCause``). The reasons here are the ones
whose answer is not the scope's passing state: an ended control session, a
shared handle, a file already at the name and a camera driver already usable
are answered again unchanged however long a client waits; a disconnected
scope, an ended motion monitor and a closing session or server are answered
so until the process restarts; a job that holds no operation is forgotten by
the same ``DELETE`` once it has ended.
"""

from __future__ import annotations

from modules.exceptions import (
    CameraDriverRepairRefusedError,
    ControlSessionRefusedError,
    HardwareCommandRefusedError,
    LiveFolderPathRefusedError,
    ServerRefusedError,
    SessionClosingError,
)
from rest import problems
from rest.problems import answered_by_member, refused_by_server

_ID = '00000000-0000-0000-0000-000000000000'


def test_each_reason_answers_the_status_its_cause_says():
    members = {
        'control_session_ended': (ControlSessionRefusedError('control_session_ended'), 422),
        'file_exists': (LiveFolderPathRefusedError('file_exists', 'a.txt', 'x'), 422),
        'driver_already_usable': (CameraDriverRepairRefusedError('driver_already_usable'), 422),
        'scope_disconnected': (HardwareCommandRefusedError('scope_disconnected', 'led_on'), 503),
        'monitor_ended': (HardwareCommandRefusedError('monitor_ended', 'move_z'), 503),
        'session_closing': (SessionClosingError('run'), 503),
    }
    server = {
        'handle_shared': (ServerRefusedError('handle_shared', 'x'), 422),
        'server_closing': (problems.server_closing(), 503),
        'not_cancellable': (ServerRefusedError('not_cancellable', 'x'), 409),
    }

    answered = {}
    for reason, (refusal, _) in members.items():
        answer = answered_by_member(refusal, _ID)
        answered[reason] = (answer.status, answer.body['reason'], 'Retry-After' in answer.headers)
    for reason, (refusal, _) in server.items():
        answer = refused_by_server(refusal, _ID)
        answered[reason] = (answer.status, answer.body['reason'], 'Retry-After' in answer.headers)

    assert answered == {
        reason: (status, reason, False) for reason, (_, status) in {**members, **server}.items()
    }
