# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The application a server serves: one route per wire member of a session.

Every call runs on a thread of its own and the route awaits it, so a call
that waits on the scope holds no worker another request needs: a shared
pool of threads let long waits hold a stop for seconds. A call's
arguments are decoded, the member reached and called, and its answer
encoded on that thread, since each may read the scope or the disk.

The Session's members are at ``/api/v1/<member>``; a live object a client
was handed has its members at ``/api/v1/handles/<type>/<id>/<member>``
(``_lumascope.rest.handles``). A call that outlives what its client will wait is a
job (``_lumascope.rest.jobs``), every answer that is not a result is a problem
(``_lumascope.rest.problems``), what the scope tells its listeners is one event
stream (``_lumascope.rest.events``), and a file in the live folder is downloaded by
its name there (``/api/v1/files/<name>``).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import pathlib
import time
import typing
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import fastapi
import fastapi.routing
import numpy as np
import pydantic
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from _lumascope.lib.real_numbers import whole_number_in
from _lumascope.modules import image_utils, wall_time, wire_encoding
from _lumascope.modules.api_surface import api, markdown_of
from _lumascope.modules.exceptions import (
    ConfigurationDefectError,
    Quiet,
    RefusalCause,
    ServerRefusedError,
    UnfitArgument,
)
from _lumascope.modules.lumascope_api.frame_record import CapturedStill
from _lumascope.modules.operation import Operation
from _lumascope.modules.protocol import PROTOCOL_FILE_CEILING_BYTES
from _lumascope.modules.protocol_runner import ProtocolRunner
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest import idempotency
from _lumascope.rest import live as live_view
from _lumascope.rest import problems
from _lumascope.rest import events as event_stream
from _lumascope.rest.events import EventStream
from _lumascope.rest.handles import HandleRegistry
from _lumascope.rest.jobs import RETRY_AFTER_S as JOB_RETRY_AFTER_S
from _lumascope.rest.jobs import Job as HeldJob
from _lumascope.rest.jobs import JobRegistry, Progress, Wait
from _lumascope.rest.problems import Answer
from _lumascope.rest.routes import (
    ANSWER_CONFIG,
    Handle,
    Job,
    PathAnswer,
    Route,
    body_of,
    job_answer,
    routes,
    sent_of_query,
)
from _lumascope.rest.served import Served, judged

# The REST log: a child of 'LVP', the main log's logger, whose REST handler
# keeps only the records marked api_request.
_request_log = logging.getLogger('LVP.rest')

VERSION = 'v1'
PREFIX = f'/api/{VERSION}'
# The request header naming the control session a call acts in
# (``ScopeSession.take_control``).
CONTROL_SESSION_HEADER = 'control-session'
# The first path segments the server answers itself; a Session member of
# one of these names would be shadowed, so one fails the build.
SERVER_SEGMENTS = frozenset(
    {'handles', 'jobs', 'events', 'files', 'previews', 'live', 'live.jpg', 'still'}
)
# Where a running job is read, and how soon: every 202 a call answers.
_JOB_HEADERS = {
    'Location': {'$ref': '#/components/headers/Location'},
    'Retry-After': {'$ref': '#/components/headers/Retry-After'},
}
# What any member answers when its call outlives what the client will wait.
_ANSWERED_AS_JOB = {
    'model': Job,
    'description': (
        'A running job, at `Location`: the operation the call started, or the '
        'call itself when it outlived the wait (`Prefer: wait`).'
    ),
    'headers': _JOB_HEADERS,
}
# The parameters more than one route takes, each declared once.
_PARAMETERS = {
    'ControlSession': {
        'name': 'Control-Session',
        'in': 'header',
        'required': False,
        'schema': {'type': 'string'},
        'description': (
            'The id of the control session the write is made in (`take_control`). '
            'While a session is held, a write that does not name it is refused 409 '
            '`control_session_held`.'
        ),
    },
    'MaxWidth': {
        'name': 'max_width',
        'in': 'query',
        'required': False,
        'schema': {'type': 'integer', 'minimum': 1},
        'description': "The most pixels wide the image is; half the frame's width when not given.",
    },
}
_CONTROL_SESSION = {'$ref': '#/components/parameters/ControlSession'}
_MAX_WIDTH = {'parameters': [{'$ref': '#/components/parameters/MaxWidth'}]}
_HEADERS = {
    'Location': {
        'description': 'Where the job is read: `/api/v1/jobs/<id>`.',
        'schema': {'type': 'string'},
    },
    'Retry-After': {
        'description': 'The seconds to wait before reading the job again.',
        'schema': {'type': 'integer'},
    },
}
# What the description says of every route, before its routes.
_INFO = """\
The Lumascope API: each member of the session, and of a live object a call
hands out, at its own route. The guide's REST chapter says how they are used.

- **Problems.** An answer that is not a result is a `Problem`
  (`application/problem+json`): 409 when the same request can succeed later,
  422 when it must change, 500 for a fault, 503 when the server cannot do it
  now. A client branches on `reason`, and retries a 409, or a 503 that
  carries `Retry-After`.
- **Jobs.** A call still running when the client stops waiting answers 202
  with its job, read at `Location` (`/api/v1/jobs/<id>`). `Prefer: wait=<s>`
  says how long to wait, 2 s when not said, at most 60;
  `Prefer: respond-async` waits for nothing. A preference taken is echoed in
  `Preference-Applied`.
- **Retries.** A call (`POST` or `PUT`) sent again with the same
  `Idempotency-Key` runs nothing and is answered its first answer; a still
  is taken again.
- **Control.** While a control session is held, a write names it in
  `Control-Session`.
- **Bodies.** A body carries only the fields its record names: an unknown
  one, nested included, is refused 422, so an answer that has gained a field
  is not sent back as a body unchanged.
- **Requests.** Every answer carries `X-Request-ID`, the id its log line and
  a problem's `instance` name.
"""


class Versions(pydantic.BaseModel):
    """The API versions a server answers."""

    model_config = ANSWER_CONFIG
    versions: list[str]


def _not_json(media_type: str, description: str, schema: dict | None = None) -> dict:
    """The ``add_api_route`` arguments declaring a route's answer as *media_type*, not JSON."""
    return {
        'response_class': fastapi.Response,
        'responses': {
            200: {
                'description': description,
                'content': {media_type: {'schema': schema or {'type': 'string'}}},
            }
        },
    }


class Closing:
    """The server's close, as the application takes part in it (``_lumascope.rest.server``).

    Once it has begun, a member asked, or a live view asked for, is refused
    ``session_closing``; jobs, handles, files and the stream are still
    served, so a client can read what the close finishes, and a live view
    already open keeps showing frames until the close finishes.
    """

    def __init__(self, stream: EventStream, jobs: JobRegistry, live: live_view.LiveView) -> None:
        self._stream = stream
        self._jobs = jobs
        self._live = live
        self.begun = False
        # Settled when the close finishes, so a client still waiting is answered then.
        self._finished: asyncio.Future[None] | None = None

    def begin(self) -> None:
        """Refuse every member asked from now on."""
        self.begun = True

    def finish(self) -> None:
        """Send ``closing`` on every stream and end it, end every live view, and answer every client waiting. Called on the server's loop."""
        self._stream.finish()
        self._live.finish()
        finished = self.finished()
        if not finished.done():
            finished.set_result(None)

    def finished(self) -> asyncio.Future[None]:
        """What a wait for a job also waits on: settled once the close has finished. Called on the server's loop."""
        if self._finished is None:
            self._finished = asyncio.get_running_loop().create_future()
        return self._finished

    def join_jobs(self, timeout_s: float) -> list[str]:
        """Wait up to *timeout_s* for the calls' threads; the names of those still running."""
        return self._jobs.join(timeout_s)


@api(in_process=True)
def build_app(session: ScopeSession, *, bound: bool = False) -> fastapi.FastAPI:
    """The application serving *session*'s wire members under ``/api/v1/``.

    The OpenAPI description is at ``/api/v1/openapi.json`` and the
    interactive reference at ``/docs``. The event stream hears the scope
    while the application is served, from its startup to its shutdown.
    Its close is ``app.state.closing``; its live view ``app.state.live``.

    When *session*'s ``rest_api`` lists ``allowed_clients``, a request from
    a client at any other address is refused ``client_not_allowed`` on every
    route, whoever serves the application; it is read from the peer alone,
    never from a header naming another.

    Args:
        session: The session the application serves, as
            ``ScopeSession.create`` brings it up; every route calls into it.
        bound: The application is served on ``rest_api``'s own address and
            port, as ``build_server`` serves it. A request whose ``Host``
            names another is then refused ``wrong_host``, and one a page
            from another site sent ``foreign_origin``, before any route
            sees it, so no web page the user opens can drive the scope. A
            host that mounts the application in its own server, or a test
            client, leaves it False: it is not served at that address.

    Raises:
        ConfigurationDefectError: ``'route_shadowed'``, a Session member is
            named as one of the server's own routes, so a client could not
            reach it.
        RestServerNotStartedError: A ``rest_api`` setting cannot be used, so
            the application would not know whom to admit.
    """
    rest_api = judged(session.setting('rest_api'))
    # The session's one protocol runner is every client's: its id is kept.
    registry = HandleRegistry(kept=lambda obj: isinstance(obj, ProtocolRunner))
    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    stream = EventStream(
        session, lambda value: _encoded_event(session, registry, value), handed_out
    )

    @contextlib.asynccontextmanager
    async def served(_app: fastapi.FastAPI) -> AsyncIterator[None]:
        stream.open(asyncio.get_running_loop())
        try:
            yield
        finally:
            stream.close()

    app = fastapi.FastAPI(
        title='Lumascope',
        version=session.app_version or 'unknown',
        description=_INFO,
        openapi_url=f'{PREFIX}/openapi.json',
        # Served from the installation (_add_reference_page): the default
        # page loads its script and style from a CDN, and is blank offline.
        docs_url=None,
        redoc_url=None,
        lifespan=served,
        responses=problems.openapi_responses(
            JSON_BODY_CEILING_BYTES,
            UPLOAD_CEILING_BYTES,
            BODY_DEADLINE_FLOOR_S,
            BODY_FLOOR_BYTES_PER_S,
        ),
    )
    # Before any route is added: every route is built of this class.
    app.router.route_class = _JsonBodyRoute

    async def versions() -> dict[str, list[str]]:
        """The API versions this server answers, each at ``/api/<version>/``."""
        return {'versions': [VERSION]}

    app.add_api_route(
        '/api',
        versions,
        methods=['GET'],
        operation_id='versions',
        **_words(versions.__doc__),
        tags=['server'],
        response_model=None,
        responses={200: {'model': Versions, 'description': 'The versions.'}},
    )
    app.state.served = rest_api
    # Innermost, so only an admitted client's request renews a session.
    app.middleware('http')(_renewing(session))
    if bound:
        app.middleware('http')(_addressed_to(rest_api))
    # Registered before the log, so it runs inside the request's id and log
    # line: a refused client's answer carries its id and is logged once.
    app.middleware('http')(_admitted_by(rest_api))
    app.middleware('http')(_identify_and_log)
    app.add_exception_handler(ServerRefusedError, _refused)
    app.add_exception_handler(_ClientGone, _client_gone)
    app.add_exception_handler(RequestValidationError, _body_does_not_fit)
    app.add_exception_handler(StarletteHTTPException, _routing_refused)
    app.add_exception_handler(Exception, _failed)

    jobs = JobRegistry(ended=stream.job_ended)
    keys = idempotency.IdempotencyKeys()
    # What each route's member says it raises, by the class it names.
    classes = problems.outcome_classes()
    live = live_view.LiveView(session)
    closing = Closing(stream, jobs, live)
    app.state.closing = closing
    app.state.live = live
    _add_handle_routes(app, session, registry, handed_out)
    _add_job_routes(app, session, jobs, closing)
    _add_event_route(app, stream, event_stream.published(session, handed_out))
    _add_live_routes(app, live, closing)
    _add_reference_page(app)
    _add_file_route(app, session)
    _add_upload_route(app, session, jobs, closing, keys)
    _add_still_route(app, session, jobs, closing)
    _add_preview_route(app, session)
    session_routes = routes(ScopeSession, handed_out=handed_out)
    shadowed = {r.path.split('/')[0] for r in session_routes} & SERVER_SEGMENTS
    if shadowed:
        raise ConfigurationDefectError(
            'route_shadowed',
            f"Session members {sorted(shadowed)} are named as the server's own routes",
        )
    for route in session_routes:
        _add(app, session, registry, jobs, stream, closing, keys, classes, route)
    for cls in sorted(handed_out, key=lambda c: c.__name__):
        for route in routes(cls, handed_out=handed_out):
            _add(app, session, registry, jobs, stream, closing, keys, classes, route)
    return app


def _encoded_event(session: ScopeSession, registry: HandleRegistry, value: object) -> object:
    """An event's value in its wire form; an event carries no running call."""

    def no_job(future: object) -> object:
        raise wire_encoding.NoWireFormError(type(future).__name__)

    return wire_encoding.encode(
        value, live_folder=session.setting('live_folder'), handle=registry.mint, job=no_job
    )


def _add_file_route(app: fastapi.FastAPI, session: ScopeSession) -> None:
    async def download(request: fastapi.Request, name: str) -> fastapi.Response:
        """A file in the live folder, by its name there: its type from its extension, as an attachment.

        The name is the one a listing and every returned path give, as path
        segments. A range of the file is answered on its own (``Range``).
        """
        _refuse_query(request)
        try:
            path = await asyncio.to_thread(session.live_folder_path, name)
        except Exception as e:
            # Refused as a Python caller is: the same door, the same words.
            return problems.answered_by_member(e, request.state.request_id).response()
        if not await asyncio.to_thread(path.is_file):
            raise problems.not_found(f'No file {name} is in the live folder.')
        return FileResponse(path, filename=path.name)

    app.add_api_route(
        f'{PREFIX}/files/{{name:path}}',
        download,
        methods=['GET'],
        tags=['files'],
        operation_id='files.download',
        **_words(download.__doc__),
        openapi_extra={
            'parameters': [
                {
                    'name': 'Range',
                    'in': 'header',
                    'required': False,
                    'schema': {'type': 'string'},
                    'description': 'The bytes of the file to send, answered 206 on their own.',
                }
            ],
            'responses': {
                '206': {
                    'description': 'The range of the file asked for.',
                    'content': {
                        'application/octet-stream': {
                            'schema': {'type': 'string', 'format': 'binary'}
                        }
                    },
                }
            },
        },
        **_not_json(
            'application/octet-stream',
            "The file's bytes, typed by its extension.",
            {'type': 'string', 'format': 'binary'},
        ),
    )


# Swagger UI's script, style and icon, unmodified from swagger-ui-dist at
# the version docs/THIRD_PARTY_NOTICES.md names, read beside this module.
_SWAGGER_UI = pathlib.Path(__file__).parent / 'swagger_ui'
_SWAGGER_UI_FILES = '/docs/files'
# Each typed here, not by the host's table, which on Windows is the
# registry's and can call a script text/plain.
_SWAGGER_UI_TYPES = {
    'swagger-ui-bundle.js': 'text/javascript',
    'swagger-ui.css': 'text/css',
    'favicon-32x32.png': 'image/png',
}


def _add_reference_page(app: fastapi.FastAPI) -> None:
    """``/docs``, the interactive reference, every file it loads served by this server.

    So it works on a computer with no internet. Swagger UI's online
    validator is turned off: its badge is an image from validator.swagger.io.
    """

    async def reference() -> fastapi.Response:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f'{app.title} - Swagger UI',
            swagger_js_url=f'{_SWAGGER_UI_FILES}/swagger-ui-bundle.js',
            swagger_css_url=f'{_SWAGGER_UI_FILES}/swagger-ui.css',
            swagger_favicon_url=f'{_SWAGGER_UI_FILES}/favicon-32x32.png',
            swagger_ui_parameters={'validatorUrl': None},
        )

    app.add_api_route('/docs', reference, methods=['GET'], include_in_schema=False)
    for name, media_type in _SWAGGER_UI_TYPES.items():
        app.add_api_route(
            f'{_SWAGGER_UI_FILES}/{name}',
            _served_file(_SWAGGER_UI / name, media_type),
            methods=['GET'],
            include_in_schema=False,
        )


def _served_file(path: pathlib.Path, media_type: str) -> Callable[[], Awaitable[fastapi.Response]]:
    """An endpoint answering *path*'s bytes as *media_type*; it takes nothing from the request."""

    async def file() -> fastapi.Response:
        return FileResponse(path, media_type=media_type)

    return file


def _add_upload_route(
    app: fastapi.FastAPI,
    session: ScopeSession,
    jobs: JobRegistry,
    closing: Closing,
    keys: idempotency.IdempotencyKeys,
) -> None:
    async def upload(request: fastapi.Request, name: str) -> fastapi.Response:
        """Write the request's body as a new file in the live folder, by its name there.

        ``ScopeSession.put_live_folder_file``, reached as every write is: on
        a call thread, as a REST client, in the control session the
        ``Control-Session`` header names. A file already at the name is
        refused, ``file_exists`` (422); it answers ``201`` with the file's
        path. Sent again with the same ``Idempotency-Key``, the same name and
        the same bytes, it writes nothing and is answered its first answer.
        """
        _refuse_query(request)
        if closing.begun:
            raise problems.session_closing()
        place = jobs.take_upload()
        try:
            data = await _body_of(request, UPLOAD_CEILING_BYTES)
            request_id = request.state.request_id
            control_id = request.headers.get(CONTROL_SESSION_HEADER)

            def work() -> object:
                with _acting_in(session, control_id):
                    path = session.put_live_folder_file(name, data)
                return wire_encoding.encode(
                    path,
                    live_folder=session.setting('live_folder'),
                    handle=_no_handle,
                    job=_no_handle,
                )

            async def put() -> fastapi.Response:
                reply = await jobs.answer(
                    jobs.run(
                        lambda: _answered(work, request_id),
                        'put_live_folder_file',
                        request_id,
                        member=request.url.path.removeprefix(f'{PREFIX}/'),
                        requested=request.state.requested,
                        progress=None,
                        place=place,
                    )
                )
                if not reply.is_problem:
                    reply = dataclasses.replace(reply, status=201)
                return reply.response()

            key = request.headers.get('idempotency-key')
            if key is None:
                return await put()
            # The body is held as its hash: a retry sends the same bytes.
            sent = {'sha256': hashlib.sha256(data).hexdigest()}
            return await keys.answer(key, idempotency.digest(request.url.path, sent), put)
        finally:
            # Nothing once the write has started: its thread gives the place back.
            place.give_back()

    app.add_api_route(
        f'{PREFIX}/files/{{name:path}}',
        upload,
        methods=['PUT'],
        tags=['files'],
        status_code=201,
        operation_id='files.upload',
        **_words(upload.__doc__),
        openapi_extra={
            'parameters': [_CONTROL_SESSION],
            'requestBody': {
                'required': True,
                'content': {
                    'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}
                },
            },
        },
        response_model=None,
        responses={201: {'model': PathAnswer, 'description': 'The file written.'}},
    )


# What a client sends a still with: ``capture_still``'s arguments but three.
# ``force_to_8bit``, since the route answers the camera's own depth;
# ``exclude_sources``, since a frame handed out to be measured never excludes
# a change; ``sum_iteration_callback``, a callback, which has no wire form.
_STILL_SENT = (
    'channel',
    'accept_dark',
    'all_ones_check',
    'timeout_s',
    'frames_summed',
    'sum_delay_s',
    'exposed_after_call',
)


def _add_still_route(
    app: fastapi.FastAPI, session: ScopeSession, jobs: JobRegistry, closing: Closing
) -> None:
    asked_by = body_of(inspect.unwrap(ScopeSession.capture_still), _STILL_SENT, 'StillAsked')

    async def still(request: fastapi.Request, asked: pydantic.BaseModel | None) -> fastapi.Response:
        """A still at the camera's own depth: its record, then its pixels.

        ``ScopeSession.capture_still`` with no 8-bit conversion and the
        arguments the body sends, each the member's own when left out, so
        with none the camera is taken as set. It is reached as every write
        is: on a call thread, as a REST client, in the control session the
        ``Control-Session`` header names. The body is the still's
        ``FrameRecord`` as UTF-8 JSON, then the image's bytes, row by row,
        little-endian, in the array's own type;
        ``X-Record-Length`` is the record's length in bytes, ``X-Image-Shape``
        the rows and columns, ``X-Image-Dtype`` the type and
        ``X-Significant-Bits`` the depth (``CapturedStill.significant_bits``).
        It is answered when the still is taken, however long that is: a job's
        answer is JSON and cannot carry the pixels. A refusal or a failure is
        the member's problem. An ``Idempotency-Key`` is not held: a still
        writes nothing, and asked again it takes another.
        """
        _refuse_query(request)
        if closing.begun:
            raise problems.session_closing()
        jobs.admit(returns_job=False)
        sent = asked.model_dump(include=asked.model_fields_set) if asked is not None else {}
        request_id = request.state.request_id
        control_id = request.headers.get(CONTROL_SESSION_HEADER)
        # The pixels travel beside the answer, which holds only the record.
        taken: list[CapturedStill] = []

        def work() -> object:
            with _acting_in(session, control_id):
                captured = session.capture_still(force_to_8bit=False, **sent)
            taken.append(captured)
            return wire_encoding.encode(
                captured.record,
                live_folder=session.setting('live_folder'),
                handle=_no_handle,
                job=_no_handle,
            )

        reply = await jobs.answer(
            jobs.run(
                lambda: _answered(work, request_id),
                'capture_still',
                request_id,
                member=request.url.path.removeprefix(f'{PREFIX}/'),
                requested=request.state.requested,
                progress=None,
            )
        )
        if reply.is_problem:
            return reply.response()
        image = taken[0].image
        record = json.dumps(reply.body).encode()
        pixels = np.ascontiguousarray(image, dtype=image.dtype.newbyteorder('<')).tobytes()
        return fastapi.Response(
            record + pixels,
            media_type='application/octet-stream',
            headers={
                'X-Record-Length': str(len(record)),
                'X-Image-Shape': ','.join(str(n) for n in image.shape),
                'X-Image-Dtype': image.dtype.name,
                'X-Significant-Bits': str(taken[0].significant_bits),
            },
        )

    # The model is built at run time, so FastAPI reads it from the signature.
    still.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                'request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=fastapi.Request
            ),
            inspect.Parameter(
                'asked',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=asked_by,
                default=fastapi.Body(None),
            ),
        ]
    )
    app.add_api_route(
        f'{PREFIX}/still',
        still,
        methods=['POST'],
        tags=['capture'],
        operation_id='still',
        **_words(still.__doc__),
        openapi_extra={
            'parameters': [_CONTROL_SESSION],
            'responses': {
                '200': {
                    'headers': {
                        'X-Record-Length': {
                            'description': "The record's length in bytes.",
                            'schema': {'type': 'integer'},
                        },
                        'X-Image-Shape': {
                            'description': 'The rows and columns, as `<rows>,<columns>`.',
                            'schema': {'type': 'string'},
                        },
                        'X-Image-Dtype': {
                            'description': "The pixels' NumPy type, such as `uint16`, little-endian.",
                            'schema': {'type': 'string'},
                        },
                        'X-Significant-Bits': {
                            'description': 'The depth the camera took the still at.',
                            'schema': {'type': 'integer'},
                        },
                    }
                }
            },
        },
        **_not_json(
            'application/octet-stream',
            "The still's record as JSON, `X-Record-Length` bytes, then its pixels.",
            {'type': 'string', 'format': 'binary'},
        ),
    )


def _no_handle(value: object) -> object:
    """A file's path and a still's record answer no handle and no job."""
    raise wire_encoding.NoWireFormError(type(value).__name__)


def _writes(member: wire_encoding.WireMember) -> bool:
    """Whether a call of *member* writes, and so is made in the control session its request names.

    A read changes nothing and a stop is anyone's, so neither is made in
    the session: the header renews a live one (``_renewing``), and one
    that has ended is answered as a request with no header.
    """
    return not (member.read or member.reads or member.is_stop)


def _acting_in(session: ScopeSession, control_id: str | None) -> contextlib.AbstractContextManager:
    """The control session a request names, entered on the call's own thread; none named, nothing.

    Entered before the member is reached, so the whole call and the lane
    work it hands on are the session's; a session no longer live refuses
    the call here.
    """
    if control_id is None:
        return contextlib.nullcontext()
    return session.acting_in_control(control_id)


def _add_preview_route(app: fastapi.FastAPI, session: ScopeSession) -> None:
    async def preview(request: fastapi.Request, name: str) -> fastapi.Response:
        """A saved image in the live folder, by its name there, as a downscaled JPEG; ``max_width`` as for ``/live``.

        Display only, as the live view is: 8-bit gray, lossy, no metadata. It
        is rendered from the file's own depth and white, so a summed still
        is as bright as it is shown; the data is the file (``/files``).
        """
        max_width = _max_width(request)
        try:
            path = await asyncio.to_thread(session.live_folder_path, name)
        except Exception as e:
            return problems.answered_by_member(e, request.state.request_id).response()
        if not await asyncio.to_thread(path.is_file):
            raise problems.not_found(f'No file {name} is in the live folder.')
        try:
            jpeg = await asyncio.to_thread(_preview_of, path, max_width)
        except Exception as e:
            # A file that is no image is refused as a Python reader of it is.
            return problems.answered_by_member(e, request.state.request_id).response()
        return fastapi.Response(jpeg, media_type='image/jpeg')

    app.add_api_route(
        f'{PREFIX}/previews/{{name:path}}',
        preview,
        methods=['GET'],
        tags=['files'],
        operation_id='previews.read',
        **_words(preview.__doc__),
        openapi_extra=_MAX_WIDTH,
        **_not_json('image/jpeg', 'The image, downscaled.', {'type': 'string', 'format': 'binary'}),
    )


def _preview_of(path: pathlib.Path, max_width: int | None) -> bytes:
    image, significant_bits, white_bits = image_utils.load_pixels(path)
    return image_utils.preview_jpeg(image, significant_bits, max_width, white_bits=white_bits)


def _add_event_route(
    app: fastapi.FastAPI, stream: EventStream, published: dict[str, type[pydantic.BaseModel]]
) -> None:
    async def events(request: fastapi.Request) -> StreamingResponse:
        """Every event the scope sends, as ``text/event-stream``: first ``status``, then each as it happens.

        A client that reconnects with ``Last-Event-ID`` is sent what it
        missed, or ``reset`` and ``status`` once that is no longer held.
        """
        _refuse_query(request)
        return StreamingResponse(
            stream.read(request.headers.get('last-event-id')),
            media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache'},
        )

    app.add_api_route(
        f'{PREFIX}/events',
        events,
        methods=['GET'],
        tags=['events'],
        operation_id='events.stream',
        **_words(events.__doc__),
        openapi_extra={
            'parameters': [
                {
                    'name': 'Last-Event-ID',
                    'in': 'header',
                    'required': False,
                    'schema': {'type': 'string'},
                    'description': 'The id of the last event a reconnecting client heard.',
                }
            ]
        },
        **_not_json('text/event-stream', event_stream.described(published)),
    )

    async def held(after: int | None = None) -> JSONResponse:
        """The events the stream still holds after event *after*, for a client that polls.

        What a stream resumed from *after* is sent, as JSON, with the id to
        ask after next; with no *after*, ``status`` and its id.
        """
        return JSONResponse(await stream.held(after))

    app.add_api_route(
        f'{PREFIX}/events/held',
        held,
        methods=['GET'],
        tags=['events'],
        operation_id='events.held',
        **_words(held.__doc__),
        response_model=None,
        responses={
            200: {'model': event_stream.held_model(published), 'description': 'The events.'}
        },
    )
    built = app.openapi

    def openapi() -> dict:
        """The description FastAPI builds, published as a client reads it.

        Each event's data is a component, since no route answers one as
        JSON; each record is one schema; the parameters and headers routes
        share are components.
        """
        if app.openapi_schema is None:
            described = built()
            _, schemas = pydantic.json_schema.models_json_schema(
                [(m, 'serialization') for m in published.values()],
                ref_template='#/components/schemas/{model}',
            )
            components = described.setdefault('components', {})
            records = components.setdefault('schemas', {})
            for name, schema in schemas.get('$defs', {}).items():
                # A name already published is the same record (no two
                # reachable classes share a name): a route's description of
                # it stands, a parameter's included.
                records.setdefault(name, schema)
            components['parameters'] = _PARAMETERS
            components['headers'] = _HEADERS
            app.openapi_schema = _one_schema_per_record(described)
        return app.openapi_schema

    app.openapi = openapi


def _one_schema_per_record(described: dict) -> dict:
    """*described* with each record one schema, where FastAPI publishes two.

    A record a call takes is checked strict and one it answers is open, so
    FastAPI names the two ``<Record>-Input`` and ``<Record>-Output``, which
    a generated client reads as two types (and a few drop an answer for
    it). They differ only in ``additionalProperties`` and the description
    the answer carries: the one schema is the answer's, stating no
    ``additionalProperties``, since a body is refused a field its record
    does not name (``_INFO``).

    Raises:
        ConfigurationDefectError: ``'record_schemas_differ'``, a record's
            two schemas differ in more than that, so one would describe
            what the server does not do.
    """
    records = described['components']['schemas']
    twins = sorted({n.rsplit('-', 1)[0] for n in records if n.endswith(('-Input', '-Output'))})
    if not twins:
        return described
    named = {
        f'#/components/schemas/{n}-{side}': f'#/components/schemas/{n}'
        for n in twins
        for side in ('Input', 'Output')
    }

    def renamed(value: object) -> object:
        if isinstance(value, dict):
            return {k: named.get(v, v) if k == '$ref' else renamed(v) for k, v in value.items()}
        if isinstance(value, list):
            return [renamed(v) for v in value]
        return value

    described = renamed(described)
    records = described['components']['schemas']
    for name in twins:
        sides = [records.pop(f'{name}-{side}') for side in ('Input', 'Output')]
        taken, answered = (
            {k: v for k, v in side.items() if k not in ('additionalProperties', 'description')}
            for side in sides
        )
        if taken != answered:
            raise ConfigurationDefectError(
                'record_schemas_differ',
                f'{name} is described one way as an argument and another as an answer',
            )
        published = dict(sides[1])
        published.pop('additionalProperties', None)
        records[name] = published
    return described


def _add_live_routes(app: fastapi.FastAPI, live: live_view.LiveView, closing: Closing) -> None:
    async def watch(request: fastapi.Request) -> fastapi.Response:
        """The camera's frames as MJPEG (``multipart/x-mixed-replace``), each the newest when the client is ready for it.

        ``max_width`` (pixels) bounds each frame's width; half the camera's
        frame by default. Each part carries ``X-Frame-Ordinal`` and
        ``X-Frame-Timestamp``. Display only: 8-bit gray, lossy, no metadata.
        """
        max_width = _max_width(request)
        if closing.begun:
            raise problems.session_closing()
        try:
            await live.watch()
        except Exception as e:
            # Refused as a Python caller is: the same door, the same words.
            return problems.answered_by_member(e, request.state.request_id).response()
        return live_view.Watching(live, max_width)

    async def snapshot(request: fastapi.Request) -> fastapi.Response:
        """The camera's newest frame as one JPEG; ``max_width`` as for ``/live``.

        Answers 503 ``no_frame_yet`` when the camera sends no frame within
        a short wait (it is not streaming, say).
        """
        max_width = _max_width(request)
        if closing.begun:
            raise problems.session_closing()
        try:
            shown = await live.snapshot(max_width)
        except Exception as e:
            return problems.answered_by_member(e, request.state.request_id).response()
        if shown is None:
            if closing.begun:
                raise problems.session_closing()
            raise problems.no_frame_yet(live_view.RETRY_AFTER_S)
        frame, jpeg = shown
        return fastapi.Response(
            jpeg, media_type='image/jpeg', headers={'Cache-Control': 'no-cache', **frame.headers()}
        )

    app.add_api_route(
        f'{PREFIX}/live',
        watch,
        methods=['GET'],
        tags=['live'],
        operation_id='live.watch',
        **_words(watch.__doc__),
        openapi_extra=_MAX_WIDTH,
        **_not_json('multipart/x-mixed-replace', 'The frames, each a JPEG part.'),
    )
    app.add_api_route(
        f'{PREFIX}/live.jpg',
        snapshot,
        methods=['GET'],
        tags=['live'],
        operation_id='live.snapshot',
        **_words(snapshot.__doc__),
        openapi_extra={
            **_MAX_WIDTH,
            'responses': {
                '200': {
                    'headers': {
                        'X-Frame-Ordinal': {
                            'description': "The frame's place in the camera's stream.",
                            'schema': {'type': 'integer'},
                        },
                        'X-Frame-Timestamp': {
                            'description': 'When the frame was taken, ISO 8601.',
                            'schema': {'type': 'string', 'format': 'date-time'},
                        },
                    }
                }
            },
        },
        **_not_json('image/jpeg', 'The newest frame.', {'type': 'string', 'format': 'binary'}),
    )


def _max_width(request: fastapi.Request) -> int | None:
    """The live view's and a preview's one query argument: a width in pixels above 0, or None when not given."""
    unknown = sorted(set(request.query_params) - {'max_width'})
    if unknown:
        raise problems.invalid_request(f'This route takes only max_width, not {unknown}.')
    given = request.query_params.getlist('max_width')
    if not given:
        return None
    if len(given) > 1:
        raise problems.invalid_request('Give max_width once.')
    width = whole_number_in(given[0])
    if width is None or width < 1:
        raise problems.invalid_request(
            f'max_width is a whole number of pixels above 0, not {given[0]!r}.'
        )
    return width


def _add_handle_routes(
    app: fastapi.FastAPI,
    session: ScopeSession,
    registry: HandleRegistry,
    handed_out: frozenset[type],
) -> None:
    async def held() -> list[dict[str, str]]:
        """Every handle held, oldest first: a client that lost an answer finds its handle here."""
        return registry.listing()

    app.add_api_route(
        f'{PREFIX}/handles',
        held,
        methods=['GET'],
        tags=['handles'],
        operation_id='handles.list',
        **_words(held.__doc__),
        response_model=None,
        responses={200: {'model': list[Handle], 'description': 'The handles.'}},
    )
    for cls in sorted(handed_out, key=lambda c: c.__name__):

        async def forget(
            request: fastapi.Request, handle_id: str, cls: type = cls
        ) -> fastapi.Response:
            # A write, as every other is: in the control session the request
            # names, and refused while another holds one before the id goes.
            control_id = request.headers.get(CONTROL_SESSION_HEADER)

            def work() -> None:
                with _acting_in(session, control_id):
                    session.as_a_guarded_write(lambda: registry.forget(handle_id, cls))

            reply = _answered(work, request.state.request_id)
            if reply.is_problem:
                return reply.response()
            return fastapi.Response(status_code=204)

        forget.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    'request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=fastapi.Request
                ),
                inspect.Parameter(
                    'handle_id',
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                    default=fastapi.Path(...),
                ),
            ]
        )
        app.add_api_route(
            f'{PREFIX}/handles/{cls.__name__}/{{handle_id}}',
            forget,
            methods=['DELETE'],
            status_code=204,
            name=f'handles/{cls.__name__}/forget',
            operation_id=f'handles.{cls.__name__}.forget',
            openapi_extra={'parameters': [_CONTROL_SESSION]},
            summary=f"Forget a {cls.__name__} handle's id, never its object.",
            description=(
                "A run goes on, and its stop is its own member. The session's protocol "
                "runner is every client's, and its id is kept (422 handle_shared). While a control "
                'session is held, only its holder forgets an id (409 control_session_held).'
            ),
            tags=['handles'],
        )


# The most a JSON body carries: a call's body is far smaller (the whole
# settings tree is about 4 KiB), and no route takes a protocol as JSON. Each
# admitted call holds its body as its parse, several times its bytes.
JSON_BODY_CEILING_BYTES = 64 * 1024
# The most a file upload carries: four times the largest protocol a load
# takes, so a new protocol ceiling moves it.
UPLOAD_CEILING_BYTES = 4 * PROTOCOL_FILE_CEILING_BYTES
# How long a body has to arrive: this floor, plus its length at the floor
# rate (a chunked body's, its ceiling). A client gone silent mid-body -- its
# link dropped -- would otherwise hold what the request took for ever: an
# upload's place and live call.
BODY_DEADLINE_FLOOR_S = 20.0
BODY_FLOOR_BYTES_PER_S = 64 * 1024


class _ClientGone(Quiet, Exception):  # noqa: N818 -- a quiet outcome, not an error
    """The client went away before its body had arrived: nobody is answered, and its log line says how much came.

    Quiet: nothing failed and nothing was declined, and the request's own
    REST-log line is its one record. The same request sent whole may succeed.

    Attributes:
        reason: ``'client_gone'``.
        arrived_bytes: How much of the body arrived before the client went.
    """

    title = 'Client Gone'
    causes: typing.ClassVar[dict[str, RefusalCause | None]] = {'client_gone': RefusalCause.STATE}

    def __init__(self, arrived_bytes: int) -> None:
        super().__init__(f'The client went away after {arrived_bytes} bytes of its body.')
        self.arrived_bytes = arrived_bytes


async def _body_of(request: fastapi.Request, ceiling_bytes: int) -> bytes:
    """The request's body, refused once it is larger than *ceiling_bytes* or slower than its deadline: the one read of a body.

    A ``Content-Length`` over the ceiling is refused before a byte is read;
    a request with neither ``Content-Length`` nor ``Transfer-Encoding`` has
    an empty body; any other is counted as it arrives, a chunked one
    included, and refused at the first byte past the ceiling, so no more
    than the ceiling is held. The whole body arrives within
    ``BODY_DEADLINE_FLOOR_S`` plus its length (a chunked body's, the
    ceiling) at ``BODY_FLOOR_BYTES_PER_S``, or is refused, its connection
    closed.

    Raises:
        ServerRefusedError: ``body_too_large``; ``body_too_slow``.
        _ClientGone: The client went away mid-body.
    """
    length = request.headers.get('content-length')
    if length is not None and int(length) > ceiling_bytes:
        raise problems.body_too_large(ceiling_bytes)
    if length is None and 'transfer-encoding' not in request.headers:
        return b''
    expected = int(length) if length is not None else ceiling_bytes
    deadline_s = BODY_DEADLINE_FLOOR_S + expected / BODY_FLOOR_BYTES_PER_S
    chunks: list[bytes] = []
    arrived = 0
    deadline = asyncio.timeout(deadline_s)
    try:
        async with deadline:
            async for chunk in request.stream():
                arrived += len(chunk)
                if arrived > ceiling_bytes:
                    raise problems.body_too_large(ceiling_bytes)
                chunks.append(chunk)
    except ClientDisconnect:
        raise _ClientGone(arrived) from None
    except TimeoutError:
        if not deadline.expired():
            raise
        raise problems.body_too_slow(deadline_s, arrived) from None
    return b''.join(chunks)


def _replaying(request: fastapi.Request, body: bytes) -> fastapi.Request:
    """*request* again, its body the bytes already read, so the framework reads them as it would the client's."""
    replayed = False

    async def receive() -> dict[str, object]:
        nonlocal replayed
        if replayed:
            # What follows the body: the client's disconnect, when it goes.
            return await request.receive()
        replayed = True
        return {'type': 'http.request', 'body': body, 'more_body': False}

    return fastapi.Request(request.scope, receive)


class _JsonBodyRoute(fastapi.routing.APIRoute):
    """A route that reads a JSON body refuses a body of another type, and one past the JSON ceiling.

    Keyed on the route reading a body parameter, so every JSON route carries
    the rule as it is built, and a route that takes its body as bytes (the
    file upload) does not. The media type is refused before the body is
    read, so it answers ahead of the body's size and the body's own
    ``invalid_request``. The body is read here, ahead of the framework,
    since a refusal raised from inside the framework's own read is answered
    as an unreadable body.
    """

    def get_route_handler(self) -> Callable[[fastapi.Request], Awaitable[fastapi.Response]]:
        handle = super().get_route_handler()
        if not self.dependant.body_params:
            return handle

        async def json_only(request: fastapi.Request) -> fastapi.Response:
            media = request.headers.get('content-type', '').split(';')[0].strip().lower()
            has_body = request.headers.get('content-length', '0') != '0' or (
                'transfer-encoding' in request.headers
            )
            if has_body and media and media != 'application/json' and not media.endswith('+json'):
                raise problems.unsupported_media_type(
                    f'A body is JSON (application/json), not {media}.'
                )
            body = await _body_of(request, JSON_BODY_CEILING_BYTES)
            return await handle(_replaying(request, body))

        return json_only


async def _identify_and_log(
    request: fastapi.Request, call_next: Callable[[fastapi.Request], Awaitable[fastapi.Response]]
) -> fastapi.Response:
    """Give the request its id, and log it.

    Every answer carries the id as ``X-Request-ID`` (a problem also as its
    ``instance``), and the REST log has one line per request under it. A
    stream's line is written when its headers are sent. A client that went
    away mid-body is answered nothing, and its line says how much arrived.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.requested = wall_time.now()
    # Set by ``_client_gone`` when the client leaves before its body has arrived.
    request.state.gone_after_bytes = None
    started = time.perf_counter()
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    gone = request.state.gone_after_bytes
    ended = response.status_code if gone is None else f'the client went away after {gone} bytes'
    _request_log.info(
        f'[{request_id}] {request.method} {request.url.path} -> {ended} '
        f'in {(time.perf_counter() - started) * 1000:.1f} ms',
        extra={'api_request': True, 'request_id': request_id},
    )
    return response


def _admitted_by(
    served: Served,
) -> Callable[
    [fastapi.Request, Callable[[fastapi.Request], Awaitable[fastapi.Response]]],
    Awaitable[fastapi.Response],
]:
    """The guard answering a client *served* does not admit, before any route sees its request."""

    async def admitted(
        request: fastapi.Request,
        call_next: Callable[[fastapi.Request], Awaitable[fastapi.Response]],
    ) -> fastapi.Response:
        peer = request.client.host if request.client else None
        if not served.admits(peer):
            return problems.refused_by_server(
                problems.client_not_allowed(peer), request.state.request_id
            ).response()
        return await call_next(request)

    return admitted


def _addressed_to(
    served: Served,
) -> Callable[
    [fastapi.Request, Callable[[fastapi.Request], Awaitable[fastapi.Response]]],
    Awaitable[fastapi.Response],
]:
    """The guard answering a request not addressed to *served*, or sent by another site's page, before any route sees it."""

    async def addressed(
        request: fastapi.Request,
        call_next: Callable[[fastapi.Request], Awaitable[fastapi.Response]],
    ) -> fastapi.Response:
        host = request.headers.get('host')
        origin = request.headers.get('origin')
        if not served.is_named_by_host(host):
            refusal = problems.wrong_host(host, served.authority)
        elif origin is not None and not served.is_own_origin(origin):
            refusal = problems.foreign_origin(origin)
        else:
            return await call_next(request)
        return problems.refused_by_server(refusal, request.state.request_id).response()

    return addressed


def _renewing(
    session: ScopeSession,
) -> Callable[
    [fastapi.Request, Callable[[fastapi.Request], Awaitable[fastapi.Response]]],
    Awaitable[fastapi.Response],
]:
    """Renew the control session a request's ``Control-Session`` header names, as it arrives.

    Every request naming it, whatever its route -- a read, a job poll, a
    write -- so a client that only polls its job is not silent
    (``ScopeSession.take_control``).
    """

    async def renewed(
        request: fastapi.Request,
        call_next: Callable[[fastapi.Request], Awaitable[fastapi.Response]],
    ) -> fastapi.Response:
        control_id = request.headers.get(CONTROL_SESSION_HEADER)
        if control_id is not None:
            session.renew_control(control_id)
        return await call_next(request)

    return renewed


async def _refused(request: fastapi.Request, refusal: ServerRefusedError) -> JSONResponse:
    return problems.refused_by_server(refusal, request.state.request_id).response()


async def _client_gone(request: fastapi.Request, gone: _ClientGone) -> fastapi.Response:
    """End a request whose client went away mid-body: its log line says so, and nothing reaches the client."""
    request.state.gone_after_bytes = gone.arrived_bytes
    # Never sent, the client having gone: a handler answers a response, and
    # this one is a request not read whole.
    return fastapi.Response(status_code=400)


async def _body_does_not_fit(
    request: fastapi.Request, error: RequestValidationError
) -> JSONResponse:
    """``invalid_request``, naming where and why each argument does not fit.

    Each error's ``input`` is left out: it echoes what was sent, and a NaN
    the reader took is no JSON, so the answer would fail where the refusal
    is due.
    """
    errors = tuple(UnfitArgument(tuple(e['loc']), e['msg'], e['type']) for e in error.errors())
    words = '; '.join(f'{"/".join(str(p) for p in e.loc)}: {e.msg}' for e in errors)
    return problems.refused_by_server(
        problems.invalid_request(words, errors=errors), request.state.request_id
    ).response()


async def _routing_refused(request: fastapi.Request, error: StarletteHTTPException) -> JSONResponse:
    """What the framework itself refused, by its status: a path no route answers (404), a
    method its route does not (405), or a request it could not read (any other 4xx)."""
    if error.status_code == 405:
        refusal = problems.method_not_allowed(
            f'{request.url.path} does not answer {request.method}.'
        )
    elif error.status_code == 404:
        refusal = problems.not_found(f'No route answers {request.url.path}.')
    else:
        refusal = problems.invalid_request(str(error.detail))
    return problems.refused_by_server(refusal, request.state.request_id).response()


async def _failed(request: fastapi.Request, error: Exception) -> JSONResponse:
    """The server's own failure, answered as the fault problem it is rather than as bare text."""
    return problems.answered_by_member(error, request.state.request_id).response()


def _words(doc: str | None) -> dict[str, str | None]:
    """A route's summary and description, as Markdown, from its docstring.

    Each parameter's words are on its property, so the description leaves
    the Args: section out rather than say them twice.
    """
    doc = inspect.cleandoc(doc or '')
    return {
        'summary': markdown_of(doc.split('\n', 1)[0]) or None,
        'description': markdown_of(doc, leave_out=('Args',)) or None,
    }


def served(route: Route) -> tuple[str, str, str]:
    """Where the server serves *route*, below ``PREFIX``: its path, its operation's name, and its tag.

    A route from a live object's class is under
    ``handles/<Class>/{handle_id}/``, the handle a client was handed; a
    Session route is at its own path.
    """
    if route.root is ScopeSession:
        return route.path, route.name, '/'.join(route.segments) or 'session'
    handles = f'handles/{route.root.__name__}'
    return (
        f'{handles}/{{handle_id}}/{route.path}',
        f'{handles}/{route.name}',
        '/'.join((handles, *route.segments)),
    )


def _add(
    app: fastapi.FastAPI,
    session: ScopeSession,
    registry: HandleRegistry,
    jobs: JobRegistry,
    stream: EventStream,
    closing: Closing,
    keys: idempotency.IdempotencyKeys,
    classes: dict[str, type],
    route: Route,
) -> None:
    doc = inspect.cleandoc(route.member.doc)
    member = route.member
    on_handle = route.root is not ScopeSession
    path, name, tag = served(route)
    if member.returns_job:
        # A member that hands back an operation is answered its job, as its
        # answer describes it, and never 200.
        answered = {
            202: {
                'model': job_answer(member.returns),
                'description': 'The job of the operation the call started.',
                'headers': _JOB_HEADERS,
            }
        }
    else:
        answered = {
            200: {
                'model': route.answer,
                'description': (
                    "The member's answer."
                    if route.sets is None
                    else 'The setting, as its read answers it after the write.'
                ),
            },
            # Any other call is a job once it has outlived the wait.
            202: _ANSWERED_AS_JOB,
        }
    common = {
        'path': f'{PREFIX}/{path}',
        'name': name,
        'operation_id': name.replace('/', '.'),
        **_words(doc),
        'tags': [tag],
        # The answer is encoded from the value (``wire_encoding.encode``) and
        # sent as it is; the models only describe it, from the member's
        # declared return type.
        'response_model': None,
        'status_code': 202 if member.returns_job else 200,
        'responses': {**answered, **problems.declared(doc, classes, member.admitted_by)},
        'openapi_extra': {'parameters': [_CONTROL_SESSION]} if _writes(member) else None,
    }

    async def answer(
        request: fastapi.Request,
        handle_id: str | None,
        act: Callable[[object, Progress, Callable[[Operation], dict]], object],
        sent: dict[str, object] | None = None,
    ) -> fastapi.Response:
        """Admit the call and answer it; a retried request (``Idempotency-Key``) is answered its first answer.

        *sent* is a POST's body, which a key is held with; a read has none,
        and runs again whatever key it is sent.
        """
        if route.query is None:
            _refuse_query(request)
        if closing.begun:
            raise problems.session_closing()
        if member.hands_out:
            registry.admit()
        jobs.admit(returns_job=member.returns_job)
        key = request.headers.get('idempotency-key') if sent is not None else None
        if key is None:
            return await respond(request, handle_id, act)
        return await keys.answer(
            key,
            idempotency.digest(request.url.path, sent),
            lambda: respond(request, handle_id, act),
        )

    async def respond(
        request: fastapi.Request,
        handle_id: str | None,
        act: Callable[[object, Progress, Callable[[Operation], dict]], object],
    ) -> JSONResponse:
        """Run the call on its own thread; answer it, or hand out its job when it outlives the wait.

        A call that hands back an operation is answered once it has
        returned, whatever the client's wait, with the operation's job: it
        returns once the operation is dispatched, within its own bounds, and
        holding it as a job of its own would hand the client a second job
        whose ``DELETE`` stops nothing.

        *act* is given the owner, the call's progress, and how to name the
        job a running call it hands back is.
        """
        request_id = request.state.request_id
        asked = request.url.path.removeprefix(f'{PREFIX}/')
        progress = Progress()

        def job_of(operation: Operation, member: str = asked, said: Progress | None = None) -> dict:
            files = operation.files
            return jobs.of_future(
                operation,
                member=member,
                answer=future_answer,
                # A member that reports its progress to a callback says it
                # there, not through its operation's progress().
                progress=(
                    (lambda: said.last)
                    if said is not None
                    else (lambda: encoded(operation.progress()))
                ),
                files=None if files is None else lambda: job_of(files, f'{member}/files'),
            )

        def encoded(value: object) -> object:
            return wire_encoding.encode(
                value,
                live_folder=session.setting('live_folder'),
                handle=registry.mint,
                job=job_of,
            )

        def future_answer(operation: Operation) -> Answer:
            return _answered(lambda: encoded(operation.result()), request_id)

        # The job of the operation the call handed back, once it has.
        handed: list[str] = []
        control_id = request.headers.get(CONTROL_SESSION_HEADER)

        def work() -> object:
            with _acting_in(session, control_id if _writes(member) else None):
                owner = registry.get(handle_id, route.root) if on_handle else session
                for segment in route.segments:
                    owner = getattr(owner, segment)
                    if owner is None:
                        raise problems.not_found(f'{name}: this scope has no {segment}.')
                value = act(owner, progress, job_of)
            if isinstance(value, Operation):
                said = progress if member.progress is not None else None
                handed.append(job_of(value, said=said)['id'])
            return encoded(value)

        # Every header is read before the call starts, so nothing about one
        # can fail once the move or capture has begun.
        wait = Wait.of(request.headers.get('prefer'))
        job = jobs.run(
            lambda: _answered(work, request_id),
            name,
            request_id,
            member=asked,
            requested=request.state.requested,
            progress=progress if member.progress else None,
        )
        if member.returns_job:
            reply = await jobs.answer(job)
            if not handed:
                return reply.response()
            return JSONResponse(reply.body, status_code=202, headers=_job_headers(handed[0]))
        reply = await jobs.answer(job, wait.seconds, closing.finished())
        if reply is not None:
            return dataclasses.replace(
                reply, headers={**reply.headers, **wait.headers()}
            ).response()
        jobs.adopt(job)
        return JSONResponse(
            job.view(),
            status_code=202,
            headers={**_job_headers(job.id), **wait.headers()},
        )

    if member.read:

        async def read(request: fastapi.Request, handle_id: str | None = None) -> JSONResponse:
            return await answer(
                request, handle_id, lambda owner, _p, _job_of: getattr(owner, member.name)
            )

        _sign(read, on_handle, None)
        app.add_api_route(endpoint=read, methods=['GET'], **common)
        return

    def decoded(sent: dict[str, object]) -> dict[str, object]:
        """The arguments *sent*, checked against the body model, as the member takes them."""
        return {
            p.name: wire_encoding.decode(
                sent[p.name],
                p.alternatives,
                resolve_path=session.live_folder_path,
                handle=registry.get,
                job=jobs.operation,
            )
            for p in member.parameters
            if p.name in sent
        }

    def invoker(
        sent: dict[str, object],
    ) -> Callable[[object, Progress, Callable[[Operation], dict]], object]:
        """How to call the member with the arguments *sent*, checked against its body model."""

        def invoke(
            owner: object, progress: Progress, job_of: Callable[[Operation], dict]
        ) -> object:
            arguments = decoded(sent)
            if member.progress is not None:
                arguments[member.progress] = progress
            if member.run_events is None:
                return getattr(owner, member.name)(**arguments)
            # The run's events name it, by its job, once the call has handed
            # it out; a call that waits for its run hands out none.
            tag = stream.run_events()
            arguments[member.run_events] = tag.events()
            try:
                value = getattr(owner, member.name)(**arguments)
            except BaseException:
                tag.settle(None)
                raise
            tag.settle({'job': job_of(value)['id']} if isinstance(value, Operation) else None)
            return value

        return invoke

    if member.reads:

        async def asked(
            request: fastapi.Request, query: pydantic.BaseModel, handle_id: str | None = None
        ) -> JSONResponse:
            try:
                checked = route.body.model_validate(sent_of_query(query))
            except pydantic.ValidationError as e:
                raise RequestValidationError(
                    [
                        {**err, 'loc': ('query', '.'.join(str(p) for p in err['loc']))}
                        for err in e.errors()
                    ]
                ) from e
            sent = checked.model_dump(include=checked.model_fields_set)
            return await answer(request, handle_id, invoker(sent))

        _sign(asked, on_handle, route, query=True)
        app.add_api_route(endpoint=asked, methods=['GET'], **common)
        return

    async def call(
        request: fastapi.Request, body: pydantic.BaseModel | None, handle_id: str | None = None
    ) -> JSONResponse:
        sent = body.model_dump(include=body.model_fields_set) if body is not None else {}
        if route.sets is None:
            return await answer(request, handle_id, invoker(sent), sent)
        write, read = invoker(sent), route.sets

        def written(
            owner: object, progress: Progress, job_of: Callable[[Operation], dict]
        ) -> object:
            # The answer is the setting as the read now answers it, not the
            # setter's own return: the value the scope took, under the URL
            # the client reads it at.
            write(owner, progress, job_of)
            if read.read:
                return getattr(owner, read.name)
            key = decoded(sent)
            return getattr(owner, read.name)(**{p.name: key[p.name] for p in read.parameters})

        return await answer(request, handle_id, written, sent)

    _sign(call, on_handle, route)
    app.add_api_route(endpoint=call, methods=[route.method.upper()], **common)


def _job_headers(job_id: str) -> dict[str, str]:
    """Where a running job is read, and how soon to read it."""
    return {'Location': f'{PREFIX}/jobs/{job_id}', 'Retry-After': str(JOB_RETRY_AFTER_S)}


def _answered(work: Callable[[], object], request_id: str) -> Answer:
    """*work*'s encoded result, or the problem it ended in: an answer, never a raise.

    The server's own refusals (a handle not held) carry their own reasons;
    any other exception is the member's outcome, as a Python caller gets it.
    """
    try:
        return problems.result(work())
    except ServerRefusedError as refusal:
        return problems.refused_by_server(refusal, request_id)
    except Exception as e:
        return problems.answered_by_member(e, request_id)


def _add_job_routes(
    app: fastapi.FastAPI, session: ScopeSession, jobs: JobRegistry, closing: Closing
) -> None:
    async def listing() -> list[dict[str, object]]:
        """Every job held, newest first."""
        # Off the loop: a run's progress is read under the run's lock.
        return await asyncio.to_thread(jobs.listing)

    app.add_api_route(
        f'{PREFIX}/jobs',
        listing,
        methods=['GET'],
        tags=['jobs'],
        operation_id='jobs.list',
        **_words(listing.__doc__),
        response_model=None,
        responses={200: {'model': list[Job], 'description': 'The jobs.'}},
    )

    async def read(request: fastapi.Request, job_id: str) -> JSONResponse:
        """A job, waiting for it to end for what the client says it will wait (``Prefer: wait``).

        200 whatever the job's status: a failed job's ``error`` is the
        problem its call ended in.
        """
        job = jobs.get(job_id)
        wait = Wait.of(request.headers.get('prefer'))
        await jobs.answer(job, wait.seconds, closing.finished())
        return JSONResponse(await asyncio.to_thread(job.view), headers=wait.headers())

    app.add_api_route(
        f'{PREFIX}/jobs/{{job_id}}',
        read,
        methods=['GET'],
        tags=['jobs'],
        operation_id='jobs.read',
        **_words(read.__doc__),
        response_model=None,
        responses={200: {'model': Job, 'description': 'The job.'}},
    )

    async def delete(request: fastapi.Request, job_id: str) -> fastapi.Response:
        """Forget a finished job (204), or stop a running one (202, with the job).

        A running job's operation is asked to stop, and the job ends as the
        stop leaves it: a stopped run ``aborted``, while a still, which has
        no stop, ends as it would have. A stop the scope refuses is the
        answer, as its member's refusal. A call held as a job when it
        outlived its wait holds no operation: 409 ``not_cancelable``. The
        forget is a write, in the control session the request names, and
        refused while another holds one: 409 ``control_session_held``. The
        stop is anyone's, made in no session, whatever the request names.
        """
        control_id = request.headers.get(CONTROL_SESSION_HEADER)

        def forget_in_session(forget: Callable[[], None]) -> None:
            with _acting_in(session, control_id):
                session.as_a_guarded_write(forget)

        def work() -> HeldJob | None:
            return jobs.delete(job_id, as_a_write=forget_in_session)

        try:
            # Off the loop: a stop is sent to the scope.
            job = await asyncio.to_thread(work)
        except ServerRefusedError:
            raise
        except Exception as e:
            return problems.answered_by_member(e, request.state.request_id).response()
        if job is None:
            return fastapi.Response(status_code=204)
        return JSONResponse(await asyncio.to_thread(job.view), status_code=202)

    app.add_api_route(
        f'{PREFIX}/jobs/{{job_id}}',
        delete,
        methods=['DELETE'],
        status_code=204,
        tags=['jobs'],
        operation_id='jobs.delete',
        **_words(delete.__doc__),
        openapi_extra={'parameters': [_CONTROL_SESSION]},
        response_model=None,
        responses={202: {'model': Job, 'description': 'The job, asked to stop.'}},
    )


def _sign(endpoint: Callable, on_handle: bool, route: Route | None, *, query: bool = False) -> None:
    """Give *endpoint* the parameters FastAPI reads: the handle's id, and a call's body or query.

    The body's model is the route's own, so FastAPI checks and describes it;
    a member every one of whose parameters has a default also takes no body.
    A read's query model is read from the query string, one key a field.
    """
    parameters = [
        inspect.Parameter(
            'request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=fastapi.Request
        )
    ]
    if query:
        parameters.append(
            inspect.Parameter(
                'query',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=typing.Annotated[route.query, fastapi.Query()],
            )
        )
    elif route is not None:
        required = any(p.required for p in route.member.parameters)
        parameters.append(
            inspect.Parameter(
                'body',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                # Not ``| None``: an absent body is the default, never checked,
                # and the description names the model itself, not "or null".
                annotation=route.body,
                default=fastapi.Body(...) if required else fastapi.Body(None),
            )
        )
    if on_handle:
        parameters.append(
            inspect.Parameter(
                'handle_id',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
                default=fastapi.Path(...),
            )
        )
    endpoint.__signature__ = inspect.Signature(parameters)


def _refuse_query(request: fastapi.Request) -> None:
    """Refuse a query string where a route reads none: a property, or a call's JSON body."""
    if request.url.query:
        raise problems.invalid_request(
            'This route takes no query string: a property takes no arguments, and a '
            "call's arguments are a JSON object; only a read method's are in the query."
        )
