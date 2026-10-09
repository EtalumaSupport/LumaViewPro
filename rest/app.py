# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The application a server serves: one route per wire member of a session.

Every call runs on a thread of its own and the route awaits it, so a call
that waits on the scope holds no worker another request needs: a shared
pool of threads let long waits hold a stop for seconds. A call's
arguments are decoded, the member reached and called, and its answer
encoded on that thread, since each may read the scope or the disk.

The Session's members are at ``/api/v1/<member>``; a live object a client
was handed has its members at ``/api/v1/handles/<type>/<id>/<member>``
(``rest.handles``). A call that outlives what its client will wait is a
job (``rest.jobs``), and every answer that is not a result is a problem
(``rest.problems``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import inspect
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import Future

import fastapi
import pydantic
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from modules import wire_encoding
from modules.protocol_runner import ProtocolRunner
from modules.scope_session import ScopeSession
from rest import problems
from rest.handles import HandleRegistry
from rest.jobs import RETRY_AFTER_S as JOB_RETRY_AFTER_S
from rest.jobs import JobRegistry, Progress, Wait
from rest.problems import Answer, ServerRefusedError
from rest.routes import Route, routes

VERSION = 'v1'
PREFIX = f'/api/{VERSION}'
# The first path segments the server answers itself; a Session member of
# one of these names would be shadowed, so one fails the build.
SERVER_SEGMENTS = frozenset({'handles', 'jobs', 'events', 'files', 'live', 'live.jpg'})


def build_app(session: ScopeSession) -> fastapi.FastAPI:
    """The application serving *session*'s wire members under ``/api/v1/``.

    The OpenAPI description is at ``/api/v1/openapi.json`` and the
    interactive reference at ``/docs``.
    """
    app = fastapi.FastAPI(
        title='Lumascope',
        version=session.app_version or 'unknown',
        openapi_url=f'{PREFIX}/openapi.json',
        docs_url='/docs',
        redoc_url=None,
    )

    async def versions() -> dict[str, list[str]]:
        """The API versions this server answers, each at ``/api/<version>/``."""
        return {'versions': [VERSION]}

    app.add_api_route('/api', versions, methods=['GET'], tags=['server'])
    app.middleware('http')(_identify_and_admit)
    app.add_exception_handler(ServerRefusedError, _refused)
    app.add_exception_handler(RequestValidationError, _body_does_not_fit)
    app.add_exception_handler(StarletteHTTPException, _routing_refused)
    app.add_exception_handler(Exception, _failed)

    # The session's one protocol runner is every client's: its id is kept.
    registry = HandleRegistry(kept=lambda obj: isinstance(obj, ProtocolRunner))
    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    jobs = JobRegistry()
    _add_handle_routes(app, registry, handed_out)
    _add_job_routes(app, jobs)
    session_routes = routes(ScopeSession, handed_out=handed_out)
    shadowed = {r.path.split('/')[0] for r in session_routes} & SERVER_SEGMENTS
    if shadowed:
        raise TypeError(f"Session members {sorted(shadowed)} are named as the server's own routes")
    for route in session_routes:
        _add(app, session, registry, jobs, route)
    for cls in sorted(handed_out, key=lambda c: c.__name__):
        for route in routes(cls, handed_out=handed_out):
            _add(app, session, registry, jobs, route)
    return app


def _add_handle_routes(
    app: fastapi.FastAPI, registry: HandleRegistry, handed_out: frozenset[type]
) -> None:
    async def held() -> list[dict[str, str]]:
        """Every handle held, oldest first: a client that lost an answer finds its handle here."""
        return registry.listing()

    app.add_api_route(f'{PREFIX}/handles', held, methods=['GET'], tags=['handles'])
    for cls in sorted(handed_out, key=lambda c: c.__name__):

        async def forget(handle_id: str, cls: type = cls) -> fastapi.Response:
            registry.forget(handle_id, cls)
            return fastapi.Response(status_code=204)

        forget.__signature__ = inspect.Signature(
            [
                inspect.Parameter(
                    'handle_id',
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                    default=fastapi.Path(...),
                )
            ]
        )
        app.add_api_route(
            f'{PREFIX}/handles/{cls.__name__}/{{handle_id}}',
            forget,
            methods=['DELETE'],
            status_code=204,
            name=f'handles/{cls.__name__}/forget',
            operation_id=f'handles.{cls.__name__}.forget',
            summary=f"Forget a {cls.__name__} handle's id, never its object.",
            description=(
                "A run goes on, and its stop is its own member. The session's protocol "
                "runner is every client's, and its id is kept (409)."
            ),
            tags=['handles'],
        )


async def _identify_and_admit(
    request: fastapi.Request, call_next: Callable[[fastapi.Request], Awaitable[object]]
) -> object:
    """Give the request its id, and refuse a body that is not JSON before any route reads it."""
    request.state.request_id = str(uuid.uuid4())
    request.state.requested = datetime.datetime.now().astimezone()
    media = request.headers.get('content-type', '').split(';')[0].strip().lower()
    has_body = request.headers.get('content-length', '0') != '0' or (
        'transfer-encoding' in request.headers
    )
    if has_body and media and media != 'application/json' and not media.endswith('+json'):
        return problems.refused_by_server(
            problems.unsupported_media_type(f'A body is JSON (application/json), not {media}.'),
            request.state.request_id,
        ).response()
    return await call_next(request)


async def _refused(request: fastapi.Request, refusal: ServerRefusedError) -> JSONResponse:
    return problems.refused_by_server(refusal, request.state.request_id).response()


async def _body_does_not_fit(
    request: fastapi.Request, error: RequestValidationError
) -> JSONResponse:
    """``invalid_request``, naming where and why each argument does not fit.

    Each error's ``input`` is left out: it echoes what was sent, and a NaN
    the reader took is no JSON, so the answer would fail where the refusal
    is due.
    """
    errors = [{k: e[k] for k in ('loc', 'msg', 'type')} for e in error.errors()]
    words = '; '.join(f'{"/".join(str(p) for p in e["loc"])}: {e["msg"]}' for e in errors)
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


def _add(
    app: fastapi.FastAPI,
    session: ScopeSession,
    registry: HandleRegistry,
    jobs: JobRegistry,
    route: Route,
) -> None:
    doc = inspect.cleandoc(route.member.doc)
    member = route.member
    on_handle = route.root is not ScopeSession
    if on_handle:
        path = f'handles/{route.root.__name__}/{{handle_id}}/{route.path}'
        name = f'handles/{route.root.__name__}/{route.path}'
        tag = '/'.join((f'handles/{route.root.__name__}', *route.segments))
    else:
        path, name, tag = route.path, route.path, '/'.join(route.segments) or 'session'
    common = {
        'path': f'{PREFIX}/{path}',
        'name': name,
        'operation_id': name.replace('/', '.'),
        'summary': doc.split('\n', 1)[0] or None,
        'description': doc or None,
        'tags': [tag],
        'response_model': None,
    }

    async def answer(
        request: fastapi.Request, handle_id: str | None, act: Callable[[object, Progress], object]
    ) -> JSONResponse:
        """Run the call on its own thread; answer it, or hand out its job when it outlives the wait."""
        _refuse_query(request)
        if member.hands_out:
            registry.admit()
        jobs.admit(returns_job=member.returns_job)
        request_id = request.state.request_id
        asked = request.url.path.removeprefix(f'{PREFIX}/')
        loop = asyncio.get_running_loop()
        progress = Progress()

        def encoded(value: object) -> object:
            return wire_encoding.encode(
                value,
                live_folder=session.get_setting('live_folder'),
                handle=registry.mint,
                job=lambda future: jobs.of_future(
                    future, member=asked, answer=future_answer, loop=loop
                ),
            )

        def future_answer(future: Future) -> Answer:
            return _answered(lambda: encoded(future.result()), request_id)

        def work() -> object:
            owner = registry.get(handle_id, route.root) if on_handle else session
            for segment in route.segments:
                owner = getattr(owner, segment)
                if owner is None:
                    raise problems.not_found(f'{name}: this scope has no {segment}.')
            return encoded(act(owner, progress))

        answered = jobs.run(lambda: _answered(work, request_id), name, request_id)
        wait = Wait.of(request.headers.get('prefer'))
        await asyncio.wait({answered}, timeout=wait.seconds)
        if answered.done():
            reply = answered.result()
            return dataclasses.replace(
                reply, headers={**reply.headers, **wait.headers()}
            ).response()
        job = jobs.adopt(
            answered,
            member=asked,
            requested=request.state.requested,
            progress=progress if member.progress else None,
        )
        return JSONResponse(
            job.view(),
            status_code=202,
            headers={
                'Location': f'{PREFIX}/jobs/{job.id}',
                'Retry-After': str(JOB_RETRY_AFTER_S),
                **wait.headers(),
            },
        )

    if member.read:

        async def read(request: fastapi.Request, handle_id: str | None = None) -> JSONResponse:
            return await answer(request, handle_id, lambda owner, _p: getattr(owner, member.name))

        _sign(read, on_handle, None)
        app.add_api_route(endpoint=read, methods=['GET'], **common)
        return

    async def call(
        request: fastapi.Request, body: pydantic.BaseModel | None, handle_id: str | None = None
    ) -> JSONResponse:
        sent = body.model_dump(include=body.model_fields_set) if body is not None else {}

        def invoke(owner: object, progress: Progress) -> object:
            arguments = {
                p.name: wire_encoding.decode(
                    sent[p.name],
                    p.alternatives,
                    resolve_path=session.live_folder_path,
                    handle=registry.get,
                )
                for p in member.parameters
                if p.name in sent
            }
            if member.progress is not None:
                arguments[member.progress] = progress
            return getattr(owner, member.name)(**arguments)

        return await answer(request, handle_id, invoke)

    _sign(call, on_handle, route)
    app.add_api_route(endpoint=call, methods=['POST'], **common)


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


def _add_job_routes(app: fastapi.FastAPI, jobs: JobRegistry) -> None:
    async def listing() -> list[dict[str, object]]:
        """Every job held, newest first."""
        return jobs.listing()

    app.add_api_route(f'{PREFIX}/jobs', listing, methods=['GET'], tags=['jobs'])

    async def read(request: fastapi.Request, job_id: str) -> JSONResponse:
        """A job, waiting for it to end for what the client says it will wait (``Prefer: wait``).

        200 whatever the job's status: a failed job's ``error`` is the
        problem its call ended in.
        """
        job = jobs.get(job_id)
        wait = Wait.of(request.headers.get('prefer'))
        if not job.finished:
            await asyncio.wait({job.answered}, timeout=wait.seconds)
        return JSONResponse(job.view(), headers=wait.headers())

    app.add_api_route(f'{PREFIX}/jobs/{{job_id}}', read, methods=['GET'], tags=['jobs'])

    async def forget(job_id: str) -> fastapi.Response:
        """Forget a finished job; one still running is 409, and its own member stops it."""
        jobs.forget(job_id)
        return fastapi.Response(status_code=204)

    app.add_api_route(
        f'{PREFIX}/jobs/{{job_id}}', forget, methods=['DELETE'], status_code=204, tags=['jobs']
    )


def _sign(endpoint: Callable, on_handle: bool, route: Route | None) -> None:
    """Give *endpoint* the parameters FastAPI reads: the handle's id, and a call's body.

    The body's model is the route's own, so FastAPI checks and describes it;
    a member every one of whose parameters has a default also takes no body.
    """
    parameters = [
        inspect.Parameter(
            'request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=fastapi.Request
        )
    ]
    if route is not None:
        required = any(p.required for p in route.member.parameters)
        parameters.append(
            inspect.Parameter(
                'body',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=route.body if required else route.body | None,
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
    if request.url.query:
        raise problems.invalid_request(
            'A member takes no query string: send its arguments as a JSON object.'
        )
