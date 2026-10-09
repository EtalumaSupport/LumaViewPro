# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The application a server serves: one route per wire member of a session.

Every call runs on a thread of its own and the route awaits it, so a call
that waits on the scope holds no worker another request needs: a shared
pool of threads let long waits hold a stop for seconds. A call's
arguments are decoded, the member reached and called, and its answer
encoded on that thread, since each may read the scope or the disk.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from concurrent.futures import Future

import fastapi
import pydantic
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from modules import wire_encoding
from modules.scope_session import ScopeSession
from rest.routes import Route, routes

VERSION = 'v1'
PREFIX = f'/api/{VERSION}'


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
    app.add_exception_handler(RequestValidationError, _body_does_not_fit)
    for route in routes(ScopeSession):
        _add(app, session, route)
    return app


async def _body_does_not_fit(
    _request: fastapi.Request, error: RequestValidationError
) -> JSONResponse:
    """A 422 naming where and why each argument does not fit.

    Not FastAPI's own, which echoes what was sent: a NaN the reader took is
    no JSON, and the answer would fail where the refusal is due.
    """
    return JSONResponse(
        {'detail': [{k: e[k] for k in ('loc', 'msg', 'type')} for e in error.errors()]},
        status_code=422,
    )


def _add(app: fastapi.FastAPI, session: ScopeSession, route: Route) -> None:
    doc = inspect.cleandoc(route.member.doc)
    common = {
        'path': f'{PREFIX}/{route.path}',
        'name': route.path,
        'operation_id': route.path.replace('/', '.'),
        'summary': doc.split('\n', 1)[0] or None,
        'description': doc or None,
        'tags': ['/'.join(route.segments) or 'session'],
        'response_model': None,
    }
    if route.member.read:

        async def read(request: fastapi.Request) -> JSONResponse:
            _refuse_query(request)
            return await _answer(session, route, lambda owner: getattr(owner, route.member.name))

        app.add_api_route(endpoint=read, methods=['GET'], **common)
        return

    async def call(request: fastapi.Request, body: pydantic.BaseModel | None) -> JSONResponse:
        _refuse_query(request)
        sent = body.model_dump(include=body.model_fields_set) if body is not None else {}

        def invoke(owner: object) -> object:
            arguments = {
                p.name: wire_encoding.decode(
                    sent[p.name],
                    p.alternatives,
                    resolve_path=session.live_folder_path,
                    handle=_no_handles_yet,
                )
                for p in route.member.parameters
                if p.name in sent
            }
            return getattr(owner, route.member.name)(**arguments)

        return await _answer(session, route, invoke)

    # The body's model is the route's own, so FastAPI checks and describes it;
    # a member every one of whose parameters has a default also takes no body.
    required = any(p.required for p in route.member.parameters)
    call.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                'request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=fastapi.Request
            ),
            inspect.Parameter(
                'body',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=route.body if required else route.body | None,
                default=fastapi.Body(...) if required else fastapi.Body(None),
            ),
        ]
    )
    app.add_api_route(endpoint=call, methods=['POST'], **common)


def _refuse_query(request: fastapi.Request) -> None:
    if request.url.query:
        raise fastapi.HTTPException(
            422, 'A member takes no query string: send its arguments as a JSON object.'
        )


async def _answer(
    session: ScopeSession, route: Route, act: Callable[[object], object]
) -> JSONResponse:
    """Reach the route's owner, *act* on it, and encode the answer, on a thread of its own."""

    def work() -> object:
        owner = session
        for segment in route.segments:
            owner = getattr(owner, segment)
            if owner is None:
                raise fastapi.HTTPException(404, f'{route.path}: this scope has no {segment}.')
        return wire_encoding.encode(
            act(owner),
            live_folder=session.get_setting('live_folder'),
            handle=_no_handles_yet,
            job=_no_handles_yet,
        )

    return JSONResponse(await _on_own_thread(work, route.path))


async def _on_own_thread(work: Callable[[], object], name: str) -> object:
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def settle(result: object, error: BaseException | None) -> None:
        # The client went away and the route was cancelled: nobody waits.
        if done.cancelled():
            return
        if error is not None:
            done.set_exception(error)
        else:
            done.set_result(result)

    def run() -> None:
        try:
            result = work()
        except Exception as e:
            loop.call_soon_threadsafe(settle, None, e)
        except BaseException as e:
            # A SystemExit on this thread ends the call, not the server: the
            # request is answered as the fault it is rather than left waiting.
            fault = RuntimeError(f'{name} ended with {type(e).__name__}')
            fault.__cause__ = e
            loop.call_soon_threadsafe(settle, None, fault)
        else:
            loop.call_soon_threadsafe(settle, result, None)

    threading.Thread(target=run, name=f'rest {name}').start()
    return await done


def _no_handles_yet(value: object, *_: object) -> object:
    kind = 'job' if isinstance(value, Future) else 'live object'
    raise wire_encoding.NoWireFormError(f'this server does not yet hand out a {kind}')
