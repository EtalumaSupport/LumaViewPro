# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The application a server serves: one route per wire member of a session.

Every call runs on a thread of its own and the route awaits it, so a call
that waits on the scope holds no worker another request needs: a shared
pool of threads let long waits hold a stop for seconds. A call's
arguments are decoded, the member reached and called, and its answer
encoded on that thread, since each may read the scope or the disk.

The Session's members are at ``/api/v1/<member>``; a live object a client
was handed has its members at ``/api/v1/handles/<type>/<id>/<member>``
(``rest.handles``).
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
from modules.protocol_runner import ProtocolRunner
from modules.scope_session import ScopeSession
from rest.handles import HandleRegistry
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
    app.add_exception_handler(RequestValidationError, _body_does_not_fit)

    # The session's one protocol runner is every client's: its id is kept.
    registry = HandleRegistry(kept=lambda obj: isinstance(obj, ProtocolRunner))
    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    _add_handle_routes(app, registry, handed_out)
    session_routes = routes(ScopeSession, handed_out=handed_out)
    shadowed = {r.path.split('/')[0] for r in session_routes} & SERVER_SEGMENTS
    if shadowed:
        raise TypeError(f"Session members {sorted(shadowed)} are named as the server's own routes")
    for route in session_routes:
        _add(app, session, registry, route)
    for cls in sorted(handed_out, key=lambda c: c.__name__):
        for route in routes(cls, handed_out=handed_out):
            _add(app, session, registry, route)
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


def _add(
    app: fastapi.FastAPI, session: ScopeSession, registry: HandleRegistry, route: Route
) -> None:
    doc = inspect.cleandoc(route.member.doc)
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

    def root_of(handle_id: str | None) -> object:
        return registry.get(handle_id, route.root) if on_handle else session

    def act_on(act: Callable[[object], object]) -> Callable[[str | None], object]:
        def work(handle_id: str | None) -> object:
            owner = root_of(handle_id)
            for segment in route.segments:
                owner = getattr(owner, segment)
                if owner is None:
                    raise fastapi.HTTPException(404, f'{name}: this scope has no {segment}.')
            return wire_encoding.encode(
                act(owner),
                live_folder=session.get_setting('live_folder'),
                handle=registry.mint,
                job=_no_jobs_yet,
            )

        return work

    if route.member.read:
        work = act_on(lambda owner: getattr(owner, route.member.name))

        async def read(request: fastapi.Request, handle_id: str | None = None) -> JSONResponse:
            _refuse_query(request)
            if route.member.hands_out:
                registry.admit()
            return JSONResponse(await _on_own_thread(lambda: work(handle_id), name))

        _sign(read, on_handle, None)
        app.add_api_route(endpoint=read, methods=['GET'], **common)
        return

    async def call(
        request: fastapi.Request, body: pydantic.BaseModel | None, handle_id: str | None = None
    ) -> JSONResponse:
        _refuse_query(request)
        if route.member.hands_out:
            registry.admit()
        sent = body.model_dump(include=body.model_fields_set) if body is not None else {}

        def invoke(owner: object) -> object:
            arguments = {
                p.name: wire_encoding.decode(
                    sent[p.name],
                    p.alternatives,
                    resolve_path=session.live_folder_path,
                    handle=registry.get,
                )
                for p in route.member.parameters
                if p.name in sent
            }
            return getattr(owner, route.member.name)(**arguments)

        work = act_on(invoke)
        return JSONResponse(await _on_own_thread(lambda: work(handle_id), name))

    _sign(call, on_handle, route)
    app.add_api_route(endpoint=call, methods=['POST'], **common)


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
        raise fastapi.HTTPException(
            422, 'A member takes no query string: send its arguments as a JSON object.'
        )


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


def _no_jobs_yet(_future: Future) -> object:
    raise wire_encoding.NoWireFormError('this server does not yet hand out a job')
