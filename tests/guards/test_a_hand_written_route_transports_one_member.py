# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A hand-written route transports one marked member, and decides no capability itself.

Most of the wire is built from the marks (``rest.routes``): a route per
marked member, nothing between the client and the member but the wire's
own form. The rest is written by hand in ``rest/app.py`` -- a file's
bytes, an upload, a still's pixels, a preview, the live view -- and a
hand-written route can do what a built one cannot: reach past the API and
decide a capability in ``rest/``, where a Python, SDK or GUI caller never
meets it. So:

1. **Named.** Every hand-written route is in ``_TRANSPORTS`` with the one
   capability it carries -- a marked member, or the subscribe and
   unsubscribe pair the live view holds open -- or as the server's own
   (its versions, its events, its handles, its jobs), and its code reaches
   each member it is named with. A new hand-written route fails until it is
   named here.
2. **Decides nothing.** The routes' code -- the ``_add_*`` builders in
   ``rest/app.py``, the helpers of that file they reach, and
   ``rest/live.py`` -- calls no ``_lumascope`` function outside ``rest/``
   but the wire's own form: ``wire_encoding``, ``wall_time``, and
   ``image_utils.preview_jpeg``, the display JPEG of a frame. A member
   reached through the session is not such a call. What it calls besides is
   ``_DECIDED_IN_REST``, an equality list that only falls.

A static read of the routes' code, and one application built on a
simulated session for the route listing.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from collections.abc import Callable

import fastapi.routing
import pytest

from _lumascope.modules import image_utils, wall_time, wire_encoding
from _lumascope.modules.api_surface import mark_of
from _lumascope.modules.lumascope_api.imaging import ImagingAPI
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest import app as rest_app
from _lumascope.rest import live as rest_live
from tests import ratchets as _ratchets
from tests.settings_fixtures import complete_settings

# A route that holds the server's own records, transporting no member.
_SERVER = 'the server'
_FRAME_SUBSCRIPTION = (
    (ImagingAPI, 'add_frame_listener', rest_live.LiveView),
    (ImagingAPI, 'remove_frame_listener', rest_live.LiveView),
)

# Each hand-written route's endpoint, by its qualified name, and what it
# carries: (class, member, where the route's code reaches it -- None for
# the endpoint itself), or the server's own.
_TRANSPORTS = {
    'build_app.<locals>.versions': _SERVER,
    '_add_file_route.<locals>.download': ((ScopeSession, 'live_folder_path', None),),
    '_add_upload_route.<locals>.upload': ((ScopeSession, 'put_live_folder_file', None),),
    '_add_still_route.<locals>.still': ((ScopeSession, 'capture_still', None),),
    '_add_preview_route.<locals>.preview': ((ScopeSession, 'live_folder_path', None),),
    '_add_live_routes.<locals>.watch': _FRAME_SUBSCRIPTION,
    '_add_live_routes.<locals>.snapshot': _FRAME_SUBSCRIPTION,
    '_add_event_route.<locals>.events': _SERVER,
    '_add_event_route.<locals>.held': _SERVER,
    '_add_handle_routes.<locals>.held': _SERVER,
    '_add_handle_routes.<locals>.forget': _SERVER,
    '_add_job_routes.<locals>.listing': _SERVER,
    '_add_job_routes.<locals>.read': _SERVER,
    '_add_job_routes.<locals>.delete': _SERVER,
}

# The built routes' endpoints are defined in ``_add``, one per member.
_BUILT = '_add.<locals>.'

# The wire's own form, which a route may call: encoding, the wall clock,
# and a frame's display JPEG.
_WIRE_OWN_MODULES = {wire_encoding.__name__, wall_time.__name__}
_WIRE_OWN_FUNCTIONS = {image_utils.preview_jpeg}

# The calls the routes' code makes that decide a capability in rest/: an
# equality list, so a fix that forgets its row fails too. GAP-14: a preview
# of a saved image has no member yet; its pixels are read from the file here.
_DECIDED_IN_REST = {
    'app.py: _lumascope.modules.image_utils.load_pixels',
}


@pytest.fixture(scope='module')
def hand_written(tmp_path_factory) -> dict[str, tuple[Callable, list[str]]]:
    """Every hand-written route of an application, by its endpoint's qualified name: the endpoint, and each route's methods and path."""
    live = tmp_path_factory.mktemp('live')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        app = rest_app.build_app(session)
    finally:
        session.shutdown()
    found: dict[str, tuple[Callable, list[str]]] = {}
    for route in app.routes:
        if not isinstance(route, fastapi.routing.APIRoute):
            continue
        name = route.endpoint.__qualname__
        if name.startswith(_BUILT):
            continue
        found.setdefault(name, (route.endpoint, []))[1].append(
            f'{sorted(route.methods)} {route.path}'
        )
    return found


def _attributes_named(code_of: object) -> set[str]:
    """Every attribute name the code of *code_of* (a function or a class) reads or calls."""
    code = getattr(code_of, '__code__', None)
    source = textwrap.dedent(inspect.getsource(code if code is not None else code_of))
    return {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)}


def test_every_hand_written_route_is_named_with_what_it_transports(hand_written):
    unnamed = sorted(set(hand_written) - set(_TRANSPORTS))
    stale = sorted(set(_TRANSPORTS) - set(hand_written))
    assert unnamed == [], (
        'a hand-written route is not named with the member it transports, or as the '
        "server's own; name it in _TRANSPORTS:\n  "
        + '\n  '.join(f'{n}: {hand_written[n][1]}' for n in unnamed)
    )
    assert stale == [], 'a _TRANSPORTS row names no hand-written route:\n  ' + '\n  '.join(stale)

    wrong = []
    for name, carried in sorted(_TRANSPORTS.items()):
        if carried == _SERVER:
            continue
        for cls, member, where in carried:
            if mark_of(vars(cls).get(member)) is None:
                wrong.append(f'{name}: {cls.__name__}.{member} is not a marked member')
            reached_in = where if where is not None else hand_written[name][0]
            if member not in _attributes_named(reached_in):
                wrong.append(
                    f'{name}: {reached_in.__qualname__} never reaches {cls.__name__}.{member}'
                )
    assert wrong == [], 'a route is named with a member it does not transport:\n  ' + '\n  '.join(
        wrong
    )


def _routes_code() -> list[tuple[str, dict[str, object], ast.AST]]:
    """The routes' code: each piece's file name, its module's names, and its tree.

    In ``rest/app.py``, the ``_add_*`` builders and every function of the
    file they name, called or handed on (``asyncio.to_thread(_preview_of,
    ...)``); the built routes' ``_add`` is not one. All of ``rest/live.py``.
    """
    top = {
        n.name: n
        for n in ast.parse(inspect.getsource(rest_app)).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name != '_add'
    }
    frontier = [n for name, n in top.items() if name.startswith('_add_')]
    reached: dict[str, ast.AST] = {}
    while frontier:
        node = frontier.pop()
        if node.name in reached:
            continue
        reached[node.name] = node
        frontier.extend(
            top[n.id] for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in top
        )
    pieces = [('app.py', vars(rest_app), node) for _, node in sorted(reached.items())]
    pieces.append(('live.py', vars(rest_live), ast.parse(inspect.getsource(rest_live))))
    return pieces


def _target(func: ast.expr, names: dict[str, object]) -> object:
    """What a call's ``f`` or ``module.f`` names in its module; None for anything else."""
    if isinstance(func, ast.Name):
        return names.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = names.get(func.value.id)
        if isinstance(base, types.ModuleType):
            return getattr(base, func.attr, None)
    return None


def _calls_outside_rest() -> tuple[set[str], set[str]]:
    """The routes' calls to a ``_lumascope`` function outside ``rest/``: those deciding something, and the wire's own."""
    decided, wire_own = set(), set()
    for file, names, tree in _routes_code():
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            target = _target(call.func, names)
            module = getattr(target, '__module__', None) or ''
            if not isinstance(target, types.FunctionType) or not module.startswith('_lumascope.'):
                continue
            if module.startswith('_lumascope.rest'):
                continue
            label = f'{file}: {module}.{target.__name__}'
            if module in _WIRE_OWN_MODULES or target in _WIRE_OWN_FUNCTIONS:
                wire_own.add(label)
            else:
                decided.add(label)
    return decided, wire_own


def test_a_hand_written_route_decides_no_capability_in_rest():
    decided, _ = _calls_outside_rest()
    _ratchets.observe(
        'hand-written routes deciding in rest/', len(decided), len(_DECIDED_IN_REST), 'equal'
    )
    assert decided - _DECIDED_IN_REST == set(), (
        "a hand-written route calls a function outside rest/ that is not the wire's own "
        'form, deciding there what a member should; transport a marked member instead:\n  '
        + '\n  '.join(sorted(decided - _DECIDED_IN_REST))
    )
    assert _DECIDED_IN_REST - decided == set(), (
        'a _DECIDED_IN_REST row is no longer called; remove it:\n  '
        + '\n  '.join(sorted(_DECIDED_IN_REST - decided))
    )


def test_the_read_sees_the_wire_s_own_calls_it_passes():
    # The read finds the calls it lets through, so its answer above is the
    # code's, not an empty read's.
    _, wire_own = _calls_outside_rest()
    assert {
        'app.py: _lumascope.modules.wire_encoding.encode',
        'app.py: _lumascope.modules.image_utils.preview_jpeg',
        'live.py: _lumascope.modules.image_utils.preview_jpeg',
    } <= wire_own
