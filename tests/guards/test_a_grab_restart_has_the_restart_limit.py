# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera command that may restart the grab waits long enough for every restart.

``restore_camera_state`` waited 15 s -- three value writes' worth -- though
it may restart the grab three times, and a grab restart on a Pylon body has
taken 11 s. A 22 s restore was declared failed while it ran on and finished
behind its caller. Pylon's conversion-gain and line-noise setters restart
the grab under the 5 s value limit the same way.

The limit is chosen by hand at each call site, so this guard derives both
sides instead of listing them: from ``drivers/``, every driver method that
enters ``update_camera_config()`` or calls ``stop_grabbing()``, directly or
through its own class's methods; from ``modules/lumascope_api/``, every
``_dispatch_camera`` and the restarts its body may reach. A command that may
restart the grab N times must wait at least N geometry limits.
"""

from __future__ import annotations

import ast
from pathlib import Path

from modules.lumascope_api.diagnostics import DiagnosticsAPI
from modules.lumascope_api.imaging import ImagingAPI
from tests.ast_seams import REPO_ROOT as _ROOT

_RESTART_PRIMITIVES = frozenset({'update_camera_config', 'stop_grabbing'})


def _receiver(value: ast.expr) -> tuple[str, ...] | None:
    """``self._driver`` as ('self', '_driver'); None for anything not a dotted name."""
    parts = []
    while isinstance(value, ast.Attribute):
        parts.insert(0, value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    return (value.id, *parts)


def _calls_on(node: ast.AST, receivers: set[tuple[str, ...]]) -> set[str]:
    """The method names called on any of ``receivers`` inside ``node``."""
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and _receiver(call.func.value) in receivers
    }


_DRIVER = {('self', '_driver'), ('self', '_scope', '_camera_driver')}


def _driver_calls(fn: ast.FunctionDef) -> set[str]:
    """The camera-driver methods ``fn`` calls, through the driver or a local name bound to it."""
    receivers = set(_DRIVER)
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and _receiver(node.value) in _DRIVER:
            receivers.update((t.id,) for t in node.targets if isinstance(t, ast.Name))
    return _calls_on(fn, receivers)


def _methods(tree: ast.AST) -> dict[str, dict[str, ast.FunctionDef]]:
    return {
        cls.name: {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
    }


def _restarting_driver_methods() -> frozenset[str]:
    """Every camera-driver method name whose body may restart the grab, on any camera.

    The camera classes are read as one namespace: the base ``Camera`` calls
    hooks (``_set_hardware_window``) that each driver implements.
    """
    classes = {}
    for path in sorted((_ROOT / 'drivers').glob('*.py')):
        for cls in (
            n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)
        ):
            classes[cls.name] = cls
    cameras = {'Camera'}
    grew = True
    while grew:
        grew = False
        for name, cls in classes.items():
            if name not in cameras and {ast.unparse(b) for b in cls.bases} & cameras:
                cameras.add(name)
                grew = True
    calls = {}
    for name in cameras:
        for fn in (m for m in classes[name].body if isinstance(m, ast.FunctionDef)):
            calls.setdefault(fn.name, set()).update(_calls_on(fn, {('self',)}))
    restarting = set(_RESTART_PRIMITIVES)
    grew = True
    while grew:
        grew = False
        for name, called in calls.items():
            if name not in restarting and called & restarting:
                restarting.add(name)
                grew = True
    return frozenset(restarting)


def _dispatches(path: Path):
    """Each ``_dispatch_camera`` in ``path``: (class, enclosing method, impl, timeout expression)."""
    tree = ast.parse(path.read_text())
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in (m for m in cls.body if isinstance(m, ast.FunctionDef)):
            for call in ast.walk(fn):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == '_dispatch_camera'
                ):
                    timeout = next(k.value for k in call.keywords if k.arg == 'timeout_s')
                    yield cls.name, fn, call.args[0], timeout


def _impl_names(cls_methods, fn: ast.FunctionDef, impl: ast.expr) -> list[str]:
    """The methods ``impl`` names: ``self.<m>``, or a parameter of ``fn`` resolved at its callers."""
    if isinstance(impl, ast.Attribute) and isinstance(impl.value, ast.Name):
        return [impl.attr]
    assert isinstance(impl, ast.Name), ast.unparse(impl)
    position = [a.arg for a in fn.args.args].index(impl.id) - 1
    names = []
    for caller in cls_methods.values():
        for call in ast.walk(caller):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == fn.name
            ):
                names.extend(_impl_names(cls_methods, caller, call.args[position]))
    assert names, f'{fn.name} dispatches {impl.id}, which no caller passes'
    return names


def _restarts(cls_methods, name: str, restarting: frozenset[str], seen=frozenset()) -> int:
    """How many grab restarts ``name``'s body may make: one per method on its call
    path that calls a restarting driver method itself."""
    if name in seen or name not in cls_methods:
        return 0
    fn = cls_methods[name]
    own = 1 if _driver_calls(fn) & restarting else 0
    return own + sum(
        _restarts(cls_methods, callee, restarting, seen | {name})
        for callee in _calls_on(fn, {('self',)})
    )


def _limit(owner: type, timeout: ast.expr) -> float | None:
    """The timeout's value from the owner's constants; None when it is the caller's declared work."""
    namespace = {'self': owner, 'imaging': ImagingAPI, 'max': max}
    try:
        return float(eval(compile(ast.Expression(timeout), '<timeout>', 'eval'), namespace))
    except NameError:
        return None


def _census():
    restarting = _restarting_driver_methods()
    owners = {'ImagingAPI': ImagingAPI, 'DiagnosticsAPI': DiagnosticsAPI}
    rows = []
    for path in sorted((_ROOT / 'modules' / 'lumascope_api').glob('*.py')):
        classes = _methods(ast.parse(path.read_text()))
        for cls_name, fn, impl, timeout in _dispatches(path):
            for name in _impl_names(classes[cls_name], fn, impl):
                rows.append(
                    (
                        f'{cls_name}.{name}',
                        _restarts(classes[cls_name], name, restarting),
                        _limit(owners[cls_name], timeout),
                        ast.unparse(timeout),
                    )
                )
    return rows


def test_the_census_finds_the_restarting_commands():
    """The instrument reports presence: the commands known to restart the grab
    are found, with the counts their bodies make."""
    counts = {name: restarts for name, restarts, _, _ in _census()}
    assert counts['ImagingAPI._set_frame_size_impl'] == 1
    assert counts['ImagingAPI._set_pixel_format_impl'] == 1
    assert counts['ImagingAPI._set_conversion_gain_mode_impl'] == 1
    assert counts['ImagingAPI._restore_camera_state_impl'] == 3
    assert counts['ImagingAPI._set_gain_db_impl'] == 0


def test_a_command_that_may_restart_the_grab_waits_for_every_restart():
    short = [
        f'{name}: may restart the grab {restarts} time(s) but waits {expr} = {limit} s'
        for name, restarts, limit, expr in _census()
        if limit is not None and limit < restarts * ImagingAPI._CAMERA_GEOMETRY_TIMEOUT_S
    ]
    assert not short, '\n'.join(short)
