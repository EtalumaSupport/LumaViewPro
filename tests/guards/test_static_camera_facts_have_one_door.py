# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the camera is has one door: ``scope.capabilities``.

The camera's model, serial number, timestamp clock, pixel formats, binning
sizes, auto modes and the largest frame the scope delivers were each
answered by two or three doors: the capabilities snapshot, an imaging
getter that read the profile or the SDK, and a diagnostics getter or a
direct driver read. Each door had its own answer for an absent camera, and
the largest frame had two sources that disagree on real hardware. The
capabilities snapshot is now the one door (ASK-5), built from the camera at
connect. This walk over the production modules refuses a read of these
facts off a driver or its profile anywhere but where the snapshot is
built, and refuses the deleted doors coming back.
"""

from __future__ import annotations

import ast

from tests.ast_seams import production_modules

_THE_DOOR = 'modules/scope_capabilities.py'

# Names whose read off a camera driver is a static camera fact.
_DRIVER_FACTS = frozenset(
    {
        'model_name',
        'get_model_name',
        '_device_serial',
        'device_serial',
        'timestamp_tick_frequency_hz',
        'get_supported_pixel_formats',
        'get_max_frame_size',
    }
)

# A camera profile's static facts.
_PROFILE_FACTS = frozenset(
    {
        'pixel_formats',
        'binning_sizes',
        'native_resolution',
        'model_name',
        'has_auto_gain',
        'has_auto_exposure',
    }
)

_DELETED_DOORS = {
    'ImagingAPI': (
        'get_supported_pixel_formats',
        'get_available_binning_sizes',
        'get_native_resolution',
        'camera_identity',
    ),
    'DiagnosticsAPI': ('get_microscope_model', 'get_camera_info'),
}


def _reads(tree: ast.AST):
    """Yield (lineno, what) for every read of a static camera fact in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node.value
            on_profile = (isinstance(base, ast.Attribute) and base.attr == 'profile') or (
                isinstance(base, ast.Name) and base.id == 'profile'
            )
            if on_profile and node.attr in _PROFILE_FACTS:
                yield node.lineno, f'profile.{node.attr}'
            elif node.attr in _DRIVER_FACTS:
                yield node.lineno, f'.{node.attr}'
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ('getattr', 'hasattr')
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _DRIVER_FACTS | _PROFILE_FACTS
        ):
            yield node.lineno, f'{node.func.id}(..., {node.args[1].value!r})'


def test_no_module_reads_a_static_camera_fact_off_the_camera():
    found = [
        f'{path}:{lineno}: {what}'
        for path, tree in production_modules()
        if path != _THE_DOOR
        for lineno, what in _reads(tree)
    ]
    assert not found, (
        'a static camera fact is read off the camera, not scope.capabilities:\n' + '\n'.join(found)
    )


def test_the_deleted_doors_stay_deleted():
    from modules.lumascope_api.diagnostics import DiagnosticsAPI
    from modules.lumascope_api.imaging import ImagingAPI

    owners = {'ImagingAPI': ImagingAPI, 'DiagnosticsAPI': DiagnosticsAPI}
    back = [
        f'{owner}.{name}'
        for owner, names in _DELETED_DOORS.items()
        for name in names
        if hasattr(owners[owner], name)
    ]
    assert not back, f'a second door for a static camera fact is back: {back}'


def test_the_walk_finds_every_kind_of_read():
    """The walk reports presence on a known positive, so its clean result means something."""
    source = (
        'a = camera.model_name\n'
        'b = self._driver.profile.binning_sizes\n'
        "c = getattr(driver, 'timestamp_tick_frequency_hz', None)\n"
        'd = driver.get_max_frame_size()\n'
        'e = profile.native_resolution\n'
    )
    assert sorted(lineno for lineno, _ in _reads(ast.parse(source))) == [1, 2, 3, 4, 5]
