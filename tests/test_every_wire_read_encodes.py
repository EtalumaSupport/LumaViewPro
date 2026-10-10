# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every value a wire client can read from a simulated session encodes.

The guard (``test_every_wire_edge_has_a_form``) holds the declared types to
a form; this holds the values. From a simulated session, homed, with the
live folder set, every wire-marked property and
published field of every live object reachable through them is read and
encoded (``wire_encoding.encode``), so a value whose runtime type is not
its annotation's -- a numpy scalar where the annotation says float, a list
where it says tuple -- is caught here rather than by a client. Paths are
named by their live-folder name where they are inside it, in either form
the live folder is stored in. An enum crosses as its name, a ``StrEnum`` as
its value, an ``IntEnum`` included, though it is an int.
"""

from __future__ import annotations

import json
import pathlib
from concurrent.futures import Future

import pytest

from modules import wire_encoding
from modules.api_surface import API, mark_of


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings

    s = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(s.scope)
        yield s
    finally:
        s.shutdown()


def _reads(cls: type) -> list[str]:
    names = list(wire_encoding._fields(cls))
    names += [
        n
        for n, v in wire_encoding._marked(cls).items()
        if isinstance(v, property) and mark_of(v) == API
    ]
    return names


def test_every_wire_read_from_the_session_encodes(session, tmp_path):
    seen, queue, read, failed = set(), [session], [], []

    def handle(obj):
        queue.append(obj)
        return {'handle': id(obj), 'type': type(obj).__name__}

    def job(future: Future):
        return {'job': id(future)}

    while queue:
        obj = queue.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        for name in _reads(type(obj)):
            where = f'{type(obj).__name__}.{name}'
            value = getattr(obj, name)
            try:
                encoded = wire_encoding.encode(
                    value, live_folder=pathlib.Path(tmp_path), handle=handle, job=job
                )
                json.dumps(encoded)
            except (wire_encoding.NoWireFormError, TypeError, ValueError) as e:
                failed.append(f'{where}: {type(e).__name__}: {e}')
            read.append(where)

    assert not failed, '\n'.join(failed)
    # The walk reached the sub-APIs and the records they hold.
    assert {'ScopeSession.status', 'ScopeSession.live_work', 'Lumascope.capabilities'} <= set(read)
    # 61 reads over 13 live objects at this writing.
    assert len(seen) >= 10, f'{len(read)} reads over {len(seen)} live objects'


def test_a_value_with_no_form_is_refused(tmp_path):
    import numpy as np

    def no_handle(obj):
        raise AssertionError(obj)

    with pytest.raises(wire_encoding.NoWireFormError, match='ndarray'):
        wire_encoding.encode(
            {'frame': np.zeros(2)}, live_folder=tmp_path, handle=no_handle, job=no_handle
        )


def test_a_path_inside_the_live_folder_is_named_in_either_form(tmp_path):
    real = tmp_path / 'real_live'
    (real / 'ProtocolData').mkdir(parents=True)
    link = tmp_path / 'link_live'
    link.symlink_to(real)

    def no_handle(obj):
        raise AssertionError(obj)

    for live, path in [(real, link / 'ProtocolData'), (link, real / 'ProtocolData')]:
        encoded = wire_encoding.encode(path, live_folder=live, handle=no_handle, job=no_handle)
        assert encoded == {'name': 'ProtocolData', 'host_path': str(path)}

    outside = wire_encoding.encode(tmp_path, live_folder=real, handle=no_handle, job=no_handle)
    assert outside == {'name': None, 'host_path': str(tmp_path)}


def test_an_int_enum_crosses_as_its_name_as_every_enum_but_a_str_enum_does(tmp_path):
    from modules.notification_center import OutcomeKind, Severity

    def no_handle(obj):
        raise AssertionError(obj)

    encoded = wire_encoding.encode(
        {'severity': Severity.WARNING, 'kind': OutcomeKind.REFUSAL},
        live_folder=tmp_path,
        handle=no_handle,
        job=no_handle,
    )

    assert encoded == {'severity': 'WARNING', 'kind': 'refusal'}
