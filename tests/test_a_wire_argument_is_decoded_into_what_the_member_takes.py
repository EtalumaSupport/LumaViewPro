# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a wire client sends for a parameter becomes what the member takes.

``wire_encoding.inbound`` reads a parameter's annotation into the forms a
client may send, and ``decode`` turns the sent value back: seconds into a
``timedelta``, an enum's form into its member, an object into a record, a
handle's id into its object, a live-folder name into a path. A type a
client could not send unambiguously -- a job, a path inside an array, a
union a value could be read as either side of -- is refused when the host
is built, so no call ever guesses. The callbacks the host fills are not
sent at all.
"""

from __future__ import annotations

import datetime
import enum
import pathlib

import pytest

from modules import wire_encoding
from modules.exceptions import Remedy


@pytest.fixture(scope='module')
def classes():
    return wire_encoding.project_classes()


@pytest.fixture(scope='module')
def aliases():
    return wire_encoding.project_aliases()


class _Speed(enum.Enum):
    SLOW = 1
    FAST = 2


class _Colour(enum.StrEnum):
    RED = 'red'


def _decode(value, text, classes, aliases, **kw):
    alternatives = wire_encoding.inbound(text, classes, aliases)
    kw.setdefault('resolve_path', lambda name: pathlib.Path('/live') / name)
    kw.setdefault('handle', lambda ident, cls: (cls.__name__, ident))
    return wire_encoding.decode(value, alternatives, **kw)


def test_each_form_is_decoded_into_its_python_value(classes, aliases):
    local = {**classes, '_Speed': _Speed, '_Colour': _Colour}

    assert _decode(1.5, 'datetime.timedelta | None', local, aliases) == datetime.timedelta(
        seconds=1.5
    )
    assert _decode(None, 'datetime.timedelta | None', local, aliases) is None
    assert _decode('FAST', '_Speed', local, aliases) is _Speed.FAST
    assert _decode('red', '_Colour', local, aliases) is _Colour.RED
    assert _decode('runs/a.tsv', 'FilePath', local, aliases) == pathlib.Path('/live/runs/a.tsv')
    assert _decode('7', 'Protocol', local, aliases) == ('Protocol', '7')
    assert _decode([1, 2], 'tuple', local, aliases) == (1, 2)
    assert _decode([1, 2], 'tuple[int, ...]', local, aliases) == (1, 2)
    assert _decode(['x'], 'Iterable[str]', local, aliases) == ['x']
    assert _decode({'a': 1}, 'dict | None', local, aliases) == {'a': 1}


def test_a_record_is_built_from_its_published_fields(classes, aliases):
    sent = {'member': 'recover_file_writer', 'confirm_text': 'Go', 'cancel_text': 'Stay'}

    assert _decode(sent, 'modules.exceptions.Remedy', classes, aliases) == Remedy(**sent)


def test_an_alias_is_read_through_to_its_value(classes, aliases):
    names = {a.name for a in wire_encoding.inbound('SettingValue', classes, aliases)}

    assert names == {'str', 'int', 'float', 'bool', 'list', 'dict', 'None'}


def test_a_parameter_the_host_fills_is_not_sent(classes, aliases):
    assert wire_encoding.inbound('Callable[[int, str], None] | None', classes, aliases) == ()
    assert wire_encoding.inbound('modules.run_events.RunEvents | None', classes, aliases) == ()


@pytest.mark.parametrize(
    'text',
    [
        'concurrent.futures.Future[None]',
        'list[FilePath]',
        'str | FilePath',
        'Protocol | str',
        'threading.Thread',
        'tuple[int, str]',
    ],
    ids=[
        'a job',
        'a path in an array',
        'a name or a path',
        'a handle or a name',
        'a thread',
        'a tuple of fixed shape',
    ],
)
def test_a_type_a_client_cannot_send_unambiguously_is_refused(classes, aliases, text):
    with pytest.raises(wire_encoding.NoWireFormError):
        wire_encoding.inbound(text, classes, aliases)


def test_every_session_parameter_has_an_inbound_form(classes, aliases):
    from modules.scope_session import ScopeSession

    seen, queue, parameters = set(), [ScopeSession], 0
    while queue:
        cls = queue.pop()
        if cls in seen:
            continue
        seen.add(cls)
        for member in wire_encoding.wire_members(cls, classes, aliases):
            parameters += len(member.parameters)
            if member.segment is not None:
                queue.append(member.segment)

    # Thirteen live objects, the Session's sub-objects among them.
    assert len(seen) == 13
    assert parameters > 100
