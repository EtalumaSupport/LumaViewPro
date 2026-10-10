# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every event declares its published record, and the record is the callback's arguments.

A listener's payload was named only in its docstring -- ``(axis, position,
state)``, ``(scan_number, scans_remaining, interval)`` -- so a host that
carries an event elsewhere (the REST server's stream) had to keep its own
table of each signature, a second store that drifts. Each event now
declares a published record where it is delivered (``modules.api_surface``):
a listener member as ``@api(event=Record)``, a ``RunEvents`` field as
``event_metadata(Record)``. This guard holds every marked ``add_*_listener`` member
and every ``RunEvents`` field to a declaration, and each record's fields,
in order, to the callback's argument types. A callback whose one argument
is a record declares that record; one with no arguments declares the
record its host reads when the event is sent, and has no arguments to
compare.

Annotations are compared as source text, normalized, as
``test_the_api_is_declared`` reads them: no session, no ``get_type_hints``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import warnings

import pytest

from modules.api_surface import event_of, field_event, fields_of, mark_of

# Declarations whose absence would mean the walk lost its way.
NOT_VACUOUS = {'MotionAPI.add_position_listener', 'RunEvents.scan_started'}


@pytest.fixture(scope='module')
def universe() -> dict[str, type]:
    import modules

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for info in pkgutil.walk_packages(modules.__path__, 'modules.'):
            importlib.import_module(info.name)
    found = {}
    for name, module in list(importlib.sys.modules.items()):
        if not name.startswith('modules'):
            continue
        for cname, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == name:
                found[cname] = obj
    return found


def _norm(text: str) -> str:
    """An annotation's text, with string forward references unwrapped."""
    node = ast.parse(text, mode='eval').body
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _norm(node.value)
    return ast.unparse(node)


def _callback_arguments(text: str) -> list[str] | None:
    """The argument types of the ``Callable`` an annotation holds, ``X | None`` allowed."""
    node = ast.parse(_norm(text), mode='eval').body
    if isinstance(node, ast.BinOp):
        sides = [node.left, node.right]
        node = next(s for s in sides if not (isinstance(s, ast.Constant) and s.value is None))
    if not (isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith('Callable')):
        return None
    arguments = node.slice.elts[0]
    return [_norm(ast.unparse(a)) for a in arguments.elts]


def _record_fields(record: type) -> list[str]:
    annotations = {}
    for klass in reversed(record.__mro__):
        annotations.update(vars(klass).get('__annotations__', {}))
    if dataclasses.is_dataclass(record):
        names = [f.name for f in dataclasses.fields(record)]
    else:
        names = list(getattr(record, '_fields', ()))
    return [
        _norm(a) if isinstance(a, str) else inspect.formatannotation(a)
        for a in (annotations[n] for n in names)
    ]


def _events(universe):
    """``(where, declared record, callback annotation text)`` for every event."""
    events = []
    for cname, cls in sorted(universe.items()):
        for member, value in vars(cls).items():
            if not (member.startswith('add_') and member.endswith('_listener')):
                continue
            if mark_of(value) is None:
                continue
            function = getattr(value, '__func__', value)
            parameters = [
                p for p in inspect.signature(function).parameters.values() if p.name != 'self'
            ]
            annotation = function.__annotations__.get(parameters[0].name)
            text = (
                annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
            )
            events.append((f'{cname}.{member}', event_of(value), text))
    from modules.run_events import RunEvents

    for field in dataclasses.fields(RunEvents):
        text = field.type if isinstance(field.type, str) else inspect.formatannotation(field.type)
        events.append((f'RunEvents.{field.name}', field_event(RunEvents, field.name), text))
    return events


def test_every_event_declares_a_published_record(universe):
    events = _events(universe)
    assert {where for where, _, _ in events} >= NOT_VACUOUS
    undeclared = [where for where, record, _ in events if record is None]
    unpublished = [
        f'{where}: {record.__name__}'
        for where, record, _ in events
        if record is not None and not fields_of(record)
    ]
    assert not (undeclared or unpublished), (
        'Every event declares the published record of its payload, where it is delivered:\n'
        f'  no record declared: {undeclared}\n'
        f'  record without @api_fields: {unpublished}'
    )


def test_each_records_fields_are_the_callbacks_arguments(universe):
    mismatched = []
    for where, record, text in _events(universe):
        if record is None:
            continue
        arguments = _callback_arguments(text)
        assert arguments is not None, f'{where}: {text} is not a callback'
        if arguments in ([], [record.__name__]):
            continue
        fields = _record_fields(record)
        if fields != arguments:
            mismatched.append(f'{where}: callback {arguments}, {record.__name__} {fields}')
    assert not mismatched, (
        "An event's record names the callback's arguments, in order and of the same "
        'types:\n  ' + '\n  '.join(mismatched)
    )
