# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each route the server answers, from the members marked for the wire.

The Session is the root: its member ``m`` is ``/api/v1/m``, and a
sub-object reached through a read (``scope``, ``scope.motion``) is a path
segment, so ``ScopeSession.scope.motion.move_absolute`` is
``/api/v1/scope/motion/move_absolute``. Names are the Python names. A read
is ``GET``; a method is ``POST``, its arguments a JSON object by parameter
name, described and checked by a model built here from the parameter's
inbound form (``wire_encoding.inbound``). Its answer is described by a
type built here from the member's outbound form (``wire_encoding.outbound``),
each record one named model.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import typing

import pydantic

from modules import wire_encoding
from modules.wire_encoding import WireMember, WireParameter

# The Python type a JSON value of each scalar name is checked as.
_SCALARS: dict[str, type] = {
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'None': type(None),
}
# Strict, so a JSON string is never read as a number nor a number as a
# bool, and a key the member does not take is refused rather than dropped.
# No NaN or infinity: Python's JSON reader takes both, and no member's
# number means either.
_MODEL_CONFIG = pydantic.ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)
# An answer's model only describes: the server sends what ``encode`` made,
# never checked against it. Open, since a record's subclass is sent with
# the keys it adds, and the description must not refuse what is sent.
ANSWER_CONFIG = pydantic.ConfigDict(extra='allow')


class Handle(pydantic.BaseModel):
    """A live object a client is handed: its id, and its class's name."""

    model_config = ANSWER_CONFIG
    handle: str
    type: str


class PathAnswer(pydantic.BaseModel):
    """A path: its name in the live folder, None when it is outside it, and its path on the host."""

    model_config = ANSWER_CONFIG
    name: str | None
    host_path: str


class JobProgress(pydantic.BaseModel):
    """How far a job's call has got, as its member last said."""

    model_config = ANSWER_CONFIG
    percent: float
    detail: str | None


class Job(pydantic.BaseModel):
    """A call running on, or ended, read at ``/api/v1/jobs/<id>``.

    ``result`` is the member's answer once the call has completed, and
    ``error`` the problem it ended in once it has failed.
    """

    model_config = ANSWER_CONFIG
    id: str
    member: str
    status: typing.Literal['pending', 'running', 'completed', 'failed']
    requested: datetime.datetime
    ended: datetime.datetime | None
    progress: JobProgress | None
    result: typing.Any = pydantic.Field(default=None)
    error: dict | None = pydantic.Field(default=None)


# One model per record class, shared by every route and event that answers
# it, so each record is one named component however many routes reach it.
_ANSWER_RECORDS: dict[type, type[pydantic.BaseModel]] = {}


@dataclasses.dataclass(frozen=True)
class Route:
    """One member's route.

    Attributes:
        root: The class the route starts from: the Session, or a live
            object's class a client holds a handle to.
        segments: The reads from the root to the member's owner.
        member: The member.
        owner: The class that declares it.
        body: The model a call's JSON body is checked against; None for a read.
        answer: The type its answer is described by (``answer_type``).
    """

    root: type
    segments: tuple[str, ...]
    member: WireMember
    owner: type
    body: type[pydantic.BaseModel] | None
    answer: object

    @property
    def path(self) -> str:
        """The route's path below the version prefix."""
        return '/'.join((*self.segments, self.member.name))


def routes(root: type, *, handed_out: frozenset[type]) -> list[Route]:
    """Every route under *root*, through its sub-objects, sorted by path.

    ``handed_out`` is the live objects a client can be handed
    (``wire_encoding.handed_out`` from the Session): a parameter of any
    other live object's class is not sent.

    Raises:
        NoWireFormError: a parameter's type has no inbound form.
        TypeError: a sub-object leads back to one of the classes it is
            reached through, so its routes would never end.
    """
    classes = wire_encoding.project_classes()
    aliases = wire_encoding.project_aliases()
    records: dict[type, type[pydantic.BaseModel]] = {}
    found: list[Route] = []

    def walk(cls: type, segments: tuple[str, ...], through: tuple[type, ...]) -> None:
        for member in wire_encoding.wire_members(cls, classes, aliases, handed_out=handed_out):
            if member.segment is not None:
                if member.segment in through:
                    raise TypeError(f'{"/".join(segments)}/{member.name} leads back to itself')
                walk(member.segment, (*segments, member.name), (*through, member.segment))
                continue
            body = None if member.read else _body(cls, member, records)
            found.append(Route(root, segments, member, cls, body, answer_type(member.returns)))

    walk(root, (), (root,))
    return sorted(found, key=lambda r: r.path)


def _body(
    owner: type, member: WireMember, records: dict[type, type[pydantic.BaseModel]]
) -> type[pydantic.BaseModel]:
    fields = {p.name: _field(p, records) for p in member.parameters}
    return pydantic.create_model(
        f'{owner.__name__}_{member.name}', __config__=_MODEL_CONFIG, **fields
    )


def _field(
    parameter: WireParameter, records: dict[type, type[pydantic.BaseModel]]
) -> tuple[object, object]:
    kind = _type(parameter.alternatives, records)
    if parameter.required:
        return kind, ...
    # A parameter the client leaves out takes the member's own default: only
    # what the client sent is passed. The default here only describes it.
    try:
        described = json.loads(json.dumps(wire_form(parameter.default)))
    except (wire_encoding.NoWireFormError, TypeError, ValueError):
        return kind, pydantic.Field(default=None, json_schema_extra=_no_default)
    return kind, pydantic.Field(default=None, json_schema_extra={'default': described})


def wire_form(value: object) -> object:
    """*value* as a client sends it: a default, or an enum's member.

    A path is a host path, not the live-folder name a client sends, and a
    live object has no id until the server hands it out: neither has a form
    here.
    """

    def refuse(v: object) -> typing.NoReturn:
        raise wire_encoding.NoWireFormError(type(v).__name__)

    if isinstance(value, pathlib.PurePath):
        refuse(value)
    return wire_encoding.encode(value, live_folder=pathlib.Path(), handle=refuse, job=refuse)


def _no_default(schema: dict) -> None:
    schema.pop('default', None)


def _type(
    alternatives: tuple[wire_encoding.Inbound, ...],
    records: dict[type, type[pydantic.BaseModel]],
) -> object:
    kinds = tuple(_one(a, records) for a in alternatives)
    return kinds[0] if len(kinds) == 1 else typing.Union[kinds]  # noqa: UP007


def _one(a: wire_encoding.Inbound, records: dict[type, type[pydantic.BaseModel]]) -> object:
    if a.form == wire_encoding.SCALAR:
        return _SCALARS[a.name]
    if a.form == wire_encoding.SECONDS:
        return typing.Annotated[float, pydantic.Field(description='Seconds.')]
    if a.form == wire_encoding.PATH:
        return typing.Annotated[
            str, pydantic.Field(description='A name in the live folder, `/`-separated.')
        ]
    if a.form == wire_encoding.HANDLE:
        return typing.Annotated[str, pydantic.Field(description=f'The id of a {a.name} handle.')]
    if a.form == wire_encoding.ENUM:
        return typing.Literal[tuple(wire_form(m) for m in a.cls)]
    if a.form == wire_encoding.ARRAY:
        return list[_type(a.items, records)] if a.items else list
    if a.form == wire_encoding.OBJECT:
        return dict[str, _type(a.items, records)] if a.items else dict
    if a.cls not in records:
        fields = {f: (_type(alts, records), ...) for f, alts in a.fields}
        records[a.cls] = pydantic.create_model(a.name, __config__=_MODEL_CONFIG, **fields)
    return records[a.cls]


def answer_type(alternatives: tuple[wire_encoding.Outbound, ...]) -> object:
    """The type a value sent as *alternatives* is described by, for the OpenAPI description."""
    kinds = tuple(_answer(a) for a in alternatives)
    return kinds[0] if len(kinds) == 1 else typing.Union[kinds]  # noqa: UP007


def _answer(a: wire_encoding.Outbound) -> object:
    if a.form == wire_encoding.SCALAR:
        return _SCALARS[a.name]
    if a.form == wire_encoding.SECONDS:
        return typing.Annotated[float, pydantic.Field(description='Seconds.')]
    if a.form == wire_encoding.ISO8601:
        return datetime.datetime
    if a.form == wire_encoding.PATH:
        return PathAnswer
    if a.form == wire_encoding.HANDLE:
        return Handle
    if a.form == wire_encoding.JOB:
        return Job
    if a.form == wire_encoding.ENUM:
        return typing.Literal[tuple(wire_form(m) for m in a.cls)]
    if a.form == wire_encoding.ARRAY:
        if a.parts:
            return tuple[tuple(answer_type(p) for p in a.parts)]
        return list[answer_type(a.items)] if a.items else list
    if a.form == wire_encoding.OBJECT:
        return dict[str, answer_type(a.items)] if a.items else dict
    if a.cls not in _ANSWER_RECORDS:
        fields = {f: (answer_type(alts), ...) for f, alts in a.fields}
        _ANSWER_RECORDS[a.cls] = pydantic.create_model(
            a.name, __config__=ANSWER_CONFIG, __doc__=a.cls.__doc__, **fields
        )
    return _ANSWER_RECORDS[a.cls]
