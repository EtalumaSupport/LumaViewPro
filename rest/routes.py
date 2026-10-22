# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each route the server answers, from the members marked for the wire.

The Session is the root: its member ``m`` is ``/api/v1/m``, and a
sub-object reached through a read (``scope``, ``scope.motion``) is a path
segment, so ``ScopeSession.scope.motion.move_xy`` is
``/api/v1/scope/motion/move_xy``. Names are the Python names. A read
-- a property, or a method marked ``reads`` -- is ``GET``, a read method's
arguments in the query string; a setter marked ``sets`` is ``PUT`` on its
read's path, its arguments a JSON object by parameter name, answered with
the read after the write; any other method is ``POST``, its arguments a
JSON object by parameter name. Each is described and checked by models
built here from the parameter's inbound form (``wire_encoding.inbound``).
Its answer is described by a type built here from the member's outbound
form (``wire_encoding.outbound``), each record one named model.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import typing

import pydantic

from modules import wire_encoding
from modules.operation import Operation
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
# A query string carries only text, so its model converts: '2' to a
# number, 'true' to a bool. What it reads is then checked against the
# member's strict body model, as a sent body is.
_QUERY_CONFIG = pydantic.ConfigDict(strict=False, extra='forbid', allow_inf_nan=False)
# The forms a query value can carry, one text value each.
_QUERY_SCALARS = (
    wire_encoding.SCALAR,
    wire_encoding.SECONDS,
    wire_encoding.ENUM,
    wire_encoding.PATH,
    wire_encoding.HANDLE,
    wire_encoding.JOB,
)
# An answer's model only describes: the server sends what ``encode`` made,
# never checked against it. Open, since a record's subclass is sent with
# the keys it adds, and the description must not refuse what is sent.
ANSWER_CONFIG = pydantic.ConfigDict(extra='allow')


class Handle(pydantic.BaseModel):
    """A live object a client is handed: its id, and its class's name."""

    model_config = ANSWER_CONFIG
    handle: str
    type: str


class RunJob(pydantic.BaseModel):
    """The run a run event is of: the id of the job its call handed out."""

    model_config = ANSWER_CONFIG
    job: str


class PathAnswer(pydantic.BaseModel):
    """A path: its name in the live folder, None when it is outside it, and its path on the host."""

    model_config = ANSWER_CONFIG
    name: str | None
    host_path: str


class RemedyAnswer(pydantic.BaseModel):
    """The one action that answers a refusal: the Session member to call, and the words that offer it."""

    model_config = ANSWER_CONFIG
    member: str
    confirm_text: str
    cancel_text: str


class RefusalAnswer(pydantic.BaseModel):
    """A refusal a record holds, as a refused call's problem carries it.

    Beside these keys, any field the refusal's type publishes.
    """

    model_config = ANSWER_CONFIG
    kind: str
    reason: str | None
    title: str
    detail: str
    remedy: RemedyAnswer | None


class JobProgress(pydantic.BaseModel):
    """How far a job's call has got, as its member last said."""

    model_config = ANSWER_CONFIG
    percent: float
    detail: str | None


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
        body: The model a call's arguments are checked against, sent as a
            JSON body or read from the query; None for a property.
        answer: The type its answer is described by (``answer_type``).
        query: The model a read method's query string is read by
            (``query_model``); None for any other member.
        sets: The read a setter's ``PUT`` sits on and answers after the
            write; None for any other member.
    """

    root: type
    segments: tuple[str, ...]
    member: WireMember
    owner: type
    body: type[pydantic.BaseModel] | None
    answer: object
    query: type[pydantic.BaseModel] | None = None
    sets: WireMember | None = None

    @property
    def path(self) -> str:
        """The route's path below the version prefix: a setter's is its read's."""
        return '/'.join((*self.segments, (self.sets or self.member).name))

    @property
    def name(self) -> str:
        """The member's own path below the version prefix, which names its operation."""
        return '/'.join((*self.segments, self.member.name))

    @property
    def method(self) -> str:
        """The HTTP method the route answers, lower case."""
        if self.sets is not None:
            return 'put'
        return 'get' if self.member.read or self.member.reads else 'post'


def routes(root: type, *, handed_out: frozenset[type]) -> list[Route]:
    """Every route under *root*, through its sub-objects, sorted by path.

    ``handed_out`` is the live objects a client can be handed
    (``wire_encoding.handed_out`` from the Session): a parameter of any
    other live object's class is not sent.

    Raises:
        NoWireFormError: a parameter's type has no inbound form, or a read
            method's has none in a query string (``query_model``).
        TypeError: a sub-object leads back to one of the classes it is
            reached through, so its routes would never end; a method
            marked ``reads`` can answer a running call or a live object,
            which a ``GET`` would hand out; or a setter's ``sets`` names no
            read it pairs with (``_paired_read``).
    """
    classes = wire_encoding.project_classes()
    aliases = wire_encoding.project_aliases()
    records: dict[type, type[pydantic.BaseModel]] = {}
    found: list[Route] = []

    def walk(cls: type, segments: tuple[str, ...], through: tuple[type, ...]) -> None:
        members = wire_encoding.wire_members(cls, classes, aliases, handed_out=handed_out)
        by_name = {m.name: m for m in members}
        for member in members:
            if member.segment is not None:
                if member.segment in through:
                    raise TypeError(f'{"/".join(segments)}/{member.name} leads back to itself')
                walk(member.segment, (*segments, member.name), (*through, member.segment))
                continue
            body = None if member.read else _body(cls, member, records)
            query = None
            if member.reads:
                if member.returns_job or member.hands_out:
                    raise TypeError(
                        f'{"/".join((*segments, member.name))} is marked reads but can answer '
                        'a running call or a live object, which a GET would hand out'
                    )
                query = query_model(cls, member)
            elif member.read and member.returns_job:
                raise TypeError(
                    f'{"/".join((*segments, member.name))} is a property but can answer a '
                    'running call, which a GET would hand out'
                )
            if member.sets is not None:
                read = _paired_read(cls, member, by_name)
                found.append(
                    Route(root, segments, member, cls, body, answer_type(read.returns), sets=read)
                )
                continue
            found.append(
                Route(root, segments, member, cls, body, answer_type(member.returns), query)
            )

    walk(root, (), (root,))
    return sorted(found, key=lambda r: r.path)


def _paired_read(owner: type, setter: WireMember, by_name: dict[str, WireMember]) -> WireMember:
    """The read *setter*'s ``sets`` names, refused unless the two are one setting.

    The read is a published read on the same class; its parameters are the
    setter's key, by name and type; and what the setter writes is what the
    read answers: its one other argument is of the read's answer type (the
    read may add None), or its other arguments are the fields of the record
    the read answers, by name and type.

    Raises:
        TypeError: the read is missing, is not a read, or does not pair.
    """
    where = f'{owner.__name__}.{setter.name} sets {setter.sets!r}'
    read = by_name.get(setter.sets)
    if read is None or not (read.read or read.reads):
        raise TypeError(f'{where}, which is no read on {owner.__name__}')
    if setter.reads or setter.returns_job:
        raise TypeError(f'{where}, but changes nothing or hands out a running call')
    parameters = {p.name: p for p in setter.parameters}
    for key in read.parameters:
        given = parameters.pop(key.name, None)
        if given is None or _kinds(given.alternatives) != _kinds(key.alternatives):
            raise TypeError(f'{where}, whose key {key.name!r} the setter does not take alike')
    answered = _kinds(read.returns) - {_NONE}
    written = list(parameters.values())
    if len(written) == 1 and _kinds(written[0].alternatives) - {_NONE} == answered:
        return read
    records = [a for a in read.returns if a.form == wire_encoding.RECORD]
    record = records[0] if len(records) == 1 else None
    if (
        record is not None
        and answered == _kinds((record,))
        and [(p.name, _kinds(p.alternatives)) for p in written]
        == [(f, _kinds(alts)) for f, alts in record.fields]
    ):
        return read
    answers = sorted(name for _, name, _ in answered)
    raise TypeError(f'{where}, which answers {answers}, not what the setter writes')


_NONE = (wire_encoding.SCALAR, 'None', None)


def _kinds(alternatives: typing.Iterable[object]) -> set[tuple[str, str, type | None]]:
    """What each alternative is, sent or answered, by its form, its name and its class."""
    return {(a.form, a.name, a.cls) for a in alternatives}


def _body(
    owner: type, member: WireMember, records: dict[type, type[pydantic.BaseModel]]
) -> type[pydantic.BaseModel]:
    fields = {p.name: _field(p, records) for p in member.parameters}
    return pydantic.create_model(
        f'{owner.__name__}_{member.name}', __config__=_MODEL_CONFIG, **fields
    )


def query_model(owner: type, member: WireMember) -> type[pydantic.BaseModel]:
    """The model a read method's arguments are read from a query string by.

    One key a parameter, its text converted to the parameter's type; a
    list is its key repeated; a record is one key a field,
    ``<parameter>.<field>``, which ``sent_of_query`` puts back together. A
    key the member does not take is refused.

    Raises:
        NoWireFormError: a parameter has no form in a query string: an
            object, a list of anything but single values, a record of
            anything but single values; a required one that may be None,
            since a query string says None only by leaving its key out; or
            one a value could be read as either a name or a number, since
            text carries no type to tell them apart.
    """
    where = f'{owner.__name__}.{member.name}'
    fields: dict[str, tuple[object, object]] = {}
    for p in member.parameters:
        present = [a for a in p.alternatives if not _is_none(a)]
        nullable = len(present) < len(p.alternatives)
        if p.required and nullable:
            raise wire_encoding.NoWireFormError(
                f'{where} parameter {p.name}: required, and may be None, which a query '
                'string can say only by leaving the key out'
            )
        if any(a.form == wire_encoding.RECORD for a in present):
            if len(present) != 1:
                raise wire_encoding.NoWireFormError(
                    f'{where} parameter {p.name}: a record or something else, in a query'
                )
            for field, alternatives in present[0].fields:
                _refuse_no_query_form(where, f'{p.name}.{field}', alternatives)
                kind = _type(alternatives, {})
                fields[f'{p.name}__{field}'] = (
                    typing.Optional[kind],  # noqa: UP045 -- kind may be Annotated
                    pydantic.Field(default=None, alias=f'{p.name}.{field}'),
                )
            continue
        _refuse_no_query_form(where, p.name, present)
        fields[p.name] = _field(p, {})
    return pydantic.create_model(
        f'{owner.__name__}_{member.name}_query', __config__=_QUERY_CONFIG, **fields
    )


def _is_none(a: wire_encoding.Inbound) -> bool:
    return a.form == wire_encoding.SCALAR and a.name == 'None'


def _refuse_no_query_form(
    where: str, key: str, alternatives: typing.Iterable[wire_encoding.Inbound]
) -> None:
    single = []
    for a in alternatives:
        if _is_none(a):
            continue
        if a.form == wire_encoding.ARRAY and all(i.form in _QUERY_SCALARS for i in a.items):
            single += list(a.items)
        elif a.form in _QUERY_SCALARS:
            single.append(a)
        else:
            raise wire_encoding.NoWireFormError(f'{where} {key}: a {a.form} has no query form')
    # Text is any value sent as a string: a str, a name, a path, an id.
    values = [
        a
        for a in single
        if a.form == wire_encoding.SECONDS or (a.form == wire_encoding.SCALAR and a.name != 'str')
    ]
    texts = [a for a in single if a not in values]
    if texts and values:
        raise wire_encoding.NoWireFormError(
            f'{where} {key}: a name or a number, which a query string cannot tell apart'
        )


def sent_of_query(query: pydantic.BaseModel) -> dict[str, object]:
    """The arguments a read's query string gave, as a JSON body would give them.

    A record's ``<parameter>.<field>`` keys are put back together as the
    record's object; a record given in part is given so, and the member's
    body model refuses it.
    """
    sent: dict[str, object] = {}
    for key, value in query.model_dump(by_alias=True, include=query.model_fields_set).items():
        parameter, dot, field = key.partition('.')
        if dot:
            sent.setdefault(parameter, {})[field] = value
        else:
            sent[key] = value
    return sent


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
    if a.form == wire_encoding.JOB:
        return typing.Annotated[
            str, pydantic.Field(description='The id of the job an operation is.')
        ]
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
    if a.form == wire_encoding.REFUSAL:
        return RefusalAnswer
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


def _operation_progress() -> object:
    """What an operation's ``progress()`` answers, described from its declared type as every answer is."""
    members = wire_encoding.wire_members(
        Operation, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    (progress,) = [m for m in members if m.name == 'progress']
    return answer_type(progress.returns)


_OPERATION_PROGRESS = _operation_progress()


class Job(pydantic.BaseModel):
    """A call running on, or ended, read at ``/api/v1/jobs/<id>``.

    ``progress`` is how far it has got: a call's own report, or for a
    running call a member handed back, its operation's (a run's
    ``RunProgress``). ``result`` is the member's answer once the call has
    completed, and ``error`` the problem it ended in once it has failed.
    """

    model_config = ANSWER_CONFIG
    id: str
    member: str
    status: typing.Literal['running', 'completed', 'failed']
    requested: datetime.datetime
    ended: datetime.datetime | None
    progress: JobProgress | _OPERATION_PROGRESS
    result: typing.Any = pydantic.Field(default=None)
    error: dict | None = pydantic.Field(default=None)
