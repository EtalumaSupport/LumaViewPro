# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""``docs/API_REFERENCE.md``: the declared API, rendered from the marks and the docstrings.

Every owner of a marked member or published field, with how a caller
reaches it; every member's signature, marks, route and docstring, and the
refusals its ``Raises:`` and its doors declare with the REST status each
answers (``rest.problems.declared``); every closed set a marked annotation
names, with its values; every outcome class with its kind, reasons, causes
and status; and the tables a reader looks things up in -- the routes, the
events, the facts a saved file states. A fact here has its home in the
code: the file is regenerated, never edited, and
``tests/guards/test_the_api_is_declared.py`` holds it equal to a fresh
render.

The render is the same text on every interpreter the suite runs on: an
annotation is written by ``annotation_text`` rather than ``inspect``,
whose form of a class changes between Python versions, and a default by
``_value_text``.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import enum
import inspect
import sys
import types
import typing

from lib.lumascope_error import Fault, LumascopeError, defined_classes
from modules import wire_encoding
from modules.api_surface import (
    IN_PROCESS,
    admitted_by_of,
    event_of,
    field_descriptions,
    field_event,
    in_process_arguments_of,
    is_engineering,
    is_record,
    markdown_of,
    own_doc,
    reads_of,
    sets_of,
    stop_of,
)
from modules.exceptions import Notice, Quiet, Refusal
from modules.image_utils import FILE_FACTS
from modules.run_events import RunEvents
from modules.scope_session import ScopeSession
from rest import problems
from rest.app import PREFIX, served
from rest.events import _LIVE_VIEW, _event_name
from rest.routes import routes
from tests.ast_seams import REPO_ROOT
from tests.marked_set import Marked

REFERENCE = REPO_ROOT / 'docs' / 'API_REFERENCE.md'
# Set, the guard writes the render to REFERENCE instead of comparing it.
WRITE = 'LVP_WRITE_API_REFERENCE'
REGENERATE = f'{WRITE}=1 python3 -m pytest tests/guards/test_the_api_is_declared.py -k reference'
# A signature longer than this is written one parameter a line.
_LINE = 88

_PREFACE = f"""\
# LumaViewPro API reference

<!-- Generated from the code; do not edit. Regenerate: {REGENERATE} -->

Every member and field of the API, rendered from the source: a member is API when it carries `@api`, a field when its class names it in `@api_fields` (`modules/api_surface.py`). Anything else is not API, whatever its name, and can change in any release. A member marked *in-process* is API for a caller in the same process (Python, MATLAB calling Python) and is kept off the wire. How the API fits together, and how to use it, is `LumascopeSkills.md`.

A member's **Raises** is its contract: each outcome class it raises, with the reasons it quotes. **Refusals** adds the refusals of the doors the member enters (a hardware lane, a guarded write); for a member on the wire they are **Refusals by REST status**, under the HTTP status each answers. Every outcome class, with each reason's cause and status, is under [Outcomes](#outcomes).
"""


def annotation_text(annotation: object) -> str:
    """An annotation as source text, the same on every Python version.

    A string annotation is the source as written; any other is written from
    its parts, a class by its qualified name.

    Raises:
        TypeError: an annotation of a kind this does not write.
    """
    if isinstance(annotation, str):
        return annotation
    if annotation is None or annotation is type(None):
        return 'None'
    if annotation is Ellipsis:
        return '...'
    if isinstance(annotation, (typing.TypeAliasType, typing.TypeVar)):
        return annotation.__name__
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return ' | '.join(annotation_text(a) for a in args)
    if origin is typing.Annotated:
        return annotation_text(args[0])
    if origin is typing.Literal:
        return f'Literal[{", ".join(_value_text(a) for a in args)}]'
    if origin is not None:
        if not args:
            return annotation_text(origin)
        inner = ', '.join(
            f'[{", ".join(annotation_text(i) for i in a)}]'
            if isinstance(a, list)
            else annotation_text(a)
            for a in args
        )
        return f'{annotation_text(origin)}[{inner}]'
    if isinstance(annotation, type):
        return annotation.__qualname__
    # A special form (``ClassVar``, ``Final``) is written by typing as its dotted name.
    if repr(annotation).startswith('typing.'):
        return repr(annotation).removeprefix('typing.')
    raise TypeError(f'no source text for the annotation {annotation!r}')


def _value_text(value: object) -> str:
    """A default or a literal as source text: an enum member by its class and name."""
    if isinstance(value, enum.Enum):
        return f'{type(value).__qualname__}.{value.name}'
    text = repr(value)
    if ' at 0x' in text:
        raise TypeError(f'no source text for the value {text}, which names an address')
    return text


def _names(text: str) -> set[str]:
    """Every name an annotation's text uses."""
    found = set()
    for node in ast.walk(ast.parse(text, mode='eval')):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found |= _names(node.value)
    return found


def member_function(value: object) -> typing.Callable | None:
    """The function a member's value calls: a property's getter, a classmethod's function; None for a field."""
    if isinstance(value, property):
        return value.fget
    if isinstance(value, (classmethod, staticmethod)):
        return value.__func__
    return value if inspect.isfunction(value) else None


def _field_annotation(cls: type, name: str) -> str | None:
    for klass in cls.__mro__:
        annotations = vars(klass).get('__annotations__', {})
        if name in annotations:
            return annotation_text(annotations[name])
    return None


def _parameter(parameter: inspect.Parameter) -> str:
    text = parameter.name
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        text = f'*{text}'
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
        text = f'**{text}'
    annotated = parameter.annotation is not inspect.Parameter.empty
    if annotated:
        text += f': {annotation_text(parameter.annotation)}'
    if parameter.default is not inspect.Parameter.empty:
        text += (
            f' = {_value_text(parameter.default)}'
            if annotated
            else f'={_value_text(parameter.default)}'
        )
    return text


def signature_text(name: str, function: typing.Callable, *, decorator: str | None = None) -> str:
    """A member's signature as a ``def`` line, wrapped one parameter a line when long."""
    signature = inspect.signature(function)
    parts, starred, positional_only = [], False, False
    for parameter in signature.parameters.values():
        if positional_only and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            parts.append('/')
            positional_only = False
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only = True
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not starred:
            parts.append('*')
        if parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_POSITIONAL):
            starred = True
        parts.append(_parameter(parameter))
    if positional_only:
        parts.append('/')
    returned = signature.return_annotation
    tail = '' if returned is inspect.Signature.empty else f' -> {annotation_text(returned)}'
    line = f'def {name}({", ".join(parts)}){tail}'
    if len(line) > _LINE:
        line = f'def {name}(\n' + ''.join(f'    {p},\n' for p in parts) + f'){tail}'
    return line if decorator is None else f'@{decorator}\n{line}'


@dataclasses.dataclass(frozen=True)
class _Owner:
    """A section of the reference: a class, or a module's functions."""

    owner: type | types.ModuleType
    name: str
    items: tuple[Marked, ...]


@dataclasses.dataclass
class _Index:
    """What the render looks up across owners, built once from the declared API."""

    universe: dict[str, type]
    outcomes: dict[str, type[LumascopeError]]
    # Each class's name -> the marked items whose annotations name it, by edge.
    named_by: dict[str, dict[str, list[str]]]
    # (owner class, member name) -> its routes, ``METHOD /path``.
    routes: dict[tuple[type, str], list[str]]
    # Each sub-object's class -> its Python path and its wire path.
    reached: dict[type, tuple[str, str]]
    handed_out: frozenset[type]
    # Each record's name -> the members and fields that deliver it as an event.
    delivers: dict[str, list[str]]


def _edges(item: Marked) -> list[tuple[str, str]]:
    """``(edge, annotation text)``: a field's annotation, or a member's return and parameters."""
    function = member_function(item.value)
    if item.field or function is None:
        text = _field_annotation(item.owner, item.member)
        return [] if text is None else [('field', text)]
    signature = inspect.signature(function)
    edges = []
    if signature.return_annotation is not inspect.Signature.empty:
        edges.append(('return', annotation_text(signature.return_annotation)))
    for parameter in signature.parameters.values():
        if parameter.annotation is not inspect.Parameter.empty:
            edges.append(('parameter', annotation_text(parameter.annotation)))
    return edges


def outcome_classes() -> dict[str, type[LumascopeError]]:
    """Every outcome class a project module defines at its top level, by name, a private one aside.

    Only those: a class a test defines under the root is held by Python
    while the suite runs, and is no part of the API; a private one is its
    module's own (a driver's fault caught inside the driver, the umbrella
    mixin).
    """
    found = {}
    for cls in defined_classes():
        if cls.__name__.startswith('_'):
            continue
        module = sys.modules.get(cls.__module__)
        if cls.__module__.split('.')[0] not in wire_encoding.PROJECT_PACKAGES | {'lib'}:
            continue
        if module is None or getattr(module, cls.__qualname__, None) is not cls:
            continue
        found[cls.__name__] = cls
    return found


def _index(marked: dict[str, Marked], universe: dict[str, type], aliases: dict[str, str]) -> _Index:
    named_by: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    delivers: dict[str, list[str]] = collections.defaultdict(list)
    for key, item in sorted(marked.items()):
        for edge, text in _edges(item):
            for used in sorted(_names(text) & universe.keys()):
                if key not in named_by[used][edge]:
                    named_by[used][edge].append(key)
        record = None if item.field else event_of(item.value)
        if record is not None:
            delivers[record.__name__].append(key)
    for field in dataclasses.fields(RunEvents):
        record = field_event(RunEvents, field.name)
        if record is not None:
            delivers[record.__name__].append(f'RunEvents.{field.name}')
    classes, project_aliases = wire_encoding.project_classes(), wire_encoding.project_aliases()
    handed_out = wire_encoding.handed_out(ScopeSession, classes, project_aliases)
    found_routes: dict[tuple[type, str], list[str]] = collections.defaultdict(list)
    for root in (ScopeSession, *sorted(handed_out, key=lambda c: c.__name__)):
        for route in routes(root, handed_out=handed_out):
            path = served(route)[0]
            found_routes[(route.owner, route.member.name)].append(
                f'{route.method.upper()} {PREFIX}/{path}'
            )
    reached = {ScopeSession: ('session', f'{PREFIX}/')}
    queue = collections.deque([ScopeSession])
    while queue:
        cls = queue.popleft()
        path, wire = reached[cls]
        for member in wire_encoding.wire_members(cls, classes, project_aliases):
            if member.segment is not None and member.segment not in reached:
                reached[member.segment] = (f'{path}.{member.name}', f'{wire}{member.name}/')
                queue.append(member.segment)
    return _Index(
        universe=universe,
        outcomes=outcome_classes(),
        named_by=named_by,
        routes=found_routes,
        reached=reached,
        handed_out=handed_out,
        delivers=delivers,
    )


def _anchor(key: str) -> str:
    return f'<a id="{key}"></a>'


def _links(keys: list[str]) -> str:
    return ', '.join(f'[`{k}`](#{k})' for k in keys)


def _reached(cls: type, index: _Index) -> list[str]:
    """How a caller comes to hold *cls*: its path from the session, its handle, and what names it."""
    lines = []
    if cls in index.reached and cls is not ScopeSession:
        path, wire = index.reached[cls]
        lines.append(f'Reached through `{path}`; on the wire, `{wire}`.')
    if cls is ScopeSession:
        lines.append(
            'Made by [`ScopeSession.create`](#ScopeSession.create); every member below is on '
            f'the wire under `{PREFIX}/` unless it is in-process.'
        )
    if cls in index.handed_out:
        lines.append(
            f'A wire client holds one as a handle: `{PREFIX}/handles/{cls.__name__}/{{handle_id}}/`.'
        )
    named = index.named_by.get(cls.__name__, {})
    for edge, words in (
        ('return', 'Returned by'),
        ('field', 'Held in'),
        ('parameter', 'Taken by'),
    ):
        keys = [k for k in named.get(edge, []) if not k.startswith(f'{cls.__name__}.')]
        if keys:
            lines.append(f'{words}: {_links(keys)}.')
    if index.delivers.get(cls.__name__):
        lines.append(f'Delivered by: {_links(index.delivers[cls.__name__])}.')
    return lines


def _marks(item: Marked) -> list[str]:
    value = item.value
    marks = []
    if item.tag == IN_PROCESS:
        marks.append('in-process')
    if reads_of(value):
        marks.append('reads (changes nothing)')
    if sets_of(value):
        marks.append(f'sets [`{sets_of(value)}`](#{item.owner_name}.{sets_of(value)})')
    if stop_of(value):
        marks.append(f'stopped by [`{stop_of(value)}`](#{stop_of(value)})')
    if is_engineering(value):
        marks.append('engineering')
    if in_process_arguments_of(value):
        names = ', '.join(f'`{n}`' for n in in_process_arguments_of(value))
        marks.append(f'in-process arguments: {names}')
    if admitted_by_of(value):
        doors = ', '.join(f'`{d.value}`' for d in admitted_by_of(value))
        marks.append(f'enters: {doors}')
    record = event_of(value)
    if record is not None:
        marks.append(f'event: [`{record.__name__}`](#{record.__name__})')
    return marks


def _said(name: str, reasons: list[str]) -> str:
    return f'[`{name}`](#{name})' + (f': {", ".join(f"`{r}`" for r in reasons)}' if reasons else '')


def _refusals(doc: str, item: Marked, index: _Index) -> str:
    """The refusals a member's ``Raises:`` and its doors declare: by the REST status each answers, for a member on the wire."""
    declared = problems.declared(doc, index.outcomes, admitted_by_of(item.value))
    if not declared:
        return ''
    if index.routes.get((item.owner, item.member)):
        lines = [
            f'- {status}: ' + '; '.join(_said(n, r) for n, r in response['x-refusals'].items())
            for status, response in declared.items()
        ]
        return '**Refusals by REST status**\n\n' + '\n'.join(lines)
    merged: dict[str, list[str]] = {}
    for response in declared.values():
        for name, reasons in response['x-refusals'].items():
            said = merged.setdefault(name, [])
            said += [r for r in reasons if r not in said]
    return '**Refusals**\n\n' + '\n'.join(f'- {_said(n, r)}' for n, r in merged.items())


def _member(item: Marked, index: _Index) -> str:
    key = f'{item.owner_name}.{item.member}'
    value = item.value
    function = member_function(value)
    if isinstance(value, property):
        returned = inspect.signature(function).return_annotation
        code = f'{item.member}: {annotation_text(returned)}  # property'
    else:
        decorator = (
            'classmethod'
            if isinstance(value, classmethod)
            else 'staticmethod'
            if isinstance(value, staticmethod)
            else None
        )
        code = signature_text(item.member, function, decorator=decorator)
    parts = [f'{_anchor(key)}\n### `{key}`', f'```python\n{code}\n```']
    facts = []
    marks = _marks(item)
    if marks:
        facts.append('Marks: ' + '; '.join(marks) + '.')
    for route in index.routes.get((item.owner, item.member), []):
        facts.append(f'Route: `{route}`.')
    if facts:
        parts.append('  \n'.join(facts))
    doc = inspect.getdoc(function) or ''
    if doc:
        parts.append(markdown_of(doc))
    refusals = _refusals(doc, item, index)
    if refusals:
        parts.append(refusals)
    return '\n\n'.join(parts)


def _fields(owner: _Owner) -> str:
    fields = [i for i in owner.items if i.field]
    if not fields:
        return ''
    described = field_descriptions(owner.owner)
    lines = []
    for item in fields:
        key = f'{owner.name}.{item.member}'
        text = _field_annotation(owner.owner, item.member)
        typed = f': `{text}`' if text is not None else ''
        words = (
            markdown_of(described[item.member]) if item.member in described else '(no description)'
        )
        lines.append(f'- {_anchor(key)}`{item.member}`{typed} -- {words}')
    return '**Fields**\n\n' + '\n'.join(lines)


def _kind(cls: type) -> str:
    for kind, words in (
        (Quiet, 'quiet outcome'),
        (Refusal, 'refusal'),
        (Notice, 'notice'),
        (Fault, 'fault'),
    ):
        if issubclass(cls, kind):
            return f'the {words} kind' if cls is kind else f'a {words}'
    return 'a subject: never raised, its children are'


def _reasons(cls: type) -> str:
    # A subclass carries its parent's reasons; a kind and a subject carry none.
    causes = getattr(cls, 'causes', None)
    if not causes:
        return ''
    rows = [
        f'| `{reason}` | {cause.value if cause is not None else "--"} | '
        f'{problems._status_of(cls, reason)} |'
        for reason, cause in sorted(causes.items())
    ]
    return '| Reason | Cause | REST status |\n|---|---|---|\n' + '\n'.join(rows)


def _class(owner: _Owner, index: _Index, *, outcome: bool = False) -> str:
    cls = owner.owner
    level = '###' if outcome else '##'
    parts = [f'{_anchor(owner.name)}\n{level} `{owner.name}`']
    facts = []
    if outcome:
        bases = ', '.join(
            f'`{b.__name__}`' for b in cls.__bases__ if not b.__name__.startswith('_')
        )
        kind = _kind(cls)
        facts.append(f'{kind[0].upper()}{kind[1:]}; bases {bases}.')
    elif is_record(cls):
        facts.append('A record: it crosses the wire as its data.')
    facts += _reached(cls, index)
    if facts:
        parts.append('  \n'.join(facts))
    doc = own_doc(cls)
    if doc:
        parts.append(markdown_of(doc, leave_out=() if outcome else ('Attributes',)))
    for text in (_fields(owner), _reasons(cls) if outcome else ''):
        if text:
            parts.append(text)
    for item in owner.items:
        if not item.field:
            parts.append(_member(item, index))
    return '\n\n'.join(parts)


def _module(owner: _Owner, index: _Index) -> str:
    parts = [
        f'{_anchor(owner.name)}\n## `{owner.name}`',
        "Functions of the module; each is in-process, since the wire routes only a class's members.",
    ]
    for item in owner.items:
        parts.append(_member(item, index))
    return '\n\n'.join(parts)


def _closed_sets(index: _Index) -> list[type[enum.Enum]]:
    """Every enum a marked annotation names, by name."""
    found = [index.universe[n] for n in index.named_by if n in index.universe]
    return sorted((c for c in found if issubclass(c, enum.Enum)), key=lambda c: c.__name__)


def _closed_set(cls: type[enum.Enum], index: _Index) -> str:
    parts = [f'{_anchor(cls.__name__)}\n### `{cls.__name__}`']
    facts = _reached(cls, index)
    if facts:
        parts.append('  \n'.join(facts))
    doc = own_doc(cls)
    if doc:
        parts.append(markdown_of(doc))
    parts.append(
        '\n'.join(
            f'- `{m.value}`' + ('' if m.value == m.name.lower() else f' (`{m.name}`)') for m in cls
        )
    )
    return '\n\n'.join(parts)


def _routes_table(index: _Index) -> str:
    rows = []
    for (owner, member), found in index.routes.items():
        for route in found:
            method, path = route.split(' ', 1)
            rows.append(
                (path, method, f'[`{owner.__name__}.{member}`](#{owner.__name__}.{member})')
            )
    lines = [f'| `{method}` | `{path}` | {member} |' for path, method, member in sorted(rows)]
    return '| Method | Path | Member |\n|---|---|---|\n' + '\n'.join(lines)


def _events_table(index: _Index) -> str:
    rows = []
    for record, keys in index.delivers.items():
        for key in keys:
            member = key.rsplit('.', 1)[1]
            name = _event_name(member)
            stream = 'live view' if name in _LIVE_VIEW else 'event stream'
            rows.append((name, key, record, stream))
    lines = [
        f'| `{name}` | [`{key}`](#{key}) | [`{record}`](#{record}) | {stream} |'
        for name, key, record, stream in sorted(rows)
    ]
    return (
        'Each event, the member or `RunEvents` field that delivers it, its record (whose fields are '
        "the callback's arguments, in order), and what carries it to a REST client.\n\n"
        '| Event | Delivered by | Record | Over REST |\n|---|---|---|---|\n' + '\n'.join(lines)
    )


def _file_facts_table() -> str:
    lines = [
        f'| `{key}` | {fact.kind.value} | {f"`{fact.plane}`" if fact.plane else "--"} | '
        f'{fact.unit or "--"} | {"yes" if fact.dropped_by_a_sum else "no"} |'
        for key, fact in FILE_FACTS.items()
    ]
    return (
        "Every fact a saved file's record can state, by its key (`modules/image_utils.FILE_FACTS`), "
        'in the order the writer writes them: what it describes (a derived file states a `capture` '
        "fact never, a `channel` fact only on its inputs' channel, a `shared` fact as all its inputs "
        "state it), its key in the file's Plane block, the unit written beside it, and whether a "
        'sum of frames drops it.\n\n'
        '| Key | Kind | Plane key | Unit | Dropped by a sum |\n|---|---|---|---|---|\n'
        + '\n'.join(lines)
    )


def _owners(
    marked: dict[str, Marked], index: _Index
) -> tuple[list[_Owner], list[_Owner], dict[str, _Owner]]:
    """The class owners in the reference's order, the module owners, and the outcome classes' owners by name."""
    grouped: dict[int, list[Marked]] = collections.defaultdict(list)
    for key in sorted(marked):
        grouped[id(marked[key].owner)].append(marked[key])
    classes, modules, outcomes = [], [], {}
    for items in grouped.values():
        first = items[0]
        owner = _Owner(first.owner, first.owner_name, tuple(items))
        if first.at_top_level:
            modules.append(owner)
        elif issubclass(first.owner, BaseException):
            outcomes[owner.name] = owner
        else:
            classes.append(owner)
    walk = list(index.reached)
    classes.sort(
        key=lambda o: (walk.index(o.owner) if o.owner in index.reached else len(walk), o.name)
    )
    modules.sort(key=lambda o: o.name)
    return classes, modules, outcomes


def render(marked: dict[str, Marked], universe: dict[str, type], aliases: dict[str, str]) -> str:
    """The reference's text for the declared API *marked*, over the classes it can name."""
    index = _index(marked, universe, aliases)
    classes, modules, outcome_owners = _owners(marked, index)
    closed = _closed_sets(index)
    contents = ['## Contents', '']
    contents += [f'- [`{o.name}`](#{o.name})' for o in classes]
    contents += [f'- [`{o.name}`](#{o.name})' for o in modules]
    contents += ['- [Closed sets](#closed-sets)']
    contents += ['- [Outcomes](#outcomes)', '- [Routes](#routes)', '- [Events](#events)']
    contents += ['- [Saved-file facts](#saved-file-facts)']
    sections = [_PREFACE.rstrip('\n'), '\n'.join(contents)]
    sections += [_class(o, index) for o in classes]
    sections += [_module(o, index) for o in modules]
    sections.append(
        "## Closed sets\n\nA closed set crosses the wire as its value, and in Python equals its value's string."
    )
    sections += [_closed_set(c, index) for c in closed]
    sections.append(
        '## Outcomes\n\nEvery outcome class: its kind, its reasons, whether the same request could '
        "succeed later (each reason's cause), and the HTTP status REST answers it with."
    )
    for name, cls in sorted(index.outcomes.items()):
        owner = outcome_owners.get(name, _Owner(cls, name, ()))
        sections.append(_class(owner, index, outcome=True))
    sections.append('## Routes\n\n' + _routes_table(index))
    sections.append('## Events\n\n' + _events_table(index))
    sections.append('## Saved-file facts\n\n' + _file_facts_table())
    return '\n\n'.join(sections) + '\n'
