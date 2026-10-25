# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The API is the marked set, the reference indexes exactly it, and it is closed.

A member is API when it carries ``@api``; a class publishes the data
attributes its ``@api_fields`` names (``modules/api_surface.py``). Before the
marker, the surface was inferred from names, so whoever named a member decided
whether it was API, and nothing asked what a published member hands out. This
guard holds four things over every class defined under ``modules/``:

1. **Equal.** The marked set equals the ``## API reference`` index in
   ``docs/LumascopeSkills.md``, both ways, in-process labels included. The
   failure names each side's extras.
2. **Real.** Every published field is a field the class has: a dataclass or
   NamedTuple field, a class attribute, or assigned as ``self.<name>`` in the
   class body.
3. **Typed.** Every marked member and field says what it hands out and takes
   in: a return annotation, every parameter's, the field's. No ``Any`` or
   ``object`` appears in one and every ``Callable`` names its arguments, a
   callback's included. The one exception is ``object`` as a callback's
   return, the typing idiom for a return the API ignores.
4. **Closed.** Every project class an annotation names is an enum, an
   exception, or a class with marked members or fields, so nothing published
   hands out or asks for an object whose own members are not API.
5. **Paths declared.** A parameter of a member on the wire (marked, not
   in-process) that takes a file-system path is annotated exactly
   ``FilePath`` or ``FilePath | None``, so a wire client can tell a path
   from a string and pass it through ``ScopeSession.live_folder_path``.
6. **One vocabulary.** No marked member's parameter and no published field
   is named in a word the API retired for another (``layer`` for
   ``channel``, ``position`` for a turret ``slot``, ``plate_key`` for
   ``labware_id``, ``idx`` for ``step_index``, ...), so a member written
   later cannot bring one back.
7. **Every quantity names its unit.** A marked member's ``float``
   parameter and a published ``float`` field end in a unit suffix
   (``UNIT_SUFFIXES``), or are in ``UNITLESS`` with the reason they carry
   none; a float inside a ``Callable`` or a ``dict`` has no name of its own
   and is not judged. No marked annotation names ``timedelta``: a duration
   is a float of seconds. A return's unit is in its member's name or its
   record, so a return is not judged by name.
8. **Every fixed word is a closed set.** No marked member's parameter or
   return and no published field names a plain ``Enum`` or an ``IntEnum``:
   a closed set is a ``StrEnum``, so it crosses the wire as its word and
   equals its plain string. Every one that holds a ``str`` -- a parameter,
   a return, a field, a dict's key or a callback's argument -- is in
   ``OPEN_STRINGS`` with the reason its words are open (a channel key, a
   catalogue key, a reason word, words a person reads, a camera's, board's
   or OS's own word, ...), so a fixed list of words written later as a
   ``str`` fails here until it is a ``StrEnum`` or its reason is given.

Annotations are read as strings and resolved by class name over the loaded
classes, never through ``typing.get_type_hints``, which raises on the
``TYPE_CHECKING`` imports the API modules use. A static read: no session.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import pkgutil
import re
import warnings

import pytest

from tests.ast_seams import REPO_ROOT, RetiredNames, Words

DOC = REPO_ROOT / 'docs' / 'LumascopeSkills.md'
INDEX_HEADING = '## API reference'
IN_PROCESS_LABEL = '(in-process)'

# Members whose absence would mean the walk lost its way, one per kind of
# owner: a Session member, a sub-API member, a run handle's, a run outcome's
# field, and an outcome delivered to a listener.
NOT_VACUOUS = {
    'ScopeSession.add_step',
    'CameraAPI.set_gain_db',
    'Operation.stop',
    'RunOutcome.status',
    'Notification.kind',
}

_BLIND = Words({'Any', 'object'})


@pytest.fixture(scope='module')
def universe() -> dict[str, type]:
    """Every class defined under modules/, by name, every module imported."""
    import modules

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for info in pkgutil.walk_packages(modules.__path__, 'modules.'):
            importlib.import_module(info.name)
    classes: dict[str, set[type]] = {}
    for name, module in list(importlib.sys.modules.items()):
        if not name.startswith('modules'):
            continue
        for cname, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == name:
                classes.setdefault(cname, set()).add(obj)
    ambiguous = sorted(n for n, found in classes.items() if len(found) > 1)
    assert not ambiguous, (
        f'two classes under modules/ share each of these names: {ambiguous}. An '
        'annotation names a class by name, so the closure check cannot tell them apart'
    )
    return {n: next(iter(found)) for n, found in classes.items()}


@pytest.fixture(scope='module')
def marked(universe: dict[str, type]) -> dict[str, str]:
    """``Class.member`` -> ``api`` or ``in_process``; fields -> ``api``."""
    from modules.api_surface import fields_of, mark_of

    found = {}
    for cname, cls in universe.items():
        for member, value in vars(cls).items():
            tag = mark_of(value)
            if tag is not None:
                found[f'{cname}.{member}'] = tag
        for field in fields_of(cls):
            found[f'{cname}.{field}'] = 'api'
    return found


def _index(text: str) -> dict[str, str]:
    """``Class.member`` -> ``api`` or ``in_process``, from the doc's index."""
    start = text.index(f'\n{INDEX_HEADING}\n')
    section = text[start + len(INDEX_HEADING) + 2 :]
    end = section.find('\n## ')
    section = section if end < 0 else section[:end]
    indexed, cls = {}, None
    for line in section.splitlines():
        heading = re.fullmatch(r'### `?([A-Za-z_][A-Za-z0-9_]*)`?', line.strip())
        if heading:
            cls = heading.group(1)
            continue
        item = re.fullmatch(r'- `([A-Za-z_][A-Za-z0-9_]*)`( \(in-process\))?', line.strip())
        if item:
            assert cls is not None, f'index item {line!r} sits under no class heading'
            indexed[f'{cls}.{item.group(1)}'] = 'in_process' if item.group(2) else 'api'
    return indexed


@pytest.fixture(scope='module')
def indexed() -> dict[str, str]:
    """pin-justified: the guard's subject is the reference's index text."""
    return _index(DOC.read_text(encoding='utf-8'))


def test_no_marked_name_is_private(marked: dict[str, str]) -> None:
    private = sorted(name for name in marked if name.split('.', 1)[1].startswith('_'))
    assert not private, f'members marked as API whose names say private: {private}'


def test_the_index_is_the_marked_set(marked: dict[str, str], indexed: dict[str, str]) -> None:
    unindexed = sorted(set(marked) - set(indexed))
    unmarked = sorted(set(indexed) - set(marked))
    mislabelled = sorted(
        f'{name}: marked {marked[name]}, indexed {indexed[name]}'
        for name in set(marked) & set(indexed)
        if marked[name] != indexed[name]
    )
    assert not (unindexed or unmarked or mislabelled), (
        f'{DOC.name} "{INDEX_HEADING}" and the @api / @api_fields marks disagree.\n'
        f'  marked, not indexed: {unindexed}\n'
        f'  indexed, not marked: {unmarked}\n'
        f'  labelled differently: {mislabelled}'
    )


def _self_assigned(cls: type) -> set[str]:
    """Names the class body assigns as ``self.<name>``, read from its source."""
    try:
        tree = ast.parse(inspect.getsource(cls).lstrip())
    except (OSError, TypeError):
        return set()
    return {
        target.attr
        for node in ast.walk(tree)
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AnnAssign, ast.AugAssign))
            else []
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == 'self'
    }


def test_every_published_field_exists(universe: dict[str, type]) -> None:
    from modules.api_surface import fields_of

    missing = []
    for cname, cls in universe.items():
        names = fields_of(cls)
        if not names:
            continue
        known = set(getattr(cls, '_fields', ())) | _self_assigned(cls)
        if dataclasses.is_dataclass(cls):
            known |= {f.name for f in dataclasses.fields(cls)}
        known |= {
            name
            for name in dir(cls)
            if not callable(inspect.getattr_static(cls, name))
            and not isinstance(
                inspect.getattr_static(cls, name), (property, classmethod, staticmethod)
            )
        }
        missing += [f'{cname}.{name}' for name in names if name not in known]
    assert not missing, f'@api_fields names a field the class does not have: {missing}'


def _text(annotation: object) -> str | None:
    """An annotation as source text; None when there is none."""
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _field_annotation(cls: type, name: str) -> str | None:
    for klass in cls.__mro__:
        annotations = vars(klass).get('__annotations__', {})
        if name in annotations:
            return _text(annotations[name])
    return None


def _edges(cls: type, member: str) -> list[tuple[str, str | None]]:
    """(edge, annotation text) for a marked member: its return and parameters, or the field."""
    value = vars(cls).get(member)
    function = value.fget if isinstance(value, property) else getattr(value, '__func__', value)
    if value is None or not callable(function) or inspect.isclass(function):
        return [('field', _field_annotation(cls, member))]
    function = inspect.unwrap(function)
    annotations = function.__annotations__
    edges = [('return', _text(annotations.get('return', inspect.Parameter.empty)))]
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in ('self', 'cls'):
            continue
        edges.append(
            (
                f'parameter {parameter.name}',
                _text(annotations.get(parameter.name, inspect.Parameter.empty)),
            )
        )
    return edges


def _named(text: str, *, holds_callbacks: bool = True) -> tuple[set[str], set[str]]:
    """(blind names, every name) an annotation uses.

    ``Any`` and ``object`` say nothing, and neither does a ``Callable`` that
    does not name the arguments it is called with. ``object`` as a
    ``Callable[[...], R]``'s ``R`` is the idiom for a return the API ignores,
    so there it is not counted blind when the annotation is one that hands
    the API a callback -- a parameter's or a field's (``holds_callbacks``) --
    and ``object`` is the whole return; nested in it, in what a member
    returns, or as ``Any`` anywhere, it is.
    """
    blind, names = set(), set()

    def visit(node: ast.AST, in_callback_return: bool) -> None:
        if isinstance(node, ast.Subscript):
            base = (
                node.value.attr
                if isinstance(node.value, ast.Attribute)
                else getattr(node.value, 'id', '')
            )
            if (
                base == 'Callable'
                and isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) == 2
            ):
                names.add(base)
                arguments, returned = node.slice.elts
                if isinstance(arguments, ast.Constant):
                    blind.add('Callable[..., R]')
                visit(arguments, in_callback_return)
                # Only the callback's own return may be a bare ``object``;
                # ``list[object]`` or ``object | None`` there still says nothing.
                bare_object = isinstance(returned, ast.Name) and returned.id == 'object'
                visit(returned, holds_callbacks and bare_object)
                return
            visit(node.value, in_callback_return)
            visit(node.slice, in_callback_return)
            return
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            names.add(name)
            ignored_return = in_callback_return and name == 'object'
            if (name in _BLIND or name == 'Callable') and not ignored_return:
                blind.add(name)
            if isinstance(node, ast.Attribute):
                visit(node.value, in_callback_return)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            inner_blind, inner = _named(node.value, holds_callbacks=holds_callbacks)
            names.update(inner)
            blind.update(inner_blind - ({'object'} if in_callback_return else set()))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, in_callback_return)

    visit(ast.parse(text, mode='eval'), False)
    return blind, names


def test_the_api_is_typed_and_closed(universe: dict[str, type], marked: dict[str, str]) -> None:
    owners = {name.split('.', 1)[0] for name in marked}
    untyped, blind, unpublished = [], [], {}
    for name in sorted(marked):
        cname, member = name.split('.', 1)
        for edge, text in _edges(universe[cname], member):
            if text is None:
                untyped.append(f'{name} {edge}')
                continue
            blind_names, names = _named(text, holds_callbacks=edge != 'return')
            if blind_names:
                blind.append(f'{name} {edge}: {text}')
            for used in names & universe.keys():
                cls = universe[used]
                if used in owners or issubclass(cls, (enum.Enum, BaseException)):
                    continue
                unpublished.setdefault(used, []).append(f'{name} {edge}')
    assert not (untyped or blind or unpublished), (
        'The API must say what every published member hands out and takes in, '
        'and publish every type it names.\n'
        f'  no annotation: {untyped}\n'
        f'  Any or object: {blind}\n'
        + ''.join(
            f'  {cls} is named by {where} but has no marked member\n'
            for cls, where in sorted(unpublished.items())
        )
    )


def test_the_walk_is_not_vacuous(marked: dict[str, str], indexed: dict[str, str]) -> None:
    assert marked.keys() >= NOT_VACUOUS, f'marks not found: {sorted(NOT_VACUOUS - marked.keys())}'
    assert indexed.keys() >= NOT_VACUOUS, (
        f'index entries not found: {sorted(NOT_VACUOUS - indexed.keys())}'
    )


# A parameter naming one of these takes a file-system path.
_PATH_NAMES = Words({'Path', 'PathLike', 'FilePath'})
_DECLARED_PATH = {'FilePath', 'FilePath | None'}
# A wire path parameter whose absence would mean the walk lost its way.
KNOWN_WIRE_PATH = 'ScopeSession.load_protocol parameter file_path'


def test_every_wire_path_is_declared(universe: dict[str, type], marked: dict[str, str]) -> None:
    seen, undeclared = set(), []
    for name in sorted(n for n, tag in marked.items() if tag == 'api'):
        cname, member = name.split('.', 1)
        for edge, text in _edges(universe[cname], member):
            if not edge.startswith('parameter ') or text is None:
                continue
            if not _named(text)[1] & _PATH_NAMES:
                continue
            seen.add(f'{name} {edge}')
            if text not in _DECLARED_PATH:
                undeclared.append(f'{name} {edge}: {text}')
    assert KNOWN_WIRE_PATH in seen, f'the walk did not reach {KNOWN_WIRE_PATH}'
    assert not undeclared, (
        'A path parameter on the wire must be annotated FilePath or FilePath | None '
        '(modules.api_surface), so a wire client knows which arguments are paths:\n  '
        + '\n  '.join(undeclared)
    )


# Words the API retired for one name each (W5a): a channel is a channel, a
# step or a scan is counted by its index, a plate is named by its labware_id,
# the turret's place is a slot, a period is period_s.
RETIRED_WORDS = RetiredNames(
    {
        'layer',
        'layers',
        'idx',
        'step_idx',
        'scan',
        'scan_number',
        'interval',
        'plate_key',
        'labware_name',
        'position',
        'turret_position',
    }
)
# A published name the rename that retires its word has not reached yet,
# with the part that renames it. Each entry must still be a hit, so the
# rename that lands deletes it here.
NOT_YET_RENAMED: dict[str, str] = {}
# A parameter whose absence would mean the walk lost its way.
KNOWN_PARAMETER = 'ScopeSession.add_step parameter protocol'


def test_the_api_speaks_one_vocabulary(universe: dict[str, type], marked: dict[str, str]) -> None:
    seen, hits = set(), set()
    for name in sorted(marked):
        cname, member = name.split('.', 1)
        for edge, _ in _edges(universe[cname], member):
            if edge == 'field':
                word, where = member, f'{name} field'
            elif edge.startswith('parameter '):
                word, where = edge.removeprefix('parameter '), f'{name} {edge}'
            else:
                continue
            seen.add(where)
            if word in RETIRED_WORDS:
                hits.add(where)
    assert KNOWN_PARAMETER in seen, f'the walk did not reach {KNOWN_PARAMETER}'
    stale = sorted(set(NOT_YET_RENAMED) - hits)
    assert not stale, f'renamed now, so delete them from NOT_YET_RENAMED: {stale}'
    offenders = sorted(hits - set(NOT_YET_RENAMED))
    assert not offenders, (
        'A published parameter or field is named in a retired word; use its one name '
        '(channel, step_index, scan_index, labware_id, slot, turret_slot, period_s):\n  '
        + '\n  '.join(offenders)
    )


# The unit suffixes a published float may end in (W5a): the unit is always
# last, so a reader of the name (Micro-Manager, PyDevice) finds it there.
UNIT_SUFFIXES = Words(
    {
        '_um',
        '_mm',
        '_um2',
        '_px2',
        '_s',
        '_ms',
        '_us',
        '_ns',
        '_ma',
        '_db',
        '_nm',
        '_hz',
        '_pct',
        '_fps',
        '_bytes',
        '_mb',
        '_degc',
        '_rpm',
        '_epoch_s',
    }
)
# A published float that carries no unit, with the reason. Each entry is
# read one by one, none admitted by pattern, and each must still be a hit,
# so a rename that gives it a unit deletes it here.
UNITLESS: dict[str, str] = {
    'ArgumentRefusedError.limits field': "the refused argument's bounds, in that argument's unit",
    'CameraAPI.set_black_level parameter level': 'a raw sensor level',
    'FrameRecord.black_level field': 'a raw sensor level',
    'CameraAPI.update_auto_gain_target_brightness parameter target_brightness': (
        'a fraction of full scale'
    ),
    'FrameRecord.saturated_fraction field': "a fraction of the frame's pixels",
    'SimulationAPI.hold_axis parameter at_fraction': "a fraction of the move's travel",
    'CameraLink.link_speed field': (
        'in the unit the camera declares (link_speed_unit), carried and never converted'
    ),
    'CellCounts.total_object_intensity field': (
        'pixel area times mean intensity in percent of full scale: an integrated '
        'intensity with no physical unit'
    ),
    'ImageStatistics.mean field': "the image's counts",
    'ImageStatistics.standard_deviation field': "the image's counts",
    'Objective.magnification field': 'a ratio',
    'Objective.numerical_aperture field': 'a ratio',
    'FilterBounds.min field': 'in the unit of the CellCountFilters field it bounds',
    'FilterBounds.max field': 'in the unit of the CellCountFilters field it bounds',
    'SizeSliderStops.positions field': 'slider positions',
    'SizeSliderStops.size_at parameter positions': 'slider positions',
    'SizeSliderStops.reach field': "a size in the measure's scale, um or px by the image's scale",
    'SizeSliderStops.sizes field': "sizes in the measure's scale, um or px by the image's scale",
    'SizeSliderStops.positions_of parameter sizes': (
        "sizes in the measure's scale, um or px by the image's scale"
    ),
}
# A float whose absence from the walk would mean it lost its way.
KNOWN_QUANTITY = 'AppliedExposure.applied_ms field'
_UNNAMED = Words({'Callable', 'dict', 'Mapping'})


def _names_a_float(text: str) -> bool:
    """Whether an annotation holds a float with a name of its own: not inside a Callable or a dict."""

    def visit(node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            base = (
                node.value.attr
                if isinstance(node.value, ast.Attribute)
                else getattr(node.value, 'id', '')
            )
            if base in _UNNAMED:
                return False
        if isinstance(node, ast.Name):
            return node.id == 'float'
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _names_a_float(node.value)
        return any(visit(child) for child in ast.iter_child_nodes(node))

    return visit(ast.parse(text, mode='eval'))


def test_every_quantity_names_its_unit(universe: dict[str, type], marked: dict[str, str]) -> None:
    seen, hits, durations = set(), set(), []
    for name in sorted(marked):
        cname, member = name.split('.', 1)
        for edge, text in _edges(universe[cname], member):
            if text is None:
                continue
            if 'timedelta' in _named(text)[1]:
                durations.append(f'{name} {edge}')
            if edge == 'field':
                word, where = member, f'{name} field'
            elif edge.startswith('parameter '):
                word, where = edge.removeprefix('parameter '), f'{name} {edge}'
            else:
                continue
            if not _names_a_float(text):
                continue
            seen.add(where)
            if not any(word.endswith(suffix) for suffix in UNIT_SUFFIXES):
                hits.add(where)
    assert KNOWN_QUANTITY in seen, f'the walk did not reach {KNOWN_QUANTITY}'
    assert not durations, f'a duration is a float of seconds, never a timedelta: {durations}'
    stale = sorted(set(UNITLESS) - hits)
    assert not stale, f'these carry a unit or are gone now, so delete them from UNITLESS: {stale}'
    offenders = sorted(hits - set(UNITLESS))
    assert not offenders, (
        "A published float's name says no unit; end it in one of UNIT_SUFFIXES, or, when it "
        'has none, enter it in UNITLESS with the reason:\n  ' + '\n  '.join(offenders)
    )


# The reasons a published str is open: its words are not a fixed list the
# API owns. Each entry below is read one by one, none admitted by pattern,
# and each must still be a hit, so a str made a StrEnum deletes it here.
_CHANNEL = "a channel key, the release's layer vocabulary (W5c)"
_OBJECTIVE = 'an objective catalogue key'
_LABWARE = 'a labware catalogue key'
_MODEL = 'a model catalogue key'
_TILING = 'a tiling catalogue key'
_FILTERSET = 'a filterset, as the catalogue or the board names it'
_CATALOGUE_TEXT = 'catalogue text, as the installation file writes it'
_WELL = 'a well label of the selected plate'
_TILE = "a tile label of the step's grid"
_STEP = "a protocol step's name or label"
_SEQUENCE = "a run's sequence name, the caller's"
_NAME = 'a name the caller gives'
_PROSE = 'words a person reads'
_REASON = 'a reason word: why a thing happened is a string; what it is, an enum'
_SETTING = 'a settings path'
_MEMBER = "an API member's or argument's name"
_TYPE = 'a type, as source text'
_OFFERED = "the values offered for the refused argument, any set's"
_PATH = 'a file-system path or file name, in process'
_CAMERA = "the camera's or its SDK's own word"
_BOARD = "the board's own report, as its firmware gives it"
_OS = "the operating system's own word"
_PLUGIN = "a plugin's name, namespace or version, or the host callback it failed in"
_CONTROL = "a control session's name or id"
_VERSION = 'a version, as its source writes it'
_COLOUR = 'a colour as text, or a channel key naming one'
_NUMBER_TEXT = 'a number given as its text'
_COLUMN = 'a column of the results file read'
_RECORD_KEYS = "a record's keys, not a set (TODO.md, outside W5b)"
_PROVENANCE = "the measuring run's record, as the characterization writes it"
_LISTENER = "a listener's own name"
_TAG = "the caller's tag for a snapshot"
_SOURCE = 'the module or function a post came from'
_CATEGORY = "composed from the GUI's label and the executor's action, the dedup identity"
_KEY = "an outcome's operation key, the caller's"
_HEADER = "the protocol file format's header line"
_LED_D = "part D: an LED by its channel's name or its address"

OPEN_STRINGS: dict[str, str] = {
    'ArgumentRefusedError.argument field': _MEMBER,
    'ArgumentRefusedError.member field': _MEMBER,
    'ArgumentRefusedError.offered field': _OFFERED,
    'ArgumentTypeRefusedError.argument field': _MEMBER,
    'ArgumentTypeRefusedError.declared field': _TYPE,
    'ArgumentTypeRefusedError.member field': _MEMBER,
    'BringUpRecord.substitution parameter setting': _SETTING,
    'BuildResult.message field': _PROSE,
    'BuildsResult.message field': _PROSE,
    'CameraAPI.apply parameter channel': _CHANNEL,
    'CameraAPI.pixel_format return': _CAMERA,
    'CameraAPI.read_pixel_format return': _CAMERA,
    'CameraAPI.save_camera_state parameter tag': _TAG,
    'CameraLink.link_speed_unit field': _CAMERA,
    'CameraLink.transport field': _CAMERA,
    'CatalogueNameRefusedError.argument field': _MEMBER,
    'CatalogueNameRefusedError.offered field': _OFFERED,
    'CellCountResult.message field': _PROSE,
    'ChannelIdentity.filterset field': _FILTERSET,
    'ChannelIdentity.find parameter key_name': _CHANNEL,
    'ChannelIdentity.model field': _MODEL,
    'ChannelIdentity.refuse_unless_on_scope parameter channel': _CHANNEL,
    'ChannelIdentity.refuse_unless_on_scope parameter member': _MEMBER,
    'ChannelIdentity.refuse_unless_on_scope parameter then': _PROSE,
    'ChannelRecord.display_color field': _COLOUR,
    'ChannelRecord.display_name field': _PROSE,
    'ChannelRecord.key_name field': _CHANNEL,
    'ChannelRecord.label field': _PROSE,
    'ControlSession.name field': _CONTROL,
    'ControlSessionRefusedError.control_session field': _CONTROL,
    'ControlsLock.control_named field': _PROSE,
    'ControlsLock.named return': _PROSE,
    'ControlsLock.run_named field': _PROSE,
    'DegradedOutput.fallback_reason field': _REASON,
    'DiagnosticRefusedError.control_session field': _CONTROL,
    'DiagnosticsAPI.get_camera_temperatures_degc return': _CAMERA,
    'DiagnosticsAPI.motor_fan_spin_times return': _RECORD_KEYS,
    'DisplayRates.frame_interval_percentiles_ms return': _RECORD_KEYS,
    'EnhanceResult.message field': _PROSE,
    'FailedCapture.cause field': _REASON,
    'FailedCapture.step_name field': _STEP,
    'FocusFailure.cause field': _REASON,
    'FocusFailure.step_name field': _STEP,
    'FrameRecord.camera_model field': _CAMERA,
    'FrameRecord.held_channel field': _CHANNEL,
    'FrameRule.provenance field': _PROVENANCE,
    'GraphOptions.x_axes field': _COLUMN,
    'GraphOptions.y_axes field': _COLUMN,
    'GraphResult.message field': _PROSE,
    'HardwareCommandRefusedError.control_session field': _CONTROL,
    'IlluminationAPI.add_led_listener parameter listener': _CHANNEL,
    'IlluminationAPI.ch2color return': _LED_D,
    'IlluminationAPI.color2ch parameter color': _LED_D,
    'IlluminationAPI.get_led_state parameter channel': _CHANNEL,
    'IlluminationAPI.led_off parameter channel': _LED_D,
    'IlluminationAPI.led_on parameter channel': _LED_D,
    'IlluminationAPI.remove_led_listener parameter listener': _CHANNEL,
    'IlluminationAPI.save_led_state parameter tag': _TAG,
    'ImagingAPI.add_frame_listener parameter name': _LISTENER,
    'ImagingAPI.set_scale_bar_color parameter color': _COLOUR,
    'InjectedFault.arguments field': _MEMBER,
    'InjectedFault.member field': _MEMBER,
    'Labware.name field': _CATALOGUE_TEXT,
    'Labware.standard field': _CATALOGUE_TEXT,
    'LedBoardInfo.firmware_date field': _BOARD,
    'LedBoardInfo.firmware_version field': _BOARD,
    'LedReading.name field': _CHANNEL,
    'LedSnapshot.tag field': _TAG,
    'LiveFolderEntry.name field': _PATH,
    'Lumascope.labware return': _LABWARE,
    'Lumascope.objectives return': _OBJECTIVE,
    'Lumascope.scope_models return': _MODEL,
    'ManualRecordingController.end_reason return': _REASON,
    'MotionAPI.move_to_well parameter well': _WELL,
    'MotionAPI.start_move_to_well parameter well': _WELL,
    'MotorBoardInfo.firmware_date field': _BOARD,
    'MotorBoardInfo.firmware_version field': _BOARD,
    'MotorBoardInfo.model field': _BOARD,
    'MotorBoardInfo.serial_number field': _BOARD,
    'NamespaceHealth.namespace field': _PLUGIN,
    'Notification.category field': _CATEGORY,
    'Notification.message field': _PROSE,
    'Notification.operation_key field': _KEY,
    'Notification.reason field': _REASON,
    'Notification.source field': _SOURCE,
    'Notification.title field': _PROSE,
    'Objective.description field': _PROSE,
    'Objective.id field': _OBJECTIVE,
    'Objective.short_name field': _CATALOGUE_TEXT,
    'ObjectiveQuestion.choices field': _OBJECTIVE,
    'ObjectiveQuestion.proposed field': _OBJECTIVE,
    'PartStatus.cause field': _REASON,
    'PartStatus.describe return': _PROSE,
    'PartStatus.detail field': _PROSE,
    'PluginNotLoaded.name field': _PLUGIN,
    'PluginNotLoaded.reason field': _REASON,
    'PluginNotLoaded.version field': _PLUGIN,
    'PluginRuntimeError.exc_type field': _TYPE,
    'PluginRuntimeError.hook field': _PLUGIN,
    'PluginRuntimeError.message field': _PROSE,
    'PluginRuntimeError.namespace field': _PLUGIN,
    'PluginRuntimeError.plugin_name field': _PLUGIN,
    'PluginStatus.name field': _PLUGIN,
    'PluginStatus.namespace field': _PLUGIN,
    'PluginStatus.version field': _PLUGIN,
    'PostProcessingAPI.draw_graph parameter x': _COLUMN,
    'PostProcessingAPI.draw_graph parameter y': _COLUMN,
    'PostProcessingAPI.draw_graph parameter x_label': _PROSE,
    'PostProcessingAPI.draw_graph parameter y_label': _PROSE,
    'PostProcessingAPI.draw_graph parameter title': _PROSE,
    'PostProcessingAPI.graph parameter x': _COLUMN,
    'PostProcessingAPI.graph parameter y': _COLUMN,
    'PostProcessingAPI.graph parameter x_label': _PROSE,
    'PostProcessingAPI.graph parameter y_label': _PROSE,
    'PostProcessingAPI.graph parameter title': _PROSE,
    'PostProcessingAPI.graph_options parameter x': _COLUMN,
    'PostProcessingAPI.graph_options parameter y': _COLUMN,
    'PostProcessingAPI.video parameter frame_rate_fps': _NUMBER_TEXT,
    'Protocol.PROTOCOL_FILE_HEADER field': _HEADER,
    'Protocol.capture_prefix return': _NAME,
    'Protocol.capture_root return': _PATH,
    'Protocol.channel_settings return': _CHANNEL,
    'Protocol.labware_id return': _LABWARE,
    'Protocol.tiling return': _TILING,
    'Protocol.validate_steps parameter objectives': _OBJECTIVE,
    'ProtocolRunRefusedError.control_session field': _CONTROL,
    'ProtocolRunner.run_autofocus parameter channel': _CHANNEL,
    'ProtocolRunner.run_autofocus parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.run_composite parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.run_protocol parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.run_single_scan parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.run_zstack parameter channel': _CHANNEL,
    'ProtocolRunner.run_zstack parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.start_autofocus parameter channel': _CHANNEL,
    'ProtocolRunner.start_autofocus parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.start_composite parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.start_protocol parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.start_single_scan parameter sequence_name': _SEQUENCE,
    'ProtocolRunner.start_zstack parameter channel': _CHANNEL,
    'ProtocolRunner.start_zstack parameter sequence_name': _SEQUENCE,
    'ProtocolSizeAdvisory.message field': _PROSE,
    'ProtocolsAPI.focus_z_um parameter then': _PROSE,
    'RecordingRefusedError.control_session field': _CONTROL,
    'Remedy.cancel_text field': _PROSE,
    'Remedy.confirm_text field': _PROSE,
    'RunFiles.not_written_reason field': _REASON,
    'RunOutcome.cleanup_failures field': _PROSE,
    'RunOutcome.merge_reason field': _REASON,
    'RunOutcome.message field': _PROSE,
    'RunOutcome.reason field': _REASON,
    'RunOutcome.title field': _PROSE,
    'RuntimeState.get_turret_config return': _OBJECTIVE,
    'RuntimeState.get_well_label return': _WELL,
    'RuntimeState.plate_to_stage parameter labware_id': _LABWARE,
    'RuntimeState.plate_transform parameter labware_id': _LABWARE,
    'RuntimeState.resolve_current_objective return': _OBJECTIVE,
    'RuntimeState.stage_to_plate parameter labware_id': _LABWARE,
    'SavedStill.channel field': _CHANNEL,
    'SavedStill.objective_id field': _OBJECTIVE,
    'SavedStill.well_label field': _WELL,
    'ScopeCapabilities.camera_model field': _CAMERA,
    'ScopeCapabilities.camera_pixel_formats field': _CAMERA,
    'ScopeCapabilities.camera_sdk_version field': _CAMERA,
    'ScopeCapabilities.camera_sensor field': _CAMERA,
    'ScopeCapabilities.camera_serial_number field': _CAMERA,
    'ScopeCapabilities.led_colors field': _CHANNEL,
    'ScopeCapabilities.model field': _MODEL,
    'ScopeModel.filterset field': _FILTERSET,
    'ScopeModel.id field': _MODEL,
    'ScopeSession.acting_in_control parameter session_id': _CONTROL,
    'ScopeSession.add_settings_listener parameter listener': _SETTING,
    'ScopeSession.add_shown_channel_listener parameter listener': _CHANNEL,
    'ScopeSession.add_step return': _STEP,
    'ScopeSession.add_step parameter channels': _CHANNEL,
    'ScopeSession.app_version return': _VERSION,
    'ScopeSession.apply_focus_to_channel_steps parameter channel': _CHANNEL,
    'ScopeSession.apply_tiling parameter tiling': _TILING,
    'ScopeSession.assign_turret_objective parameter objective_id': _OBJECTIVE,
    'ScopeSession.channel_acquire parameter channel': _CHANNEL,
    'ScopeSession.channel_auto_gain parameter channel': _CHANNEL,
    'ScopeSession.channel_auto_gain_lock parameter channel': _CHANNEL,
    'ScopeSession.channel_colors return': _CHANNEL,
    'ScopeSession.confirm_objective parameter objective_id': _OBJECTIVE,
    'ScopeSession.create parameter sim_model': _MODEL,
    'ScopeSession.end_control parameter session_id': _CONTROL,
    'ScopeSession.get_setting parameter path': _SETTING,
    'ScopeSession.labware_id return': _LABWARE,
    'ScopeSession.live_folder_listing parameter name': _PATH,
    'ScopeSession.live_folder_path parameter name': _PATH,
    'ScopeSession.model return': _MODEL,
    'ScopeSession.model_at_next_start return': _MODEL,
    'ScopeSession.new_protocol parameter tiling': _TILING,
    'ScopeSession.objective_id return': _OBJECTIVE,
    'ScopeSession.point_steps_at_turret_slot parameter objective_id': _OBJECTIVE,
    'ScopeSession.protocol_files_stuck_write return': _PROSE,
    'ScopeSession.put_live_folder_file parameter name': _PATH,
    'ScopeSession.remove_settings_listener parameter listener': _SETTING,
    'ScopeSession.remove_shown_channel_listener parameter listener': _CHANNEL,
    'ScopeSession.rename_step return': _STEP,
    'ScopeSession.rename_step parameter name': _STEP,
    'ScopeSession.save_channel_focus parameter channel': _CHANNEL,
    'ScopeSession.save_focus parameter channel': _CHANNEL,
    'ScopeSession.saved_focus_z_um parameter channel': _CHANNEL,
    'ScopeSession.select_labware parameter labware_id': _LABWARE,
    'ScopeSession.select_model parameter model': _MODEL,
    'ScopeSession.select_objective parameter objective_id': _OBJECTIVE,
    'ScopeSession.set_channel_acquire parameter channel': _CHANNEL,
    'ScopeSession.set_channel_auto_gain parameter channel': _CHANNEL,
    'ScopeSession.set_live_folder parameter folder': _PATH,
    'ScopeSession.set_protocol_capture_root parameter root': _PATH,
    'ScopeSession.set_protocol_filepath parameter file_path': _PATH,
    'ScopeSession.set_protocol_labware return': _LABWARE,
    'ScopeSession.set_protocol_labware parameter labware_id': _LABWARE,
    'ScopeSession.show_channel parameter channel': _CHANNEL,
    'ScopeSession.take_control return': _CONTROL,
    'ScopeSession.take_control parameter name': _CONTROL,
    'ScopeSession.turret_objective return': _OBJECTIVE,
    'ScopeSession.update_settings parameter path': _SETTING,
    'ScopeSession.update_step return': _STEP,
    'ScopeSession.update_step parameter channel': _CHANNEL,
    'ScopeSession.update_step parameter label': _STEP,
    'ScopeSession.user_settings_file return': _PATH,
    'SettingRefusedError.member field': _MEMBER,
    'SettingRefusedError.path field': _SETTING,
    'SettingsChanged.paths field': _SETTING,
    'SettingsSetAside.reason field': _REASON,
    'ShownChannel.channel field': _CHANNEL,
    'SimulatedModelAnswer.model field': _MODEL,
    'SimulationAPI.answer_model parameter model': _MODEL,
    'SimulationAPI.bind_camera_usb_driver parameter hardware_id': _OS,
    'SimulationAPI.bind_camera_usb_driver parameter service': _OS,
    'SimulationAPI.drop_replies parameter command': _BOARD,
    'SkippedBuild.message field': _PROSE,
    'StepRecord.channel field': _CHANNEL,
    'StepRecord.label field': _STEP,
    'StepRecord.name field': _STEP,
    'StepRecord.objective field': _OBJECTIVE,
    'StepRecord.stim_config field': _CHANNEL,
    'StepRecord.tile field': _TILE,
    'StepRecord.well field': _WELL,
    'Substitution.setting field': _SETTING,
    'SupportReportSaved.message return': _PROSE,
    'SupportReportSaved.title return': _PROSE,
    'TakenCapture.step_name field': _STEP,
    'TilingConfig.available field': _TILING,
    'TilingConfig.available_configs return': _TILING,
    'TilingConfig.default field': _TILING,
    'TilingConfig.default_config return': _TILING,
    'TurretSlotQuestion.objective_id field': _OBJECTIVE,
    'UsbDeviceBinding.date field': _OS,
    'UsbDeviceBinding.inf field': _OS,
    'UsbDeviceBinding.instance_id field': _OS,
    'UsbDeviceBinding.provider field': _OS,
    'UsbDeviceBinding.service field': _OS,
    'UsbDeviceBinding.version field': _OS,
    'UsbIdentityBindings.hardware_id field': _OS,
    'UsbIdentityBindings.name field': _OS,
    'UsbIdentityBindings.needs field': _OS,
    'Video.frame_rate_fps field': _NUMBER_TEXT,
    'WorkItem.name field': _PROSE,
}
# A string whose absence from the walk would mean it lost its way.
KNOWN_OPEN = 'Notification.category field'


def test_every_fixed_word_is_a_closed_set(
    universe: dict[str, type], marked: dict[str, str]
) -> None:
    loose, hits = [], set()
    for name in sorted(marked):
        cname, member = name.split('.', 1)
        for edge, text in _edges(universe[cname], member):
            if text is None:
                continue
            names = _named(text, holds_callbacks=edge != 'return')[1]
            for used in names & universe.keys():
                cls = universe[used]
                if issubclass(cls, enum.Enum) and not issubclass(cls, enum.StrEnum):
                    loose.append(f'{name} {edge}: {used}')
            if 'str' in names:
                hits.add(f'{name} {edge}')
    assert KNOWN_OPEN in hits, f'the walk did not reach {KNOWN_OPEN}'
    assert not loose, f'a closed set the API names is a StrEnum, never an Enum or IntEnum: {loose}'
    stale = sorted(set(OPEN_STRINGS) - hits)
    assert not stale, (
        f'these are a closed set or gone now, so delete them from OPEN_STRINGS: {stale}'
    )
    offenders = sorted(hits - set(OPEN_STRINGS))
    assert not offenders, (
        'A published str takes or answers a word: make its fixed list a StrEnum, or, when '
        'its words are open, enter it in OPEN_STRINGS with the reason:\n  ' + '\n  '.join(offenders)
    )
