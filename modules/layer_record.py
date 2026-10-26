# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Layer identity: what a layer IS on the attached unit.

One record per layer (stable key name, display name, LED board address,
excitation wavelength) plus the unit's filterset, resolved once into an
immutable snapshot. Identity is unit data, not code: different filtersets
carry different LEDs, so the truth lives in the unit's motorconfig
Channels block when it has one, and in the model's `scopes.json` rows for
units built before per-unit blocks existed.

Internally a layer is identified by its integer id. The names are fields
that get WRITTEN OUT (display, and serialisation under the stable
`key_name`); nothing in memory looks a layer up by its name past the
deserialisation boundary this module implements.

Resolution never raises: a scope with no resolvable identity gets the
empty `unresolved` snapshot, and the illumination API is where that
state becomes a loud, named error on first use. Failing construction
here would take down a scope whose camera and stage are fine.
"""

from __future__ import annotations

import enum
import os
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields

from lvp_logger import logger
from modules.exceptions import (
    ArgumentRefusedError,
    ChannelBlockSupersededNotice,
    ChannelBlockUnusableNotice,
    ChannelRowsSkippedNotice,
    ConfigurationDefectError,
    HardwareCommandRefusedError,
    InstallationFileError,
    MissingPart,
    Notice,
    ScopeModelUnknownError,
)
from modules.path_utils import read_installation_file, resolve_data_file
from modules.api_surface import api, api_fields
from modules.closed_sets import member_of
from modules.scope_capabilities import FrameSize


class Modality(enum.StrEnum):
    """The kind of light a layer images."""

    TRANSMITTED = 'transmitted'
    FLUORESCENCE = 'fluorescence'
    LUMINESCENCE = 'luminescence'


class Background(enum.StrEnum):
    """What a layer's field of view looks like, which is not its modality's:
    darkfield is transmitted light on a dark field."""

    BRIGHT = 'bright'
    DARK = 'dark'


# The one encoding of a stored colour: catalogue, board block and the user's
# override all write '#rrggbb', lower case, so one colour is one string.
_DISPLAY_COLOR = re.compile(r'#[0-9a-f]{6}')


# A colour as the red, green and blue weights, each 0 to 1, a mono image is
# multiplied by. White is grey: it is what a transmitted channel is and what
# any channel is shown in with false colour off.
Color = tuple[float, float, float]
GREY: Color = (1.0, 1.0, 1.0)


def parse_display_color(text: object) -> Color:
    """The red, green and blue weights, each 0 to 1, of a ``'#rrggbb'`` colour.

    Lower-case hex digits only: an upper-case colour is refused, so a
    colour has one stored form. Its brightest component is ``ff``: a
    channel baked in its colour then keeps the mono whole in its brightest
    plane, so a build reads it back without knowing the colour it was
    baked in, which a colour changed since the capture would contradict.
    How bright a channel is shown is the display range's, not its colour's.

    Raises:
        ValueError: ``text`` is not a ``'#rrggbb'`` string, or its
            brightest component is not ``ff``.
    """
    if not isinstance(text, str) or not _DISPLAY_COLOR.fullmatch(text):
        raise ValueError(f"{text!r} is not a '#rrggbb' colour")
    color = tuple(int(text[i : i + 2], 16) / 255 for i in (1, 3, 5))
    if max(color) != 1:
        raise ValueError(f'{text!r} is not at full brightness: its brightest component is not ff')
    return color


def format_display_color(color: Color) -> str:
    """The ``'#rrggbb'`` text of a colour, the inverse of ``parse_display_color``:
    each weight to the nearest of the 256 steps."""
    return '#' + ''.join(f'{round(weight * 255):02x}' for weight in color)


@dataclass(frozen=True)
class CatalogueLayer:
    """What a layer of one key is in this release, on every unit that has it.

    `background` is what the scale bar and annotation text are drawn
    against.
    `manual_max_exposure_ms` is the exposure slider's ceiling, None where the
    layer has none (luminescence integrates as long as the sensor allows).
    `display_color` is the release's colour for the layer, ``'#rrggbb'``.
    `step_order` is where the layer's steps are inserted in a new protocol,
    which is not its display order.
    `sim_light_per_ma_s` is the light the simulated cameras render for the
    layer, per mA of LED current and second of exposure at unity gain;
    None for a layer with no LED. Each value puts the layer's shipped
    defaults (`settings.json`) at brightness 1.0, the specimen at its own
    grey levels: 1 / (exposure s x linear gain x mA), so BF's 2 ms, 1 dB
    and 5 mA give 89.1 and fluorescence's 100 ms, 10 dB and 250 mA 0.0126.
    The defaults are what makes each layer look right on a scope, so they
    carry the light each modality delivers.
    """

    key: str
    modality: Modality
    background: Background
    manual_max_exposure_ms: float | None
    display_color: str
    step_order: int
    sim_light_per_ma_s: float | None


@api_fields(
    'background',
    'display_color',
    'display_name',
    'excitation_nm',
    'id',
    'key_name',
    'label',
    'led_addresses',
    'modality',
)
@dataclass(frozen=True)
class ChannelRecord:
    """Identity of one layer on one unit.

    `id` is the internal key and the display position -- assigned from
    the release catalogue's order, never persisted to any per-unit or
    user file, so it can renumber freely between releases.

    `key_name` is the stable serialisation name (settings block keys,
    protocol TSV `Color`, filename tokens, metadata `channel`). It never
    changes when a layer is renamed.

    `display_name` is the unit's name for the layer. Freely changeable: a
    rename touches this field and nothing persisted. None for a
    fluorescence channel a board's Channels block names by its excitation
    alone.

    `led_addresses` is the LED board address(es) this layer drives, empty
    for a layer with no LED (luminescence). A tuple rather than a single
    value because a future channel (quantitative phase) drives several
    switches as one layer; today every shipped row carries zero or one.
    A row in a model's file names them ``led_channel``, read into this
    field by the one row reader.

    `excitation_nm` is the excitation wavelength -- named for what it
    holds, because the layer NAME is the emission colour the operator
    sees on screen, and an ambiguous "wavelength" on a layer called
    Green invites silently wrong metadata. None is the truth for
    broadband transmitted light and for LED-less layers.

    `modality` and `background` are the layer's catalogue record's
    (`CatalogueLayer`). `display_color` is the unit's colour for the layer:
    its board block row's, else the catalogue record's.

    `label` is what the layer is called wherever it is shown or written:
    a fluorescence layer's excitation (``'405 nm'``), any other layer's
    `display_name`.
    """

    id: int
    key_name: str
    display_name: str | None
    led_addresses: tuple[int, ...]
    excitation_nm: float | None
    modality: Modality
    background: Background
    display_color: str
    label: str

    @property
    def led_channel(self) -> tuple[int, ...]:
        """``led_addresses`` by its old name, unpublished.

        Read by the engineering plugin releases before the one that reads
        ``led_addresses``; it goes when ``MINIMUM_PLUGIN_VERSIONS`` rises
        past them.
        """
        return self.led_addresses


class IdentitySource(enum.StrEnum):
    """Which rung of the identity resolver answered."""

    MOTORCONFIG = 'motorconfig'
    SCOPES = 'scopes'
    UNRESOLVED = 'unresolved'


@api_fields('channels', 'filterset', 'model', 'source')
@dataclass(frozen=True)
class ChannelIdentity:
    """The resolved per-unit identity snapshot.

    `source` records which rung answered: ``MOTORCONFIG`` (the unit's own
    block -- authoritative and complete when present), ``SCOPES`` (the
    model's rows -- what a pre-block unit of this model has), or
    ``UNRESOLVED`` (no block and no resolvable model; empty, and loud at
    the point of LED use rather than silently wrong here).

    `model` is the scope model the resolver settled on, whichever rung
    answered: the override, else the model the board reports, else the
    operator's selection -- the only identity an FX2 scope has, since it
    cannot report its own. A model the catalogue lacks is still carried,
    with no channels; `None` means there was no model at all. Every reader
    of "which model is this scope" takes it from here, so the channels, the
    capabilities and the saved files name one model.
    """

    channels: tuple[ChannelRecord, ...]
    filterset: str
    source: IdentitySource
    model: str | None

    @api(in_process=True, reads=True)
    def find(self, key_name: str) -> ChannelRecord | None:
        """Return the layer whose stable key name matches, else None.

        The deserialisation boundary: a name arriving from disk or from
        an API caller is mapped to its record exactly once, here.

        Raises:
            Nothing.
        """
        for layer in self.channels:
            if layer.key_name == key_name:
                return layer
        return None

    @api(in_process=True)
    def refuse_unless_on_scope(self, channel: str, member: str, *, then: str) -> None:
        """Refuse ``channel`` unless it is a layer of this release and this unit has it.

        The one question for a command that takes a layer: its settings,
        its focus, a still or a recording named for it. The name first
        (``refuse_unknown_layer``), so a name that is no layer is never
        told it is missing hardware.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``.
            HardwareCommandRefusedError: ``'axis_absent'``, naming the
                layer (``MissingPart.layer``).
            ConfigurationDefectError: ``'layers_unresolved'``, this unit's
                layers could not be resolved, so ``channel`` cannot ``then``.
        """
        refuse_unknown_layer(channel)
        if self.find(channel) is not None:
            return
        if self.channels:
            part = MissingPart.layer(channel)
            raise HardwareCommandRefusedError(part.reason, member, missing=part)
        raise ConfigurationDefectError(
            'layers_unresolved',
            f"this scope's layers could not be resolved (model {self.model}), "
            f'so {channel} cannot {then}',
        )


UNRESOLVED = ChannelIdentity(
    channels=(), filterset='', source=IdentitySource.UNRESOLVED, model=None
)

# The release catalogue is loaded once per process: it ships with the
# release (the file is version-paired), so nothing invalidates it at
# runtime. Tests hand the resolver a catalogue explicitly and may
# replace this cache to exercise a different one.
_CATALOGUE_CACHE: tuple[CatalogueLayer, ...] | None = None


def refuse_unknown_layer(layer: object, argument: str = 'channel') -> None:
    """Refuse a value that is not one of this release's layers.

    The one check of a layer name, asked by every member that takes one
    before it asks whether the unit has the layer: a name that is no layer
    is the request's fault on every scope. Names are exact: ``'bf'`` is
    not ``'BF'``.

    Raises:
        ArgumentRefusedError: ``'no_channel_selected'`` for None,
            ``'channel_unknown'`` for anything else not in the catalogue;
            each offers the catalogue.
    """
    catalogue = release_catalogue()
    if layer is None:
        raise ArgumentRefusedError(
            'no_channel_selected', argument=argument, value=layer, offered=catalogue
        )
    if not isinstance(layer, str) or layer not in catalogue:
        raise ArgumentRefusedError(
            'channel_unknown', argument=argument, value=layer, offered=catalogue
        )


def release_catalogue() -> tuple[str, ...]:
    """The release's layer vocabulary (stable key names, display order).

    The single source every vocabulary consumer derives from -- layer
    lists, protocol validation, metadata channel acceptance.

    Raises:
        InstallationFileError: as ``release_layers``.
    """
    return tuple(layer.key for layer in release_layers())


def release_layers() -> tuple[CatalogueLayer, ...]:
    """The release catalogue's records, in display order.

    Read from the installation's own folder, not a scope's: the settings
    bring-up needs it before any scope exists.

    Raises:
        InstallationFileError: the installation's ``scopes.json`` is
            unusable, states no layer order, or has a record that is not
            whole. Nothing is cached, so the next call reads the file again.
    """
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        path = resolve_data_file('scopes.json')
        _CATALOGUE_CACHE = load_layer_catalogue(read_installation_file(path), path)
    return _CATALOGUE_CACHE


def release_layer(key: object) -> CatalogueLayer:
    """The release catalogue's record for ``key``.

    Raises:
        ArgumentRefusedError: as ``refuse_unknown_layer``.
        InstallationFileError: as ``release_layers``.
    """
    refuse_unknown_layer(key)
    return next(layer for layer in release_layers() if layer.key == key)


def release_color(key: object) -> Color:
    """The release's colour for the layer ``key`` (``parse_display_color``).

    Every unit's layer of this key has this colour today: a unit's record
    takes its colour from the release record.

    Raises:
        ArgumentRefusedError: as ``refuse_unknown_layer``.
    """
    return parse_display_color(release_layer(key).display_color)


# Every layer of the release and the colour it is shown and written in,
# from the one read of it, ``ScopeSession.channel_colors``. A module below
# the Session is handed this and reads no colour of its own.
type ChannelColors = Mapping[str, Color]


def release_colors() -> dict[str, Color]:
    """Every release layer's colour, ``release_color`` of each key."""
    return {key: release_color(key) for key in release_catalogue()}


def output_color(name: object, colors: ChannelColors) -> Color:
    """The colour an image named for ``name`` is written in: the layer's
    colour in ``colors``, and ``GREY`` for a composite
    (``PostFunction.COMPOSITE``), which is RGB and carries its own -- and is
    grey once collapsed into a stack.

    Raises:
        ArgumentRefusedError: ``name`` is neither a layer of this release
            nor a composite (``refuse_unknown_layer``).
    """
    from modules.common_utils import PostFunction

    if name == PostFunction.COMPOSITE.value:
        return GREY
    refuse_unknown_layer(name)
    return colors[name]


def shown_color(name: object, false_color_on: bool, colors: ChannelColors) -> Color:
    """The colour an image named for ``name`` is shown in: its
    ``output_color`` with false colour on, ``GREY`` with it off.

    Raises:
        ArgumentRefusedError: as ``output_color``, whether false colour is
            on or off.
    """
    color = output_color(name, colors)
    return color if false_color_on else GREY


# The keys a model's layer row may carry; each but the colour is required.
# A row's colour is optional: without one the layer takes the release's.
_REQUIRED_ROW_FIELDS = ('key_name', 'display_name', 'led_channel', 'excitation_nm')
_ROW_FIELDS = frozenset(_REQUIRED_ROW_FIELDS) | {'display_color'}


class LedBoardFamily(enum.StrEnum):
    """The kind of board a model's LEDs are on, as its catalogue row names it.

    An EL-0940 is an LED board of its own; an FX2 drives its scope's camera
    and LEDs both.
    """

    EL_0940 = 'EL-0940'
    FX2 = 'FX2'


class MotorBoardFamily(enum.StrEnum):
    """The kind of board a model's motor axes are on, as its catalogue row names it."""

    EL_0940 = 'EL-0940'
    TMCM_6110 = 'TMCM-6110'


@api_fields(
    'id',
    'has_focus',
    'has_xy_stage',
    'has_turret',
    'led_board',
    'motor_board',
    'channels',
    'filterset',
    'pixel_size_um',
    'lens_focal_length_mm',
    'max_frame',
)
@dataclass(frozen=True)
class ScopeModel:
    """One model of the installation's catalogue (``scopes.json``'s ``Models``), read once at bring-up.

    ``id`` is the model's name, the key the operator selects and a board
    reports. ``has_focus``, ``has_xy_stage`` and ``has_turret`` are the motor
    axes the model is built with: Z, X and Y, T. ``channels`` are the model's
    channel rows, resolved against the release's layer vocabulary: what a unit
    of the model has when its board carries no Channels block of its own.
    ``led_board`` and ``motor_board`` name the model's boards, ``motor_board``
    None for a model with no motor axes; production finds the boards it has
    and reads neither, and the simulator builds from them. ``filterset`` is
    None where the row names none. ``pixel_size_um`` (the sensor's pixel
    pitch) and ``lens_focal_length_mm`` (the tube lens) are a Classic
    model's optics, which no motorconfig carries; ``max_frame`` is the
    largest frame, unbinned, of a model whose lens images less than its
    sensor. Each is None where the row declares none.
    """

    id: str
    has_focus: bool
    has_xy_stage: bool
    has_turret: bool
    led_board: LedBoardFamily
    motor_board: MotorBoardFamily | None
    channels: tuple[ChannelRecord, ...]
    filterset: str | None
    pixel_size_um: float | None
    lens_focal_length_mm: float | None
    max_frame: FrameSize | None


# Each flag's key in scopes.json and the motor axes it stands for: a focus
# drive is Z, a stage is X and Y, a turret is T.
_FLAG_KEYS = {'has_focus': 'Focus', 'has_xy_stage': 'XYStage', 'has_turret': 'Turret'}
_AXES_BY_FLAG = (('has_focus', 'Z'), ('has_xy_stage', 'XY'), ('has_turret', 'T'))
# The keys of a model entry: the flags, ``LEDBoard`` and ``Layers`` required,
# the rest stated only by the models that have them.
_MODEL_KEYS = frozenset(_FLAG_KEYS.values()) | {
    'LEDBoard',
    'Layers',
    'MotorBoard',
    'Filterset',
    'Optics',
    'MaxFrame',
}
# An Optics block's keys and the field each fills.
_OPTICS_KEYS = {'PixelSize': 'pixel_size_um', 'LensFocalLength': 'lens_focal_length_mm'}


def load_scope_models(data_file: str | None = None) -> Mapping[str, ScopeModel]:
    """The model catalogue -- scopes.json's `Models` section -- as records, or a refusal.

    The scope reads it once, from the folder it was started on, before
    anything is started, and everything else reads the scope's copy. With
    no catalogue the declared model has no entry, `model_has_turret`
    answers False, and a turret scope whose board is not talking is
    configured as turretless -- answering with its stored objective
    instead of the one in the light path. So an unusable file or a
    missing section refuses, naming the file. Each model's layer rows are
    resolved against the release's layer vocabulary (``release_layers``).

    ``data_file`` None reads the installation's own folder.

    Raises:
        InstallationFileError: as ``read_scope_models``, or the release's
            vocabulary is unusable (``release_layers``).
    """
    path = data_file if data_file is not None else resolve_data_file('scopes.json')
    return read_scope_models(read_installation_file(path), path, release_layers())


def read_scope_models(
    scopes_data: dict, path: str | os.PathLike, catalogue: tuple[CatalogueLayer, ...]
) -> Mapping[str, ScopeModel]:
    """``scopes_data``'s models as a read-only mapping of records by name, or a refusal naming ``path``.

    Every row is whole or the catalogue is refused: the file ships with the
    release, so a row it cannot read is the release's defect, and read in
    part it would build a scope unlike the hardware (a missing ``Turret``
    makes a turret scope turretless).

    Raises:
        InstallationFileError: there is no Models section, or an entry is
            not an object, misses a required key, carries an undeclared
            one, or gives a key a value it cannot hold; a layer row as
            ``_model_layers`` refuses it. Each names the model and the key.
    """
    models = scopes_data.get('Models')
    if not isinstance(models, dict):
        raise InstallationFileError(
            path, f'has no usable Models section (found {type(models).__name__})'
        )
    return types.MappingProxyType(
        {model: _scope_model(model, entry, path, catalogue) for model, entry in models.items()}
    )


def _scope_model(
    model: str, entry: object, path: str | os.PathLike, catalogue: tuple[CatalogueLayer, ...]
) -> ScopeModel:
    """The record for one model entry, or a refusal naming the file, the model and the key."""

    def refuse(problem: str) -> InstallationFileError:
        return InstallationFileError(path, f'gives model {model!r} {problem}')

    if not isinstance(entry, dict):
        raise InstallationFileError(
            path, f'has a model {model!r} whose entry is a {type(entry).__name__}, not an object'
        )
    undeclared = sorted(set(entry) - _MODEL_KEYS)
    if undeclared:
        raise refuse(f'the undeclared key {undeclared[0]!r}')
    flags = {}
    for field, key in _FLAG_KEYS.items():
        if not isinstance(entry.get(key), bool):
            raise refuse(f'a {key!r} that is not true or false')
        flags[field] = entry[key]
    text = {}
    for key in ('LEDBoard', 'MotorBoard', 'Filterset'):
        value = entry.get(key)
        if (key == 'LEDBoard' or key in entry) and not (isinstance(value, str) and value):
            raise refuse(f'a {key!r} that is not a name')
        text[key] = value
    for key, family in (('LEDBoard', LedBoardFamily), ('MotorBoard', MotorBoardFamily)):
        if text[key] is not None and member_of(family, text[key]) is None:
            raise refuse(f'a {key!r} {text[key]!r} that is not one of {", ".join(family)}')
    optics = entry.get('Optics', {})
    if not isinstance(optics, dict) or set(optics) - set(_OPTICS_KEYS):
        raise refuse(f"an 'Optics' that holds other than {sorted(_OPTICS_KEYS)}")
    for key, value in optics.items():
        if not _is_number(value) or value <= 0:
            raise refuse(f'an Optics {key!r} that is not a positive number')
    max_frame = None
    if 'MaxFrame' in entry:
        raw = entry['MaxFrame']
        if not (
            isinstance(raw, dict)
            and set(raw) == {'width', 'height'}
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in raw.values())
        ):
            raise refuse("a 'MaxFrame' that is not a positive width and height in pixels")
        max_frame = FrameSize(width=raw['width'], height=raw['height'])
    return ScopeModel(
        id=model,
        **flags,
        led_board=text['LEDBoard'],
        motor_board=text['MotorBoard'],
        channels=_model_layers(entry.get('Layers'), catalogue, refuse),
        filterset=text['Filterset'],
        **{
            field: float(optics[key]) if key in optics else None
            for key, field in _OPTICS_KEYS.items()
        },
        max_frame=max_frame,
    )


def model_axes(models: Mapping[str, ScopeModel], model: str) -> frozenset[str]:
    """The motor axes the catalogue says a model has, or a refusal.

    A model the catalogue does not list is refused rather than answered
    as axis-less: an axis-less answer builds a manual scope, and a typo in
    the model name would then look like a scope with no motor board.
    """
    entry = models.get(model)
    if entry is None:
        raise ScopeModelUnknownError(model, sorted(models))
    return entry_axes(entry)


def entry_axes(entry: ScopeModel) -> frozenset[str]:
    """The motor axes one catalogue entry declares. Empty for a manual
    scope, which has no motor board at all."""
    return frozenset(axis for flag, axes in _AXES_BY_FLAG if getattr(entry, flag) for axis in axes)


def entry_expects_motion(entry: ScopeModel | None) -> bool:
    """Whether a scope with this catalogue entry has a motor board.

    No entry expects one: only the catalogue can say a scope is manual,
    and answering "manual" for a model it does not list would let a
    motorized scope's missing board pass as expected.
    """
    return entry is None or bool(entry_axes(entry))


def load_layer_catalogue(scopes_data: dict, path: str | os.PathLike) -> tuple[CatalogueLayer, ...]:
    """The release's layer records, in display order, or a refusal naming ``path``.

    The catalogue is the single authored order: a layer's id IS its
    position here, and every identity row (model or per-unit block) must
    name a catalogued key to resolve. Deriving the order from the
    per-model row lists instead would make id assignment depend on which
    model happens to be listed first, so the order is stated once. An
    empty vocabulary would let the settings check pass with no layers to
    check, so a missing one refuses. Every fact about a layer that is
    the same on every unit is a field of its record, and a record missing
    one has no answer for it, so it refuses too.

    Raises:
        InstallationFileError: ``LayerOrder`` is missing, empty, or not a
            list of records; or a record has a field missing or of the
            wrong kind, or repeats another's key or step order.
    """
    raw = scopes_data.get('LayerOrder')
    if not isinstance(raw, list) or not raw:
        raise InstallationFileError(path, f'has no usable LayerOrder (found {raw!r})')
    layers = tuple(_catalogue_layer(entry, path) for entry in raw)
    for field in ('key', 'step_order'):
        values = [getattr(layer, field) for layer in layers]
        repeated = sorted({v for v in values if values.count(v) > 1}, key=str)
        if repeated:
            raise InstallationFileError(
                path, f'has LayerOrder records repeating {field} {repeated}'
            )
    return layers


# The fields every catalogue record states.
_CATALOGUE_FIELDS = tuple(field.name for field in fields(CatalogueLayer))


def _catalogue_layer(entry: object, path: str | os.PathLike) -> CatalogueLayer:
    """One ``LayerOrder`` record, or the refusal naming ``path`` and the field."""

    def refuse(why: str) -> InstallationFileError:
        return InstallationFileError(path, f'has a LayerOrder record {entry!r} {why}')

    if not isinstance(entry, dict):
        raise refuse('that is not an object')
    missing = [f for f in _CATALOGUE_FIELDS if f not in entry]
    if missing:
        raise refuse(f'missing {missing}')
    key = entry['key']
    if not isinstance(key, str) or not key:
        raise refuse('whose key is not a name')
    modality = member_of(Modality, entry['modality'])
    if modality is None:
        raise refuse(f'whose modality is not one of {", ".join(Modality)}')
    background = member_of(Background, entry['background'])
    if background is None:
        raise refuse(f'whose background is not one of {", ".join(Background)}')
    ceiling = entry['manual_max_exposure_ms']
    if ceiling is not None and not (_is_number(ceiling) and ceiling > 0):
        raise refuse('whose manual_max_exposure_ms is not a positive number or null')
    try:
        parse_display_color(entry['display_color'])
    except ValueError as e:
        raise refuse(f'whose display_color {e}') from e
    step_order = entry['step_order']
    if not isinstance(step_order, int) or isinstance(step_order, bool):
        raise refuse('whose step_order is not an integer')
    light = entry['sim_light_per_ma_s']
    if light is not None and not (_is_number(light) and light > 0):
        raise refuse('whose sim_light_per_ma_s is not a positive number or null')
    return CatalogueLayer(
        key=key,
        modality=modality,
        background=background,
        manual_max_exposure_ms=None if ceiling is None else float(ceiling),
        display_color=entry['display_color'],
        step_order=step_order,
        sim_light_per_ma_s=None if light is None else float(light),
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _record(
    key_name: str, row: dict, release: CatalogueLayer, position: int
) -> ChannelRecord | str:
    """The record ``row`` describes for ``key_name``, or why it describes none.

    One reading of a row for both sources, the model's rows and a board's
    Channels block: the LED address, the excitation, the name, and an
    optional colour. A fluorescence channel is labelled by its excitation,
    so it must carry one and needs no name; any other channel is labelled by
    its name. A row's colour is held to the one stored form and full
    brightness (`parse_display_color`); without one the channel takes the
    catalogue's.
    """
    raw_channel = row.get('led_channel')
    if raw_channel is None:
        channel: tuple[int, ...] = ()
    elif isinstance(raw_channel, int) and not isinstance(raw_channel, bool):
        channel = (raw_channel,)
    elif isinstance(raw_channel, list) and all(
        isinstance(c, int) and not isinstance(c, bool) for c in raw_channel
    ):
        channel = tuple(raw_channel)
    else:
        return f'has malformed led_channel {raw_channel!r}'
    if len(channel) > 1:
        return f'drives multiple channels {channel}; multi-channel layers are not supported yet'
    raw_nm = row.get('excitation_nm')
    if raw_nm is None:
        excitation: float | None = None
    elif _is_number(raw_nm):
        excitation = float(raw_nm)
    else:
        return f'has malformed excitation_nm {raw_nm!r}'
    display = row.get('display_name')
    if display is not None and (not isinstance(display, str) or not display):
        return f'has malformed display_name {display!r}'
    if release.modality == Modality.FLUORESCENCE:
        if excitation is None:
            return 'has no excitation_nm'
        label = f'{excitation:g} nm'
    elif display is None:
        return 'has no display_name'
    else:
        label = display
    color = row.get('display_color')
    if color is None:
        color = release.display_color
    else:
        try:
            parse_display_color(color)
        except ValueError as e:
            return f'has display_color {e}'
    return ChannelRecord(
        id=position,
        key_name=key_name,
        display_name=display,
        led_addresses=channel,
        excitation_nm=excitation,
        modality=release.modality,
        background=release.background,
        display_color=color,
        label=label,
    )


def _shared_address(records: list[ChannelRecord]) -> str | None:
    """Why ``records`` cannot all be one unit's: two of them on one LED address."""
    holders: dict[int, str] = {}
    for record in records:
        for address in record.led_addresses:
            if address in holders:
                return f'{holders[address]} and {record.key_name} both name LED address {address}'
            holders[address] = record.key_name
    return None


def _model_layers(
    rows: object,
    catalogue: tuple[CatalogueLayer, ...],
    refuse: Callable[[str], InstallationFileError],
) -> tuple[ChannelRecord, ...]:
    """A model's layer rows as records, in id order, or ``refuse``'s refusal.

    Every row is whole or the model is refused, as its entry is: a row
    whose key is not in the release's vocabulary, that misses a field or
    carries an undeclared one, that ``_record`` cannot read, or that shares
    an LED address with another. Each record takes its key's catalogue
    fields from ``catalogue``.
    """
    by_key = {layer.key: (position, layer) for position, layer in enumerate(catalogue)}
    if not isinstance(rows, list):
        raise refuse("a 'Layers' that is not a list of rows")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise refuse(f'a layer row that is not an object: {row!r}')
        key_name = row.get('key_name')
        if not isinstance(key_name, str) or key_name not in by_key:
            raise refuse(f'the layer {key_name!r}, not one of this release {tuple(by_key)}')
        missing = [k for k in _REQUIRED_ROW_FIELDS if k not in row]
        if missing:
            raise refuse(f'a {key_name} layer row with no {missing[0]!r}')
        undeclared = sorted(set(row) - _ROW_FIELDS)
        if undeclared:
            raise refuse(f'a {key_name} layer row with the undeclared key {undeclared[0]!r}')
        position, release = by_key[key_name]
        record = _record(key_name, row, release, position)
        if isinstance(record, str):
            raise refuse(f'a {key_name} layer row that {record}')
        records.append(record)
    shared = _shared_address(records)
    if shared is not None:
        raise refuse(f'layer rows where {shared}')
    return tuple(sorted(records, key=lambda r: r.id))


def _parse_channel_block(
    channels: object, catalogue: tuple[CatalogueLayer, ...], report: Callable[[Notice], None]
) -> tuple[ChannelRecord, ...]:
    """Parse a board's Channels block: a mapping from catalogue key to its row.

    A row this release cannot read costs that row, and every such row is
    reported in one notice: a block is the unit's, written by a release
    that may be newer than this one. A block that is not a mapping,
    or that names one LED address twice, describes no assembly that can be
    trusted in part: the unit has no channels, reported.
    """
    if not isinstance(channels, dict):
        report(ChannelBlockUnusableNotice(f'its Channels is a {type(channels).__name__}'))
        return ()
    by_key = {layer.key: (position, layer) for position, layer in enumerate(catalogue)}
    records, skipped = [], {}
    for key_name, row in channels.items():
        if key_name not in by_key:
            skipped[str(key_name)] = 'not a channel this release knows'
            continue
        if not isinstance(row, dict) or 'led_channel' not in row:
            skipped[key_name] = 'not a mapping holding its led_channel'
            continue
        position, release = by_key[key_name]
        record = _record(key_name, row, release, position)
        if isinstance(record, str):
            skipped[key_name] = record
            continue
        records.append(record)
    if skipped:
        report(ChannelRowsSkippedNotice(skipped))
    shared = _shared_address(records)
    if shared is not None:
        report(ChannelBlockUnusableNotice(shared))
        return ()
    return tuple(sorted(records, key=lambda r: r.id))


def resolve_layer_identity(
    *,
    board_block: dict | None,
    board_config_read_ok: bool,
    motor_model: str | None,
    configured_model: str | None,
    models: Mapping[str, ScopeModel],
    catalogue: tuple[CatalogueLayer, ...],
    report: Callable[[Notice], None],
    override_model: str | None = None,
) -> ChannelIdentity:
    """Resolve the unit's layer identity from the first authoritative source.

    Precedence: an explicit `override_model` (a lab/engineering request
    to impersonate a model for this session -- it wins over everything,
    loudly, and is never persisted); else the unit's own motorconfig
    Channels block with its Filterset, taken WHOLE (a block describes one
    physical filterset assembly, so merging it field-by-field with model
    data could describe hardware that does not exist); else the model's
    `scopes.json` rows, with the motor-reported model outranking the
    configured one because hardware truth beats a user selection; else
    the empty `unresolved` snapshot. Whichever rung answers, the snapshot
    carries the model the resolver settled on (`ChannelIdentity.model`).

    A block that is absent because the board's config could not be READ
    is not the same as a unit with no block: the failed read is logged
    as an error here (the one place both facts are in hand) and the
    model rung answers, so the scope stays usable while the failure
    stays visible.

    A board carrying the old Layers block, or a Filterset with no
    Channels, resolves from its model, reported through ``report``; so are
    the board block's skipped rows and an unusable block.

    Reads nothing: ``models`` is the scope's model catalogue and
    ``catalogue`` the release's layer records, each read once by its
    owner, so rows are resolved against the one vocabulary every layer
    list in the process uses.
    """

    def _from_model(model: str) -> ChannelIdentity:
        entry = models.get(model)
        if entry is None:
            return ChannelIdentity(
                channels=(), filterset='', source=IdentitySource.UNRESOLVED, model=model
            )
        return ChannelIdentity(
            channels=entry.channels,
            filterset=entry.filterset or '',
            source=IdentitySource.SCOPES,
            model=model,
        )

    if override_model is not None:
        logger.warning(
            f'[LAYER_RECORD] identity override active: resolving as model '
            f'{override_model!r} for this session'
        )
        identity = _from_model(override_model)
        if identity.source == IdentitySource.UNRESOLVED:
            logger.error(
                f'[LAYER_RECORD] override model {override_model!r} has no scopes '
                f'entry; identity is unresolved'
            )
        return identity

    # The configured model is consulted only when the hardware reports no
    # model at all. A motor-reported model with no scopes entry (a newer
    # unit than this release knows) goes unresolved and loud rather than
    # silently adopting whatever the user last selected.
    model = motor_model or configured_model

    board = board_block or {}
    if 'Channels' in board:
        layers = _parse_channel_block(board['Channels'], catalogue, report)
        filterset = board.get('Filterset', '')
        if not isinstance(filterset, str):
            filterset = ''
        return ChannelIdentity(
            channels=layers, filterset=filterset, source=IdentitySource.MOTORCONFIG, model=model
        )
    if 'Layers' in board or 'Filterset' in board:
        report(ChannelBlockSupersededNotice(model))

    if not board_config_read_ok:
        logger.error(
            '[LAYER_RECORD] board config could not be read; a per-unit LED '
            'block may exist but is unavailable -- resolving from the model '
            'instead'
        )

    if not model:
        return UNRESOLVED
    identity = _from_model(model)
    if identity.source == IdentitySource.UNRESOLVED:
        logger.error(f'[LAYER_RECORD] model {model!r} has no scopes entry; identity is unresolved')
    return identity
