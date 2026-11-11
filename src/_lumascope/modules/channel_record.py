# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Channel identity: what a channel IS on the attached unit.

One record per channel (stable key name, display name, LED board address,
excitation wavelength) plus the unit's filterset, resolved once into an
immutable snapshot. Identity is unit data, not code: different filtersets
carry different LEDs, so the truth lives in the unit's motorconfig
Channels block when it has one, and in the model's `scopes.json` rows for
units built before per-unit blocks existed.

A channel is known by its stable key (`key_name`), in memory as on disk:
settings blocks, protocol steps, file names and metadata all carry it,
and `ChannelIdentity.find` maps a key to the unit's record. A record's
`id` is its position in the release's order: it sorts a unit's records
and is never stored.

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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields

from _lumascope.lvp_logger import logger
from _lumascope.modules.exceptions import (
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
from _lumascope.modules.scope_parts import Part
from _lumascope.modules.path_utils import read_installation_file, resolve_data_file
from _lumascope.modules.api_surface import api, api_fields
from _lumascope.modules.closed_sets import member_of
from _lumascope.modules.scope_capabilities import FrameSize


class Modality(enum.StrEnum):
    """The kind of light a channel images."""

    TRANSMITTED = 'transmitted'
    FLUORESCENCE = 'fluorescence'
    LUMINESCENCE = 'luminescence'


class Background(enum.StrEnum):
    """What a channel's field of view looks like, which is not its modality's:
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


# The objective a channel's simulated light is stated through: the release's
# default objective (`settings.json` `objective_id`, the 20x), so a default
# simulated session renders each channel at its stated light.
SIM_LIGHT_MAGNIFICATION = 20.0


@dataclass(frozen=True)
class CatalogueChannel:
    """What a channel of one key is in this release, on every unit that has it.

    `background` is whether the channel's field of view is bright or dark,
    which its modality does not say: darkfield is transmitted light that
    shows bright subjects on a dark field.
    `manual_max_exposure_ms` is the exposure slider's ceiling, None where the
    channel has none (luminescence integrates as long as the sensor allows).
    `manual_max_illumination_ma` is the illumination slider's ceiling, None
    where the slider goes to the LED driver's cap. It is the slider's range,
    not the light's limit: the text entry reaches past it, up to what the
    connected board can be asked for.
    `display_color` is the release's colour for the channel, ``'#rrggbb'``.
    `sim_light_per_ma_s` is the light the simulated cameras render for the
    channel, per mA of LED current and second of exposure at unity gain,
    through an objective of `SIM_LIGHT_MAGNIFICATION`; None for a channel
    with no LED. Each value puts the channel's shipped defaults
    (`settings.json`) at brightness 1.0, the specimen at its own grey
    levels: 1 / (exposure s x linear gain x mA), so BF's 2 ms, 1 dB and
    5 mA give 89.1 and fluorescence's 100 ms, 10 dB and 250 mA 0.0126.
    The defaults are what makes each channel look right on a scope, so they
    carry the light each modality delivers.
    `sim_light_magnification_power` is how that light falls with the
    objective: it is multiplied by (`SIM_LIGHT_MAGNIFICATION` / the
    objective's magnification) to this power; None where
    `sim_light_per_ma_s` is. On the LS850T a clear brightfield field falls
    as the square (/3.84 and /3.99 from 10x to 20x to 40x), so BF's is 2;
    a channel the bench has not read is 0.
    """

    key: str
    modality: Modality
    background: Background
    manual_max_exposure_ms: float | None
    manual_max_illumination_ma: int | None
    display_color: str
    sim_light_per_ma_s: float | None
    sim_light_magnification_power: float | None


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
    """Identity of one channel on one unit.

    Attributes:
        id: The internal key and the display position, assigned from the
            release catalogue's order; never stored in any per-unit or user
            file, so it can renumber freely between releases.
        key_name: The stable name the channel is stored under (settings
            block keys, the protocol TSV's ``Color``, file name tokens, the
            metadata's ``channel``); it never changes when a channel is
            renamed.
        display_name: The unit's name for the channel, free to change: a
            rename touches this field and nothing stored. None for a
            fluorescence channel a board's Channels block names by its
            excitation alone.
        led_addresses: The LED board addresses this channel drives, empty
            for a channel with no LED (luminescence). A tuple because a
            channel may drive several switches as one; every shipped row
            carries zero or one. A row in a model's file names them
            ``led_channel``, read into this field by the one row reader.
        excitation_nm: The excitation wavelength, in nm, and the number a
            fluorescence key names (``Ex<nm>``). None for broadband
            transmitted light and for a channel with no LED.
        modality: The kind of light the channel images, from its catalogue
            record (``CatalogueChannel``).
        background: Whether the channel's field of view is bright or dark,
            from its catalogue record.
        display_color: The unit's colour for the channel, ``'#rrggbb'``: its
            board block row's, else the catalogue record's.
        label: What the channel is called wherever it is shown or written:
            a fluorescence channel's excitation (``'405 nm'``), any other
            channel's ``display_name``.
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


@api_fields('slider_max_exposure_ms', 'slider_max_illumination_ma')
@dataclass(frozen=True)
class ChannelSliderMaxima:
    """Where one channel's manual sliders stop: display hints, never a bound.

    Each is the connected part's cap narrowed to the channel's manual ceiling,
    so the brightfield slider stops at a useful range while a typed value
    still reaches past it, up to what the camera or the LED board can be
    asked for. A value above that limit is refused by its writer
    (``CameraAPI.set_exposure_ms``, the illumination and settings writers),
    never by this record.

    Attributes:
        slider_max_exposure_ms: The exposure slider's maximum, in
            milliseconds; None with no camera connected.
        slider_max_illumination_ma: The illumination slider's maximum, in
            mA; None with no LED board, or for a channel with no LED on this
            scope (luminescence).
    """

    slider_max_exposure_ms: float | None
    slider_max_illumination_ma: int | None


class IdentitySource(enum.StrEnum):
    """Which rung of the identity resolver answered."""

    MOTORCONFIG = 'motorconfig'
    SCOPES = 'scopes'
    STOCK = 'stock'
    UNRESOLVED = 'unresolved'


# The optics module every OEM unit carries: a board naming a model the
# catalogue does not list (an OEM unit's own name) has this model's
# channels. It is the shipped template's ``microscope``.
STOCK_OPTICS_MODEL = 'LS850'


@api_fields('channels', 'filterset', 'model', 'model_named_by_board', 'source')
@dataclass(frozen=True)
class ChannelIdentity:
    """The resolved per-unit identity snapshot.

    Attributes:
        channels: The unit's channel records, in the release's order
            (``ChannelRecord.id``); empty when ``source`` is ``UNRESOLVED``.
        filterset: The name of the unit's filterset: its board block's
            ``Filterset``, else its model's row's; an empty string where
            neither names one.
        model: The scope model the resolver settled on, whichever rung
            answered: the override, else the model the board reports, else
            the operator's selection -- the only identity an FX2 scope has,
            since it cannot report its own. A model the catalogue lacks is
            still carried: with the stock module's channels, or with none
            when an override names it; None means there was no model at
            all. Every reader of "which model is this scope" takes it from
            here, so the channels, the capabilities and the saved files name
            one model.
        model_named_by_board: Whether the motor board named the model
            itself, so the board's name outranked the operator's
            selection: every EL-0940 board answering FULLINFO, the LS720's
            board, every OEM unit. Such a scope runs as the board's model
            at every start, whatever is selected. False where the selection
            decided: a Classic FX2, which has no board to ask, and a
            simulated scope built as the selection. It describes the
            board, not an engineering override.
        source: Which rung answered: ``MOTORCONFIG`` (the unit's own block,
            authoritative and complete when present), ``SCOPES`` (the
            model's rows, what a pre-block unit of this model has),
            ``STOCK`` (the model is one the catalogue does not list, an OEM
            unit's own name, so the rows are those of the stock optics
            module, which every OEM unit carries), or ``UNRESOLVED`` (no
            block and no model, or an override naming a model the catalogue
            lacks; empty, and loud at the point of LED use rather than
            silently wrong here).
    """

    channels: tuple[ChannelRecord, ...]
    filterset: str
    source: IdentitySource
    model: str | None
    model_named_by_board: bool

    @api(in_process=True, reads=True)
    def find(self, key_name: str) -> ChannelRecord | None:
        """Return the channel whose stable key name matches, else None.

        The deserialisation boundary: a name arriving from disk or from
        an API caller is mapped to its record exactly once, here.

        Args:
            key_name: The channel's stable key name, such as ``'BF'`` or
                ``'Ex488'``.

        Raises:
            Nothing.
        """
        for record in self.channels:
            if record.key_name == key_name:
                return record
        return None

    def key_at(self, address: int) -> str | None:
        """The key of the channel whose LED is at board ``address``, else None."""
        for record in self.channels:
            if address in record.led_addresses:
                return record.key_name
        return None

    @api(in_process=True)
    def refuse_unless_on_scope(self, channel: str, member: str, *, then: str) -> None:
        """Refuse ``channel`` unless it is a channel of this release and this unit has it.

        The one question for a command that takes a channel: its settings,
        its focus, a still or a recording named for it. The name first
        (``refuse_unknown_channel``), so a name that is no channel is never
        told it is missing hardware.

        Args:
            channel: The channel's key name the command was given.
            member: The name of the command asking, which the hardware
                refusal names as the command refused.
            then: What the command would have the channel do, as words
                completing "so <channel> cannot ...", such as ``'be
                captured'``; the unresolved refusal's message ends with it.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``.
            HardwareCommandRefusedError: ``'not_on_this_scope'``, naming the
                channel (``Part.CHANNEL``).
            ConfigurationDefectError: ``'channels_unresolved'``, this unit's
                channels could not be resolved, so ``channel`` cannot ``then``.
        """
        refuse_unknown_channel(channel)
        if self.find(channel) is not None:
            return
        if self.channels:
            part = MissingPart(Part.CHANNEL, channel=channel)
            raise HardwareCommandRefusedError(part.reason, member, missing=part)
        raise ConfigurationDefectError(
            'channels_unresolved',
            f"this scope's channels could not be resolved (model {self.model}), "
            f'so {channel} cannot {then}',
        )

    def unit_key(self, name: str) -> str:
        """The key this unit holds a stored ``name``'s channel under.

        An old name meant an LED, not a wavelength: it is this unit's
        fluorescence channel on that LED address, else -- a unit with no LED
        there -- its stock key, a key this unit lacks, which the refusals then
        name in their own words. Any other name is returned as it is.
        """
        old = _old_name(name)
        if old is None:
            return name
        for record in self.channels:
            if record.modality == Modality.FLUORESCENCE and old.led_address in record.led_addresses:
                return record.key_name
        return old.stock_key


UNRESOLVED = ChannelIdentity(
    channels=(),
    filterset='',
    source=IdentitySource.UNRESOLVED,
    model=None,
    model_named_by_board=False,
)

# The release catalogue is loaded once per process: it ships with the
# release (the file is version-paired), so nothing invalidates it at
# runtime. Tests hand the resolver a catalogue explicitly and may
# replace this cache to exercise a different one.
_CATALOGUE_CACHE: tuple[CatalogueChannel, ...] | None = None
# The release's old names, loaded once per process beside the catalogue and
# replaced by tests the same way.
_OLD_NAMES_CACHE: tuple[OldChannelName, ...] | None = None


@dataclass(frozen=True)
class OldChannelName:
    """A fluorescence key an earlier release stored, and the LED it meant.

    Earlier releases named the fluorescence channels for their colour, and
    every unit that stored one drove it on the same LED address, so the
    address is what an old name means. `stock_key` is the release key of
    the stock filterset's LED on that address: what a stored old name reads
    as where no unit is known.
    """

    name: str
    led_address: int
    stock_key: str


def refuse_unknown_channel(channel: object, argument: str = 'channel') -> None:
    """Refuse a value that is not one of this release's channels.

    The one check of a channel name, asked by every member that takes one
    before it asks whether the unit has the channel: a name that is no channel
    is the request's fault on every scope. Names are exact: ``'bf'`` is
    not ``'BF'``.

    Raises:
        ArgumentRefusedError: ``'no_channel_selected'`` for None,
            ``'channel_unknown'`` for anything else not in the catalogue;
            each offers the catalogue.
    """
    catalogue = release_catalogue()
    if channel is None:
        raise ArgumentRefusedError(
            'no_channel_selected', argument=argument, value=channel, offered=catalogue
        )
    if not isinstance(channel, str) or channel not in catalogue:
        raise ArgumentRefusedError(
            'channel_unknown', argument=argument, value=channel, offered=catalogue
        )


def release_catalogue() -> tuple[str, ...]:
    """The release's channel vocabulary (stable key names, display order).

    The single source every vocabulary consumer derives from -- channel
    lists, protocol validation, metadata channel acceptance.

    Raises:
        InstallationFileError: as ``release_channels``.
    """
    return tuple(record.key for record in release_channels())


def release_channels() -> tuple[CatalogueChannel, ...]:
    """The release catalogue's records, in display order.

    Read from the installation's own folder, not a scope's: the settings
    bring-up needs it before any scope exists.

    Raises:
        InstallationFileError: the installation's ``scopes.json`` is
            unusable, states no channel order, or has a record that is not
            whole. Nothing is cached, so the next call reads the file again.
    """
    global _CATALOGUE_CACHE
    if _CATALOGUE_CACHE is None:
        path = resolve_data_file('scopes.json')
        _CATALOGUE_CACHE = load_channel_catalogue(read_installation_file(path), path)
    return _CATALOGUE_CACHE


def release_old_names() -> tuple[OldChannelName, ...]:
    """The fluorescence keys earlier releases stored, from the release's own file.

    Read from the installation's own folder, as the catalogue is: a stored
    name is read where no scope is known (an old step name, an old run
    folder).

    Raises:
        InstallationFileError: as ``load_old_channel_names``, or as
            ``release_channels``. Nothing is cached, so the next call reads the
            file again.
    """
    global _OLD_NAMES_CACHE
    if _OLD_NAMES_CACHE is None:
        path = resolve_data_file('scopes.json')
        _OLD_NAMES_CACHE = load_old_channel_names(
            read_installation_file(path), path, release_channels()
        )
    return _OLD_NAMES_CACHE


def load_old_channel_names(
    scopes_data: dict, path: str | os.PathLike, catalogue: tuple[CatalogueChannel, ...]
) -> tuple[OldChannelName, ...]:
    """``scopes_data``'s ``OldChannelNames`` as records, or a refusal naming ``path``.

    Each entry maps an old name to its LED address and its stock key. The
    table is a fact of history the release states once, so a table that
    cannot be true refuses: an old name that is also a key of this release
    would be read two ways, a stock key that is no fluorescence key of this
    release gives an old name nothing to read as, and two old names on one
    address cannot both be that LED.

    Raises:
        InstallationFileError: the table is missing or not a mapping; or an
            entry is as above, or not an ``led_address`` and a ``stock_key``.
    """
    raw = scopes_data.get('OldChannelNames')
    if not isinstance(raw, dict):
        raise InstallationFileError(path, f'has no usable OldChannelNames (found {raw!r})')
    fluorescence = {record.key for record in catalogue if record.modality == Modality.FLUORESCENCE}
    keys = {record.key for record in catalogue}
    names: list[OldChannelName] = []
    for name, entry in raw.items():

        def refuse(why: str, name=name) -> InstallationFileError:
            return InstallationFileError(path, f'has an OldChannelNames entry {name!r} {why}')

        if name in keys:
            raise refuse('that is also a key of this release')
        if not isinstance(entry, dict) or set(entry) != {'led_address', 'stock_key'}:
            raise refuse('that is not an led_address and a stock_key')
        address = entry['led_address']
        if not isinstance(address, int) or isinstance(address, bool) or address < 0:
            raise refuse(f'whose led_address {address!r} is not an LED address')
        if entry['stock_key'] not in fluorescence:
            raise refuse(
                f'whose stock_key {entry["stock_key"]!r} is not a fluorescence key of this release'
            )
        names.append(OldChannelName(name=name, led_address=address, stock_key=entry['stock_key']))
    shared = _shared_address((old.name, (old.led_address,)) for old in names)
    if shared is not None:
        raise InstallationFileError(path, f'has OldChannelNames where {shared}')
    return tuple(names)


def _old_name(name: object) -> OldChannelName | None:
    return next((old for old in release_old_names() if old.name == name), None)


def stored_key(name: object) -> object:
    """The release key a stored channel name reads as where no unit is known.

    An old name reads as its stock key, any other name as itself. For a file
    read with no unit -- post-processing an old run folder: the colour,
    modality and settings of its ``Blue`` are the stock key's, and the name
    it is read and written under stays ``Blue``. Nothing measured is ever
    taken from the stock key's record; the file states its own.

    Raises:
        InstallationFileError: as ``release_old_names``.
    """
    old = _old_name(name)
    return name if old is None else old.stock_key


def stored_channel_names() -> tuple[str, ...]:
    """Every channel name a saved file may hold: this release's keys, then the old names.

    Raises:
        InstallationFileError: as ``release_old_names``.
    """
    return (*release_catalogue(), *(old.name for old in release_old_names()))


def release_channel(key: object) -> CatalogueChannel:
    """The release catalogue's record for ``key``.

    Raises:
        ArgumentRefusedError: as ``refuse_unknown_channel``.
        InstallationFileError: as ``release_channels``.
    """
    refuse_unknown_channel(key)
    return next(record for record in release_channels() if record.key == key)


def release_color(key: object) -> Color:
    """The release's colour for the channel ``key`` (``parse_display_color``).

    Every unit's channel of this key has this colour today: a unit's record
    takes its colour from the release record.

    Raises:
        ArgumentRefusedError: as ``refuse_unknown_channel``.
    """
    return parse_display_color(release_channel(key).display_color)


# Every channel of the release and the colour it is shown and written in,
# from the one read of it, ``ScopeSession.channel_colors``. A module below
# the Session is handed this and reads no colour of its own.
type ChannelColors = Mapping[str, Color]


def release_colors() -> dict[str, Color]:
    """Every release channel's colour, ``release_color`` of each key."""
    return {key: release_color(key) for key in release_catalogue()}


def output_color(name: object, colors: ChannelColors) -> Color:
    """The colour an image named for ``name`` is written in: the channel's
    colour in ``colors``, and ``GREY`` for a composite
    (``PostFunction.COMPOSITE``), which is RGB and carries its own -- and is
    grey once collapsed into a stack. A stored old name is its stock key's
    colour (``stored_key``).

    Raises:
        ArgumentRefusedError: ``name`` is neither a channel of this release,
            an old name nor a composite (``refuse_unknown_channel``).
    """
    from _lumascope.modules.common_utils import PostFunction

    if name == PostFunction.COMPOSITE.value:
        return GREY
    key = stored_key(name)
    refuse_unknown_channel(key)
    return colors[key]


def shown_color(name: object, false_color_on: bool, colors: ChannelColors) -> Color:
    """The colour an image named for ``name`` is shown in: its
    ``output_color`` with false colour on, ``GREY`` with it off.

    Raises:
        ArgumentRefusedError: as ``output_color``, whether false colour is
            on or off.
    """
    color = output_color(name, colors)
    return color if false_color_on else GREY


# The keys a model's channel row may carry, by its channel's modality; each but
# the colour is required. A row's colour is optional: without one the channel
# takes the release's. A fluorescence channel is labelled by its excitation, so
# its row stores no name to publish beside the label.
_REQUIRED_ROW_FIELDS = ('key_name', 'led_channel', 'excitation_nm')
_NAMED_ROW_FIELDS = (*_REQUIRED_ROW_FIELDS, 'display_name')


def _row_fields(modality: Modality) -> tuple[tuple[str, ...], frozenset[str]]:
    """The fields a model row for a channel of ``modality`` must carry, and every
    field it may."""
    required = _REQUIRED_ROW_FIELDS if modality == Modality.FLUORESCENCE else _NAMED_ROW_FIELDS
    return required, frozenset(required) | {'display_color'}


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

    Attributes:
        id: The model's name, the key the operator selects and a board
            reports.
        has_focus: Whether the model is built with a focus drive (Z).
        has_xy_stage: Whether the model is built with an XY stage (X and Y).
        has_turret: Whether the model is built with an objective turret (T):
            the answer ``ScopeCapabilities.has_turret`` gives when the
            scope's motor board did not come up.
        led_board: The kind of board the model's LEDs are on. Production
            finds the boards it has and does not read it; the simulator
            builds from it.
        motor_board: The kind of motor board the model has; None for a
            model with no motor axes. Read, like ``led_board``, only by the
            simulator.
        channels: The model's channel rows, resolved against the release's
            channel vocabulary and in its order: what a unit of the model
            has when its board carries no Channels block of its own.
        filterset: The name of the model's filterset; None where the row
            names none.
        pixel_size_um: The sensor's pixel pitch, in um, for a Classic
            model, whose optics no motorconfig carries; None where the row
            declares none.
        lens_focal_length_mm: The tube lens's focal length, in mm, for a
            Classic model; None where the row declares none.
        max_frame: The largest frame, unbinned, of a model whose lens
            images less than its sensor; None where the row declares none.
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
    no catalogue the declared model has no entry, `capabilities.has_turret`
    answers False, and a turret scope whose board did not come up is
    configured as turretless -- answering with its stored objective
    instead of the one in the light path. So an unusable file or a
    missing section refuses, naming the file. Each model's channel rows are
    resolved against the release's channel vocabulary (``release_channels``).

    ``data_file`` None reads the installation's own folder.

    Raises:
        InstallationFileError: as ``read_scope_models``, or the release's
            vocabulary is unusable (``release_channels``).
    """
    path = data_file if data_file is not None else resolve_data_file('scopes.json')
    return read_scope_models(
        read_installation_file(path), path, release_channels(), release_old_names()
    )


def read_scope_models(
    scopes_data: dict,
    path: str | os.PathLike,
    catalogue: tuple[CatalogueChannel, ...],
    old_names: tuple[OldChannelName, ...],
) -> Mapping[str, ScopeModel]:
    """``scopes_data``'s models as a read-only mapping of records by name, or a refusal naming ``path``.

    Every row is whole or the catalogue is refused: the file ships with the
    release, so a row it cannot read is the release's defect, and read in
    part it would build a scope unlike the hardware (a missing ``Turret``
    makes a turret scope turretless).

    Raises:
        InstallationFileError: there is no Models section, or an entry is
            not an object, misses a required key, carries an undeclared
            one, or gives a key a value it cannot hold; a channel row as
            ``_model_channels`` refuses it. Each names the model and the key.
    """
    models = scopes_data.get('Models')
    if not isinstance(models, dict):
        raise InstallationFileError(
            path, f'has no usable Models section (found {type(models).__name__})'
        )
    return types.MappingProxyType(
        {
            model: _scope_model(model, entry, path, catalogue, old_names)
            for model, entry in models.items()
        }
    )


def _scope_model(
    model: str,
    entry: object,
    path: str | os.PathLike,
    catalogue: tuple[CatalogueChannel, ...],
    old_names: tuple[OldChannelName, ...],
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
        channels=_model_channels(entry.get('Layers'), catalogue, old_names, refuse),
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


def load_channel_catalogue(
    scopes_data: dict, path: str | os.PathLike
) -> tuple[CatalogueChannel, ...]:
    """The release's channel records, in display order, or a refusal naming ``path``.

    The catalogue is the single authored order: a channel's id IS its
    position here, and every identity row (model or per-unit block) must
    name a catalogued key to resolve. Deriving the order from the
    per-model row lists instead would make id assignment depend on which
    model happens to be listed first, so the order is stated once. An
    empty vocabulary would let the settings check pass with no channels to
    check, so a missing one refuses. Every fact about a channel that is
    the same on every unit is a field of its record, and a record missing
    one has no answer for it, so it refuses too.

    Raises:
        InstallationFileError: ``LayerOrder`` is missing, empty, or not a
            list of records; or a record has a field missing or of the
            wrong kind, or repeats another's key.
    """
    raw = scopes_data.get('LayerOrder')
    if not isinstance(raw, list) or not raw:
        raise InstallationFileError(path, f'has no usable LayerOrder (found {raw!r})')
    records = tuple(_catalogue_channel(entry, path) for entry in raw)
    keys = [record.key for record in records]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise InstallationFileError(path, f'has LayerOrder records repeating key {repeated}')
    return records


# The fields every catalogue record states.
_CATALOGUE_FIELDS = tuple(field.name for field in fields(CatalogueChannel))


def _catalogue_channel(entry: object, path: str | os.PathLike) -> CatalogueChannel:
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
    illumination_ceiling = entry['manual_max_illumination_ma']
    if illumination_ceiling is not None and not (
        isinstance(illumination_ceiling, int)
        and not isinstance(illumination_ceiling, bool)
        and illumination_ceiling > 0
    ):
        raise refuse('whose manual_max_illumination_ma is not a positive integer or null')
    try:
        parse_display_color(entry['display_color'])
    except ValueError as e:
        raise refuse(f'whose display_color {e}') from e
    light = entry['sim_light_per_ma_s']
    if light is not None and not (_is_number(light) and light > 0):
        raise refuse('whose sim_light_per_ma_s is not a positive number or null')
    power = entry['sim_light_magnification_power']
    if light is None and power is not None:
        raise refuse('whose sim_light_magnification_power is set on a channel with no light')
    if light is not None and not _is_number(power):
        raise refuse('whose sim_light_magnification_power is not a number')
    return CatalogueChannel(
        key=key,
        modality=modality,
        background=background,
        manual_max_exposure_ms=None if ceiling is None else float(ceiling),
        manual_max_illumination_ma=illumination_ceiling,
        display_color=entry['display_color'],
        sim_light_per_ma_s=None if light is None else float(light),
        sim_light_magnification_power=None if power is None else float(power),
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _record(
    key_name: str, row: dict, release: CatalogueChannel, position: int
) -> ChannelRecord | str:
    """The record ``row`` describes for ``key_name``, or why it describes none.

    One reading of a row for both sources, the model's rows and a board's
    Channels block: the LED address, the excitation, the name, and an
    optional colour. A fluorescence channel is labelled by its excitation,
    so it must carry one and needs no name; any other channel is labelled by
    its name. A fluorescence key is named for its excitation (``Ex<nm>``) and
    states that excitation a second time, so a row whose ``excitation_nm`` is
    another describes no channel. A row's colour is held to the one stored
    form and full brightness (`parse_display_color`); without one the
    channel takes the catalogue's.
    """
    raw_addresses = row.get('led_channel')
    if raw_addresses is None:
        led_addresses: tuple[int, ...] = ()
    elif isinstance(raw_addresses, int) and not isinstance(raw_addresses, bool):
        led_addresses = (raw_addresses,)
    elif isinstance(raw_addresses, list) and all(
        isinstance(a, int) and not isinstance(a, bool) for a in raw_addresses
    ):
        led_addresses = tuple(raw_addresses)
    else:
        return f'has malformed led_channel {raw_addresses!r}'
    if len(led_addresses) > 1:
        return f'drives multiple LED addresses {led_addresses}; a channel on several LEDs is not supported yet'
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
        if key_name != f'Ex{excitation:g}':
            return f'has excitation_nm {excitation:g}, not the excitation its key names'
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
        led_addresses=led_addresses,
        excitation_nm=excitation,
        modality=release.modality,
        background=release.background,
        display_color=color,
        label=label,
    )


def _shared_address(named: Iterable[tuple[str, tuple[int, ...]]]) -> str | None:
    """Why ``named`` -- each a name and its LED addresses -- cannot all be one
    unit's: two of them on one LED address."""
    holders: dict[int, str] = {}
    for name, addresses in named:
        for address in addresses:
            if address in holders:
                return f'{holders[address]} and {name} both name LED address {address}'
            holders[address] = name
    return None


def _model_channels(
    rows: object,
    catalogue: tuple[CatalogueChannel, ...],
    old_names: tuple[OldChannelName, ...],
    refuse: Callable[[str], InstallationFileError],
) -> tuple[ChannelRecord, ...]:
    """A model's channel rows as records, in id order, or ``refuse``'s refusal.

    Every row is whole or the model is refused, as its entry is: a row
    whose key is not in the release's vocabulary, that misses a field or
    carries an undeclared one, that ``_record`` cannot read, or that shares
    an LED address with another. Each record takes its key's catalogue
    fields from ``catalogue``. A row keyed by an old name's stock key sits
    on that old name's address: a unit of the model stored its old name for
    that LED, and on any other address its old data would light another.
    """
    by_key = {record.key: (position, record) for position, record in enumerate(catalogue)}
    if not isinstance(rows, list):
        raise refuse("a 'Layers' that is not a list of rows")
    records = []
    for row in rows:
        if not isinstance(row, dict):
            raise refuse(f'a channel row that is not an object: {row!r}')
        key_name = row.get('key_name')
        if not isinstance(key_name, str) or key_name not in by_key:
            raise refuse(f'the channel {key_name!r}, not one of this release {tuple(by_key)}')
        position, release = by_key[key_name]
        required, declared = _row_fields(release.modality)
        missing = [k for k in required if k not in row]
        if missing:
            raise refuse(f'a {key_name} channel row with no {missing[0]!r}')
        undeclared = sorted(set(row) - declared)
        if undeclared:
            raise refuse(f'a {key_name} channel row with the undeclared key {undeclared[0]!r}')
        record = _record(key_name, row, release, position)
        if isinstance(record, str):
            raise refuse(f'a {key_name} channel row that {record}')
        for old in old_names:
            if old.stock_key == key_name and record.led_addresses != (old.led_address,):
                on = ', '.join(map(str, record.led_addresses)) or 'none'
                raise refuse(
                    f'a {key_name} channel row on LED address {on}, where '
                    f'{old.name}, which reads as {key_name}, was LED address {old.led_address}'
                )
        records.append(record)
    shared = _shared_address((r.key_name, r.led_addresses) for r in records)
    if shared is not None:
        raise refuse(f'channel rows where {shared}')
    return tuple(sorted(records, key=lambda r: r.id))


def _parse_channel_block(
    channels: object, catalogue: tuple[CatalogueChannel, ...], report: Callable[[Notice], None]
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
    by_key = {record.key: (position, record) for position, record in enumerate(catalogue)}
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
    shared = _shared_address((r.key_name, r.led_addresses) for r in records)
    if shared is not None:
        report(ChannelBlockUnusableNotice(shared))
        return ()
    return tuple(sorted(records, key=lambda r: r.id))


def resolve_channel_identity(
    *,
    board_block: dict | None,
    board_config_read_ok: bool,
    motor_model: str | None,
    configured_model: str | None,
    models: Mapping[str, ScopeModel],
    catalogue: tuple[CatalogueChannel, ...],
    report: Callable[[Notice], None],
) -> ChannelIdentity:
    """Resolve the unit's channel identity from the first authoritative source.

    Precedence: the unit's own motorconfig Channels block with its
    Filterset, taken WHOLE (a block describes one physical filterset
    assembly, so merging it field-by-field with model data could describe
    hardware that does not exist); else the model's
    `scopes.json` rows, with the motor-reported model outranking the
    configured one because hardware truth beats a user selection; else,
    for a model the catalogue does not list, the stock optics module's
    rows (`stock`); else, with no model at all, the empty `unresolved`
    snapshot. Whichever rung answers, the snapshot
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
    ``catalogue`` the release's channel records, each read once by its
    owner, so rows are resolved against the one vocabulary every channel
    list in the process uses.

    ``motor_model`` is the name the board gives as its own, None where the
    scope has no board to name one (``ChannelIdentity.model_named_by_board``).
    """
    # Decided here, where the board's name is weighed against the selection.
    named_by_board = bool(motor_model)

    def _from_model(model: str) -> ChannelIdentity:
        entry = models.get(model)
        if entry is None:
            return ChannelIdentity(
                channels=(),
                filterset='',
                source=IdentitySource.UNRESOLVED,
                model=model,
                model_named_by_board=named_by_board,
            )
        return ChannelIdentity(
            channels=entry.channels,
            filterset=entry.filterset or '',
            source=IdentitySource.SCOPES,
            model=model,
            model_named_by_board=named_by_board,
        )

    # The configured model is consulted only when the hardware reports no
    # model at all. A motor-reported model with no scopes entry is an OEM
    # unit's own name: it keeps that name and has the stock optics module's
    # channels (source STOCK), never whatever the user last selected.
    model = motor_model or configured_model

    board = board_block or {}
    if 'Channels' in board:
        channels = _parse_channel_block(board['Channels'], catalogue, report)
        filterset = board.get('Filterset', '')
        if not isinstance(filterset, str):
            filterset = ''
        return ChannelIdentity(
            channels=channels,
            filterset=filterset,
            source=IdentitySource.MOTORCONFIG,
            model=model,
            model_named_by_board=named_by_board,
        )
    if 'Layers' in board or 'Filterset' in board:
        report(ChannelBlockSupersededNotice(model))

    if not board_config_read_ok:
        logger.error(
            '[CHANNEL_RECORD] board config could not be read; a per-unit LED '
            'block may exist but is unavailable -- resolving from the model '
            'instead'
        )

    if not model:
        return UNRESOLVED
    identity = _from_model(model)
    if identity.source != IdentitySource.UNRESOLVED:
        return identity
    stock = _from_model(STOCK_OPTICS_MODEL)
    logger.info(
        f'[CHANNEL_RECORD] model {model!r} has no scopes entry; its channels are '
        f"the stock optics module's ({STOCK_OPTICS_MODEL})"
    )
    return ChannelIdentity(
        channels=stock.channels,
        filterset=stock.filterset,
        source=IdentitySource.STOCK,
        model=model,
        model_named_by_board=named_by_board,
    )
