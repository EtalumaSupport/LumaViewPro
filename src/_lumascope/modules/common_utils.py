# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import csv
import ctypes
import dataclasses
import enum
import numbers
import os
import pathlib
import platform
import re
import threading
import time as _time
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import psutil

from _lumascope.modules.exceptions import Refusal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd

    from _lumascope.modules.channel_record import Modality

    from _lumascope.modules.lumascope_api.illumination import LedReading
    from _lumascope.modules.scope_capabilities import FieldOfView, ScopeCapabilities


# ---------------------------------------------------------------------------
# Hardware defaults (fallbacks when motorconfig/scope not available)
# ---------------------------------------------------------------------------
# LS850 full travel range -- used as default stage limits when scope is not connected.
DEFAULT_STAGE_TRAVEL_UM = {'x': 120000.0, 'y': 80000.0}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@enum.unique
class PostFunction(enum.Enum):
    COMPOSITE = 'Composite'
    STITCHED = 'Stitched'
    ZPROJECT = 'ZProject'
    VIDEO = 'Video'
    HYPERSTACK = 'Hyperstack'

    @classmethod
    def list_values(cls):
        return [c.value for c in cls]


@dataclasses.dataclass(frozen=True)
class StepNameComponents:
    """The semantic fields a step name encodes -- the single source of truth.

    A step name is RENDERED from these fields by build_step_name and (only at
    load boundaries for files that predate the Label column) RECOVERED by
    parse_legacy_step_name. Holding the fields, not the rendered string, is what makes
    renaming idempotent: changing one field and rebuilding always yields exactly
    one token for it. The legacy builder appended a token only if the rendered
    string did not already contain it, so feeding a built name back in as the
    seed left the stale token beside the new one (e.g. a channel change produced
    'A2_BF_Green'); rebuilding from components cannot do that.
    """

    well: str = ''  # 'A2'; the base when no custom_prefix is set
    custom_prefix: str = ''  # step label (user text or 'custom0001'); renders as the base when set
    channel: str | None = None  # 'BF' | 'Ex488' | 'Composite' | None
    tile: str | None = None  # tile label, rendered '_T<tile>'
    objective: str | None = None  # short objective name (turret builds)
    turret_position: int | None = None  # rendered '_Turret<n>'
    z_index: int | None = None  # rendered '_Z<n>'
    scan_count: int | None = None  # rendered zero-padded to 4 digits
    # Ordered post-output suffixes. A chain, not a single value: a stitched
    # output that is later z-projected carries both ('stitched', 'zproj_median')
    # and renders '_stitched_zproj_median'. Each element is a single suffix
    # token ('stitched'|'video'|'stack'|'hyperstack') or 'zproj_<method>'.
    post: tuple[str, ...] = ()


# The 'video' post token is the one suffix other modules build on: the
# recording engine derives video-recording folder names and frame-file
# tokens from it (src/_lumascope/modules/recording_frames.py), so it gets a named
# constant instead of living only inside the frozenset literal.
POST_TOKEN_VIDEO = 'video'

# Single-token post-output suffixes occupying StepNameComponents.post. A
# z-projection is two segments ('zproj_<method>'); the rest are single tokens.
_POST_SUFFIX_TOKENS = frozenset({'stitched', POST_TOKEN_VIDEO, 'stack', 'hyperstack'})


def build_step_name(c: StepNameComponents) -> str:
    """Render a step name deterministically from its components.

    Replace-not-append: the name is assembled fresh from the fields, with no
    'skip this token if the string already has it' guard and no path that folds
    an already-built name back in as the seed. Feeding a built name back in as
    the seed and appending a changed field was what left the stale token beside
    the new one (a channel change produced 'A2_BF_Green'); building from
    components in a fixed token order cannot do that.
    """
    base = c.custom_prefix if c.custom_prefix else c.well
    parts = [base]
    if c.channel not in (None, ''):
        parts.append(c.channel)
    if c.tile not in (None, '', -1):
        parts.append(f'T{c.tile}')
    if c.objective not in (None, '', -1):
        parts.append(c.objective)
    if c.turret_position is not None:
        parts.append(f'Turret{c.turret_position}')
    if c.z_index not in (None, '', -1):
        parts.append(f'Z{c.z_index}')
    if c.scan_count not in (None, ''):
        parts.append(f'{c.scan_count:0>4}')
    parts.extend(c.post)
    return '_'.join(p for p in parts if p != '')


def _token_kind(seg: str, channels: set, objectives: set) -> str | None:
    """Classify one step-name segment into the component it fills, or None.

    The single definition of "what is a step-name token", shared by
    parse_legacy_step_name's custom-prefix accretion (which only needs to know
    whether a segment classifies), its field-assignment loop (which needs
    which field it fills), and the load-boundary machine-shape test in
    recover_step_label. One table means they cannot disagree, and adding a
    token type is a single edit.

    Order is the disambiguation: a turret token ('Turret<n>') is tested
    before the tile token, whose prefix is the same 'T' letter. The tile
    shape is 'T<row-letters><col-number>', where the row label is one OR
    MORE uppercase letters (a mosaic past 26 rows carries 'AA', 'AB', ...).
    Requiring uppercase letters then digits means it cannot swallow
    'Turret<n>' (whose 'urret' is lowercase) and lose the turret position,
    which would otherwise surface as a bogus tile that broke callers
    parsing the tile's trailing number.
    """
    if seg in channels or seg == 'Composite':
        return 'channel'
    if re.fullmatch(r'Turret\d+', seg):
        return 'turret_position'
    if re.fullmatch(r'T[A-Z]+\d+', seg):
        return 'tile'
    if seg in objectives or re.fullmatch(r'\d+x.*', seg):
        return 'objective'
    if re.fullmatch(r'Z\d+', seg):
        return 'z_index'
    if re.fullmatch(r'\d{4,}', seg):
        return 'scan_count'
    if seg in _POST_SUFFIX_TOKENS:
        return 'post'
    if seg == 'zproj':
        return 'zproj'
    return None


def _machine_tokens_only(segs: list[str], channels: set, objectives: set) -> bool:
    """True when every segment classifies as a known step-name token.

    A machine-generated name -- even one carrying stale tokens from old
    channel-change bugs, or one whose row columns were later output-adjusted
    by a post processor -- consists only of vocabulary tokens after its
    anchor. User-typed text always contains at least one segment the
    vocabulary cannot classify.
    """
    i = 0
    while i < len(segs):
        kind = _token_kind(segs[i], channels, objectives)
        if kind is None:
            return False
        if kind == 'zproj':
            i += 1  # the method segment rides with its zproj token
        i += 1
    return True


def parse_legacy_step_name(
    name: str,
    known_channels: 'Iterable[str] | None' = None,
    known_objectives: 'Iterable[str]' = (),
) -> StepNameComponents:
    """Recover the components from a rendered step name, by SHAPE not position.

    LEGACY LOAD BOUNDARY ONLY: the run-time pipeline never decodes a name --
    fields travel as columns (Label et al.) -- so the only remaining caller
    is the one-shot migration of files that predate the Label column
    (recover_step_label). Each '_'-separated segment is classified by its
    shape (plus the known-channel / known-objective vocabularies), so a token
    never depends on a hard-coded segment index that silently breaks when
    the token set shifts. Round-trips every name build_step_name produces:
    build_step_name(parse_legacy_step_name(s)) == s.

    A custom prefix may itself contain underscores, so leading segments that do
    not classify as a known token accrete into custom_prefix; a custom prefix
    that embeds a token-shaped segment is inherently ambiguous and is not
    guaranteed to round-trip -- only auto-generated 'custom<N>' prefixes are.
    That ambiguity is exactly why the parse is quarantined to the legacy
    boundary and never trusted with user-typed text.
    """
    if known_channels is None:
        known_channels = get_stored_channel_names()
    channels = set(known_channels)
    objectives = set(known_objectives)

    stem = pathlib.Path(name).name
    segs = stem.split('_')
    if not segs or segs == ['']:
        return StepNameComponents()

    def token_kind(seg):
        return _token_kind(seg, channels, objectives)

    def classifies(seg):
        return token_kind(seg) is not None

    well = ''
    custom_prefix = ''
    if re.fullmatch(r'[A-Z]{1,2}\d+', segs[0]):
        well = segs[0]
        i = 1
    else:
        custom_parts = [segs[0]]
        i = 1
        while i < len(segs) and not classifies(segs[i]):
            custom_parts.append(segs[i])
            i += 1
        custom_prefix = '_'.join(custom_parts)

    channel = tile = objective = None
    turret_position = z_index = scan_count = None
    post = []
    while i < len(segs):
        seg = segs[i]
        kind = token_kind(seg)
        if kind == 'channel':
            channel = seg
        elif kind == 'tile':
            tile = seg[1:]
        elif kind == 'objective':
            objective = seg
        elif kind == 'turret_position':
            turret_position = int(seg[len('Turret') :])
        elif kind == 'z_index':
            z_index = int(seg[1:])
        elif kind == 'scan_count':
            scan_count = int(seg)
        elif kind == 'post':
            post.append(seg)
        elif kind == 'zproj' and i + 1 < len(segs):
            post.append(f'zproj_{segs[i + 1]}')
            i += 1
        i += 1

    return StepNameComponents(
        well=well,
        custom_prefix=custom_prefix,
        channel=channel,
        tile=tile,
        objective=objective,
        turret_position=turret_position,
        z_index=z_index,
        scan_count=scan_count,
        post=tuple(post),
    )


def _blank_to_none(value):
    """Normalize an absent step-column value (empty string or the -1 sentinel) to None."""
    return None if value in (None, '', -1) else value


def step_components(
    step: 'pd.Series | Mapping[str, object]', **overrides: object
) -> StepNameComponents:
    """Map a protocol step / post-record row to its canonical name components.

    The single place that reads a step's columns into StepNameComponents, so
    every name-building site renders one identity from one source. Well, channel,
    Tile, Z-Slice and Label are the authoritative columns. Label is the step's
    base text -- a user-typed name kept verbatim, or the machine-assigned
    'custom<NNNN>' prefix of an added step; empty means the base is the Well.
    A label is never parsed back out of the rendered Name, so user text that
    happens to embed a token-shaped segment ('Treatment_10x') can never be
    truncated by the token vocabulary. The objective is deliberately not part
    of a step's identity: it is a capture-time turret detail the writer stamps
    onto the saved filename, never the Name. Pass overrides (channel=, tile=,
    z_index=, scan_count=, objective=, turret_position=, post=) to set the
    capture- or output-specific fields a given site adds.
    """
    well = step['Well']
    well = '' if well in (None, '') else str(well)
    label = step['Label']
    label = '' if label in (None, '') else str(label)
    base = StepNameComponents(
        well=well,
        custom_prefix=label,
        channel=_blank_to_none(step['channel']),
        tile=_blank_to_none(step['Tile']),
        z_index=_blank_to_none(step['Z-Slice']),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def recover_step_label(step: 'pd.Series | Mapping[str, object]') -> tuple[str, bool]:
    """Recover (label, is_auto) for a legacy row that predates the Label column.

    Load-boundary classification, run once per row when migrating a pre-Label
    protocol or post-record file. Two machine shapes are recognized:

    1. A Name byte-equal to the render of the row's structured columns -- the
       ordinary auto-generated name.
    2. A Name that is its own anchor (the row's Well, or a 'custom<NNNN>'
       prefix) followed only by vocabulary tokens. This is the shape of names
       written by old releases whose channel change appended a token instead
       of replacing it ('A2_BF_Green'), and of post-record rows whose columns
       were output-adjusted (a stitch blanks the tile, a composite rewrites
       the color) while Name kept the source step's text.

    Both recover the machine base ('' for a well anchor, the prefix for a
    custom step), so the re-render cleans stale tokens exactly as the old
    run-time parse did. Anything else is user text kept verbatim -- the lossy
    token-shape parse is never trusted with it, so a label like
    'Treatment_10x' survives whole. Comparing shapes (rather than trusting a
    stored auto/user flag) also absorbs legacy rows whose flag says
    user-named but whose Name is really machine-shaped; taking such a Name as
    a verbatim label would re-suffix it on the next render ('A2_BF_BF').
    """
    name = str(step['Name'])
    well = step['Well']
    well = '' if well in (None, '') else str(well)
    candidate_prefix = '' if well else parse_legacy_step_name(name).custom_prefix
    rendered = build_step_name(
        step_components(
            {
                'Well': well,
                'Label': candidate_prefix,
                'channel': step['channel'],
                'Tile': step['Tile'],
                'Z-Slice': step['Z-Slice'],
            }
        )
    )
    if rendered == name:
        return candidate_prefix, True

    segs = name.split('_')
    if well:
        anchored = segs[0] == well
        machine_base = ''
    else:
        anchored = re.fullmatch(r'custom\d+', segs[0]) is not None
        machine_base = segs[0]
    if anchored and _machine_tokens_only(segs[1:], set(get_stored_channel_names()), set()):
        return machine_base, True

    return name, False


def resolve_step_rename(raw_text: str) -> str | None:
    """Resolve a step-name field value to the name to rename to, or None.

    Auto-named custom steps blank the name field so the default name shows
    as a hint placeholder rather than editable text. A blank field
    therefore means "no rename intended": persisting the empty string
    would wipe the auto-assigned name, leaving added steps unnamed and
    colliding on the same default name. Returns None for a blank field so
    callers keep the existing name; anything else is a real rename, passed
    as typed, and the protocol cleans it or refuses it.

    Args:
        raw_text: the raw text from the step-name input field.

    Returns:
        The text to rename to, or None if the field is blank.
    """
    return raw_text if raw_text.strip() else None


def first_few(items: list[str], *, separator: str, limit: int = 5) -> str:
    """The first *limit* of *items*, then how many more there are.

    A refusal that names everything it found grows with the protocol: a
    96-step list is taller than the popup that shows it, and the count it
    opens with scrolls out of sight. Keep the order the caller gives, so the
    names read in the order the person built them.
    """
    shown = separator.join(items[:limit])
    remaining = len(items) - limit
    return f'{shown}{separator}and {remaining} more' if remaining > 0 else shown


def is_valid_gain_db(value: object) -> bool:
    """True when a camera gain reading is usable (a number >= 0 dB).

    Negative is the drivers' not-connected answer (a read that fails
    raises); 0 dB is a legal gain. A non-number (e.g. a stringified value
    from a hand-built snapshot) is invalid rather than a comparison
    TypeError, so every consumer can branch on the predicate without its
    own type guard. One shared predicate so every consumer validates the
    same way.
    """
    return isinstance(value, numbers.Real) and value >= 0


def is_valid_exposure_ms(value: object) -> bool:
    """True when a camera exposure reading is usable (a positive number).

    Negative is the drivers' not-connected answer (a read that fails
    raises) and 0 is the API's inactive-camera return -- neither is a
    physical exposure. A non-number is invalid rather than a comparison
    TypeError. One shared predicate so every consumer validates the same
    way.
    """
    return isinstance(value, numbers.Real) and value > 0


def is_valid_frame_size(value: object) -> bool:
    """True when a frame-size reading is usable (positive width and height).

    The drivers answer None (or an empty dict for the max/min variants)
    when the camera is not connected (a read that fails raises), and a
    zero dimension is never a deliverable frame. One shared predicate so
    every consumer validates the same way.
    """
    if not isinstance(value, dict):
        return False
    width = value.get('width')
    height = value.get('height')
    return (
        isinstance(width, numbers.Real)
        and isinstance(height, numbers.Real)
        and width > 0
        and height > 0
    )


def is_valid_pixel_format(value: object) -> bool:
    """True when a pixel-format reading is usable (a non-empty string).

    None is the drivers' not-connected answer (a read that fails raises)
    -- distinct from every real format name so it cannot be mistaken for
    one.
    """
    return isinstance(value, str) and value != ''


def is_valid_binning_size(value: object) -> bool:
    """True when a binning reading is usable (a whole factor >= 1).

    Below 1 is never a binning a camera holds (a read that fails raises);
    binning is always at least 1x1 on real hardware.
    """
    return isinstance(value, numbers.Real) and value >= 1


def get_channels() -> list[str]:
    """The release's channel key names, in display order.

    Derived from the release catalogue rather than authored here, so the
    vocabulary has exactly one source and a catalogue change cannot leave
    this list silently disagreeing with it.
    """
    from _lumascope.modules.channel_record import release_catalogue

    return list(release_catalogue())


def get_stored_channel_names() -> list[str]:
    """Every channel name a saved file may hold: the release's keys, then the
    old names earlier releases stored (``channel_record.stored_channel_names``)."""
    from _lumascope.modules.channel_record import stored_channel_names

    return list(stored_channel_names())


def _channels_of_modality(modality: 'Modality') -> list[str]:
    from _lumascope.modules.channel_record import release_channels

    return [record.key for record in release_channels() if record.modality == modality]


def get_transmitted_channels() -> list[str]:
    from _lumascope.modules.channel_record import Modality

    return _channels_of_modality(Modality.TRANSMITTED)


def get_fluorescence_channels() -> list[str]:
    from _lumascope.modules.channel_record import Modality

    return _channels_of_modality(Modality.FLUORESCENCE)


def get_luminescence_channels() -> list[str]:
    from _lumascope.modules.channel_record import Modality

    return _channels_of_modality(Modality.LUMINESCENCE)


def get_image_channels() -> list[str]:
    """Fluorescence + luminescence channels (false-color, displayed as colored images)."""
    return get_fluorescence_channels() + get_luminescence_channels()


def get_channels_with_led() -> list[str]:
    return get_transmitted_channels() + get_fluorescence_channels()


DEFAULT_CHANNEL = 'BF'


class Light(NamedTuple):
    """What lit one frame, from one read of the LEDs.

    Attributes:
        channel: The channel the frame is named for (``light_of``).
        illumination_by_channel_ma: The current of every lit LED a channel
            names, by that channel's key, in address order; empty when none
            was lit.
    """

    channel: str
    illumination_by_channel_ma: 'Mapping[str, float]'


def light_of(readings: 'tuple[LedReading, ...] | None', shown_channel: str) -> Light:
    """The channel a recorded frame was imaged in, and every lit channel's current, from one read.

    The channel is the lit LED, else ``shown_channel``, the channel the recording
    is named for. A recording refuses to start with another LED lit and
    refuses a light of any LED not lit above 0 mA while it runs
    (``ManualRecordingController.start``, ``IlluminationAPI.lights_a_dark_led``),
    and a protocol's video step lights its own channel alone, so a frame's
    lit LED is its channel's; the shown channel names a frame with no LED lit,
    as luminescence images. With more than one LED lit, the first in
    address order wins; a lit address the identity names no channel for
    names no channel and has no current here.

    Lit is the frame record's meaning: commanded on at a current above 0 mA,
    since the board takes 0 mA as off (``snapshot_lit_pairs``). An LED left
    on at 0 mA names nothing. ``readings`` is None when no LED board is
    installed: nothing is lit, so a board-less scope falls through to the
    channel.
    """
    from _lumascope.modules.lumascope_api.illumination import snapshot_lit_pairs

    lit = {address for address, _ma in snapshot_lit_pairs(readings)}
    named = [r for r in readings or () if r.address in lit and r.name is not None]
    return Light(
        channel=named[0].name if named else shown_channel,
        illumination_by_channel_ma={r.name: r.illumination_ma for r in named},
    )


def read_table(lines: 'Iterable[str]', *, sep: str) -> tuple[list[str], list[list[str]]]:
    """The header and rows of a delimited table, every cell the text written.

    The inverse of ``csv.writer``, which every LumaViewPro table writer is:
    no cell is turned into a missing value, a number or a date here, so a
    step named ``NA`` reads back as ``NA``; each reader types its own
    columns by its own rule. Blank lines are skipped.

    ``lines`` (an open file, or the rest of one) is read once, as it comes,
    so the table's text is never held whole. A cell whose text was read
    before is the object read before: a protocol's cells are a few thousand
    texts repeated on every row, and an object per cell held a large
    protocol's load at several times the table it returned.

    Raises:
        ValueError: the table has no header, a header naming one column
            twice, a NUL, a quote the csv module cannot close, or a row
            whose cell count is not the header's. No LumaViewPro writer
            produces any of them.
    """
    texts: dict[str, str] = {}
    try:
        rows = [
            [texts.setdefault(cell, cell) for cell in row]
            for row in csv.reader(lines, delimiter=sep, strict=True)
            if row
        ]
    except csv.Error as e:
        raise ValueError(f'it is not a well-formed table ({e})') from None
    if any('\x00' in text for text in texts):
        raise ValueError('it holds a NUL character, which no table cell can hold')
    if not rows:
        raise ValueError('its table has no header row')
    header, body = rows[0], rows[1:]
    repeated = sorted({column for column in header if header.count(column) > 1})
    if repeated:
        raise ValueError(f'its table header names {", ".join(map(repr, repeated))} more than once')
    for number, row in enumerate(body, 2):
        if len(row) != len(header):
            raise ValueError(
                f'row {number} of its table has {len(row)} cells where the header has {len(header)}'
            )
    return header, body


def to_bool(val) -> bool:
    if isinstance(val, str):
        return val.lower() == 'true'
    elif val in ('', None):
        return False
    else:
        return bool(float(val))


def to_float(val) -> float:
    if 'numpy' in str(type(val)):
        return val.astype(float)
    else:
        return float(val)


def get_pixel_size(
    focal_length: float,
    binning_size: int | None,
    *,
    capabilities: 'ScopeCapabilities',
) -> float | None:
    """Effective um/pixel for the given objective focal length and binning.

    Reads the tube focal length and sensor pixel size from the scope
    capabilities the caller hands in (the single source of truth for image
    scale). Returns None when the scope cannot report its optics (unknown
    camera, no declared optics) or the binning is unknown -- callers then
    degrade honestly (no scale bar, no field of view, no PhysicalSizeX)
    rather than using an invented scale. There is deliberately no hardcoded fallback: a guessed pixel size
    is written into every image and cannot be told from a measured one.

    The capabilities are required, never looked up: every producer of a
    scale holds the scope it is imaging with, and a lookup through a
    process-wide store answered None for every headless run, so every
    headless image was written with no scale.
    """
    tube_focal_length = capabilities.lens_focal_length_mm
    pixel_width = capabilities.pixel_size_um
    if tube_focal_length is None or pixel_width is None or binning_size is None:
        return None
    magnification = tube_focal_length / focal_length
    um_per_pixel = pixel_width / magnification

    return um_per_pixel * binning_size


def get_field_of_view(
    focal_length: float,
    frame_size: dict,
    binning_size: int,
    *,
    capabilities: 'ScopeCapabilities',
) -> dict | None:
    um_per_pixel = get_pixel_size(
        focal_length=focal_length,
        binning_size=binning_size,
        capabilities=capabilities,
    )
    # No scale, no field of view. A fabricated extent reads downstream as a
    # measurement (tiling steps, FOV readouts); None is the honest signal that
    # this scope cannot report its field of view.
    if um_per_pixel is None:
        return None
    fov_x = um_per_pixel * frame_size['width']
    fov_y = um_per_pixel * frame_size['height']

    return {'width': fov_x, 'height': fov_y}


FOV_UNKNOWN_TEXT = 'unknown'


def format_field_of_view(fov: 'FieldOfView | Refusal') -> tuple[str, str]:
    """Render (width, height) FOV label text for the settings readouts.

    Returns the rounded micron strings, or ('unknown', 'unknown') for the
    refusal that says why the scope cannot report a field of view
    (``RuntimeState.optics_status``).
    """
    if isinstance(fov, Refusal):
        return (FOV_UNKNOWN_TEXT, FOV_UNKNOWN_TEXT)
    return (str(round(fov.width_um, 0)), str(round(fov.height_um, 0)))


def max_decimal_precision(parameter: str) -> int:
    DEFAULT_PRECISION = 5
    PRECISION_MAP = {'x': 4, 'y': 4, 'z': 5}

    return PRECISION_MAP.get(parameter, DEFAULT_PRECISION)


_IS_WINDOWS = platform.system() == 'Windows'

# The PDH statuses whose value is usable: VALID_DATA and NEW_DATA.
_PDH_VALID_STATUSES = (0, 1)


class CounterFailed(NamedTuple):
    """A performance counter the query added at its start whose read then
    failed, with the PDH status the read returned."""

    status: int


class _PdhQueryOnce:
    """A PDH query opened on its first call and kept for every later one.

    A start that fails is kept and raised again by every later call, so a
    failed query reads as failed on each tick rather than once, and costs
    no retry.
    """

    _PDH_FMT_DOUBLE = 0x00000200
    _ERROR_SUCCESS = 0

    def __init__(self):
        self._initialized = False
        self._init_error: Exception | None = None
        self._query = None

    def _collected(self):
        """Start the query if it has not started, then collect it.

        Raises:
            OSError: the start or the collect failed.
        """
        if self._init_error is not None:
            raise self._init_error.with_traceback(None)
        if not self._initialized:
            try:
                self._init()
            except Exception as failed:
                self._init_error = failed
                raise
        ret = self._PdhCollectQueryData(self._query)
        if ret != self._ERROR_SUCCESS:
            raise OSError(f'PdhCollectQueryData failed: 0x{ret & 0xFFFFFFFF:08x}')


# ---------------------------------------------------------------------------
# Windows perf-counter query (PDH) -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Added on branch `perf-instrumentation-4.0.0-beta` to capture standby cache
# growth, nonpaged pool, and system file cache as part of the buffer-churn
# investigation. These counters are NOT in psutil. The PDH layer ("\Memory\..."
# perf-counter paths) is the same data PowerShell `Get-Counter` returns.
#
# Lifetime: this entire `_PdhCountersOnce` helper plus the [PDH METRICS]
# line built from it are temporary. Remove once the buffer-reuse fixes
# land and the standby-cache trend is verified flat.
#
# Performance: PdhCollectQueryData is one syscall per counter, ~1 ms total.
# Safe to call once per minute.
# ---------------------------------------------------------------------------


class _PdhCountersOnce(_PdhQueryOnce):
    """Lazy-initialized PDH query for a fixed set of counters.

    Opens the PDH query on first call, caches the counter handles, and
    re-collects on each subsequent call.
    """

    # Counter paths -- match `Get-Counter` PowerShell paths exactly.
    # `\Memory\Available Bytes` is what Windows considers "available" -- equals
    # standby + free + zero pages. Useful as a sanity check against the breakdown.
    _COUNTERS: ClassVar[dict] = {
        'standby_normal_bytes': r'\Memory\Standby Cache Normal Priority Bytes',
        'standby_reserve_bytes': r'\Memory\Standby Cache Reserve Bytes',
        'standby_core_bytes': r'\Memory\Standby Cache Core Bytes',
        'pool_nonpaged_bytes': r'\Memory\Pool Nonpaged Bytes',
        'pool_paged_bytes': r'\Memory\Pool Paged Bytes',
        'system_cache_bytes': r'\Memory\Cache Bytes',
        'modified_page_bytes': r'\Memory\Modified Page List Bytes',
        'free_zero_bytes': r'\Memory\Free & Zero Page List Bytes',
        'available_bytes': r'\Memory\Available Bytes',
        'commit_bytes': r'\Memory\Committed Bytes',
        'commit_limit_bytes': r'\Memory\Commit Limit',
    }

    def __init__(self):
        super().__init__()
        self._handles = {}  # name -> counter handle

    def _init(self):
        pdh = ctypes.WinDLL('pdh')
        self._pdh = pdh

        self._PdhOpenQueryW = pdh.PdhOpenQueryW
        self._PdhAddCounterW = pdh.PdhAddCounterW
        self._PdhCollectQueryData = pdh.PdhCollectQueryData
        self._PdhGetFormattedCounterValue = pdh.PdhGetFormattedCounterValue

        class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
            _fields_ = [
                ('CStatus', ctypes.c_ulong),
                ('doubleValue', ctypes.c_double),
            ]

        self._PDH_FMT_COUNTERVALUE = _PDH_FMT_COUNTERVALUE

        query = ctypes.c_void_p()
        ret = self._PdhOpenQueryW(None, 0, ctypes.byref(query))
        if ret != self._ERROR_SUCCESS:
            raise OSError(f'PdhOpenQueryW failed: 0x{ret & 0xFFFFFFFF:08x}')
        self._query = query

        for name, path in self._COUNTERS.items():
            handle = ctypes.c_void_p()
            ret = self._PdhAddCounterW(query, path, 0, ctypes.byref(handle))
            if ret != self._ERROR_SUCCESS:
                # Some counters (e.g. Standby Cache Core) may be absent
                # on older Windows builds -- mark this one missing and
                # continue. PdhCollectQueryData still works on the rest.
                continue
            self._handles[name] = handle

        # First collect "primes" the counters; second collect gives real
        # values. Some counters (rate-based) need 2 samples -- for the byte
        # counters we use, a single collect is enough.
        self._PdhCollectQueryData(query)

        self._initialized = True

    def query(self) -> dict[str, float | CounterFailed]:
        """Collect every counter this Windows build has.

        Returns:
            {field: bytes} for each counter added at the start, a counter
            whose read failed as its CounterFailed; a counter this build
            lacks is absent.

        Raises:
            OSError: the query's start or its collect failed; a failed
                start is raised again on every call.
        """
        self._collected()
        out = {}
        for name, handle in self._handles.items():
            value = self._PDH_FMT_COUNTERVALUE()
            ret = self._PdhGetFormattedCounterValue(
                handle, self._PDH_FMT_DOUBLE, None, ctypes.byref(value)
            )
            out[name] = (
                float(value.doubleValue)
                if ret == self._ERROR_SUCCESS
                else CounterFailed(ret & 0xFFFFFFFF)
            )
        return out


_pdh_counters_singleton = _PdhCountersOnce() if _IS_WINDOWS else None


def query_windows_perf_counters() -> dict[str, float | CounterFailed] | None:
    """One-shot snapshot of selected Windows memory perf counters.

    TEMPORARY -- see `_PdhCountersOnce`'s comment.

    Returns:
        {field: bytes} as ``_PdhCountersOnce.query`` answers it; None off
        Windows, where the counters do not exist.

    Raises:
        OSError: as ``_PdhCountersOnce.query``.
    """
    if _pdh_counters_singleton is None:
        return None
    return _pdh_counters_singleton.query()


# ---------------------------------------------------------------------------
# Windows GPU utilization + memory via PDH "GPU Engine" counters
#
# Vendor-agnostic: the GPU Engine / GPU Process Memory counter sets are
# populated by the WDDM driver model for ANY adapter -- AMD integrated, Intel,
# NVIDIA alike -- the same data Task Manager's GPU column reads. No vendor SDK
# (pynvml / GPUtil are NVIDIA-only and would be dead on an AMD box).
#
# Unlike the fixed memory counters above, GPU Engine is a WILDCARD counter:
# one instance per (process, adapter, engine), named like
# "pid_12345_luid_0x0..._phys_0_eng_0_engtype_3D". We add the wildcard path and
# enumerate matching instances each collect via PdhGetFormattedCounterArray,
# then keep only THIS process's pid and aggregate. engtype_3D is the OpenGL /
# Kivy render engine; on an integrated GPU there is no dedicated VRAM, so
# Shared Usage is the meaningful memory figure (Dedicated reads ~0).
# ---------------------------------------------------------------------------

_PDH_MORE_DATA = 0x800007D2


class GpuReading(NamedTuple):
    """This process's GPU use. A field is None where this Windows build
    lacks its counter, and its CounterFailed where one of this process's
    instances failed its read."""

    util_total_percent: float | CounterFailed | None
    util_3d_percent: float | CounterFailed | None
    shared_mem_mb: float | CounterFailed | None
    dedicated_mem_mb: float | CounterFailed | None


class _GpuPdhCountersOnce(_PdhQueryOnce):
    _UTIL_PATH = r'\GPU Engine(*)\Utilization Percentage'
    _SHARED_MEM_PATH = r'\GPU Process Memory(*)\Shared Usage'
    _DEDICATED_MEM_PATH = r'\GPU Process Memory(*)\Dedicated Usage'

    def __init__(self):
        super().__init__()
        self._counters = {}  # name -> counter handle
        self._pid_tag = f'pid_{os.getpid()}_'

    def _init(self):
        pdh = ctypes.WinDLL('pdh')
        self._PdhOpenQueryW = pdh.PdhOpenQueryW
        self._PdhAddCounterW = pdh.PdhAddCounterW
        self._PdhCollectQueryData = pdh.PdhCollectQueryData
        self._PdhGetFormattedCounterArrayW = pdh.PdhGetFormattedCounterArrayW

        class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
            _fields_ = [
                ('CStatus', ctypes.c_ulong),
                ('doubleValue', ctypes.c_double),
            ]

        class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
            _fields_ = [
                ('szName', ctypes.c_wchar_p),
                ('FmtValue', _PDH_FMT_COUNTERVALUE),
            ]

        self._ITEM = _PDH_FMT_COUNTERVALUE_ITEM_W

        query = ctypes.c_void_p()
        ret = self._PdhOpenQueryW(None, 0, ctypes.byref(query))
        if ret != self._ERROR_SUCCESS:
            raise OSError(f'PdhOpenQueryW failed: 0x{ret & 0xFFFFFFFF:08x}')
        self._query = query

        for name, path in (
            ('util', self._UTIL_PATH),
            ('shared_mem', self._SHARED_MEM_PATH),
            ('dedicated_mem', self._DEDICATED_MEM_PATH),
        ):
            handle = ctypes.c_void_p()
            ret = self._PdhAddCounterW(query, path, 0, ctypes.byref(handle))
            if ret == self._ERROR_SUCCESS:
                self._counters[name] = handle

        if not self._counters:
            raise OSError('no GPU PDH counters available')

        # Prime: utilization is rate-based and needs two collects before
        # the first real value.
        self._PdhCollectQueryData(query)
        self._initialized = True

    def _read_array(self, handle) -> list[tuple[str, int, float]]:
        """Every instance of a wildcard counter, as (name, status, value).

        Raises:
            OSError: either call of the read failed.
        """
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        # First pass with a null buffer reports the required size.
        ret = (
            self._PdhGetFormattedCounterArrayW(
                handle,
                self._PDH_FMT_DOUBLE,
                ctypes.byref(size),
                ctypes.byref(count),
                None,
            )
            & 0xFFFFFFFF
        )
        if ret == self._ERROR_SUCCESS or (ret == _PDH_MORE_DATA and size.value == 0):
            return []  # no instance of the counter exists
        if ret != _PDH_MORE_DATA:
            raise OSError(f'PdhGetFormattedCounterArrayW failed: 0x{ret:08x}')
        buf = (ctypes.c_byte * size.value)()
        ret = self._PdhGetFormattedCounterArrayW(
            handle,
            self._PDH_FMT_DOUBLE,
            ctypes.byref(size),
            ctypes.byref(count),
            ctypes.cast(buf, ctypes.POINTER(self._ITEM)),
        )
        if ret != self._ERROR_SUCCESS:
            raise OSError(f'PdhGetFormattedCounterArrayW failed: 0x{ret & 0xFFFFFFFF:08x}')
        items = ctypes.cast(buf, ctypes.POINTER(self._ITEM * count.value)).contents
        return [
            (it.szName or '', it.FmtValue.CStatus, float(it.FmtValue.doubleValue)) for it in items
        ]

    def _this_process(self, counter: str) -> list[tuple[str, int, float]] | None:
        """This process's instances of ``counter``; None where this build
        lacks the counter."""
        if counter not in self._counters:
            return None
        return [
            item for item in self._read_array(self._counters[counter]) if self._pid_tag in item[0]
        ]

    @staticmethod
    def _summed(items, engine: str = '', unit: float = 1.0) -> float | CounterFailed | None:
        """The instances' sum in ``unit``, over those of ``engine`` when it
        is named; the first failed instance's CounterFailed instead."""
        if items is None:
            return None
        total = 0.0
        for name, status, value in items:
            if not name.endswith(engine):
                continue
            if status not in _PDH_VALID_STATUSES:
                return CounterFailed(status)
            total += value
        return total / unit

    def query(self) -> GpuReading:
        """This process's GPU use, summed over its instances.

        A process with no instance of a counter reads 0.0 there.

        Raises:
            OSError: the query's start, its collect or a counter's read
                failed; a failed start is raised again on every call.
        """
        self._collected()
        util = self._this_process('util')
        mib = 1024 * 1024
        return GpuReading(
            util_total_percent=self._summed(util),
            util_3d_percent=self._summed(util, engine='engtype_3D'),
            shared_mem_mb=self._summed(self._this_process('shared_mem'), unit=mib),
            dedicated_mem_mb=self._summed(self._this_process('dedicated_mem'), unit=mib),
        )


_gpu_pdh_singleton = _GpuPdhCountersOnce() if _IS_WINDOWS else None


def query_gpu_metrics() -> GpuReading | None:
    """Vendor-agnostic GPU utilization + memory for this process.

    Works for AMD / Intel / NVIDIA via the WDDM GPU Engine counters (the
    source Task Manager's GPU column reads).

    Returns:
        This process's GpuReading; None off Windows, where the counters
        do not exist.

    Raises:
        OSError: as ``_GpuPdhCountersOnce.query``.
    """
    if _gpu_pdh_singleton is None:
        return None
    return _gpu_pdh_singleton.query()


def gui_resources() -> tuple[int, int] | None:
    """This process's GDI and USER object counts; None off Windows, where
    they do not exist.

    GDI is what causes Windows-wide slowdown after 24h+ runs. Every
    `Texture.create()` and unclosed matplotlib figure adds a GDI handle.
    Process limit is 10,000; Windows desktop degrades around 5,000.
    """
    if not _IS_WINDOWS:
        return None
    GR_GDIOBJECTS = 0
    GR_USEROBJECTS = 1
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    # Declare argtypes / restype so the 64-bit Windows HANDLE
    # is passed without truncation. Default ctypes types are
    # c_int (4 bytes signed), which silently truncates a 64-bit
    # pseudo-handle and causes GetGuiResources to return 0 -- a
    # broken metric that masquerades as "no GDI usage" in
    # metrics.log on healthy systems. (Repeated assignments are
    # idempotent, so doing this each call is safe.)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    user32.GetGuiResources.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.GetGuiResources.restype = ctypes.c_uint
    handle = kernel32.GetCurrentProcess()
    return user32.GetGuiResources(handle, GR_GDIOBJECTS), user32.GetGuiResources(
        handle, GR_USEROBJECTS
    )


# ---------------------------------------------------------------------------
# Rate-tracker for cumulative counters -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Several metrics we want (page faults/sec, IO write/sec, MsMpEng read/sec)
# come from cumulative counters. Cache the last value+timestamp and compute
# a delta-rate on each call. Per-key state.
# ---------------------------------------------------------------------------

_rate_state = {}  # {key: (last_value, last_ts)}

# The shortest gap over which an interval reading is taken: under it a
# rate would divide by next to nothing.
_MIN_INTERVAL_S = 0.5


def delta_rate(key: str, current_value: float, now: float | None = None) -> float | None:
    """The per-second rate of a cumulative counter since its last reading.

    Returns:
        The rate; None with no earlier reading of ``key``, or one under
        half a second ago.
    """
    if now is None:
        now = _time.monotonic()
    prev = _rate_state.get(key)
    _rate_state[key] = (current_value, now)
    if prev is None:
        return None
    last_value, last_ts = prev
    dt = now - last_ts
    if dt < _MIN_INTERVAL_S:
        return None
    return (current_value - last_value) / dt


# The one psutil handle for THIS process, so every reading of it is taken
# through the same object.
_SELF_PROC = None


def self_process() -> psutil.Process:
    """This process's one psutil handle."""
    global _SELF_PROC
    if _SELF_PROC is None:
        _SELF_PROC = psutil.Process(os.getpid())
    return _SELF_PROC


def process_cpu_percent() -> float | None:
    """This process's CPU since the previous call; 100.0 is one full core.

    The same figure psutil's ``Process.cpu_percent`` gives, from a baseline
    this module holds, so the first call answers that it has none rather
    than psutil's 0.0.

    Returns:
        The percent; None on the first call, or one under half a second
        after the previous.
    """
    times = self_process().cpu_times()
    rate = delta_rate('process_cpu_s', times.user + times.system)
    return None if rate is None else rate * 100.0


# The system's CPU times at the previous call. psutil's own cpu_percent()
# keeps its baseline per calling thread, and the metrics tick runs on a
# fresh timer thread each time, so a thread with no baseline read 0.0.
_system_cpu_last = {}  # {'times': psutil cpu_times}

# What psutil counts as idle, and the guest time Linux also counts in user.
_IDLE_CPU_FIELDS = ('idle', 'iowait')
_GUEST_CPU_FIELDS = ('guest', 'guest_nice')


def system_cpu_percent() -> float | None:
    """The whole host's CPU busy since the previous call, in percent.

    Computed as psutil's ``cpu_percent`` computes it, from one baseline
    whatever thread calls.

    Returns:
        The percent; None on the first call, or where no CPU time has
        passed since the previous.
    """
    now = psutil.cpu_times()
    last = _system_cpu_last.get('times')
    _system_cpu_last['times'] = now
    if last is None:
        return None
    return cpu_busy_percent(last, now)


def cpu_busy_percent(before: NamedTuple, after: NamedTuple) -> float | None:
    """The whole host's CPU busy between two ``psutil.cpu_times()`` readings, in percent.

    Computed as psutil's ``cpu_percent`` computes it: each field's delta
    floored at zero, the guest fields left out of the total and iowait
    counted idle where the host has it.

    Args:
        before: The earlier reading.
        after: The later reading, of the same host.

    Returns:
        The percent; None where no CPU time passed between the two.
    """
    deltas = {
        field: max(0.0, getattr(after, field) - getattr(before, field)) for field in after._fields
    }
    total = sum(v for field, v in deltas.items() if field not in _GUEST_CPU_FIELDS)
    if total <= 0.0:
        return None
    idle = sum(deltas[field] for field in _IDLE_CPU_FIELDS if field in deltas)
    return (total - idle) / total * 100.0


# A process's CPU seconds so far, keyed by (pid, create_time) so a pid the OS
# gave to a new process is never paired with the one that had it before.
ProcessCpuSeconds = dict[tuple[int, float], tuple[str, float]]


def process_cpu_seconds() -> tuple[ProcessCpuSeconds, int]:
    """Every process's CPU seconds so far, and how many the OS would not let this one read.

    A process that ends while the table is read is left out. One whose
    CPU time or start the OS keeps from this process is counted, not read:
    on macOS a non-root user cannot read about a third, ``WindowServer``
    and ``kernel_task`` among them.

    Returns:
        ``({(pid, create_time): (name, user + system seconds)}, denied)``.
    """
    seconds: ProcessCpuSeconds = {}
    denied = 0
    for proc in psutil.process_iter(['name', 'create_time', 'cpu_times']):
        times, created = proc.info['cpu_times'], proc.info['create_time']
        if times is None or created is None:
            denied += 1
            continue
        seconds[(proc.pid, created)] = (
            proc.info['name'] or f'pid {proc.pid}',
            times.user + times.system,
        )
    return seconds, denied


def program_cpu_shares(
    before: ProcessCpuSeconds,
    after: ProcessCpuSeconds,
    window_s: float,
    cores: int,
    left_out: frozenset[int],
) -> list[tuple[str, int, float]]:
    """Each process's share of the whole machine's CPU between two ``process_cpu_seconds`` tables.

    100% is every logical core busy for the whole window, as Task Manager
    counts it. Only a process in both tables is shared: one that ended or
    started inside the window has no share.

    Args:
        before: The table at the window's start.
        after: The table at its end.
        window_s: The seconds between the two.
        cores: The host's logical cores.
        left_out: The pids that are not other programs (this process; on
            Windows pid 0, the System Idle Process).

    Returns:
        ``(name, pid, percent)`` for each process, largest share first;
        empty for a window with no length.
    """
    if window_s <= 0.0:
        return []
    capacity_s = window_s * cores
    shares = [
        (name, pid, (cpu_s - before[(pid, created)][1]) / capacity_s * 100.0)
        for (pid, created), (name, cpu_s) in after.items()
        if pid not in left_out and (pid, created) in before
    ]
    return sorted(shares, key=lambda share: share[2], reverse=True)


# ---------------------------------------------------------------------------
# Per-thread CPU attribution
#
# psutil.Process.threads() returns cumulative (user + system) CPU seconds per
# OS thread id. Caching the previous snapshot and dividing the delta by the
# wall-clock interval gives each thread's average CPU fraction since the last
# call. Percentages are per-core-normalized like psutil.cpu_percent (100.0 =
# one full logical core), so the per-thread values sum to roughly the
# [PROCESS METRICS] process CPU figure -- a built-in cross-check. OS thread
# ids map back to Python thread names via threading.enumerate() native_id, so
# the output lines up with the names in [THREAD METRICS]. Module-level state
# mirrors _rate_state.
#
# Per-thread detail is available on Windows and Linux. macOS psutil reports
# only a single aggregate thread (id=1), so on Mac this collapses to one
# entry -- acceptable, since the deployment + bench target is Windows.
# ---------------------------------------------------------------------------

_thread_cpu_state = {'ts': None, 'cpu': {}}  # the previous call's wall time and {tid: cpu_seconds}


def thread_cpu_percentages(
    proc: psutil.Process | None = None, now: float | None = None
) -> dict[str, float] | None:
    """Per-thread CPU percent averaged over the interval since the last call.

    Same-named threads (e.g. pool workers) are summed under one label.
    100.0 means one full logical core; the sum across threads can exceed
    100 on a multi-core host.

    Returns:
        {thread_label: percent} for each thread that had a reading at the
        previous call: a thread born since, or whose id a new thread took,
        has none. None on the first call, or one under half a second after
        the previous.

    Raises:
        psutil.Error: the process's threads could not be read.
    """
    if proc is None:
        proc = self_process()
    if now is None:
        now = _time.monotonic()
    cpu_now = {th.id: th.user_time + th.system_time for th in proc.threads()}
    last_ts, cpu_last = _thread_cpu_state['ts'], _thread_cpu_state['cpu']
    # Replaced whole, so a thread that exited leaves no state behind.
    _thread_cpu_state['ts'], _thread_cpu_state['cpu'] = now, cpu_now
    if last_ts is None or now - last_ts < _MIN_INTERVAL_S:
        return None
    dt = now - last_ts

    # native_id -> python thread name; native threads with no Python wrapper
    # fall back to tid_<id>.
    name_by_tid = {t.native_id: t.name for t in threading.enumerate() if t.native_id is not None}

    out = {}
    for tid, cpu in cpu_now.items():
        last_cpu = cpu_last.get(tid)
        if last_cpu is None or cpu < last_cpu:
            continue
        label = name_by_tid.get(tid, f'tid_{tid}')
        out[label] = out.get(label, 0.0) + (cpu - last_cpu) / dt * 100.0
    return out


# ---------------------------------------------------------------------------
# Defender (MsMpEng.exe) metrics -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Direct signal on the "Defender memory-maps every TIFF write" hypothesis.
# If MsMpEng's IO read rate climbs proportional to our save rate, that's
# the smoking gun for Defender being on the slowdown's critical path.
#
# Cache the process across calls (psutil.process_iter is expensive).
# Re-resolve only if the cached one has died.
# ---------------------------------------------------------------------------

_defender_process_cache = {'process': None}


class DefenderReading(NamedTuple):
    """MsMpEng.exe's memory and the bytes it has read. Its I/O fields are
    the refusal where Windows refuses this process a read of its I/O, and
    the rate is None with no earlier reading."""

    private_mb: float
    rss_mb: float
    io_read_total_mb: float | psutil.AccessDenied
    io_read_mbps: float | psutil.AccessDenied | None


def _defender_process() -> psutil.Process | None:
    """The running MsMpEng.exe, or None when none is running."""
    proc = _defender_process_cache['process']
    if proc is not None and proc.is_running():
        return proc
    _defender_process_cache['process'] = None
    for p in psutil.process_iter(['name']):
        if (p.info['name'] or '').lower() == 'msmpeng.exe':
            _defender_process_cache['process'] = p
            return p
    return None


def query_defender_metrics() -> DefenderReading | None:
    """MsMpEng.exe's memory and I/O.

    Returns:
        Its DefenderReading; None off Windows, or with MsMpEng not running.

    Raises:
        psutil.Error: its memory could not be read; a process that has
            exited is resolved again on the next call.
    """
    if not _IS_WINDOWS:
        return None
    proc = _defender_process()
    if proc is None:
        return None
    try:
        mem = proc.memory_info()
    except psutil.NoSuchProcess:
        _defender_process_cache['process'] = None
        raise
    mib = 1024 * 1024
    try:
        io = proc.io_counters()
    except psutil.AccessDenied as refused:
        io_read_total_mb, io_read_mbps = refused, refused
    else:
        io_read_total_mb = io.read_bytes / mib
        rate = delta_rate('defender_io_read', io.read_bytes)
        io_read_mbps = None if rate is None else rate / mib
    return DefenderReading(
        private_mb=getattr(mem, 'private', mem.rss) / mib,
        rss_mb=mem.rss / mib,
        io_read_total_mb=io_read_total_mb,
        io_read_mbps=io_read_mbps,
    )


class HostBoot(NamedTuple):
    """When the host booted (seconds since the epoch) and how long it has
    been up (hours)."""

    boot_epoch_s: float
    uptime_hr: float


def host_boot() -> HostBoot:
    """The host's boot instant and uptime, read once.

    Uptime is the wall clock now less the boot instant: the host states its
    boot on the wall clock alone, so the two are read on one clock.
    """
    boot_epoch_s = psutil.boot_time()
    return HostBoot(boot_epoch_s=boot_epoch_s, uptime_hr=(_time.time() - boot_epoch_s) / 3600.0)


def read_video_config(step) -> dict:
    """Return a step's Video Config as a dict, or {} when absent or malformed.

    The one safe reader of a step's Video Config, shared by the disk-write
    estimate and the time estimate. A protocol step is a pandas Series (or a
    dict, or None at some defaults), and its 'Video Config' cell can be an
    unpopulated NaN -- a truthy float, so a plain `or {}` does NOT guard it. A
    caller that read it unguarded raised (None.get / nan.get), and the disk
    check that wraps the call in a broad except then silently skipped, leaving a
    near-full disk undetected. Returning {} for every non-dict keeps the reader
    total so the guard can never be disabled by malformed config.

    Args:
        step: A protocol step (pandas Series / dict / None).

    Returns:
        The step's Video Config dict, or {} if the step or the cell is not a dict.
    """
    get = getattr(step, 'get', None)
    if get is None:
        return {}
    video_config = get('Video Config')
    return video_config if isinstance(video_config, dict) else {}


# Free-space floor (2 GB) below which capture work refuses to start and
# a running manual recording stops gracefully. One canonical value: the
# protocol scan loop and the recording controllers share it so the
# capture paths cannot drift on what "disk almost full" means.
MIN_REQUIRED_DISK_MB = 2048

# Minimum free disk for a SINGLE protocol write, in MB. Distinct from the
# whole-run MIN_REQUIRED_DISK_MB above: each individual capture must have
# at least this much headroom beyond its own estimated size before it is
# allowed to write, so one write cannot fill the last sliver of a disk.
MIN_PER_WRITE_DISK_MB = 500

# How often a recording's write lane re-probes free disk while draining.
# Frequent enough that a filling disk stops the recording within a few
# frames' worth of writes; rare enough that the probe never shapes drain
# speed. Shared by both recording controllers.
DISK_FLOOR_CHECK_INTERVAL_S = 2.0

# Step count above which a protocol is large enough to tell the user about
# before they run it. Advisory only -- it never refuses anything, and a
# legitimate large protocol proceeds untouched. Distinct from the disk
# floors above: those decide whether a run may start, this decides whether
# the user is told how big the thing they just built is.
PROTOCOL_SIZE_ADVISORY_STEPS = 10_000


def format_disk_size_mb(mb: float) -> str:
    """Render a megabyte figure for a human, in binary units.

    Binary throughout, matching what MB already means here -- check_disk_space_ok
    reads free space as ``disk.free / (1024**2)``, so a decimal rendering would
    print a number the disk checks disagree with. GB up to 1024 GB, TB above,
    one decimal place.
    """
    if mb < 1024:
        return f'{mb:.1f} MB'
    gb = mb / 1024
    if gb < 1024:
        return f'{gb:.1f} GB'
    return f'{gb / 1024:.1f} TB'


def check_disk_space_ok(path: str | os.PathLike, required_mb: float) -> tuple[bool, float]:
    """Probe free disk space and compare against a threshold.

    Single canonical disk probe shared by protocol_image_writer, the
    protocol scan loop, and the record pre-flight in main_display. Each
    caller keeps its own threshold + abort/notification policy; only
    the probe itself is shared so the three sites cannot drift on
    backend choice (shutil vs psutil disagree on Windows partition
    mounts) or unit conversion.

    Args:
        path: Filesystem path to probe (str or pathlib.Path). It need not
            exist yet: a run's output folder is often made only when its
            first file is written, and it lands on the volume of its
            nearest existing ancestor, which is what is measured.
        required_mb: Minimum free space the caller needs, in MB.

    Returns:
        (ok, free_mb) -- ok is True iff free_mb >= required_mb.

    Raises:
        OSError / PermissionError: propagated from psutil.disk_usage so
            callers can decide whether to swallow (best-effort probes)
            or abort (load-bearing probes).
    """
    probe = pathlib.Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    disk = psutil.disk_usage(str(probe))
    free_mb = disk.free / (1024**2)
    return (free_mb >= required_mb, free_mb)


def get_extra_disks_info(exclude_path: str = '/') -> str | None:
    """The free space of every disk but the one holding ``exclude_path``.

    A partition whose usage cannot be read is left out.

    Returns:
        A string like "D: 250.5 GB free (15.2% used) | E: 100.3 GB free
        (8.5% used)"; None when no other disk is mounted.

    Raises:
        OSError: the partitions could not be listed.
    """
    disk_partitions = psutil.disk_partitions(all=False)

    # Find which partition contains the exclude_path
    excluded_device = None
    for partition in disk_partitions:
        if exclude_path.startswith(partition.mountpoint):
            excluded_device = partition.device
            break

    disk_info_list = []
    for partition in disk_partitions:
        # Skip the excluded device/path
        if excluded_device and partition.device == excluded_device:
            continue

        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue
        disk_info_list.append(
            f'{partition.device}: {usage.free / (1024**3):.1f} GB free ({usage.percent:.1f}% used)'
        )

    return ' | '.join(disk_info_list) if disk_info_list else None
