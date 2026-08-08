# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import ctypes
import dataclasses
import enum
import gc
import json
import numbers
import os
import pathlib
import platform
import re
import threading
import time as _time
from typing import ClassVar

import numpy as np
import psutil

from lvp_logger import logger


# ---------------------------------------------------------------------------
# Hardware defaults (fallbacks when motorconfig/scope not available)
# ---------------------------------------------------------------------------
# LS850 full travel range -- used as default stage limits when scope is not connected.
DEFAULT_STAGE_TRAVEL_UM = {'x': 120000.0, 'y': 80000.0}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@enum.unique
class ColorChannel(enum.Enum):
    Blue = 0
    Green = 1
    Red = 2
    BF = 3
    PC = 4
    DF = 5
    Lumi = 6


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


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


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
    channel: str | None = None  # 'BF' | 'Green' | 'Composite' | None
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


# Single-token post-output suffixes occupying StepNameComponents.post. A
# z-projection is two segments ('zproj_<method>'); the rest are single tokens.
_POST_SUFFIX_TOKENS = frozenset({'stitched', 'video', 'stack', 'hyperstack'})


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


def _token_kind(seg: str, layers: set, objectives: set) -> str | None:
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
    if seg in layers or seg == 'Composite':
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


def _machine_tokens_only(segs: list[str], layers: set, objectives: set) -> bool:
    """True when every segment classifies as a known step-name token.

    A machine-generated name -- even one carrying stale tokens from old
    channel-change bugs, or one whose row columns were later output-adjusted
    by a post processor -- consists only of vocabulary tokens after its
    anchor. User-typed text always contains at least one segment the
    vocabulary cannot classify.
    """
    i = 0
    while i < len(segs):
        kind = _token_kind(segs[i], layers, objectives)
        if kind is None:
            return False
        if kind == 'zproj':
            i += 1  # the method segment rides with its zproj token
        i += 1
    return True


def parse_legacy_step_name(name, known_layers=None, known_objectives=()) -> StepNameComponents:
    """Recover the components from a rendered step name, by SHAPE not position.

    LEGACY LOAD BOUNDARY ONLY: the run-time pipeline never decodes a name --
    fields travel as columns (Label et al.) -- so the only remaining caller
    is the one-shot migration of files that predate the Label column
    (recover_step_label). Each '_'-separated segment is classified by its
    shape (plus the known-layer / known-objective vocabularies), so a token
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
    if known_layers is None:
        known_layers = get_layers()
    layers = set(known_layers)
    objectives = set(known_objectives)

    stem = pathlib.Path(name).name
    segs = stem.split('_')
    if not segs or segs == ['']:
        return StepNameComponents()

    def token_kind(seg):
        return _token_kind(seg, layers, objectives)

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


def step_components(step, **overrides) -> StepNameComponents:
    """Map a protocol step / post-record row to its canonical name components.

    The single place that reads a step's columns into StepNameComponents, so
    every name-building site renders one identity from one source. Well, Color,
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
        channel=_blank_to_none(step['Color']),
        tile=_blank_to_none(step['Tile']),
        z_index=_blank_to_none(step['Z-Slice']),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def recover_step_label(step) -> tuple[str, bool]:
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
                'Color': step['Color'],
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
    if anchored and _machine_tokens_only(segs[1:], set(get_layers()), set()):
        return machine_base, True

    return name, False


def resolve_step_rename(raw_text: str, sanitize) -> str | None:
    """Resolve a step-name field value to the name to persist, or None.

    Auto-named custom steps blank the name field so the default name shows
    as a hint placeholder rather than editable text. A blank field
    therefore means "no rename intended": persisting the empty string
    would wipe the auto-assigned name, leaving added steps unnamed and
    colliding on the same default name. Returns None for a blank field so
    callers keep the existing name; a non-empty entry is a real rename and
    is returned sanitized.

    Args:
        raw_text: the raw text from the step-name input field.
        sanitize: callable that cleans a name (e.g. strips invalid chars).

    Returns:
        The sanitized name to persist, or None if the field is blank.
    """
    cleaned = sanitize(raw_text)
    return cleaned if cleaned else None


def convert_zstack_reference_position_setting_to_config(text_label: str) -> str:
    LABEL_MAP = {
        'Current Position at Top': 'top',
        'Current Position at Center': 'center',
        'Current Position at Bottom': 'bottom',
    }

    if text_label in LABEL_MAP:
        return LABEL_MAP[text_label]

    raise Exception(f'Unknown Z-stack position reference: {text_label}')


def is_valid_gain_db(value) -> bool:
    """True when a camera gain reading is usable (a number >= 0 dB).

    Negative is the drivers' failed-read / inactive sentinel; 0 dB is a
    legal gain. A non-number (e.g. a stringified value from a hand-built
    snapshot) is invalid rather than a comparison TypeError, so every
    consumer can branch on the predicate without its own type guard.
    One shared predicate so every consumer of the sentinel contract
    validates the same way.
    """
    return isinstance(value, numbers.Real) and value >= 0


def is_valid_exposure_ms(value) -> bool:
    """True when a camera exposure reading is usable (a positive number).

    Negative is the drivers' failed-read sentinel and 0 is the API's
    inactive-camera return -- neither is a physical exposure. A
    non-number is invalid rather than a comparison TypeError. One shared
    predicate so every consumer of the sentinel contract validates the
    same way.
    """
    return isinstance(value, numbers.Real) and value > 0


def is_valid_frame_size(value) -> bool:
    """True when a frame-size reading is usable (positive width and height).

    The drivers return None (or an empty dict for the max/min variants) as
    the failed-read / inactive sentinel, and a zero dimension is never a
    deliverable frame. One shared predicate so every consumer of the
    sentinel contract validates the same way.
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


def is_valid_pixel_format(value) -> bool:
    """True when a pixel-format reading is usable (a non-empty string).

    None is the drivers' shared failed-read / inactive sentinel -- distinct
    from every real format name so it cannot be mistaken for one.
    """
    return isinstance(value, str) and value != ''


def is_valid_binning_size(value) -> bool:
    """True when a binning reading is usable (a whole factor >= 1).

    Negative is the drivers' failed-read sentinel; binning is always at
    least 1x1 on real hardware.
    """
    return isinstance(value, numbers.Real) and value >= 1


# Distinct non-format inputs raw_bytes_per_pixel has already warned about;
# the caller cadence is per-stats-tick, so an unknown format warns once per
# distinct value instead of flooding the log every second.
_RAW_BPP_WARNED: set[str] = set()


def raw_bytes_per_pixel(pixel_format: str, is_color_native: bool = False) -> int:
    """Bytes per pixel of the RAW camera buffer (for data-rate readouts).

    Mono8 is one byte; every other Mono format (Mono10 / Mono12 / Mono16 and
    the packed variants such as Mono10g40IDS) is delivered in a uint16
    container, so two bytes. Color-native cameras (none in the shipping fleet)
    carry three channels.

    A non-string input (the pixel-format cache before any format was ever
    read) is warned about and treated as a 2-byte container: the camera value
    getters answer last-known-good, so a sentinel reaching this math means a
    consumer bypassed that containment -- loud, not silently classified.

    Args:
        pixel_format: SDK pixel-format name (e.g. 'Mono8', 'Mono12', 'Mono16').
        is_color_native: Whether the camera delivers 3-channel color frames.

    Returns:
        Bytes occupied by one pixel of the raw camera frame.
    """
    if not is_valid_pixel_format(pixel_format):
        marker = repr(pixel_format)
        if marker not in _RAW_BPP_WARNED:
            _RAW_BPP_WARNED.add(marker)
            logger.warning(
                f'raw_bytes_per_pixel: no pixel format known ({marker}); '
                f'assuming a 2-byte container for the data-rate readout'
            )
        bytes_per_channel = 2
    else:
        bytes_per_channel = 1 if pixel_format == 'Mono8' else 2
    channels = 3 if is_color_native else 1
    return bytes_per_channel * channels


def get_layers() -> list[str]:
    return ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi']


def get_transmitted_layers() -> list[str]:
    return ['BF', 'PC', 'DF']


def get_fluorescence_layers() -> list[str]:
    return ['Blue', 'Green', 'Red']


def get_bright_background_layers() -> list[str]:
    """Layers whose field of view is bright, so overlays drawn on them
    (scale bar, annotation text) must be dark to stay visible. Brightfield
    and phase contrast have a bright background; darkfield does not -- it
    shows bright subjects on a dark field -- so it is excluded here even
    though it is a transmitted-light mode.
    """
    return ['BF', 'PC']


def get_luminescence_layers() -> list[str]:
    return ['Lumi']


def get_image_layers() -> list[str]:
    """Fluorescence + luminescence layers (false-color, displayed as colored images)."""
    return get_fluorescence_layers() + get_luminescence_layers()


def get_layers_with_led() -> list[str]:
    return get_transmitted_layers() + get_fluorescence_layers()


def get_opened_layer(lumaview_imagesettings) -> str | None:
    for layer in get_layers():
        try:
            layer_accordion_obj = lumaview_imagesettings.accordion_item_lookup(layer=layer)
            if not layer_accordion_obj.collapse:
                return layer
        except Exception as e:
            logger.debug(
                '[common_utils] get_opened_layer: accordion lookup for '
                'layer=%s raised; skipping: %s: %s',
                layer,
                type(e).__name__,
                e,
            )
            continue

    return None


def get_opened_layer_obj(lumaview_imagesettings):
    return lumaview_imagesettings.layer_lookup(layer=get_opened_layer(lumaview_imagesettings))


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


def to_int(val) -> int | None:
    if 'numpy' in str(type(val)):
        return int(val.astype(float))
    elif val in ('', None):
        return -1
    else:
        return int(float(val))


def get_pixel_size(
    focal_length: float,
    binning_size: int,
) -> float | None:
    """Effective um/pixel for the given objective focal length and binning.

    Reads the tube focal length and sensor pixel size from the active scope's
    capabilities (the single source of truth for image scale). Returns None
    when there is no active scope, or when the scope cannot report its optics
    (unknown camera, no declared optics) -- callers then degrade honestly (no
    scale bar, no field of view, no PhysicalSizeX) rather than using an
    invented scale. There is deliberately no hardcoded fallback: a guessed
    pixel size is written into every image and cannot be told from a measured
    one.
    """
    import modules.app_context as _app_ctx

    ctx = _app_ctx.ctx
    if ctx is None or ctx.scope is None:
        return None
    tube_focal_length = ctx.scope.capabilities.lens_focal_length_mm
    pixel_width = ctx.scope.capabilities.pixel_size_um
    if tube_focal_length is None or pixel_width is None:
        return None
    magnification = tube_focal_length / focal_length
    um_per_pixel = pixel_width / magnification

    return um_per_pixel * binning_size


def get_field_of_view(
    focal_length: float,
    frame_size: dict,
    binning_size: int,
) -> dict | None:
    um_per_pixel = get_pixel_size(
        focal_length=focal_length,
        binning_size=binning_size,
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


def format_field_of_view(fov_size: dict | None) -> tuple[str, str]:
    """Render (width, height) FOV label text for the settings readouts.

    Returns the rounded micron strings, or ('unknown', 'unknown') when the
    scope cannot report a field of view (no known pixel size). The three FOV
    readouts share this so an unmeasurable scope reads the same everywhere.
    """
    if fov_size is None:
        return (FOV_UNKNOWN_TEXT, FOV_UNKNOWN_TEXT)
    return (str(round(fov_size['width'], 0)), str(round(fov_size['height'], 0)))


def max_decimal_precision(parameter: str) -> int:
    DEFAULT_PRECISION = 5
    PRECISION_MAP = {'x': 4, 'y': 4, 'z': 5}

    return PRECISION_MAP.get(parameter, DEFAULT_PRECISION)


_IS_WINDOWS = platform.system() == 'Windows'


# ---------------------------------------------------------------------------
# Windows perf-counter query (PDH) -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Added on branch `perf-instrumentation-4.0.0-beta` to capture standby cache
# growth, nonpaged pool, and system file cache as part of the buffer-churn
# investigation. These counters are NOT in psutil. The PDH layer ("\Memory\..."
# perf-counter paths) is the same data PowerShell `Get-Counter` returns.
#
# Lifetime: this entire `_PdhCountersOnce` helper plus the PDH fields injected
# into `system_metrics()` are temporary. Remove once the buffer-reuse fixes
# land and the standby-cache trend is verified flat.
#
# Performance: PdhCollectQueryData is one syscall per counter, ~1 ms total.
# Safe to call once per minute.
# ---------------------------------------------------------------------------


class _PdhCountersOnce:
    """Lazy-initialized PDH query for a fixed set of counters.

    Opens the PDH query on first call, caches the counter handles, and
    re-collects on each subsequent call. On any failure, marks itself
    disabled so subsequent calls return {} without retry overhead.
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

    # PDH return codes / format flags (winperf.h)
    _PDH_FMT_DOUBLE = 0x00000200
    _ERROR_SUCCESS = 0

    def __init__(self):
        self._initialized = False
        self._disabled = False
        self._query = None
        self._handles = {}  # name -> counter handle

    def _init(self):
        try:
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
                raise OSError(f'PdhOpenQueryW failed: 0x{ret:08x}')
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
        except Exception:
            self._disabled = True

    def query(self):
        """Return {field: bytes_value} for all counters that are working.

        Returns {} if PDH unavailable or initialization failed.
        """
        if self._disabled:
            return {}
        if not self._initialized:
            self._init()
            if self._disabled:
                return {}

        try:
            ret = self._PdhCollectQueryData(self._query)
            if ret != self._ERROR_SUCCESS:
                return {}

            out = {}
            for name, handle in self._handles.items():
                value = self._PDH_FMT_COUNTERVALUE()
                ret = self._PdhGetFormattedCounterValue(
                    handle, self._PDH_FMT_DOUBLE, None, ctypes.byref(value)
                )
                if ret == self._ERROR_SUCCESS:
                    out[name] = float(value.doubleValue)
            return out
        except Exception:
            return {}


_pdh_counters_singleton = _PdhCountersOnce() if _IS_WINDOWS else None


def query_windows_perf_counters():
    """One-shot snapshot of selected Windows memory perf counters.

    Returns dict mapping field name to bytes value. Returns {} on non-Windows
    or if PDH is unavailable. TEMPORARY -- see `_PdhCountersOnce` docstring.
    """
    if _pdh_counters_singleton is None:
        return {}
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


class _GpuPdhCountersOnce:
    _UTIL_PATH = r'\GPU Engine(*)\Utilization Percentage'
    _SHARED_MEM_PATH = r'\GPU Process Memory(*)\Shared Usage'
    _DEDICATED_MEM_PATH = r'\GPU Process Memory(*)\Dedicated Usage'

    _PDH_FMT_DOUBLE = 0x00000200
    _ERROR_SUCCESS = 0

    def __init__(self):
        self._initialized = False
        self._disabled = False
        self._query = None
        self._counters = {}  # name -> counter handle
        self._pid_tag = f'pid_{os.getpid()}_'

    def _init(self):
        try:
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
        except Exception:
            self._disabled = True

    def _read_array(self, handle):
        """Return [(instance_name, value), ...] for a wildcard counter."""
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        # First pass with a null buffer reports the required size.
        ret = self._PdhGetFormattedCounterArrayW(
            handle,
            self._PDH_FMT_DOUBLE,
            ctypes.byref(size),
            ctypes.byref(count),
            None,
        )
        if (ret & 0xFFFFFFFF) != _PDH_MORE_DATA or size.value == 0:
            return []
        buf = (ctypes.c_byte * size.value)()
        ret = self._PdhGetFormattedCounterArrayW(
            handle,
            self._PDH_FMT_DOUBLE,
            ctypes.byref(size),
            ctypes.byref(count),
            ctypes.cast(buf, ctypes.POINTER(self._ITEM)),
        )
        if ret != self._ERROR_SUCCESS:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(self._ITEM * count.value)).contents
        out = []
        for it in items:
            # CStatus 0 (VALID_DATA) / 1 (NEW_DATA) both carry a usable value.
            if it.FmtValue.CStatus in (0, 1):
                out.append((it.szName or '', float(it.FmtValue.doubleValue)))
        return out

    def query(self):
        """Return GPU metrics dict for this process, or {} if unavailable."""
        if self._disabled:
            return {}
        if not self._initialized:
            self._init()
            if self._disabled:
                return {}
        try:
            if self._PdhCollectQueryData(self._query) != self._ERROR_SUCCESS:
                return {}
            out = {}
            if 'util' in self._counters:
                util_total = 0.0
                util_3d = 0.0
                for name, val in self._read_array(self._counters['util']):
                    if self._pid_tag in name:
                        util_total += val
                        if name.endswith('engtype_3D'):
                            util_3d += val
                out['gpu_util_total_percent'] = util_total
                out['gpu_util_3d_percent'] = util_3d
            for key, field in (
                ('shared_mem', 'gpu_shared_mem_mb'),
                ('dedicated_mem', 'gpu_dedicated_mem_mb'),
            ):
                if key in self._counters:
                    total = sum(
                        v for n, v in self._read_array(self._counters[key]) if self._pid_tag in n
                    )
                    out[field] = total / (1024 * 1024)
            return out
        except Exception:
            return {}


_gpu_pdh_singleton = _GpuPdhCountersOnce() if _IS_WINDOWS else None


def query_gpu_metrics():
    """Vendor-agnostic GPU utilization + memory for this process.

    Returns {} on non-Windows or if GPU PDH counters are unavailable. Keys when
    present: gpu_util_total_percent, gpu_util_3d_percent, gpu_shared_mem_mb,
    gpu_dedicated_mem_mb. Works for AMD / Intel / NVIDIA via the WDDM GPU Engine
    counters (the source Task Manager's GPU column reads).
    """
    if _gpu_pdh_singleton is None:
        return {}
    return _gpu_pdh_singleton.query()


# ---------------------------------------------------------------------------
# Rate-tracker for cumulative counters -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Several metrics we want (page faults/sec, IO write/sec, MsMpEng read/sec)
# come from cumulative counters. Cache the last value+timestamp and compute
# a delta-rate on each call. Per-key state.
# ---------------------------------------------------------------------------

_rate_state = {}  # {key: (last_value, last_ts)}


def _delta_rate(key, current_value, now=None):
    """Return per-second rate for a cumulative counter. Returns 0.0 on first
    call or if time delta is too small to be meaningful."""
    if now is None:
        now = _time.monotonic()
    prev = _rate_state.get(key)
    _rate_state[key] = (current_value, now)
    if prev is None:
        return 0.0
    last_value, last_ts = prev
    dt = now - last_ts
    if dt < 0.5:  # avoid divide-by-near-zero on rapid back-to-back calls
        return 0.0
    return (current_value - last_value) / dt


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

_thread_cpu_state = {}  # {tid: (cpu_seconds, wall_ts)}


def thread_cpu_percentages(proc=None, now=None):
    """Per-thread CPU percent averaged over the interval since the last call.

    Returns {thread_label: percent}. Empty on the first call (no baseline yet)
    or on error. Same-named threads (e.g. pool workers) are summed under one
    label. 100.0 means one full logical core; the sum across threads can
    exceed 100 on a multi-core host.
    """
    if proc is None:
        proc = psutil.Process(os.getpid())
    if now is None:
        now = _time.monotonic()
    try:
        threads = proc.threads()
    except Exception:
        return {}

    # native_id -> python thread name; native threads with no Python wrapper
    # fall back to tid_<id>.
    name_by_tid = {}
    try:
        for t in threading.enumerate():
            nid = getattr(t, 'native_id', None)
            if nid is not None:
                name_by_tid[nid] = t.name
    except Exception:
        pass

    out = {}
    seen = set()
    for th in threads:
        tid = th.id
        seen.add(tid)
        cpu = th.user_time + th.system_time
        prev = _thread_cpu_state.get(tid)
        _thread_cpu_state[tid] = (cpu, now)
        if prev is None:
            continue
        last_cpu, last_ts = prev
        dt = now - last_ts
        if dt < 0.5:  # avoid divide-by-near-zero on rapid back-to-back calls
            continue
        pct = (cpu - last_cpu) / dt * 100.0
        if pct < 0.0:  # counter reset or tid reuse -- treat as no sample
            pct = 0.0
        label = name_by_tid.get(tid, f'tid_{tid}')
        out[label] = out.get(label, 0.0) + pct

    # Drop state for threads that exited so the cache can't grow unbounded.
    for dead in [tid for tid in _thread_cpu_state if tid not in seen]:
        del _thread_cpu_state[dead]

    return out


# ---------------------------------------------------------------------------
# Defender (MsMpEng.exe) metrics -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Direct signal on the "Defender memory-maps every TIFF write" hypothesis.
# If MsMpEng's IO read rate climbs proportional to our save rate, that's
# the smoking gun for Defender being on the slowdown's critical path.
#
# Cache the PID across calls (psutil.process_iter is expensive). Re-resolve
# only if the cached PID has died.
# ---------------------------------------------------------------------------

_defender_pid_cache = {'pid': None, 'process': None}


def query_defender_metrics():
    """Return MsMpEng.exe metrics dict, or {} if not running / not Windows.

    Cumulative IO read MB is included; the caller is expected to also call
    _delta_rate for the per-sec rate.
    """
    if not _IS_WINDOWS:
        return {}

    proc = _defender_pid_cache.get('process')
    try:
        if proc is None or not proc.is_running():
            for p in psutil.process_iter(['pid', 'name']):
                try:
                    if p.info['name'] and p.info['name'].lower() == 'msmpeng.exe':
                        proc = psutil.Process(p.info['pid'])
                        _defender_pid_cache['pid'] = proc.pid
                        _defender_pid_cache['process'] = proc
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            else:
                _defender_pid_cache['pid'] = None
                _defender_pid_cache['process'] = None
                return {}
    except Exception:
        return {}

    try:
        mem = proc.memory_info()
        out = {
            'defender_private_mb': getattr(mem, 'private', mem.rss) / (1024 * 1024),
            'defender_rss_mb': mem.rss / (1024 * 1024),
        }
        try:
            io = proc.io_counters()
            out['defender_io_read_mb_total'] = io.read_bytes / (1024 * 1024)
            out['defender_io_read_mbps'] = _delta_rate('defender_io_read', io.read_bytes) / (
                1024 * 1024
            )
        except (psutil.AccessDenied, AttributeError):
            pass
        return out
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        _defender_pid_cache['pid'] = None
        _defender_pid_cache['process'] = None
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# tracemalloc top-N allocators -- TEMPORARY INSTRUMENTATION (2026-04-30)
#
# Off by default. Enable via tracemalloc_enabled: true in data/settings.json.
# When on, tracemalloc is started at first call and a snapshot is taken;
# top-N allocators are returned. Overhead: ~10-30% process memory. Use only
# when needed.
# ---------------------------------------------------------------------------

_tracemalloc_started = False


def _read_tracemalloc_gate():
    # Reuse lvp_logger.lvp_appdata so the production-installed path
    # (~/Documents/LumaViewPro <version>/data/) resolves the same way
    # the logger's debug-mode gate does. Fall back to the source root
    # when lvp_logger isn't importable (e.g. unit tests that exercise
    # this module in isolation).
    from modules.settings_init import load_tracemalloc_setting

    try:
        import lvp_logger

        base_dir = lvp_logger.lvp_appdata
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_tracemalloc_setting(base_dir)


# Read once at import from the tracemalloc_enabled setting. This is a
# diagnostic gate, not the runtime-toggleable debug_mode shape: there is no
# setter, so flipping the setting takes effect only on the next restart.
_tracemalloc_enabled = _read_tracemalloc_gate()


def query_tracemalloc_top_n(n=5):
    """Return list of top-N allocators by current size, or [] if disabled.

    Enable via tracemalloc_enabled: true in data/settings.json.
    """
    if not _tracemalloc_enabled:
        return []
    global _tracemalloc_started
    try:
        import tracemalloc

        if not _tracemalloc_started:
            tracemalloc.start(20)  # 20 frames of context
            _tracemalloc_started = True
            return []  # first call has no baseline
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics('lineno')
        out = []
        for s in stats[:n]:
            frame = s.traceback[0]
            out.append(
                {
                    'file': frame.filename,
                    'line': frame.lineno,
                    'size_kb': s.size / 1024,
                    'count': s.count,
                }
            )
        return out
    except Exception:
        return []


# A single persistent Process handle for THIS process. psutil's
# Process.cpu_percent() reports CPU since the PRIOR call on the SAME object, so a
# fresh Process per snapshot always reads 0.0 -- no reference point -- which made
# the [PROCESS METRICS] process-CPU figure log 0.0% on every snapshot. Caching
# the handle (and priming it once on creation) makes each snapshot's process CPU
# the average over the inter-snapshot interval, the way the module-level
# psutil.cpu_percent() already reports the system figure.
_SELF_PROC = None


def _self_process():
    global _SELF_PROC
    if _SELF_PROC is None:
        _SELF_PROC = psutil.Process(os.getpid())
        _SELF_PROC.cpu_percent()  # prime the delta reference (this first read is 0.0)
    return _SELF_PROC


def system_metrics(path='/'):
    """Return a one-shot snapshot of process and host resource state.

    Used by `log_system_metrics()` (called hourly from `lumaviewpro.py`).
    Failures on individual metrics return -1 / None / 0.0 so callers
    can log "this metric isn't available on this platform" without the
    whole snapshot blowing up.

    Long-run leak detection: GDI/handle/thread/GC counts should plateau
    in steady state. A steady upward trend across hourly snapshots
    indicates a leak. See `docs/LOG_ANALYSIS_GUIDE.md` "Resource Health"
    section for healthy/unhealthy patterns.
    """
    proc = _self_process()
    disk = psutil.disk_usage(path)
    vmem = psutil.virtual_memory()

    metrics = {
        # CPU
        'cpu_percent_total': psutil.cpu_percent(),
        'cpu_percent_python': proc.cpu_percent(),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        # RAM
        'ram_available_gb': vmem.available / 1e9,
        'ram_percent_total': vmem.percent,
        'ram_used_python_percent': proc.memory_percent(),
        'ram_used_python_mb': proc.memory_info().rss / 1e6,
        'ram_used_total_mb': vmem.used / 1e6,
        # Disk
        'disk_free_gb': disk.free / 1024**3,
        'disk_used_percent': disk.percent,
    }

    # --- Private memory bytes (Windows-specific; falls back to RSS) ---
    # Working Set / RSS underreports because Windows can trim it while the
    # process still holds committed virtual memory. Private bytes is what
    # Task Manager calls "Memory (private working set)" and is the actual
    # leak indicator.
    try:
        mem = proc.memory_info()
        private = getattr(mem, 'private', mem.rss)
        metrics['ram_private_mb'] = private / 1e6
    except Exception:
        metrics['ram_private_mb'] = -1

    # --- System swap (catches page-file pressure even when "RAM looks low") ---
    try:
        swap = psutil.swap_memory()
        metrics['swap_percent'] = swap.percent
        metrics['swap_used_gb'] = swap.used / 1e9
    except Exception:
        metrics['swap_percent'] = -1
        metrics['swap_used_gb'] = -1

    # --- OS handles (Windows) / file descriptors (POSIX) ---
    # Windows caps process handles around 16M but typical apps run
    # 500-2000. A steady climb of a few handles per minute is a leak
    # of file/socket/thread handles. POSIX fds equivalent.
    try:
        if _IS_WINDOWS:
            metrics['os_handles'] = proc.num_handles()
        else:
            metrics['os_handles'] = proc.num_fds()
    except Exception:
        metrics['os_handles'] = -1

    # --- Open files count ---
    # Most actionable diagnostic when handles climb: tells you exactly
    # which files are leaked. We log only the count here; if it crosses
    # a threshold, the operator can dump the list manually via
    # `psutil.Process().open_files()`.
    try:
        metrics['open_files_count'] = len(proc.open_files())
    except Exception:
        metrics['open_files_count'] = -1

    # --- Process I/O bytes (cumulative + rates, per-process) ---
    # Distinguishes "we wrote 50 GB this hour" from "Windows Defender did".
    # Both bytes counters reset only when the process restarts.
    # Rates added 2026-04-30 (TEMPORARY) -- at 60 s sampling interval, the
    # rate gives MB/sec of TIFF writes; cross-reference with Defender IO
    # read rate to confirm "Defender mmaps every TIFF" hypothesis.
    try:
        io = proc.io_counters()
        metrics['io_read_mb'] = io.read_bytes / 1e6
        metrics['io_write_mb'] = io.write_bytes / 1e6
        metrics['io_read_mbps'] = _delta_rate('proc_io_read', io.read_bytes) / 1e6
        metrics['io_write_mbps'] = _delta_rate('proc_io_write', io.write_bytes) / 1e6
    except Exception:
        metrics['io_read_mb'] = -1
        metrics['io_write_mb'] = -1
        metrics['io_read_mbps'] = -1
        metrics['io_write_mbps'] = -1

    # --- Page faults (rate) ---
    # Sustained > 1000 pf/sec on a desktop = real memory pressure (paging
    # working set in/out). Useful as a sanity signal -- if pf/sec stays low
    # while standby grows, the slowdown is allocator/standby-cache, not
    # real paging. If pf/sec spikes during slow state, real paging.
    try:
        mem = proc.memory_info()
        pf = getattr(mem, 'pfaults', None) or getattr(mem, 'num_page_faults', None)
        if pf is not None:
            metrics['page_faults_total'] = pf
            metrics['page_faults_per_sec'] = _delta_rate('page_faults', pf)
    except Exception:
        pass

    # --- GDI / USER objects (Windows only -- main long-run-stability concern) ---
    # GDI is what causes Windows-wide slowdown after 24h+ runs. Every
    # `Texture.create()` and unclosed matplotlib figure adds a GDI handle.
    # Process limit is 10,000; Windows desktop degrades around 5,000.
    if _IS_WINDOWS:
        try:
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
            metrics['gdi_objects'] = user32.GetGuiResources(handle, GR_GDIOBJECTS)
            metrics['user_objects'] = user32.GetGuiResources(handle, GR_USEROBJECTS)
        except Exception:
            metrics['gdi_objects'] = -1
            metrics['user_objects'] = -1
    else:
        metrics['gdi_objects'] = -1
        metrics['user_objects'] = -1

    # --- Thread count + names ---
    # Should plateau within ~30s of startup at ~20-25 (8 executors * 2
    # threads + camera + main + a few Kivy). Steady growth means an
    # executor/handler is spawning without joining.
    try:
        metrics['thread_count'] = threading.active_count()
        metrics['thread_names'] = sorted(t.name for t in threading.enumerate())
    except Exception:
        metrics['thread_count'] = -1
        metrics['thread_names'] = []

    # --- Python GC (catches reference-cycle / closure-capture leaks) ---
    # `gc.get_objects()` is somewhat expensive (iterates all tracked
    # objects) -- fine at hourly cadence. Steady linear growth indicates
    # accumulation, typically from observers/callbacks holding refs.
    try:
        metrics['gc_objects'] = len(gc.get_objects())
        gc_stats = gc.get_stats()
        metrics['gc_gen0_collections'] = gc_stats[0]['collections']
        metrics['gc_gen1_collections'] = gc_stats[1]['collections']
        metrics['gc_gen2_collections'] = gc_stats[2]['collections']
    except Exception:
        metrics['gc_objects'] = -1
        metrics['gc_gen0_collections'] = -1
        metrics['gc_gen1_collections'] = -1
        metrics['gc_gen2_collections'] = -1

    # --- Windows PDH memory counters ---
    # Standby cache + nonpaged pool are the specific signals for
    # diagnosing buffer-churn / slow-onset memory growth on Windows.
    try:
        pdh = query_windows_perf_counters()
        for k, v in pdh.items():
            metrics[f'pdh_{k}'] = v
    except Exception:
        pass

    # --- Defender (MsMpEng.exe) metrics ---
    try:
        for k, v in query_defender_metrics().items():
            metrics[k] = v
    except Exception:
        pass

    # --- Live GC depth (uncollected objects per generation) ---
    # Existing gc_genN_collections counts collections-since-start (a counter).
    # gc.get_count() is the CURRENT depth -- pairs with the counter to show
    # both rate (collections/min) and steady-state pressure (depth growing
    # = generation 2 leaks).
    try:
        c = gc.get_count()
        if len(c) >= 3:
            metrics['gc_count_gen0'] = c[0]
            metrics['gc_count_gen1'] = c[1]
            metrics['gc_count_gen2'] = c[2]
    except Exception:
        pass

    return metrics


def check_disk_space(path='/') -> float:
    """
    Returns free disk space in MB
    """

    disk = psutil.disk_usage(path)
    free_space_mb = disk.free / (1024**2)
    return free_space_mb


# Per-step disk-write estimates (MB). An image step writes one file; a video
# step scales with its recording length, so a flat per-video figure under-counts
# a long capture by orders of magnitude.
ESTIMATED_IMAGE_STEP_MB = 8  # ~1900x1900 16-bit TIFF (~7.2 MB) + metadata
ESTIMATED_VIDEO_STEP_MB = 50  # floor for a short compressed MP4
_MP4_COMPRESSION_FRACTION = 0.1  # MP4 ~ a tenth of the raw per-frame bytes


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


def _coerce_positive_float(value) -> float:
    """Coerce a Video Config numeric to a float >= 0, defaulting bad input to 0.

    Video Config values come from a user-edited protocol and may be a
    non-numeric string or a NaN; float() raises on the former and NaN poisons
    the frame-count arithmetic. Both collapse to 0 (a missing dimension), which
    floors the estimate rather than raising and disabling the disk guard.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    # NaN != NaN; a NaN duration/fps means "unknown", treated as 0.
    return result if result == result and result > 0 else 0.0


def estimate_step_write_mb(step, *, video_as_frames: bool = False) -> float:
    """Estimate the disk a single protocol step will write, in MB.

    One owner for write-size estimation, shared by the pre-scan free-space check
    and the per-write threshold so the two cannot drift. An image step writes
    one file. A video step scales with the recording: the frame count is
    duration_s * fps, and each frame costs about one image when saved as
    individual TIFFs (video_as_frames) or a compressed fraction of that in an
    MP4. Deriving from duration/fps -- not a flat per-video figure -- is what
    lets a long recording be sized before it fills the disk; the MP4 path is
    floored at the historical estimate so a short clip is never under-counted.

    Total by construction: a None / malformed step or Video Config sizes to the
    single-image estimate rather than raising, so the broad except around the
    disk check can never silently disable the guard.

    Args:
        step: A protocol step (pandas Series / dict / None); reads Acquire +
            Video Config.
        video_as_frames: Run-level flag -- video saved as individual frames
            rather than a compressed MP4.

    Returns:
        Estimated megabytes the step will write to disk.
    """
    get = getattr(step, 'get', None)
    if get is None or get('Acquire') != 'video':
        return ESTIMATED_IMAGE_STEP_MB
    video_config = read_video_config(step)
    duration_s = _coerce_positive_float(video_config.get('duration'))
    fps = _coerce_positive_float(video_config.get('fps'))
    frames = max(1, int(duration_s * fps))
    if video_as_frames:
        return frames * ESTIMATED_IMAGE_STEP_MB
    return max(
        ESTIMATED_VIDEO_STEP_MB, frames * ESTIMATED_IMAGE_STEP_MB * _MP4_COMPRESSION_FRACTION
    )


# Free-space floor (2 GB) below which capture work refuses to start and
# a running manual recording stops gracefully. One canonical value: the
# protocol scan loop and the manual-record controller share it so the
# two capture paths cannot drift on what "disk almost full" means.
MIN_REQUIRED_DISK_MB = 2048


def check_disk_space_ok(path, required_mb: float) -> tuple[bool, float]:
    """Probe free disk space and compare against a threshold.

    Single canonical disk probe shared by protocol_image_writer, the
    protocol scan loop, and the record pre-flight in main_display. Each
    caller keeps its own threshold + abort/notification policy; only
    the probe itself is shared so the three sites cannot drift on
    backend choice (shutil vs psutil disagree on Windows partition
    mounts) or unit conversion.

    Args:
        path: Filesystem path to probe (str or pathlib.Path).
        required_mb: Minimum free space the caller needs, in MB.

    Returns:
        (ok, free_mb) -- ok is True iff free_mb >= required_mb.

    Raises:
        OSError / PermissionError: propagated from psutil.disk_usage so
            callers can decide whether to swallow (best-effort probes)
            or abort (load-bearing probes).
    """
    disk = psutil.disk_usage(str(path))
    free_mb = disk.free / (1024**2)
    return (free_mb >= required_mb, free_mb)


def get_extra_disks_info(exclude_path: str = '/') -> str | None:
    """
    Returns formatted disk information for extra disks (excluding the disk containing exclude_path).
    Returns None if only the excluded disk exists or no extra disks are found.
    Returns formatted string like: "D: 250.5 GB free (15.2% used) | E: 100.3 GB free (8.5% used)"
    """
    try:
        import psutil

        disk_partitions = psutil.disk_partitions(all=False)

        # Find which partition contains the exclude_path
        excluded_device = None
        try:
            for partition in disk_partitions:
                if exclude_path.startswith(partition.mountpoint):
                    excluded_device = partition.device
                    break
        except Exception:
            pass

        disk_info_list = []
        for partition in disk_partitions:
            # Skip the excluded device/path
            if excluded_device and partition.device == excluded_device:
                continue

            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info_list.append(
                    f'{partition.device}: {usage.free / (1024**3):.1f} GB free ({usage.percent:.1f}% used)'
                )
            except (PermissionError, OSError):
                continue

        return ' | '.join(disk_info_list) if disk_info_list else None

    except Exception:
        return None
