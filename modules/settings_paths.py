# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a settings write must pass before it reaches the store.

``ScopeSession.update_settings(path, value)`` is the one write to the live
settings, for the GUI, a plugin, a script and REST alike. This module is
its check, kept free of the Session so it can be read and tested alone:
the path is one leaf of the shipped template, the value is that leaf's
kind, the setting has no Session member of its own and is not one only
the installation sets, and the value is in the range the writer owns for
it.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import typing

import modules.binning as binning
import modules.common_utils as common_utils
from modules.exceptions import (
    ArgumentRefusedError,
    SettingRefusedError,
    StoredSettingReplacedNotice,
)
from modules.image_mode import (
    VALID_LIVE_OUTPUT_FORMATS,
    VALID_SEQUENCED_OUTPUT_FORMATS,
    available_modes,
)
from modules.finite_number import is_finite_number
from modules.layer_record import parse_display_color
from modules.lumascope_api._constants import SIMULATOR_TIERS, refuse_acceleration_pct
from modules.protocol import Protocol, ProtocolScheduleRefusedError, schedule_from_units
from modules.tiling_config import TilingConfig


@dataclasses.dataclass(frozen=True)
class SettingMember:
    """The member that changes a setting: its owning class and its name.

    ``owner`` is ``'ScopeSession'`` or ``'CameraAPI'``, the class the member
    is looked up on; ``path`` is where a caller reaches it from the API
    root, which is the session: the member's own name on the session, and
    ``scope.camera.<name>`` on the camera, so a REST client is sent to the
    route that exists.
    """

    owner: str
    name: str

    @property
    def path(self) -> str:
        return self.name if self.owner == 'ScopeSession' else f'scope.camera.{self.name}'

    @property
    def words(self) -> str:
        """The member as a refusal names it."""
        return f'ScopeSession.{self.name}' if self.owner == 'ScopeSession' else self.path


def _session(name: str) -> SettingMember:
    return SettingMember('ScopeSession', name)


def _camera(name: str) -> SettingMember:
    return SettingMember('CameraAPI', name)


# A setting changed by its own member, because the member does more than
# store it: it checks the setting against the scope, applies it, or changes
# another setting with it. A path at or under a key here is refused, naming
# the member. ``*`` stands for any layer.
SETTINGS_WITH_A_MEMBER: typing.Final[dict[str, SettingMember]] = {
    'microscope': _session('select_model'),
    'live_folder': _session('set_live_folder'),
    'protocol.filepath': _session('set_protocol_filepath'),
    'objective_id': _session('select_objective'),
    'objective_confirmed': _session('confirm_objective'),
    'turret_objectives': _session('assign_turret_objective'),
    'protocol.labware': _session('select_labware'),
    'image_mode': _camera('set_image_mode'),
    'binning': _camera('set_binning_size'),
    'frame': _camera('set_frame_size'),
    'camera.high_conversion_gain': _camera('set_high_conversion_gain'),
    'camera.line_noise_reduction': _camera('set_line_noise_reduction'),
    'scale_bar.enabled': _session('set_scale_bar'),
    'motion.acceleration_max_pct': _session('set_acceleration_limit'),
    'bookmark': _session('save_bookmark'),
    '*.acquire': _session('set_layer_acquire'),
    '*.auto_gain': _session('set_layer_auto_gain'),
    '*.focus': _session('save_layer_focus'),
}

# A setting only the installation's settings file sets: it decides how the
# next start reaches the scope, the machine or its files -- the server and
# its key, the start mode, the single-instance port, what is profiled and
# where the profile is written. No write changes it, from any caller, so
# no caller can reconfigure the scope it is driving. A path at or under a
# key here is refused.
INSTALLATION_ONLY: typing.Final[frozenset[str]] = frozenset(
    {
        'rest_api',
        'mode',
        'lvp_lock_port',
        'profile_trace_output_dir',
        'debug_mode',
        'cprofile_enabled',
        'profile_trace_enabled',
        'tracemalloc_enabled',
        'memory_profile_enabled',
        'memory_profile_interval_s',
        'fx2_debug_wire_enabled',
    }
)

# A setting derived from the release when the template is read, never a
# person's: ``step_channel_order``, from the catalogue's ``LayerOrder``. No
# write changes it, and a stored copy that differs is replaced at load by
# the derived one.
DERIVED: typing.Final[frozenset[str]] = frozenset({'step_channel_order'})

# A switch: its kind, true or false, is its whole range. ``*`` stands for
# any layer.
KIND_IS_THE_RANGE: typing.Final[frozenset[str]] = frozenset(
    {
        'separate_folder_per_channel',
        '*.autofocus',
        '*.false_color',
        '*.stim_config.enabled',
        'logging.autofocus',
        'focus_score_enabled',
        'preview_host_downscale',
        'profiling.enabled',
        'profiling.handle_trace_enabled',
        'profiling.open_files_enabled',
        'video_as_frames',
        'video.timestamp_overlay',
        'show_tooltips',
        'show_step_locations',
        'protocol_led_on',
        'keep_led_between_steps',
        'stimulation_enabled',
    }
)

# Any finite number, whose bound another owner holds where it is used: the
# stage offset, whose position past travel is ``MotionAPI.get_axis_limits``'
# refusal at the move.
BOUND_AT_USE: typing.Final[frozenset[str]] = frozenset({'stage_offset.x', 'stage_offset.y'})

# The kinds a setting shipped as null holds when it is not null. ``*`` stands
# for any layer as a first segment, and for any key as a last one.
_NULLABLE_KINDS: typing.Final[dict[str, tuple[str, ...]]] = {
    '*.acquire': ('string',),
    '*.focus': ('number',),
    '*.display_color': ('string',),
    'turret_objectives.*': ('string',),
    # True is a confirmation stored before the topology was recorded.
    'objective_confirmed': ('bool', 'string'),
    'profiling.metrics_interval_s': ('number',),
    'profile_trace_output_dir': ('string',),
    'rest_api.api_key': ('string',),
}

VIDEO_MAX_FPS_LIMIT: typing.Final = 200
"""The highest recording-rate cap a person may set; 0 means no cap."""

VIDEO_MAX_DURATION_S_RANGE: typing.Final = (1, 3600)
"""The manual recording's time limit, in seconds, inclusive."""


JPG_QUALITY_RANGE: typing.Final = (1, 100)
"""The JPG encoder's quality, inclusive."""

VIDEO_STEP_DURATION_S_MAX: typing.Final = 3600
"""The longest video a protocol step records, in seconds."""

STIM_FREQUENCY_HZ_RANGE: typing.Final = (0.5, 6.0)
STIM_PULSE_WIDTH_MS_RANGE: typing.Final = (5, 25)
STIM_PULSE_COUNT_RANGE: typing.Final = (1, 300)
"""The stimulation ranges, as the layer panel's sliders have always held them."""


def _refuse_outside(path: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise SettingRefusedError(
            'out_of_range', path, f'it must be between {low} and {high}, not {value!r}'
        )


def _refuse_below(path: str, value: float, floor: float, *, inclusive: bool) -> None:
    if not (value >= floor if inclusive else value > floor):
        bound = f'at least {floor}' if inclusive else f'above {floor}'
        raise SettingRefusedError('out_of_range', path, f'it must be {bound}, not {value!r}')


def _refuse_non_count(path: str, value: float) -> None:
    if not (float(value).is_integer() and value >= 1):
        raise SettingRefusedError(
            'out_of_range', path, f'it must be a whole number of at least 1, not {value!r}'
        )


def _refuse_unknown_format(path: str, value: object, formats: frozenset) -> None:
    # The kind first: a member asks this rule before any kind check, and an
    # unhashable value would fail the membership test as a bare TypeError.
    if not isinstance(value, str) or value not in formats:
        raise SettingRefusedError(
            'out_of_range', path, f'{value!r} is not one of {", ".join(sorted(formats))}'
        )


def _overlap(path: str, value: float) -> None:
    try:
        TilingConfig.validate_overlap_percent(value)
    except ValueError as e:
        raise SettingRefusedError('out_of_range', path, str(e)) from e


def _binning_label(path: str, value: str) -> None:
    # A label is arithmetic (a positive square ``NxN``), judged here before
    # bring-up parses it: parsed there, one malformed label dropped the
    # whole file for the shipped template.
    try:
        binning.binning_size_str_to_int(text=value)
    except ArgumentRefusedError as e:
        raise SettingRefusedError('out_of_range', path, str(e)) from e


def _acceleration(path: str, value: float) -> None:
    try:
        refuse_acceleration_pct(value)
    except ValueError as e:
        raise SettingRefusedError('out_of_range', path, str(e)) from e


def _schedule(path: str, value: object) -> None:
    try:
        schedule_from_units(path.rpartition('.')[2], value)
    except ProtocolScheduleRefusedError as e:
        raise SettingRefusedError('out_of_range', path, str(e)) from e


def _live_format(path: str, value: str) -> None:
    _refuse_unknown_format(path, value, VALID_LIVE_OUTPUT_FORMATS)


def _sequenced_format(path: str, value: str) -> None:
    _refuse_unknown_format(path, value, VALID_SEQUENCED_OUTPUT_FORMATS)


def _zstack_position(path: str, value: str) -> None:
    _refuse_unknown_format(path, value, frozenset(common_utils.ZSTACK_POSITION_LABELS))


def _simulator_tier(path: str, value: str) -> None:
    _refuse_unknown_format(path, value, frozenset(SIMULATOR_TIERS))


def _image_mode(path: str, value: str) -> None:
    _refuse_unknown_format(path, value, frozenset(available_modes()))


def _acquire(path: str, value: str | None) -> None:
    # None is a layer that acquires nothing.
    modes = Protocol.VALID_ACQUIRE_MODES
    if value is not None and not (isinstance(value, str) and value in modes):
        raise SettingRefusedError(
            'out_of_range',
            path,
            f'it must be None or one of {", ".join(sorted(modes))}, not {value!r}',
        )


# Which light path the objective was confirmed on: a turret's slots, or the
# one objective a turretless scope selects. A confirmation is for its own
# light path, so a scope brought up with the other owes the question again.
OBJECTIVE_TOPOLOGIES: typing.Final[tuple[str, str]] = ('turret', 'turretless')


def _objective_confirmed(path: str, value: object) -> None:
    # None is never confirmed. True is a confirmation stored before the
    # topology was recorded, which no load can date: whether the scope has a
    # turret is known only at bring-up, which rewrites it to the one it has.
    if value is None or value is True or value in OBJECTIVE_TOPOLOGIES:
        return
    raise SettingRefusedError(
        'out_of_range',
        path,
        f'it must be None or one of {", ".join(OBJECTIVE_TOPOLOGIES)}, not {value!r}',
    )


def _log_level(path: str, value: str) -> None:
    # A level's name in any case: the reader upper-cases it (Eric, 2026-10-09).
    levels = logging.getLevelNamesMapping()
    if value.upper() not in levels:
        raise SettingRefusedError(
            'out_of_range', path, f'{value!r} is not a log level: {", ".join(sorted(levels))}'
        )


def _video_max_fps(path: str, value: float) -> None:
    _refuse_outside(path, value, 0, VIDEO_MAX_FPS_LIMIT)


def _video_max_duration(path: str, value: float) -> None:
    _refuse_outside(path, value, *VIDEO_MAX_DURATION_S_RANGE)


def _jpg_quality(path: str, value: float) -> None:
    _refuse_outside(path, value, *JPG_QUALITY_RANGE)


def _non_negative(path: str, value: float) -> None:
    _refuse_below(path, value, 0, inclusive=True)


def _positive(path: str, value: float) -> None:
    _refuse_below(path, value, 0, inclusive=False)


def _fraction(path: str, value: float) -> None:
    if not 0 < value <= 1:
        raise SettingRefusedError(
            'out_of_range', path, f'it must be above 0 and at most 1, not {value!r}'
        )


def _metrics_interval(path: str, value: float | None) -> None:
    # Shipped null, the default cadence; a value set is seconds between samples.
    if value is not None:
        _positive(path, value)


def _video_step_duration(path: str, value: float) -> None:
    if not 0 < value <= VIDEO_STEP_DURATION_S_MAX:
        raise SettingRefusedError(
            'out_of_range',
            path,
            f'it must be above 0 and at most {VIDEO_STEP_DURATION_S_MAX}, not {value!r}',
        )


def _composite_threshold(path: str, value: float) -> None:
    _refuse_outside(path, value, 0, 100)


def _stim_frequency(path: str, value: float) -> None:
    _refuse_outside(path, value, *STIM_FREQUENCY_HZ_RANGE)


def _stim_pulse_width(path: str, value: float) -> None:
    _refuse_outside(path, value, *STIM_PULSE_WIDTH_MS_RANGE)


def _stim_pulse_count(path: str, value: float) -> None:
    _refuse_non_count(path, value)
    _refuse_outside(path, value, *STIM_PULSE_COUNT_RANGE)


def _display_color(path: str, value: str | None) -> None:
    # None is no override: the channel is shown in its catalogue colour.
    if value is None:
        return
    try:
        parse_display_color(value)
    except ValueError:
        raise SettingRefusedError(
            'out_of_range', path, f"it must be None or a '#rrggbb' colour, not {value!r}"
        ) from None


# The range the writer owns for a setting: what any value of it must satisfy
# whatever hardware is attached. A ceiling the attached hardware declares (a
# camera's exposure or gain, an LED board's current) is not a range here: the
# stored value is the person's intent, and the apply drives the hardware at
# its limit with the intent kept (``CameraAPI.applied_gain_db_for`` and kin).
# A member's setting is here too: its range is held at load as well, though a
# write to it is refused for its member before the range is read. ``*`` stands
# for any layer, as in ``SETTINGS_WITH_A_MEMBER``. Each rule takes the
# concrete path and the value.
_RANGES: typing.Final[dict[str, typing.Callable[[str, typing.Any], None]]] = {
    'binning.size': _binning_label,
    'simulator_tier': _simulator_tier,
    'zstack.position': _zstack_position,
    'image_mode': _image_mode,
    'objective_confirmed': _objective_confirmed,
    'logging.default.level': _log_level,
    'motion.acceleration_max_pct': _acceleration,
    'protocol.period': _schedule,
    'protocol.duration': _schedule,
    'tiling_overlap_percent': _overlap,
    'image_output_format.live': _live_format,
    'image_output_format.sequenced': _sequenced_format,
    'video.max_fps': _video_max_fps,
    'video.max_duration_seconds': _video_max_duration,
    'jpg_quality': _jpg_quality,
    'live_view_fps': _non_negative,
    'zstack.step_size': _non_negative,
    'zstack.range': _non_negative,
    'protocol.autogain.target_brightness': _fraction,
    'protocol.autogain.max_duration_seconds': _positive,
    'protocol.autogain.min_gain_db': _non_negative,
    'protocol.autogain.max_gain_db': _non_negative,
    'profiling.handle_trace_obj_sample_every': _refuse_non_count,
    'profiling.metrics_interval_s': _metrics_interval,
    'ag_ae_max_exposure_ms.transmitted': _positive,
    'ag_ae_max_exposure_ms.fluorescence': _positive,
    'ag_ae_max_exposure_ms.luminescence': _positive,
    '*.acquire': _acquire,
    '*.exposure_ms': _positive,
    '*.gain_db': _non_negative,
    '*.illumination_ma': _non_negative,
    '*.sum': _refuse_non_count,
    '*.video_config.fps': _positive,
    '*.video_config.duration': _video_step_duration,
    '*.composite_brightness_threshold': _composite_threshold,
    '*.stim_config.frequency': _stim_frequency,
    '*.stim_config.pulse_width': _stim_pulse_width,
    '*.stim_config.pulse_count': _stim_pulse_count,
    '*.stim_config.illumination_ma': _non_negative,
    '*.display_color': _display_color,
}


def _pattern_for(patterns: typing.Collection[str], path: str) -> str | None:
    """The one of ``patterns`` naming ``path``: itself, else its layer's ``*.``, else its block's ``.*``."""
    layer, _, rest = path.partition('.')
    block, _, _key = path.rpartition('.')
    for pattern in (
        path,
        f'*.{rest}' if rest and layer in common_utils.get_layers() else None,
        f'{block}.*' if block else None,
    ):
        if pattern in patterns:
            return pattern
    return None


def _range_for(path: str) -> typing.Callable[[str, typing.Any], None] | None:
    pattern = _pattern_for(_RANGES, path)
    return _RANGES[pattern] if pattern else None


def is_classified(path: str) -> bool:
    """Whether the setting at ``path`` says what holds it beyond its kind.

    A member, the installation, the release's derivation, a range, a switch's
    kind, or an owner that bounds it where it is used.
    """
    return (
        member_for(path) is not None
        or is_installation_only(path)
        or path in DERIVED
        or _range_for(path) is not None
        or _pattern_for(KIND_IS_THE_RANGE, path) is not None
        or path in BOUND_AT_USE
    )


def nullable_kinds_for(path: str) -> tuple[str, ...] | None:
    """The kinds the setting at ``path``, shipped as null, holds when it is not null."""
    pattern = _pattern_for(_NULLABLE_KINDS, path)
    return _NULLABLE_KINDS[pattern] if pattern else None


def refuse_outside_range(path: str, value: object) -> None:
    """Refuse ``value`` for the setting at ``path`` by the rule the writer and the load hold.

    For a caller that stores a ranged setting without the writer: a Session
    member that owns its setting, and a settings dict handed to a session,
    which never went through the load. ``path`` is a setting with a range.

    Raises:
        SettingRefusedError: ``'out_of_range'``, the value is outside the
            setting's range.
    """
    _range_for(path)(path, value)


def _kind(value: object) -> str:
    # bool first: True is an int. Exact types, so a numpy scalar -- a float
    # subclass that json cannot save -- is its own kind and refused. NaN and
    # an infinity are their own kind: every range compares, and a comparison
    # with NaN passes, so a leaf with no range would take them as numbers.
    if type(value) is bool:
        return 'bool'
    if type(value) in (int, float):
        return 'number' if is_finite_number(value) else 'non-finite number'
    if type(value) is str:
        return 'string'
    if type(value) is list:
        return 'list'
    if type(value) is dict:
        return 'mapping'
    if value is None:
        return 'null'
    return type(value).__name__


def member_for(path: str) -> SettingMember | None:
    """The member that owns the setting at ``path``, or None."""
    segments = path.split('.')
    layers = common_utils.get_layers()
    for owned, member in SETTINGS_WITH_A_MEMBER.items():
        owned_segments = owned.split('.')
        if len(segments) < len(owned_segments):
            continue
        if all(
            want == got or (want == '*' and got in layers)
            for want, got in zip(owned_segments, segments, strict=False)
        ):
            return member
    return None


def is_installation_only(path: str) -> bool:
    """Whether the setting at ``path`` is set only by the installation's settings file."""
    return path.split('.')[0] in INSTALLATION_ONLY


def check_write(
    template: dict, path: str, value: object, *, led_max_ma: float | None = None
) -> None:
    """Refuse a write the store must not take; return when ``value`` may be stored at ``path``.

    ``template`` is the shipped ``settings.json``, which describes every
    setting there is. A leaf the template holds as null is one with no
    shipped value; it takes null or the kinds declared for it
    (``_NULLABLE_KINDS``). ``led_max_ma`` is the current the
    attached LED board declares it can drive, or None with no board: a
    layer's illumination above it is refused, since the board refuses to
    drive it and over-driving an LED is a damage mode; a camera's limits
    are not checked here, since the apply drives the camera at its limit
    with the stored intent kept.

    Raises:
        SettingRefusedError: ``path`` is owned by a member (named),
            is set only by the installation's settings file, is derived
            from the release (``'derived'``), is not a setting, or names a
            block rather than one setting; or ``value``
            is not the setting's kind, or is outside its range (a protocol
            period or duration no protocol can run, and an illumination
            above the attached board's maximum, among them).
    """
    member = member_for(path)
    if member is not None:
        raise SettingRefusedError(
            'has_member', path, f'it is changed by {member.words}', member=member.path
        )
    if is_installation_only(path):
        raise SettingRefusedError(
            'installation_only', path, "it is read from the installation's settings file"
        )
    if path in DERIVED:
        raise SettingRefusedError('derived', path, "it is derived from the release's catalogue")
    shipped: object = template
    for segment in path.split('.'):
        if not isinstance(shipped, dict) or segment not in shipped:
            raise SettingRefusedError('not_a_setting', path, 'there is no such setting')
        shipped = shipped[segment]
    if isinstance(shipped, dict):
        raise SettingRefusedError(
            'block', path, 'it is a block of settings; change each by its own path'
        )
    _refuse_kind_or_range(shipped, path, value)
    if led_max_ma is not None and is_led_current(path) and value > led_max_ma:
        raise SettingRefusedError(
            'out_of_range',
            path,
            f"it must be at most {led_max_ma:g} mA, this LED board's maximum, not {value!r}",
        )


def is_led_current(path: str) -> bool:
    """Whether ``path`` is a layer's illumination or stimulation current."""
    layer, _, rest = path.partition('.')
    return layer in common_utils.get_layers() and rest in (
        'illumination_ma',
        'stim_config.illumination_ma',
    )


def _refuse_kind_or_range(shipped: object, path: str, value: object) -> None:
    want, got = _kind(shipped), _kind(value)
    if want == 'null':
        # A leaf shipped null with no kind declared holds null alone; the
        # guard over the template's leaves keeps every one declared.
        kinds = nullable_kinds_for(path) or ()
        if got != 'null' and got not in kinds:
            held = ' or '.join(f'a {kind}' for kind in kinds)
            raise SettingRefusedError(
                'wrong_kind', path, f'it holds {held + " or " if held else ""}null, not {value!r}'
            )
    elif got != want:
        raise SettingRefusedError('wrong_kind', path, f'it holds a {want}, not {value!r}')
    if path in DERIVED and value != shipped:
        raise SettingRefusedError(
            'derived', path, f"it is derived from the release's catalogue as {shipped!r}"
        )
    rule = _range_for(path)
    if rule is not None:
        rule(path, value)


def _leaf_paths(block: dict, prefix: str = '') -> typing.Iterator[tuple[str, object]]:
    """Every setting in ``block`` that is not itself a block, as (dotted path, value)."""
    for key, value in block.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from _leaf_paths(value, f'{path}.')
        else:
            yield path, value


def _values_the_writer_refuses(
    settings: dict, template: dict
) -> typing.Iterator[tuple[str, dict, str, object, SettingRefusedError]]:
    """Every value in ``settings`` the writer's own rule refuses, with the refusal.

    Every setting the template holds is checked by the writer's rule, so
    the load and a handed dict hold one rule with the write. Yields the
    dotted path, the dict holding the value and its key, the shipped value,
    and the refusal the writer would raise.
    """
    for path, shipped in _leaf_paths(template):
        *parents, leaf = path.split('.')
        stored = settings
        for segment in parents:
            stored = stored.get(segment)
            if not isinstance(stored, dict):
                break
        if not isinstance(stored, dict) or leaf not in stored:
            continue
        try:
            _refuse_kind_or_range(shipped, path, stored[leaf])
        except SettingRefusedError as refused:
            yield path, stored, leaf, shipped, refused


def replace_refused_stored_values(
    settings: dict, template: dict
) -> StoredSettingReplacedNotice | None:
    """Replace each stored value the writer would refuse with the shipped one.

    The load's half of the writer: a file written before a rule was held,
    or edited by hand, can hold a value no write could store -- of the wrong
    kind, NaN, or outside a range -- and nobody can fix a file at load, so
    that key alone takes the template's value, in the settings the app runs
    on and so in the file at its next save; every other setting stays the
    person's.

    Returns:
        One notice naming every replaced value, for the Session to report
        once a host can hear it; None when nothing was replaced.
    """
    replaced = []
    for path, holder, leaf, shipped, _refused in _values_the_writer_refuses(settings, template):
        replaced.append((path, holder[leaf], shipped))
        holder[leaf] = copy.deepcopy(shipped)
    return StoredSettingReplacedNotice(replaced) if replaced else None


def refuse_handed_values(settings: dict, template: dict) -> None:
    """Refuse the first value in a dict a caller handed in that the writer refuses.

    The write's half, for a dict handed to a session whole: the caller can
    fix its dict, so a value no write could store is refused as a write of
    it would be, by the same rule, before anything runs on it.

    Raises:
        SettingRefusedError: as ``check_write`` raises it for that value.
    """
    for *_, refused in _values_the_writer_refuses(settings, template):
        raise refused
