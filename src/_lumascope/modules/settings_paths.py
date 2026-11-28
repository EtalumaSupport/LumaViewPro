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
import enum
import functools
import logging
import typing

# Imported for its registration of the VERBOSE level, which _log_level allows.
import _lumascope.lib.log_helpers  # noqa: F401
import _lumascope.modules.binning as binning
import _lumascope.modules.common_utils as common_utils
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    SettingRefusedError,
    StoredSettingReplacedNotice,
)
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.closed_sets import member_of
from _lumascope.modules.image_mode import (
    LIVE_OUTPUT_FORMATS,
    ImageMode,
    OutputFormat,
    RecordingFormat,
)
from _lumascope.lib.real_numbers import is_count, is_finite_number, is_integer_number
from _lumascope.modules.channel_record import parse_display_color
from _lumascope.modules.lumascope_api._constants import (
    ACCELERATION_PCT_MAX,
    ACCELERATION_PCT_MIN,
    SIMULATOR_TIERS,
    refuse_acceleration_pct,
    refuse_inverted_gain_range,
)
from _lumascope.modules.protocol import ProtocolScheduleRefusedError, schedule_from_units
from _lumascope.modules.zstack_config import ZStackReference
from _lumascope.modules.python_values import python_values
from _lumascope.modules.tiling_config import OVERLAP_PERCENT_RANGE, TilingGrids


@dataclasses.dataclass(frozen=True)
class SettingMember:
    """The member that changes a setting: its owning class and its name.

    ``owner`` is ``'ScopeSession'`` or ``'CameraAPI'``, the class the member
    is looked up on; ``path`` is where a caller reaches it from the API
    root, which is the session: the member's own name on the session, and
    ``scope.camera.<name>`` on the camera, so a REST client is sent to the
    route that exists.

    ``read`` names the owner's member that reads the setting when the
    writer is an action, not a setter: a setter names its read itself
    (``@api(sets=...)``), and an action stores the setting as one effect
    among others, so the table says where it is read.
    """

    owner: str
    name: str
    read: str | None = None

    @property
    def path(self) -> str:
        return self.path_of(self.name)

    def path_of(self, name: str) -> str:
        """The path from the API root of the owner's member ``name``: the setter's read, say."""
        return name if self.owner == 'ScopeSession' else f'scope.camera.{name}'

    @property
    def words(self) -> str:
        """The member as a refusal names it."""
        return f'ScopeSession.{self.name}' if self.owner == 'ScopeSession' else self.path


def _session(name: str, *, read: str | None = None) -> SettingMember:
    return SettingMember('ScopeSession', name, read)


def _camera(name: str) -> SettingMember:
    return SettingMember('CameraAPI', name)


# A setting changed by its own member, because the member does more than
# store it: it checks the setting against the scope, applies it, or changes
# another setting with it. A path at or under a key here is refused, naming
# the member. ``*`` stands for any channel.
SETTINGS_WITH_A_MEMBER: typing.Final[dict[str, SettingMember]] = {
    'microscope': _session('select_model'),
    'live_folder': _session('set_live_folder'),
    'protocol.filepath': _session('set_protocol_filepath'),
    'objective_id': _session('start_select_objective', read='objective_id'),
    'objective_confirmed': _session('confirm_objective', read='objective_question'),
    'turret_objectives': _session('assign_turret_objective'),
    'protocol.labware': _session('select_labware'),
    'image_mode': _camera('set_image_mode'),
    'binning': _camera('set_binning'),
    'frame': _camera('set_frame_size'),
    'camera.high_conversion_gain': _camera('set_high_conversion_gain'),
    'camera.line_noise_reduction': _camera('set_line_noise_reduction'),
    'scale_bar.enabled': _session('set_scale_bar_enabled'),
    'motion.acceleration_max_pct': _session('set_acceleration_limit_pct'),
    'bookmark': _session('save_bookmark', read='bookmark'),
    '*': _session('set_channel_settings'),
}

# A setting only the installation's settings file sets: it decides how the
# next start reaches the scope, the machine or its files -- the server, the
# start mode, the single-instance port, what is profiled and
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
        'profile_trace_enabled',
        'memory_profile_enabled',
        'memory_profile_interval_s',
        'profiling',
    }
)

# A section whose one reader judges it where it is used and answers a value
# it cannot use by not acting on it: the load neither judges nor replaces
# it, since the shipped value is not always the narrower one (an empty
# client list admits every client), and a save writes back what the user
# wrote. ``rest_api``'s reader is the REST server's judge (``_lumascope.rest.served``).
JUDGED_BY_ITS_READER: typing.Final[frozenset[str]] = frozenset({'rest_api'})

# A preference of the person at the display: it changes no capture, no file
# and no hardware -- each is read only by the GUI's own widgets, or the
# plugin's -- so it is written by anyone while a control session is held,
# where every other setting is the session holder's alone.
PREFERENCES: typing.Final[frozenset[str]] = frozenset(
    {
        'show_tooltips',
        'show_step_locations',
        'live_view_fps',
        'capture_shown_s',
        'stimulation_enabled',
        'focus_score_enabled',
    }
)

# A switch: its kind, true or false, is its whole range. ``*`` stands for
# any channel.
KIND_IS_THE_RANGE: typing.Final[frozenset[str]] = frozenset(
    {
        'separate_folder_per_channel',
        '*.autofocus',
        '*.false_color',
        '*.stim_config.enabled',
        'logging.autofocus',
        'focus_score_enabled',
        'preview_host_downscale',
        'profiling.handle_trace_enabled',
        'profiling.open_files_enabled',
        'video.timestamp_overlay',
        'motion.restore_after_home',
        'show_tooltips',
        'show_step_locations',
        'protocol_led_on',
        'keep_led_between_steps',
        'stimulation_enabled',
        'quick_enhance.*',
    }
)

# Any finite number, whose bound another owner holds where it is used: the
# stage offset, whose position past travel is ``MotionAPI.axis_limits``'
# refusal at the move.
BOUND_AT_USE: typing.Final[frozenset[str]] = frozenset({'stage_offset.x', 'stage_offset.y'})

# What a setting no member writes means, in one line with its unit, for the
# reference's settings table: a member's setting means what its member says.
# The one place these meanings are written.
SETTING_MEANINGS: typing.Final[dict[str, str]] = {
    'simulator_tier': (
        'How a simulated scope is built: `fast`, Python stand-ins for the boards with no '
        "timing; `firmware`, the production driver on the motor board's own firmware, its "
        'motors moving at once; `realistic`, the same with moves and the camera taking real '
        'time. Read when the session is created; `fast` where this computer has no MicroPython '
        'runtime for the others.'
    ),
    'image_output_format.live': (
        'The format a still is saved in, and each frame of a manual recording saved as frames.'
    ),
    'image_output_format.sequenced': (
        "The format a run saves its images in, and each frame of a run's recording saved as "
        'frames; a composite run whose format cannot be merged saves OME-TIFF and logs it.'
    ),
    'jpg_quality': (
        'The JPEG quality a still or run image saved as JPG is encoded at; higher keeps more '
        'detail in a larger file.'
    ),
    'separate_folder_per_channel': (
        "Stills and a run's images saved in a folder per channel, named for it; off for an "
        'autofocus scan.'
    ),
    'stage_offset.x': (
        'Where the plate is mounted on the stage along X, in um: the offset between plate and '
        'stage coordinates.'
    ),
    'stage_offset.y': (
        'Where the plate is mounted on the stage along Y, in um: the offset between plate and '
        'stage coordinates.'
    ),
    'zstack.step_size': (
        'The spacing between the slices of a z-stack, in um; 0, the new-install value, is no '
        'stack, and stacking is refused.'
    ),
    'zstack.range': (
        'The Z extent a z-stack covers, in um; 0, the new-install value, is no stack, and '
        'stacking is refused.'
    ),
    'zstack.position': "Where a step's Z sits in its z-stack: at the stack's top, center or bottom.",
    'protocol.period': (
        "The time between a new protocol's scans, in minutes; a protocol loaded keeps its own."
    ),
    'protocol.duration': (
        "How long a new protocol's run lasts, in hours; a protocol loaded keeps its own."
    ),
    'protocol.autogain.target_brightness': (
        'The brightness auto-gain drives a frame to, as a fraction of full scale, in the live '
        'view and in a run alike.'
    ),
    'protocol.autogain.min_gain_db': (
        'The lowest gain auto-gain may set, in dB, in the live view and in a run alike.'
    ),
    'protocol.autogain.max_gain_db': (
        'The highest gain auto-gain may set, in dB, in the live view and in a run alike.'
    ),
    'logging.default.level': (
        'The level of the main, serial, camera and API logs; DEBUG instead while `debug_mode` '
        'is on.'
    ),
    'logging.autofocus': "Each autofocus step's focus score written to the main log.",
    'focus_score_enabled': (
        "The GUI's live view computes and shows each frame's focus score; autofocus is not changed."
    ),
    'preview_host_downscale': (
        "The GUI's live view shrinks each frame on the CPU before it is drawn, in place of the "
        'GPU scaling the whole frame, for a computer whose full-size upload stutters.'
    ),
    'quick_enhance.auto_levels_enabled': (
        "Quick Enhance stretches an image's 1st to 99th percentile of brightness to the full range."
    ),
    'quick_enhance.gamma_enabled': (
        'Quick Enhance applies a gamma of 0.9 after the stretch, lifting the midtones; skipped '
        'for a flat image.'
    ),
    'quick_enhance.illumination_correction_enabled': (
        'Quick Enhance divides out a brightness plane fitted across the field, flattening '
        'uneven illumination.'
    ),
    'quick_enhance.denoise_enabled': 'Quick Enhance applies a 3-pixel median filter.',
    'quick_enhance.sharpen_enabled': (
        'Quick Enhance sharpens the signal with an unsharp mask, leaving the background as it is.'
    ),
    'video_recording_format': (
        "How a recording, manual or a run's, is saved: `mp4` one video; `frames` a folder of "
        'frame files and their manifest; `hyperstack` the frames and an OME-TIFF hyperstack '
        'built from them.'
    ),
    'motion.restore_after_home': (
        'A home puts the turret back on its slot and Z at its target, each when its position '
        'was known before the home; X and Y are never put back. Read when a home names no '
        '`put_back`.'
    ),
    'video.max_fps': (
        "The highest rate a recording, manual or a run's, samples at, in frames per second; 0 "
        'keeps every frame the camera delivers.'
    ),
    'video.max_duration_s': (
        "How long a manual recording runs before it stops, in seconds; a run's recording "
        "takes its step's own."
    ),
    'video.timestamp_overlay': (
        "Each frame's time burned into a recording saved as MP4, manual or a run's; frames "
        'saved as files carry none.'
    ),
    'show_tooltips': "The GUI shows a control's tooltip while the pointer rests on it.",
    'live_view_fps': "The most frames per second the GUI's live view draws; 0 draws every frame.",
    'capture_shown_s': (
        "How long the GUI shows a still or a run's composite just taken before the live view "
        'returns, in seconds.'
    ),
    'ag_ae_max_exposure_ms.transmitted': (
        'The longest exposure auto-gain may set on a transmitted-light channel, in ms.'
    ),
    'ag_ae_max_exposure_ms.fluorescence': (
        'The longest exposure auto-gain may set on a fluorescence channel, and on any channel '
        'neither transmitted nor luminescence, in ms.'
    ),
    'ag_ae_max_exposure_ms.luminescence': (
        'The longest exposure auto-gain may set on a luminescence channel, in ms.'
    ),
    'tiling_overlap_percent': (
        'How much neighbouring tiles overlap when a protocol is tiled, in percent of a field.'
    ),
    'show_step_locations': "The GUI's stage map marks the loaded protocol's step positions.",
    'protocol_led_on': (
        "Going to a protocol step lights the step's channel at its current, on the step's first "
        'visit; off leaves every channel dark. A run is not changed.'
    ),
    'keep_led_between_steps': (
        'A run keeps the LED lit across a move when the next capture uses the same channel; off '
        'for an autofocus scan.'
    ),
    'stimulation_enabled': (
        'The GUI shows the stimulation controls on the fluorescence channels, on a board '
        'whose firmware stimulates.'
    ),
    'mode': (
        '`engineering` starts the GUI in engineering mode: a still or run image on a turret '
        'scope names its slot, and system metrics are logged each minute. Read by the GUI '
        'alone.'
    ),
    'debug_mode': (
        'Every log at DEBUG from its first line, in place of `logging.default.level`; read '
        'before the settings are prepared.'
    ),
    'profile_trace_enabled': (
        'Timing traces (serial commands, motion polls, frame readiness, camera callbacks, lock '
        'waits) written as CSV; read when the session is created, and never turned off in the '
        'process.'
    ),
    'profile_trace_output_dir': (
        "The folder the timing traces are written to; null, the installation's `logs/profile`."
    ),
    'memory_profile_enabled': (
        'Resident memory logged on a cadence, and the top allocators at fixed points; read when '
        'the session is created.'
    ),
    'memory_profile_interval_s': 'How often the memory profiler samples, in seconds.',
    'profiling.handle_trace_enabled': (
        "The growth in the process's open handles logged around saves and camera calls; read "
        'when the session is created.'
    ),
    'profiling.open_files_enabled': "The process's open-file count in each system-metrics line.",
    'profiling.metrics_interval_s': (
        'How often system metrics are logged, in seconds; null, hourly, or each minute in '
        'engineering mode.'
    ),
    'lvp_lock_port': (
        'The local port the single-instance lock holds, so one program at a time drives a '
        'scope from this installation; read before the settings are prepared.'
    ),
    'rest_api.enabled': 'Whether the GUI serves the REST API beside its window.',
    'rest_api.port': 'The TCP port the REST server listens on.',
    'rest_api.address': (
        'The one IP address of this computer the REST server listens on; a loopback address '
        'serves this computer alone.'
    ),
    'rest_api.allowed_clients': (
        'The addresses and networks whose requests the REST server answers, any other refused '
        '`client_not_allowed`; empty answers every client that reaches the address.'
    ),
}

# The kinds a setting shipped as null holds when it is not null. ``*`` stands
# for any channel as a first segment, and for any key as a last one.
_NULLABLE_KINDS: typing.Final[dict[str, tuple[str, ...]]] = {
    '*.acquire': ('string',),
    '*.focus': ('number',),
    '*.display_color': ('string',),
    'turret_objectives.*': ('string',),
    # True is a confirmation stored before the topology was recorded.
    'objective_confirmed': ('bool', 'string'),
    'profiling.metrics_interval_s': ('number',),
    'profile_trace_output_dir': ('string',),
}

VIDEO_MAX_FPS_LIMIT: typing.Final = 200
"""The highest recording-rate cap a person may set; 0 means no cap."""

LIVE_VIEW_FPS_LIMIT: typing.Final = 60
"""The highest live-view display-rate cap a person may set; 0 means no cap."""

VIDEO_MAX_DURATION_S_RANGE: typing.Final = (1, 3600)
"""The manual recording's time limit, in seconds, inclusive."""


JPG_QUALITY_RANGE: typing.Final = (1, 100)
"""The JPG encoder's quality, inclusive."""

CAPTURE_SHOWN_S_RANGE: typing.Final = (1, 10)
"""How long a still or a composite is shown once taken, in whole seconds, inclusive."""

VIDEO_STEP_DURATION_S_MAX: typing.Final = 3600
"""The longest video a protocol step records, in seconds."""

STIM_FREQUENCY_HZ_RANGE: typing.Final = (0.5, 6.0)
STIM_PULSE_WIDTH_MS_RANGE: typing.Final = (5, 25)
STIM_PULSE_COUNT_RANGE: typing.Final = (1, 300)
"""The stimulation ranges, as the channel panel's sliders have always held them."""


COMPOSITE_THRESHOLD_RANGE: typing.Final = (0, 100)
"""A composite's brightness threshold, in percent, inclusive."""


def _refuse_outside(
    path: str,
    value: float,
    limits: tuple[float, float],
    *,
    whole: bool = False,
    count: bool = False,
    low_open: bool = False,
) -> None:
    # ``whole`` and ``count`` are judged before the range, each in its own
    # words. An open low bound has no ``limits`` to publish: no number is the
    # lowest a client may send.
    if whole:
        _refuse_unless_int(path, value)
    if count:
        _refuse_non_count(path, value)
    low, high = limits
    if low_open:
        if not low < value <= high:
            takes = f'above {low} and at most {high}'
            raise SettingRefusedError(
                'out_of_range', path, f'it must be {takes}, not {value!r}', takes=takes
            )
    elif not low <= value <= high:
        takes = f'between {low} and {high}'
        raise SettingRefusedError(
            'out_of_range',
            path,
            f'it must be {takes}, not {value!r}',
            limits=(low, high),
            takes=takes,
        )


def _refuse_unless_int(path: str, value: float) -> None:
    if not is_integer_number(value):
        raise SettingRefusedError('out_of_range', path, f'it must be a whole number, not {value!r}')


def _refuse_below(path: str, value: float, floor: float, *, inclusive: bool) -> None:
    if not (value >= floor if inclusive else value > floor):
        bound = f'at least {floor}' if inclusive else f'above {floor}'
        raise SettingRefusedError(
            'out_of_range', path, f'it must be {bound}, not {value!r}', takes=bound
        )


def _refuse_non_count(path: str, value: float) -> None:
    if not (is_count(value) and value >= 1):
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


def _refuse_unless_member(
    path: str,
    value: object,
    closed_set: type[enum.StrEnum],
    within: tuple[enum.StrEnum, ...] | None = None,
) -> None:
    # A setting that holds a closed set is judged by the set's one judge, in
    # the setting's own refusal, which the load replaces and reports; a
    # setting that takes part of the set offers that part.
    offered = tuple(closed_set) if within is None else within
    if member_of(closed_set, value) not in offered:
        shown = str(value) if isinstance(value, enum.StrEnum) else value
        raise SettingRefusedError(
            'not_offered', path, f'it must be one of {", ".join(offered)}, not {shown!r}'
        )


def _judged(
    judge: typing.Callable[[typing.Any], object],
    refused: type[Exception],
    path: str,
    value: object,
    *,
    takes: str,
    limits: tuple[float, float] | None = None,
) -> None:
    """Refuse what another owner's ``judge`` refuses, as this setting's refusal.

    The judge's own words are kept; ``refused`` is the one exception it
    raises for a value it does not take. ``takes`` says what the setting
    takes, since the judge's row carries no numbers to say it from.
    """
    try:
        judge(value)
    except refused as e:
        raise SettingRefusedError('out_of_range', path, str(e), limits=limits, takes=takes) from e


def _or_none(rule: typing.Callable[[str, typing.Any], None], path: str, value: object) -> None:
    """``rule``, for a setting shipped null: None is always taken."""
    if value is not None:
        rule(path, value)


def _log_level(path: str, value: str, *, takes: str) -> None:
    # A level's name in any case: the reader upper-cases it (Eric, 2026-10-09).
    if value.upper() not in logging.getLevelNamesMapping():
        raise SettingRefusedError(
            'out_of_range', path, f'it must be {takes}, not {value!r}', takes=takes
        )


def _display_color(path: str, value: str | None, *, takes: str) -> None:
    # None is no override: the channel is shown in its catalogue colour.
    if value is None:
        return
    try:
        parse_display_color(value)
    except ValueError:
        raise SettingRefusedError(
            'out_of_range', path, f'it must be {takes}, not {value!r}', takes=takes
        ) from None


_NON_NEGATIVE = functools.partial(_refuse_below, floor=0, inclusive=True)
_POSITIVE = functools.partial(_refuse_below, floor=0, inclusive=False)
_COUNT = functools.partial(_refuse_non_count)


# The range the writer owns for a setting: what any value of it must satisfy
# whatever hardware is attached. A limit the attached hardware declares (a
# camera's exposure or gain, an LED board's current, the frames one capture
# may sum) is not a range here: the channel write clamps a value written past
# it and tells, and a loaded one is kept as the person's intent while the
# apply drives the hardware at its limit (``CameraAPI.applied_gain_db_for``
# and kin).
# A member's setting is here too: its range is held at load as well, though a
# write to it is refused for its member before the range is read. ``*`` stands
# for any channel, as in ``SETTINGS_WITH_A_MEMBER``. Each rule takes the
# concrete path and the value. A row is one of the refusers above with its
# numbers, or another owner's judge through ``_judged``; a rule written for one
# setting is refused by the guard over this table. A row whose numbers do not
# say what it takes -- a judge's, a rule with logic of its own -- says it in
# its ``takes``, the one statement of its range.
_RANGES: typing.Final[dict[str, typing.Callable[[str, typing.Any], None]]] = {
    # A label is arithmetic (a positive square ``NxN``), judged here before
    # bring-up parses it: parsed there, one malformed label dropped the
    # whole file for the shipped template.
    'binning.size': functools.partial(
        _judged,
        binning.binning_size_str_to_int,
        ArgumentRefusedError,
        takes="a square label 'NxN', N a whole number of at least 1",
    ),
    'simulator_tier': functools.partial(_refuse_unknown_format, formats=frozenset(SIMULATOR_TIERS)),
    'zstack.position': functools.partial(_refuse_unless_member, closed_set=ZStackReference),
    'image_mode': functools.partial(_refuse_unless_member, closed_set=ImageMode),
    'logging.default.level': functools.partial(
        _log_level,
        takes=f'a log level, in any case: {", ".join(sorted(logging.getLevelNamesMapping()))}',
    ),
    'motion.acceleration_max_pct': functools.partial(
        _judged,
        refuse_acceleration_pct,
        ArgumentRefusedError,
        takes=f'a whole number between {ACCELERATION_PCT_MIN} and {ACCELERATION_PCT_MAX}',
        limits=(ACCELERATION_PCT_MIN, ACCELERATION_PCT_MAX),
    ),
    'protocol.period': functools.partial(
        _judged,
        functools.partial(schedule_from_units, 'period'),
        ProtocolScheduleRefusedError,
        takes='0 (one scan) or at least one second',
    ),
    'protocol.duration': functools.partial(
        _judged,
        functools.partial(schedule_from_units, 'duration'),
        ProtocolScheduleRefusedError,
        takes='0 (one scan) or more',
    ),
    'tiling_overlap_percent': functools.partial(
        _judged,
        TilingGrids.validate_overlap_percent,
        ValueError,
        takes=f'between {OVERLAP_PERCENT_RANGE[0]:g} and {OVERLAP_PERCENT_RANGE[1]:g}',
        limits=OVERLAP_PERCENT_RANGE,
    ),
    'image_output_format.live': functools.partial(
        _refuse_unless_member, closed_set=OutputFormat, within=LIVE_OUTPUT_FORMATS
    ),
    'image_output_format.sequenced': functools.partial(
        _refuse_unless_member, closed_set=OutputFormat
    ),
    'video_recording_format': functools.partial(_refuse_unless_member, closed_set=RecordingFormat),
    'video.max_fps': functools.partial(_refuse_outside, limits=(0, VIDEO_MAX_FPS_LIMIT)),
    'video.max_duration_s': functools.partial(_refuse_outside, limits=VIDEO_MAX_DURATION_S_RANGE),
    # The encoder takes a whole quality; a fraction would be truncated.
    'jpg_quality': functools.partial(_refuse_outside, limits=JPG_QUALITY_RANGE, whole=True),
    'live_view_fps': functools.partial(_refuse_outside, limits=(0, LIVE_VIEW_FPS_LIMIT)),
    # Whole seconds: a half second's difference is not a preference anyone states.
    'capture_shown_s': functools.partial(_refuse_outside, limits=CAPTURE_SHOWN_S_RANGE, whole=True),
    'zstack.step_size': _NON_NEGATIVE,
    'zstack.range': _NON_NEGATIVE,
    'protocol.autogain.target_brightness': functools.partial(
        _refuse_outside, limits=(0, 1), low_open=True
    ),
    'protocol.autogain.min_gain_db': _NON_NEGATIVE,
    'protocol.autogain.max_gain_db': _NON_NEGATIVE,
    # Shipped null, the default cadence; a value set is seconds between samples.
    'profiling.metrics_interval_s': functools.partial(_or_none, _POSITIVE),
    'memory_profile_interval_s': _POSITIVE,
    'ag_ae_max_exposure_ms.transmitted': _POSITIVE,
    'ag_ae_max_exposure_ms.fluorescence': _POSITIVE,
    'ag_ae_max_exposure_ms.luminescence': _POSITIVE,
    # None is a channel that acquires nothing.
    '*.acquire': functools.partial(
        _or_none, functools.partial(_refuse_unless_member, closed_set=AcquireMode)
    ),
    '*.exposure_ms': _POSITIVE,
    '*.gain_db': _NON_NEGATIVE,
    '*.illumination_ma': _NON_NEGATIVE,
    '*.sum': _COUNT,
    '*.video_config.duration': functools.partial(
        _refuse_outside, limits=(0, VIDEO_STEP_DURATION_S_MAX), low_open=True
    ),
    '*.composite_brightness_threshold': functools.partial(
        _refuse_outside, limits=COMPOSITE_THRESHOLD_RANGE
    ),
    '*.stim_config.frequency': functools.partial(_refuse_outside, limits=STIM_FREQUENCY_HZ_RANGE),
    '*.stim_config.pulse_width': functools.partial(
        _refuse_outside, limits=STIM_PULSE_WIDTH_MS_RANGE
    ),
    '*.stim_config.pulse_count': functools.partial(
        _refuse_outside, limits=STIM_PULSE_COUNT_RANGE, count=True
    ),
    '*.stim_config.illumination_ma': _NON_NEGATIVE,
    '*.display_color': functools.partial(_display_color, takes="None or a '#rrggbb' color"),
}


def _pattern_for(patterns: typing.Collection[str], path: str) -> str | None:
    """The one of ``patterns`` naming ``path``: itself, else its channel's ``*.``, else its block's ``.*``."""
    channel, _, rest = path.partition('.')
    block, _, _key = path.rpartition('.')
    for pattern in (
        path,
        f'*.{rest}' if rest and channel in common_utils.get_channels() else None,
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

    A member, the installation, a range, a switch's kind, or an owner that
    bounds it where it is used.
    """
    return (
        member_for(path) is not None
        or is_installation_only(path)
        or _range_for(path) is not None
        or _pattern_for(KIND_IS_THE_RANGE, path) is not None
        or path in BOUND_AT_USE
    )


def nullable_kinds_for(path: str) -> tuple[str, ...] | None:
    """The kinds the setting at ``path``, shipped as null, holds when it is not null."""
    pattern = _pattern_for(_NULLABLE_KINDS, path)
    return _NULLABLE_KINDS[pattern] if pattern else None


def refused_channel_values(
    channel: str, values: typing.Mapping[str, object]
) -> list[SettingRefusedError]:
    """Every one of a channel's values the channel rules refuse, each as its refusal.

    The one table of channel ranges, read for values that are not yet in the
    settings: a protocol step's, or its Layer Settings row's. ``values``
    maps a ranged number under the channel (``'exposure_ms'``,
    ``'video_config.duration'``, ``'stim_config.frequency'``) to its value. The
    rule is the channel's whatever ``channel`` names, so a step whose color is
    no channel is still judged; ``channel`` names the refusal's path. No
    hardware limit is judged: a value past one is clamped where it is
    written to the channel or run.

    Returns:
        A ``SettingRefusedError`` for each refused value, in ``values``'
        order: ``'not_a_number'`` for one that is not a finite number,
        ``'out_of_range'`` for one outside its range. Empty when every value
        is in range.
    """
    refused = []
    for key, value in values.items():
        path = f'{channel}.{key}'
        # As the writer takes it: a protocol's cells may be numpy scalars.
        value = python_values(value)
        try:
            if _kind(value) != 'number':
                raise SettingRefusedError(
                    'not_a_number', path, f'it must be a finite number, not {value!r}'
                )
            _RANGES[f'*.{key}'](path, value)
        except SettingRefusedError as e:
            refused.append(e)
    return refused


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
    # bool first: True is an int. Exact types, so a numpy scalar (np.float64
    # is a float subclass; np.int64 and np.float32, which json cannot save,
    # are not) is its own kind and refused: the store holds plain Python
    # values. A kind dispatch, not a test of a number. NaN and
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
    channels = common_utils.get_channels()
    for owned, member in SETTINGS_WITH_A_MEMBER.items():
        owned_segments = owned.split('.')
        if len(segments) < len(owned_segments):
            continue
        if all(
            want == got or (want == '*' and got in channels)
            for want, got in zip(owned_segments, segments, strict=False)
        ):
            return member
    return None


def is_installation_only(path: str) -> bool:
    """Whether the setting at ``path`` is set only by the installation's settings file."""
    return path.split('.')[0] in INSTALLATION_ONLY


def check_write(template: dict, path: str, value: object) -> None:
    """Refuse a write the store must not take; return when ``value`` may be stored at ``path``.

    ``template`` is ``check_value``'s. A channel's settings are a member's
    (``*`` in ``SETTINGS_WITH_A_MEMBER``), so no LED current reaches this
    judge.

    Raises:
        SettingRefusedError: ``path`` is owned by a member (named),
            is set only by the installation's settings file, is not a
            setting, or names a
            block rather than one setting; or ``value``
            is not the setting's kind, or is outside its range (a protocol
            period or duration no protocol can run among them).
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
    check_value(template, path, value)


def check_value(template: dict, path: str, value: object) -> None:
    """Refuse a value the setting at ``path`` does not take, whoever its writer.

    ``check_write`` without its question of who writes: a member that owns
    a setting judges a value it stores by this.

    ``template`` is the shipped ``settings.json``, which describes every
    setting there is. A leaf the template holds as null is one with no
    shipped value; it takes null or the kinds declared for it
    (``_NULLABLE_KINDS``). No hardware limit is judged here: the channel
    write clamps a value past one (``ScopeSession._write_channels``).

    Raises:
        SettingRefusedError: ``path`` is not a setting or names a block, or
            ``value`` is not the setting's kind or is outside its range.
    """
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


def _refuse_kind_or_range(shipped: object, path: str, value: object) -> None:
    want, got = _kind(shipped), _kind(value)
    if want == 'null':
        # A leaf shipped null with no kind declared holds null alone; the
        # guard over the template's leaves keeps every one declared.
        kinds = nullable_kinds_for(path) or ()
        if got != 'null' and got not in kinds:
            held = ' or '.join(f'a {kind}' for kind in kinds)
            raise SettingRefusedError(
                _not_its_kind(kinds),
                path,
                f'it holds {held + " or " if held else ""}null, not {value!r}',
            )
    elif got != want:
        raise SettingRefusedError(_not_its_kind((want,)), path, f'it holds a {want}, not {value!r}')
    rule = _range_for(path)
    if rule is not None:
        rule(path, value)


def _not_its_kind(kinds: tuple[str, ...]) -> str:
    """The reason a value not of a setting's ``kinds`` is refused for.

    A number setting given anything but a finite number says
    ``'not_a_number'``, as an argument does for it; any other kind,
    ``'wrong_kind'``.
    """
    return 'not_a_number' if kinds == ('number',) else 'wrong_kind'


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
    the load and a handed dict hold one rule with the write, but for a
    section judged by its reader (``JUDGED_BY_ITS_READER``). Yields the
    dotted path, the dict holding the value and its key, the shipped value,
    and the refusal the writer would raise.
    """
    for path, shipped in _leaf_paths(template):
        *parents, leaf = path.split('.')
        if parents and parents[0] in JUDGED_BY_ITS_READER:
            continue
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


def _is_a_count(path: str) -> bool:
    """Whether the setting at ``path`` holds a count: its range row carries the count rule."""
    rule = _range_for(path)
    return isinstance(rule, functools.partial) and (
        rule.func is _refuse_non_count or bool(rule.keywords.get('count'))
    )


def store_counts_as_integers(settings: dict, template: dict) -> None:
    """Keep each count in ``settings`` as an ``int``: a ``2.0`` the count rule took is stored as 2.

    The count rule is whole by value, since JSON and a caller's float may
    write a count as ``2.0``; every reader after the store (a capture's
    ``sum_count``, a ``range``) takes an integer, so the store holds one.
    Called once the values are judged, so each count here is whole.
    """
    for path, _shipped in _leaf_paths(template):
        if not _is_a_count(path):
            continue
        *parents, leaf = path.split('.')
        stored = settings
        for segment in parents:
            stored = stored.get(segment)
            if not isinstance(stored, dict):
                break
        if isinstance(stored, dict) and is_count(stored.get(leaf)):
            stored[leaf] = int(stored[leaf])


def replace_refused_stored_values(
    settings: dict, template: dict
) -> StoredSettingReplacedNotice | None:
    """Replace each stored value the writer would refuse with the shipped one.

    The load's half of the writer: a file written before a rule was held,
    or edited by hand, can hold a value no write could store -- of the wrong
    kind, NaN, or outside a range -- and nobody can fix a file at load, so
    that key alone takes the template's value, in the settings the app runs
    on and so in the file at its next save; every other setting stays the
    person's. The auto-gain range is judged after its bounds, by the arm's
    own rule: each bound may be in range while the pair is inverted, and an
    inverted pair stopped bring-up at the arm, so both bounds take the
    template's.

    Returns:
        One notice naming every replaced value, for the Session to report
        once a host can hear it; None when nothing was replaced.
    """
    replaced = []
    for path, holder, leaf, shipped, _refused in _values_the_writer_refuses(settings, template):
        replaced.append((path, holder[leaf], shipped))
        holder[leaf] = copy.deepcopy(shipped)
    stored, shipped = settings['protocol']['autogain'], template['protocol']['autogain']
    try:
        refuse_inverted_gain_range(stored, 'protocol.autogain')
    except ArgumentRefusedError:
        for leaf in ('min_gain_db', 'max_gain_db'):
            replaced.append((f'protocol.autogain.{leaf}', stored[leaf], shipped[leaf]))
            stored[leaf] = shipped[leaf]
    return StoredSettingReplacedNotice(replaced) if replaced else None


def refuse_handed_values(settings: dict, template: dict) -> None:
    """Refuse the first value in a dict a caller handed in that the writer refuses.

    The write's half, for a dict handed to a session whole: the caller can
    fix its dict, so a value no write could store is refused as a write of
    it would be, by the same rule, before anything runs on it. The auto-gain
    range is judged after its bounds, by the arm's own rule, as the load's
    half judges it: otherwise an inverted pair is met at the first arm,
    which at bring-up is partway through the session's creation.

    Raises:
        SettingRefusedError: as ``check_write`` raises it for that value.
        ArgumentRefusedError: ``'gain_range_inverted'``, an auto-gain range
            whose minimum gain is above its maximum, naming
            ``protocol.autogain`` and both bounds.
    """
    for *_, refused in _values_the_writer_refuses(settings, template):
        raise refused
    refuse_inverted_gain_range(settings['protocol']['autogain'], 'protocol.autogain')
