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

import typing

import modules.common_utils as common_utils
from modules.exceptions import SettingRefusedError, StoredSettingReplacedNotice
from modules.image_mode import VALID_LIVE_OUTPUT_FORMATS, VALID_SEQUENCED_OUTPUT_FORMATS
from modules.lumascope_api._constants import refuse_acceleration_pct
from modules.protocol import ProtocolScheduleRefusedError, schedule_from_units
from modules.tiling_config import TilingConfig

# A setting changed by its own Session member, because the member does more
# than store it: it checks the setting against the scope, applies it, or
# changes another setting with it. A path at or under a key here is refused,
# naming the member. ``*`` stands for any layer.
SETTINGS_WITH_A_MEMBER: typing.Final[dict[str, str]] = {
    'microscope': 'select_model',
    'live_folder': 'set_live_folder',
    'protocol.filepath': 'set_protocol_filepath',
    'objective_id': 'select_objective',
    'objective_confirmed': 'confirm_objective',
    'turret_objectives': 'assign_turret_objective',
    'protocol.labware': 'select_labware',
    'image_mode': 'set_image_mode',
    'binning': 'set_binning_size',
    'frame': 'set_frame_size',
    'camera.high_conversion_gain': 'set_high_conversion_gain',
    'camera.line_noise_reduction': 'set_line_noise_reduction',
    'scale_bar.enabled': 'set_scale_bar',
    'motion.acceleration_max_pct': 'set_acceleration_limit',
    'bookmark': 'save_bookmark',
    '*.acquire': 'set_layer_acquire',
    '*.auto_gain': 'set_layer_auto_gain',
    '*.focus': 'save_layer_focus',
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

VIDEO_MAX_FPS_LIMIT: typing.Final = 200
"""The highest recording-rate cap a person may set; 0 means no cap."""

VIDEO_MAX_DURATION_S_RANGE: typing.Final = (1, 3600)
"""The manual recording's time limit, in seconds, inclusive."""


JPG_QUALITY_RANGE: typing.Final = (1, 100)
"""The JPG encoder's quality, inclusive."""


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


def _refuse_unknown_format(path: str, value: str, formats: frozenset) -> None:
    if value not in formats:
        raise SettingRefusedError(
            'out_of_range', path, f'{value!r} is not one of {", ".join(sorted(formats))}'
        )


def _overlap(path: str, value: float) -> None:
    try:
        TilingConfig.validate_overlap_percent(value)
    except ValueError as e:
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


# The range the writer owns for a setting: what any value of it must satisfy
# whatever hardware is attached. A ceiling the attached hardware declares (a
# camera's exposure or gain, an LED board's current) is the Session's to
# apply at the write, not a range here. A member's setting is here too: its
# range is held at load as well, though a write to it is refused for its
# member before the range is read. ``*`` stands for any layer, as in
# ``SETTINGS_WITH_A_MEMBER``. Each rule takes the concrete path and the value.
_RANGES: typing.Final[dict[str, typing.Callable[[str, typing.Any], None]]] = {
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
    '*.exposure_ms': _positive,
    '*.gain_db': _non_negative,
    '*.illumination_ma': _non_negative,
    '*.sum': _refuse_non_count,
    '*.video_config.fps': _positive,
    '*.video_config.duration': _positive,
}


def _range_for(path: str) -> typing.Callable[[str, typing.Any], None] | None:
    rule = _RANGES.get(path)
    if rule is None:
        layer, _, rest = path.partition('.')
        if rest and layer in common_utils.get_layers():
            rule = _RANGES.get(f'*.{rest}')
    return rule


def _ranged_paths(settings: dict) -> list[str]:
    """Every concrete path ``_RANGES`` covers in ``settings``, the layers expanded."""
    layers = [layer for layer in common_utils.get_layers() if isinstance(settings.get(layer), dict)]
    paths = []
    for pattern in _RANGES:
        if pattern.startswith('*.'):
            paths.extend(f'{layer}.{pattern[2:]}' for layer in layers)
        else:
            paths.append(pattern)
    return paths


def _kind(value: object) -> str:
    # bool first: True is an int. Exact types, so a numpy scalar -- a float
    # subclass that json cannot save -- is its own kind and refused.
    if type(value) is bool:
        return 'bool'
    if type(value) in (int, float):
        return 'number'
    if type(value) is str:
        return 'string'
    if type(value) is list:
        return 'list'
    if type(value) is dict:
        return 'mapping'
    if value is None:
        return 'null'
    return type(value).__name__


def member_for(path: str) -> str | None:
    """The Session member that owns the setting at ``path``, or None."""
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


def check_write(template: dict, path: str, value: object) -> None:
    """Refuse a write the store must not take; return when ``value`` may be stored at ``path``.

    ``template`` is the shipped ``settings.json``, which describes every
    setting there is. A leaf the template holds as null is one with no
    shipped value; it takes any scalar.

    Raises:
        SettingRefusedError: ``path`` is owned by a Session member (named),
            is set only by the installation's settings file, is not a
            setting, or names a block rather than one setting; or ``value``
            is not the setting's kind, or is outside its range (a protocol
            period or duration no protocol can run among them).
    """
    member = member_for(path)
    if member is not None:
        raise SettingRefusedError(
            'has_member', path, f'it is changed by ScopeSession.{member}', member=member
        )
    if is_installation_only(path):
        raise SettingRefusedError(
            'installation_only', path, "it is read from the installation's settings file"
        )
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
    scalar = ('bool', 'number', 'string', 'null')
    if want == 'null' and got not in scalar:
        raise SettingRefusedError('wrong_kind', path, f'it holds a single value, not a {got}')
    if want != 'null' and got != want:
        raise SettingRefusedError('wrong_kind', path, f'it holds a {want}, not {value!r}')
    rule = _range_for(path)
    if rule is not None:
        rule(path, value)


def replace_refused_stored_values(
    settings: dict, template: dict
) -> StoredSettingReplacedNotice | None:
    """Replace each stored value the writer would refuse with the shipped one.

    The load's half of the writer's ranges: a file written before a range
    was held, or edited by hand, can hold a value no write could store. That
    key alone takes the template's value, in the settings the app runs on and
    so in the file at its next save; every other setting stays the person's.

    Returns:
        One notice naming every replaced value, for the Session to report
        once a host can hear it; None when nothing was replaced.
    """
    replaced = []
    for path in _ranged_paths(settings):
        *parents, leaf = path.split('.')
        stored = settings
        for segment in parents:
            stored = stored.get(segment)
            if not isinstance(stored, dict):
                break
        if not isinstance(stored, dict) or leaf not in stored:
            continue
        shipped = template
        for segment in path.split('.'):
            shipped = shipped[segment]
        try:
            _refuse_kind_or_range(shipped, path, stored[leaf])
        except SettingRefusedError:
            replaced.append((path, stored[leaf], shipped))
            stored[leaf] = shipped
    return StoredSettingReplacedNotice(replaced) if replaced else None
