# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a settings write must pass before it reaches the store.

``ScopeSession.update_settings(path, value)`` is the one write to the live
settings, for the GUI, a plugin, a script and REST alike. This module is
its check, kept free of the Session so it can be read and tested alone:
the path is one leaf of the shipped template, the value is that leaf's
kind, the setting has no Session member of its own, and the value is in
the range the writer owns for it.
"""

from __future__ import annotations

import typing

import modules.common_utils as common_utils
import modules.settings_init as settings_init
from lvp_logger import logger
from modules.exceptions import SettingRefusedError
from modules.image_mode import VALID_LIVE_OUTPUT_FORMATS, VALID_SEQUENCED_OUTPUT_FORMATS
from modules.protocol import schedule_from_units
from modules.tiling_config import TilingConfig

# A setting changed by its own Session member, because the member does more
# than store it: it checks the setting against the scope, applies it, or
# changes another setting with it. A path at or under a key here is refused,
# naming the member. ``*`` stands for any layer.
SETTINGS_WITH_A_MEMBER: typing.Final[dict[str, str]] = {
    'microscope': 'select_model',
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
    '*.focus': 'save_focus',
}

VIDEO_MAX_FPS_LIMIT: typing.Final = 200
"""The highest recording-rate cap a person may set; 0 means no cap."""

VIDEO_MAX_DURATION_S_RANGE: typing.Final = (1, 3600)
"""The manual recording's time limit, in seconds, inclusive."""


def _refuse_outside(path: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise SettingRefusedError(
            'out_of_range', path, f'it must be between {low} and {high}, not {value!r}'
        )


def _refuse_unknown_format(path: str, value: str, formats: frozenset) -> None:
    if value not in formats:
        raise SettingRefusedError(
            'out_of_range', path, f'{value!r} is not one of {", ".join(sorted(formats))}'
        )


def _overlap(value: float) -> None:
    try:
        TilingConfig.validate_overlap_percent(value)
    except ValueError as e:
        raise SettingRefusedError('out_of_range', 'tiling_overlap_percent', str(e)) from e


_RANGES: typing.Final[dict[str, typing.Callable[[typing.Any], None]]] = {
    'protocol.period': lambda value: schedule_from_units('period', value),
    'protocol.duration': lambda value: schedule_from_units('duration', value),
    'tiling_overlap_percent': _overlap,
    'image_output_format.live': lambda value: _refuse_unknown_format(
        'image_output_format.live', value, VALID_LIVE_OUTPUT_FORMATS
    ),
    'image_output_format.sequenced': lambda value: _refuse_unknown_format(
        'image_output_format.sequenced', value, VALID_SEQUENCED_OUTPUT_FORMATS
    ),
    'video.max_fps': lambda value: _refuse_outside('video.max_fps', value, 0, VIDEO_MAX_FPS_LIMIT),
    'video.max_duration_seconds': lambda value: _refuse_outside(
        'video.max_duration_seconds', value, *VIDEO_MAX_DURATION_S_RANGE
    ),
}


def _live_folder(value: str, installation: str) -> str:
    try:
        return settings_init.bring_up_live_folder(logger, value, installation)
    except ValueError as e:
        # pathlib's answer to a string no file system can name (a NUL byte).
        raise SettingRefusedError('out_of_range', 'live_folder', f'{value!r} is not a path') from e


# A setting stored in a form of its own rather than as given, by the same
# rule its value takes when the settings file is loaded.
_STORED_FORM: typing.Final[dict[str, typing.Callable[[typing.Any, str], typing.Any]]] = {
    'live_folder': _live_folder,
}


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


def check_write(template: dict, path: str, value: object, *, installation: str) -> object:
    """The value to store at ``path``, or a refusal of a write the store must not take.

    ``template`` is the shipped ``settings.json``, which describes every
    setting there is. A leaf the template holds as null is one with no
    shipped value; it takes any scalar. ``installation`` is the folder the
    scope was started on: a live folder given relative to it is stored
    absolute, and created.

    Raises:
        SettingRefusedError: ``path`` is owned by a Session member (named),
            is not a setting, or names a block rather than one setting; or
            ``value`` is not the setting's kind, or is outside its range.
        ProtocolScheduleRefusedError: a protocol period or duration no
            protocol can run.
    """
    member = member_for(path)
    if member is not None:
        raise SettingRefusedError(
            'has_member', path, f'it is changed by ScopeSession.{member}', member=member
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
    want, got = _kind(shipped), _kind(value)
    scalar = ('bool', 'number', 'string', 'null')
    if want == 'null' and got not in scalar:
        raise SettingRefusedError('wrong_kind', path, f'it holds a single value, not a {got}')
    if want != 'null' and got != want:
        raise SettingRefusedError('wrong_kind', path, f'it holds a {want}, not {value!r}')
    rule = _RANGES.get(path)
    if rule is not None:
        rule(value)
    stored_form = _STORED_FORM.get(path)
    return value if stored_form is None else stored_form(value, installation)
