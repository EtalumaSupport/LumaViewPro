# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The settings a panel shows as a toggle or a picker, one row each.

A row names the widget that shows a setting, the setting's path, and for a
toggle the name its person record carries, for a picker the values it
offers and the label of a value. The panels' one handler pair and one show
(``SettingControls`` in ``ui/ui_helpers.py``) read these rows, so adding a
setting control is one row here and its widget in the kv.

Nothing here imports Kivy: another toolkit's panel reads the same rows.

A label is display: the store, the API, a protocol and REST hold the value,
so a label is renamed with no migration. Each picker's labels must stay
distinct, because a pick is read back to its value through them.
"""

from __future__ import annotations

import dataclasses
import typing

from _lumascope.modules.image_mode import LIVE_OUTPUT_FORMATS, OutputFormat, RecordingFormat
from _lumascope.modules.zstack_config import ZStackReference


@dataclasses.dataclass(frozen=True)
class Toggle:
    """A setting shown on and off by one widget, recorded under ``record``."""

    widget: str
    path: str
    record: str


@dataclasses.dataclass(frozen=True)
class Picker:
    """A setting chosen from ``choices``, each shown as ``label(value)``.

    ``label`` takes any value the store may hold, a choice or not, so a
    value a REST client stored off the choices is shown as itself.
    """

    widget: str
    path: str
    choices: tuple[object, ...]
    label: typing.Callable[[object], str]

    def value_of(self, label: str) -> object:
        """The choice shown as ``label``; a label no choice has is a defect, raised."""
        for choice in self.choices:
            if self.label(choice) == label:
                return choice
        raise KeyError(f'{self.widget} offers no choice labeled {label!r}')


SettingControl = Toggle | Picker


# The output-format selectors' labels, in selector order. The hyperstack's
# stored spelling is too wide for the closed selector, which would shorten it
# mid-word.
OUTPUT_FORMAT_LABELS = {
    OutputFormat.TIFF: 'TIFF',
    OutputFormat.OME_TIFF: 'OME-TIFF',
    OutputFormat.HYPERSTACK: 'Hyperstack',
    OutputFormat.JPG: 'JPG',
}

# The recording format picker's labels, in its order.
RECORDING_FORMAT_LABELS = {
    RecordingFormat.MP4: 'MP4',
    RecordingFormat.FRAMES: 'Frames',
    RecordingFormat.HYPERSTACK: 'Hyperstack',
}

# The z-stack reference picker's labels, in its order.
ZSTACK_REFERENCE_LABELS = {
    ZStackReference.TOP: 'Current Position at Top',
    ZStackReference.CENTER: 'Current Position at Center',
    ZStackReference.BOTTOM: 'Current Position at Bottom',
}


def _percent(value: object) -> str:
    """``10%`` for 10 or 10.0, ``12.5%`` for 12.5: a stored value shown as itself."""
    return f'{float(value):g}%'


ADVANCED_SETTINGS: tuple[SettingControl, ...] = (
    Toggle('restore_after_home_id', 'motion.restore_after_home', 'RESTORE_AFTER_HOME'),
    Toggle('video_timestamp_overlay_id', 'video.timestamp_overlay', 'VIDEO_TIMESTAMP_OVERLAY'),
    Toggle('separate_folder_per_channel_id', 'separate_folder_per_channel', 'SEPARATE_FOLDERS'),
    Toggle('protocol_led_on_btn', 'protocol_led_on', 'PROTOCOL_LED_ON'),
    Toggle('keep_led_between_steps_btn', 'keep_led_between_steps', 'KEEP_LED_BETWEEN_STEPS'),
    Toggle('stimulation_settings_btn', 'stimulation_enabled', 'STIMULATION_ENABLED'),
    Toggle('show_step_locations_id', 'show_step_locations', 'SHOW_STEP_LOCATIONS'),
    Picker('tiling_overlap_spinner', 'tiling_overlap_percent', (0.0, 10.0, 15.0, 20.0), _percent),
)

MICROSCOPE_SETTINGS: tuple[SettingControl, ...] = (
    Picker(
        'live_image_output_format_spinner',
        'image_output_format.live',
        tuple(fmt for fmt in OUTPUT_FORMAT_LABELS if fmt in LIVE_OUTPUT_FORMATS),
        OUTPUT_FORMAT_LABELS.__getitem__,
    ),
    Picker(
        'sequenced_image_output_format_spinner',
        'image_output_format.sequenced',
        tuple(OUTPUT_FORMAT_LABELS),
        OUTPUT_FORMAT_LABELS.__getitem__,
    ),
    Picker(
        'video_recording_format_spinner',
        'video_recording_format',
        tuple(RECORDING_FORMAT_LABELS),
        RECORDING_FORMAT_LABELS.__getitem__,
    ),
    Toggle('show_tooltips_btn', 'show_tooltips', 'SHOW_TOOLTIPS'),
)

ZSTACK: tuple[SettingControl, ...] = (
    Picker(
        'zstack_spinner',
        'zstack.position',
        tuple(ZSTACK_REFERENCE_LABELS),
        ZSTACK_REFERENCE_LABELS.__getitem__,
    ),
)

# Each Quick Enhance step's switch has the id of its setting under quick_enhance.
QUICK_ENHANCE: tuple[SettingControl, ...] = tuple(
    Toggle(step, f'quick_enhance.{step}', f'QUICK_ENHANCE_{step.upper()}')
    for step in (
        'auto_levels_enabled',
        'gamma_enabled',
        'illumination_correction_enabled',
        'denoise_enabled',
        'sharpen_enabled',
    )
)

# Every row, for the census and the guards that read the table whole.
ALL: tuple[SettingControl, ...] = ADVANCED_SETTINGS + MICROSCOPE_SETTINGS + ZSTACK + QUICK_ENHANCE
