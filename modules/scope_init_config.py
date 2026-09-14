# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

import modules.binning as binning
import modules.image_mode as image_mode
from drivers.motorboard import ACCELERATION_PCT_MAX, ACCELERATION_PCT_MIN
from modules.exceptions import ConfigError
from lvp_logger import logger


def _bounded_acceleration_pct(raw: object) -> int:
    """The stored acceleration percentage, forced into what the driver accepts.

    A settings dict is not a trusted input. It can be hand-edited on disk, and
    it can be handed to a session directly instead of being read from a file,
    so neither the slider that used to be the only thing holding this in range
    nor any load-time repair is on every path that gets here. Bounding it at
    this read -- the one place stored settings become hardware commands -- is
    what makes the limit hold for a caller that never draws a GUI.

    A non-numeric value is repaired rather than raised on, because this runs
    during bring-up: refusing to start is a worse answer than starting at the
    acceleration a fresh install already uses.
    """
    try:
        val_pct = int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            f'[Settings ] acceleration_max_pct {raw!r} is not a number; '
            f'using {ACCELERATION_PCT_MAX}'
        )
        return ACCELERATION_PCT_MAX

    bounded = max(ACCELERATION_PCT_MIN, min(ACCELERATION_PCT_MAX, val_pct))
    if bounded != val_pct:
        logger.warning(
            f'[Settings ] acceleration_max_pct {val_pct} is outside '
            f'[{ACCELERATION_PCT_MIN}, {ACCELERATION_PCT_MAX}]; using {bounded}'
        )
    return bounded


@dataclass
class ScopeInitConfig:
    """Configuration bundle for Lumascope.initialize().

    Captures all scope-level hardware settings needed to go from
    "connected" to "ready-to-use".  Does NOT include per-layer camera
    settings (gain, exposure, auto-gain).

    `expects_motion` / `expects_led` reflect what the selected scope's
    `scopes.json` entry says it should have, used by `initialize()` to
    filter the partial-hardware notification (LS620 correctly has no
    motor -- don't pop a "Motor Controller missing" warning). Defaults
    are True so callers that don't supply scope_config preserve the
    pre-filter behavior.
    """

    labware: object
    objective_id: str
    turret_config: dict | None
    binning_size: int
    frame_width: int
    frame_height: int
    acceleration_pct: int
    stage_offset: dict
    scale_bar_enabled: bool
    capture_depth: int
    expects_motion: bool = True
    expects_led: bool = True
    high_conversion_gain: bool = False
    line_noise_reduction: bool = False

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        labware: object,
        scope_config: dict | None = None,
        layer_identity: object | None = None,
    ) -> 'ScopeInitConfig':
        """Build config from LVP settings dict and labware object.

        scope_config: the entry for the active scope from scopes.json
        (e.g. ``{"Focus": false, "XYStage": false, "Turret": false, ...}``).
        When provided, drives expects_motion for the partial-hardware
        notification filter.

        layer_identity: the scope's resolved layer identity snapshot.
        When provided, drives expects_led: identity carrying at least one
        LED-driving layer means an LED board is expected, so its absence
        deserves the notification. Identity outranks the scopes.json
        entry here because a unit's own config can differ from its model.

        Raises:
            ConfigError: ``frame`` or ``objective_id`` is missing. Every
                other field has a value ``initialize`` can apply harmlessly
                when absent; these two do not -- a frame the camera never
                held is silent-wrong geometry, and an objective default that
                names no shipped objective was prefix-matched to a real one
                and stamped into every saved image's scale.
        """
        missing = [key for key in ('frame', 'objective_id') if key not in settings]
        if missing:
            raise ConfigError(
                f'settings cannot configure a scope: missing {missing}; '
                'a factory-built session needs both, a file-sourced one has them'
            )
        binning_size = binning.binning_size_str_to_int(
            text=settings.get('binning', {}).get('size', '1x1')
        )
        capture_depth = image_mode.resolve_image_mode(
            image_mode.resolve_settings_image_mode(settings)
        )['capture_depth']
        if scope_config is None:
            expects_motion = True
        else:
            expects_motion = bool(
                scope_config.get('Focus')
                or scope_config.get('XYStage')
                or scope_config.get('Turret')
            )
        if layer_identity is None:
            expects_led = True
        else:
            expects_led = any(layer.led_channel for layer in layer_identity.layers)
        return cls(
            labware=labware,
            objective_id=settings['objective_id'],
            turret_config=settings.get('turret_objectives'),
            binning_size=binning_size,
            frame_width=settings['frame']['width'],
            frame_height=settings['frame']['height'],
            acceleration_pct=_bounded_acceleration_pct(
                settings.get('motion', {}).get('acceleration_max_pct', ACCELERATION_PCT_MAX)
            ),
            stage_offset=settings.get('stage_offset', {'x': 0, 'y': 0}),
            scale_bar_enabled=settings.get('scale_bar', {}).get('enabled', False),
            capture_depth=capture_depth,
            expects_motion=expects_motion,
            expects_led=expects_led,
            high_conversion_gain=settings.get('camera', {}).get('high_conversion_gain', False),
            line_noise_reduction=settings.get('camera', {}).get('line_noise_reduction', False),
        )
