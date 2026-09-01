# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

import modules.binning as binning
import modules.image_mode as image_mode


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
        cls, settings, labware, scope_config: dict | None = None, layer_identity=None
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
        """
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
            objective_id=settings.get('objective_id', '4x'),
            turret_config=settings.get('turret_objectives', None),
            binning_size=binning_size,
            frame_width=settings['frame']['width'],
            frame_height=settings['frame']['height'],
            acceleration_pct=settings.get('motion', {}).get('acceleration_max_pct', 100),
            stage_offset=settings.get('stage_offset', {'x': 0, 'y': 0}),
            scale_bar_enabled=settings.get('scale_bar', {}).get('enabled', False),
            capture_depth=capture_depth,
            expects_motion=expects_motion,
            expects_led=expects_led,
            high_conversion_gain=settings.get('camera', {}).get('high_conversion_gain', False),
            line_noise_reduction=settings.get('camera', {}).get('line_noise_reduction', False),
        )
