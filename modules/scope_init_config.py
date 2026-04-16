# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

import modules.binning as binning


@dataclass
class ScopeInitConfig:
    """Configuration bundle for Lumascope.initialize().

    Captures all scope-level hardware settings needed to go from
    "connected" to "ready-to-use".  Does NOT include per-layer camera
    settings (gain, exposure, auto-gain).

    `expects_motion` / `expects_led` reflect what the selected scope's
    `scopes.json` entry says it should have, used by `initialize()` to
    filter the partial-hardware notification (LS620 correctly has no
    motor — don't pop a "Motor Controller missing" warning). Defaults
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
    expects_motion: bool = True
    expects_led: bool = True

    @classmethod
    def from_settings(cls, settings, labware, scope_config: dict | None = None) -> 'ScopeInitConfig':
        """Build config from LVP settings dict and labware object.

        scope_config: the entry for the active scope from scopes.json
        (e.g. ``{"Focus": false, "XYStage": false, "Turret": false,
        "Layers": {...}}``). When provided, drives expects_motion /
        expects_led for the partial-hardware notification filter.
        """
        binning_size = binning.binning_size_str_to_int(
            text=settings.get('binning', {}).get('size', '1x1')
        )
        if scope_config is None:
            expects_motion = True
            expects_led = True
        else:
            expects_motion = bool(
                scope_config.get('Focus')
                or scope_config.get('XYStage')
                or scope_config.get('Turret')
            )
            layers = scope_config.get('Layers', {})
            expects_led = bool(layers) and any(layers.values())
        return cls(
            labware=labware,
            objective_id=settings.get('objective_id', '4x'),
            turret_config=settings.get('turret_objectives', None),
            binning_size=binning_size,
            frame_width=settings['frame']['width'] * binning_size,
            frame_height=settings['frame']['height'] * binning_size,
            acceleration_pct=settings.get('motion', {}).get('acceleration_max_pct', 100),
            stage_offset=settings.get('stage_offset', {'x': 0, 'y': 0}),
            scale_bar_enabled=settings.get('scale_bar', {}).get('enabled', False),
            expects_motion=expects_motion,
            expects_led=expects_led,
        )
