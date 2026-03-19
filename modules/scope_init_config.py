# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

import modules.binning as binning


@dataclass
class ScopeInitConfig:
    """Configuration bundle for Lumascope.initialize().

    Captures all scope-level hardware settings needed to go from
    "connected" to "ready-to-use".  Does NOT include per-layer camera
    settings (gain, exposure, auto-gain).
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

    @classmethod
    def from_settings(cls, settings, labware) -> 'ScopeInitConfig':
        """Build config from LVP settings dict and labware object."""
        binning_size = binning.binning_size_str_to_int(
            text=settings.get('binning', {}).get('size', '1x1')
        )
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
        )
