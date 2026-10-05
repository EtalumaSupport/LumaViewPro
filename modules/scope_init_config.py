# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

import modules.binning as binning
import modules.image_mode as image_mode
import modules.layer_record as layer_record
from modules.exceptions import ConfigError
from modules.lumascope_api._constants import (
    is_turret_slot,
    refuse_acceleration_pct,
)
from lvp_logger import logger


@dataclass
class ScopeInitConfig:
    """Configuration bundle for Lumascope.initialize().

    Captures all scope-level hardware settings needed to go from
    "connected" to "ready-to-use".  Does NOT include per-layer camera
    settings (gain, exposure, auto-gain).

    `expects_motion` / `expects_led` reflect what the selected scope's
    `scopes.json` entry says it should have. A scope that expects no motor
    board (an LS620 has none) is complete without one: `initialize()` does
    not warn that it is missing, the connection check does not require it,
    and startup does not home it. Defaults are True so callers that don't
    supply scope_config are held to every board.

    The labware, stage offset, turret map, selected objective and scale
    bar are not here: the scope reads them from the session's settings,
    their one store, whenever it acts on them.
    """

    # The session's one has-a-turret answer. Required, never defaulted: a
    # turreted scope treated as turretless would answer with its stored
    # objective instead of the one in the light path.
    turreted: bool
    # The saved turret position: the slot a person last turned to, which
    # the slot lookup prefers when two slots carry one objective. None when
    # nothing usable was saved.
    preferred_turret_slot: int | None
    binning_size: int
    frame_width: int
    frame_height: int
    acceleration_pct: int
    # The saved image mode, which names the capture depth bring-up applies;
    # a camera without that depth starts in 8-bit and says so.
    image_mode: str
    expects_motion: bool = True
    expects_led: bool = True
    high_conversion_gain: bool = False
    line_noise_reduction: bool = False

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        scope_config: dict | None = None,
        layer_identity: object | None = None,
        *,
        turreted: bool,
    ) -> 'ScopeInitConfig':
        """Build config from the LVP settings dict.

        turreted: the session's one has-a-turret answer
        (``ScopeSession.scope_has_turret``). On a turreted scope the stored
        ``objective_id`` is not required: the objective is the slot's
        assignment, derived when asked. It is never carried: the scope
        reads the selected objective from the settings.

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
            ConfigError: ``frame``, ``binning``, ``stage_offset``,
                ``turret_objectives``, ``scale_bar.enabled`` or
                ``motion.acceleration_max_pct`` is missing, or
                ``objective_id`` is missing on a scope with no turret. Every
                other field has a value ``initialize`` can apply harmlessly
                when absent; these do not -- a frame the
                camera never held is silent-wrong geometry, the binning is the
                other half of that geometry and bring-up stores what the
                camera delivered at into the same slot, an invented stage
                offset puts every plate position somewhere else on the stage,
                an objective default that names no shipped objective was
                prefix-matched to a real one and stamped into every saved
                image's scale, an invented acceleration limit commands the
                motors at a limit nobody chose, and the scope reads the turret
                map and the scale bar from the settings at every use, so one
                missing would fail every objective read or capture rather
                than once, here. A present one no board may be
                given is refused too, before bring-up commands anything: a
                dict handed to a session never went through the load that
                replaces such a value.
        """
        required = ('frame', 'binning', 'stage_offset', 'turret_objectives', 'scale_bar', 'motion')
        if not turreted:
            required += ('objective_id',)
        missing = [key for key in required if key not in settings]
        if 'motion' in settings and 'acceleration_max_pct' not in settings['motion']:
            missing.append('motion.acceleration_max_pct')
        if 'scale_bar' in settings and 'enabled' not in settings['scale_bar']:
            missing.append('scale_bar.enabled')
        if missing:
            raise ConfigError(
                f'settings cannot configure a scope: missing {missing}; '
                'a factory-built session needs them, a file-sourced one has them'
            )
        acceleration_pct = settings['motion']['acceleration_max_pct']
        try:
            refuse_acceleration_pct(acceleration_pct)
        except ValueError as e:
            raise ConfigError(f'settings cannot configure a scope: {e}') from e
        binning_size = binning.binning_size_str_to_int(text=settings['binning']['size'])
        expects_motion = layer_record.entry_expects_motion(scope_config)
        preferred_turret_slot = settings.get('turret_position')
        if preferred_turret_slot is not None and not is_turret_slot(preferred_turret_slot):
            # A preference, not a position: a value that names no slot
            # leaves no preference, which the lookup already handles, and
            # is said here so a hand-edited file is visible in the record.
            logger.warning(
                f'[Session  ] saved turret_position {preferred_turret_slot!r} is not a slot '
                '1-4; no preferred slot'
            )
            preferred_turret_slot = None
        if layer_identity is None:
            expects_led = True
        else:
            expects_led = any(layer.led_channel for layer in layer_identity.layers)
        return cls(
            turreted=turreted,
            preferred_turret_slot=preferred_turret_slot,
            binning_size=binning_size,
            frame_width=settings['frame']['width'],
            frame_height=settings['frame']['height'],
            acceleration_pct=acceleration_pct,
            image_mode=image_mode.resolve_settings_image_mode(settings),
            expects_motion=expects_motion,
            expects_led=expects_led,
            high_conversion_gain=settings.get('camera', {}).get('high_conversion_gain', False),
            line_noise_reduction=settings.get('camera', {}).get('line_noise_reduction', False),
        )
