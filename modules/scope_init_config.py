# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

from dataclasses import dataclass

import modules.binning as binning
import modules.layer_record as layer_record


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

        ``settings`` is a prepared dict (``settings_init.prepare_loaded``,
        which every session's settings went through): every key is
        present and every value is inside the writer's range, so each is
        indexed here, never hedged.
        """
        acceleration_pct = settings['motion']['acceleration_max_pct']
        binning_size = binning.binning_size_str_to_int(text=settings['binning']['size'])
        expects_motion = layer_record.entry_expects_motion(scope_config)
        if layer_identity is None:
            expects_led = True
        else:
            expects_led = any(layer.led_channel for layer in layer_identity.layers)
        return cls(
            turreted=turreted,
            binning_size=binning_size,
            frame_width=settings['frame']['width'],
            frame_height=settings['frame']['height'],
            acceleration_pct=acceleration_pct,
            image_mode=settings['image_mode'],
            expects_motion=expects_motion,
            expects_led=expects_led,
            high_conversion_gain=settings['camera']['high_conversion_gain'],
            line_noise_reduction=settings['camera']['line_noise_reduction'],
        )
