# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ProtocolsAPI -- protocol-author surface: construct Protocol objects.

The protocol constructors need one thing the Protocol data class cannot
resolve for itself: where `data/tiling.json` lives. That path is a
property of the running INSTALLATION, not of any protocol, so the
constructors resolve it here and callers never pass
`tiling_configs_file_loc` by hand.

`source_path` arrives after construction rather than as a constructor
argument: the application builds the scope before it knows its own data
root. That ordering is why `register_source_path` is a separate call,
and why the constructors raise instead of guessing when it was never
made -- a wrong tiling config silently produces a protocol whose tiling
geometry does not match the instrument.

This surface BUILDS protocols; it does not run them. The runner is
`ScopeSession.create_protocol_runner()`.

It also owns the one rule that decides whether a protocol is admissible
on this scope at all -- whether the scope can put the glass it names in
the light path. That rule lives here because the facts it needs are the
scope's, and because every moment that admits a protocol (run, load,
new, navigating to a step) has to ask the SAME question. They used to
ask four different ones.
"""

from __future__ import annotations

import logging
import pathlib
import typing
from collections.abc import Iterable
from typing import TYPE_CHECKING

from modules.exceptions import ProtocolRunRefusedError

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from modules.protocol import Protocol

_api_log = logging.getLogger('LVP.api')


class ProtocolsAPI:
    """Protocol-construction sub-API on a Lumascope.

    Hosts the two public constructors (`load_protocol`, `create_protocol`)
    and the installation data root both resolve against.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope
        self._source_path = None

    def register_source_path(self, source_path) -> None:
        """Register the LVP source/data path the constructors resolve against.

        Internal session-composition wiring -- called by ScopeSession at
        construction and not part of the L2 API surface.

        Called once at startup, after the scope is constructed. Tests that
        don't drive the protocol API can skip it.

        Args:
            source_path: Path-like to the LVP source/data root.
        """
        self._source_path = source_path

    def tiling_configs_path(self) -> pathlib.Path:
        """Resolve data/tiling.json from the registered source path.

        The one owner of that path: the protocol constructors resolve it
        here, and the run engine takes it from here at run start for the
        post-run composite merge and hyperstack build, so a headless run
        reads the same tiling config the session was built with instead
        of whatever the process's script root holds. An engine seam, not
        part of the L2 API surface: a caller never needs the path itself.
        """

        if self._source_path is None:
            raise RuntimeError(
                'scope.protocols.load_protocol/create_protocol require '
                'scope.protocols.register_source_path() to have been called.'
            )
        return pathlib.Path(self._source_path) / 'data' / 'tiling.json'

    def load_protocol(self, file_path: str | pathlib.Path) -> Protocol:
        """Load a Protocol from disk.

        Wraps ``Protocol.from_file(...)`` and resolves
        ``data/tiling.json`` from the registered source_path.

        Args:
            file_path: Path to the protocol file.

        Returns:
            Protocol: The loaded Protocol instance.

        Raises:
            ProtocolFormatError: On format issues (same surface as
                Protocol.from_file), or when the file names a plate this
                installation's labware catalogue does not have. Refused
                here, by name, before any object exists: a protocol whose
                plate the scope cannot be on must never be adopted.
            ProtocolRunRefusedError: The file names glass this scope
                cannot put in the light path, with the reason a run would
                give for the same file. Refused for the same reason the
                plate is: a caller must not be handed a protocol it can
                edit, navigate and save but never perform.
        """
        from modules import labware_loader
        from modules.protocol import Protocol

        protocol = Protocol.from_file(
            file_path=file_path,
            tiling_configs_file_loc=self.tiling_configs_path(),
            led_max_ma=self._scope.capabilities.led_max_ma,
            wellplate_loader=labware_loader.WellPlateLoader(source_path=self._source_path),
        )
        # After the parse, so a file that is not a protocol at all is
        # answered as that rather than as a turret problem, and before the
        # return, so no caller ever holds an inadmissible protocol.
        self.refuse_unaddressable_objectives(protocol.steps()['Objective'].to_list())
        return protocol

    def create_protocol(
        self,
        *,
        config: dict | None = None,
        input_config: dict | None = None,
        empty_config: dict | None = None,
    ) -> Protocol:
        """Construct a Protocol in-memory.

        Three modes (pass exactly one):
          - config={...}: full config dict passed to Protocol() directly.
          - input_config={...}: partial config (positions, layer_configs,
            etc.); routed through Protocol.from_config which fills defaults.
          - empty_config={...}: labware, period, duration, frame_dimensions
            and binning_size for an empty-steps protocol, which needs no
            objective; routed through Protocol.create_empty.
        tiling_configs_file_loc is resolved internally from the registered
        source_path.

        Args:
            config: Full config dict, or None.
            input_config: Partial config dict, or None.
            empty_config: Empty-steps config dict, or None.

        Returns:
            Protocol: Newly constructed Protocol instance.

        Raises:
            ValueError: If exactly one of config/input_config/empty_config
                was not provided.
        """
        from modules.protocol import Protocol

        provided = sum(1 for x in (config, input_config, empty_config) if x is not None)
        if provided != 1:
            raise ValueError(
                'create_protocol(): pass exactly one of config=, input_config=, or empty_config='
            )
        tcfg = self.tiling_configs_path()
        if input_config is not None:
            return Protocol.from_config(
                input_config=input_config,
                tiling_configs_file_loc=tcfg,
                capabilities=self._scope.capabilities,
            )
        if empty_config is not None:
            return Protocol.create_empty(
                config=empty_config,
                tiling_configs_file_loc=tcfg,
                capabilities=self._scope.capabilities,
            )
        return Protocol(
            tiling_configs_file_loc=tcfg,
            config=config,
        )

    def add_step(
        self,
        protocol: Protocol,
        *,
        layer_configs: dict,
        stim_configs: dict,
        plate_position: dict,
        objective_id: str | None,
        channel_order: list[str] | None = None,
        before_step: int | None = None,
        after_step: int | None = None,
    ) -> list[str]:
        """Add one step per acquiring layer to ``protocol`` at ``plate_position``.

        The GUI's Add Step and a script's add are this one call. A layer
        whose ``acquire`` is None contributes no step, and when no layer
        acquires the add is refused rather than silently doing nothing:
        the click used to return bare, so the user saw nothing happen and
        nothing said why. On a turret scope the current slot must name an
        objective, or the step would record glass the scope cannot say it
        has.

        ``channel_order`` names the layers whose steps come first, in that
        order; layers it does not name follow in the order given. Each
        layer's step is placed after the previous layer's, so the protocol
        reads in that order -- the composite path associates channels by
        their step order.

        ``objective_id`` is the active objective, or None when no one can say
        which objective is in the light path.

        Returns the inserted step names, in protocol order.

        Raises:
            ProtocolRunRefusedError: no layer acquires, the turret's current
                slot has no objective, or the active objective is unknown.
                Logged and notified once.
            ProtocolError: an impossible ``before_step`` / ``after_step``
                (raised by the protocol).
        """
        if self._scope.capabilities.has_turret and (
            not self._scope.motion.is_current_turret_position_objective_set()
        ):
            self._refuse(
                reason='turret_objective_unset',
                title='Protocol Add Step Error',
                message=(
                    'Cannot add step to protocol. Please set objective for current turret position.'
                ),
            )
        if objective_id is None:
            self._refuse(
                reason='objective_unknown',
                title='Protocol Add Step Error',
                message=(
                    'Cannot add step: the objective in the light path is unknown, so the '
                    'step could not say which objective it was taken with.'
                ),
            )
        if not any(cfg['acquire'] is not None for cfg in layer_configs.values()):
            self._refuse(
                reason='no_acquiring_layer',
                title='Protocol Add Step Error',
                message=(
                    'Cannot add step: no channel is set to acquire. '
                    'Set a channel to Image or Video first.'
                ),
            )

        ordered = [layer for layer in (channel_order or []) if layer in layer_configs]
        ordered += [layer for layer in layer_configs if layer not in ordered]
        stim_configs = self._stim_configs_with_invalid_channels_disabled(stim_configs)

        names: list[str] = []
        for layer in ordered:
            layer_config = layer_configs[layer]
            if layer_config['acquire'] is None:
                continue
            name = protocol.insert_step(
                step_name=None,
                layer=layer,
                layer_config=layer_config,
                stim_configs=stim_configs,
                plate_position=plate_position,
                objective_id=objective_id,
                before_step=before_step,
                after_step=after_step,
            )
            names.append(name)
            # The next layer goes after this one. Handing every layer the
            # same index puts each new step AT that index, which reads the
            # channels back in reverse. The index is arithmetic, not a
            # name lookup: a loaded file may carry duplicate names.
            inserted_at = before_step if before_step is not None else after_step + 1
            before_step, after_step = None, inserted_at
        return names

    @staticmethod
    def _stim_configs_with_invalid_channels_disabled(stim_configs: dict) -> dict:
        """Disable an enabled stim channel whose frequency or current is not positive.

        Warned and disabled rather than refused: the stim feature is not
        in use yet (Eric, 2026-09-21), so its range has no owner and a
        refusal would be a rule for nobody. When it is used, the range is
        refused at this boundary, never disabled.
        """
        for stim_color, sc in stim_configs.items():
            if not isinstance(sc, dict) or not sc.get('enabled', False):
                continue
            freq = sc.get('frequency', 0)
            if not isinstance(freq, (int, float)) or freq <= 0:
                _api_log.warning(
                    f'[API] Stim channel {stim_color}: frequency {freq} Hz is invalid '
                    f'(must be > 0). Disabling channel.'
                )
                sc['enabled'] = False
            exp = sc.get('exposure', 0)
            if isinstance(exp, (int, float)) and exp == 0 and sc.get('enabled', False):
                _api_log.warning(
                    f'[API] Stim channel {stim_color}: exposure is 0. '
                    f'This may produce no visible pulses.'
                )
            illum = sc.get('illumination_ma', 0)
            if isinstance(illum, (int, float)) and illum <= 0 and sc.get('enabled', False):
                _api_log.warning(
                    f'[API] Stim channel {stim_color}: illumination {illum} mA is invalid '
                    f'(must be > 0). Disabling channel.'
                )
                sc['enabled'] = False
        return stim_configs

    def refuse_unaddressable_objectives(self, objective_ids: Iterable[str]) -> None:
        """Refuse unless this scope can put every objective named here in the light path.

        The one rule behind every moment that admits a protocol: starting
        a run, loading a file, building a new protocol, and navigating to
        a single step. Each of those carried its own version of it and the
        versions disagreed, so a protocol the run refused could still be
        loaded, displayed and navigated -- and the step pointer moved
        before anything had been asked at all.

        With a turret, an objective is addressable when a slot is assigned
        to it. An unassigned turret therefore addresses NOTHING and
        refuses everything. That is deliberate and it is the part that
        changed: the exemption this replaces existed to keep a fresh
        install runnable, but the startup objective question already does
        that -- it assigns the current position before any protocol loads,
        behind a popup that cannot be dismissed. A scope arriving here
        with every slot empty is one whose slots were CLEARED.

        Without a turret, the objective in the light path is whichever one
        is mounted and nothing can change it, so a protocol may name only
        one.

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this rule by loading a protocol or starting a run, both of which
        ask it here and deliver the same refusal. A separate public
        "may I?" would be a second way to ask one question, and its answer
        could go stale between the asking and the doing.

        Args:
            objective_ids: The objective ids that have to be addressable
                -- every step's when admitting a whole protocol, one
                step's when navigating to it. Order and duplicates are
                irrelevant; naming nothing is admitted, because a protocol
                with no steps is the empty-protocol refusal's to answer
                and not this one's. Compared as given: whether an id names
                real glass is validation's question, and it is asked
                first.

        Raises:
            ProtocolRunRefusedError: This scope cannot address them. It
                has been logged and shown to the user before it is raised,
                the way the protocol builder's refusal is: this runs
                before any run exists, so no run funnel has seen it.
        """
        named = set(objective_ids)

        if self._scope.capabilities.has_turret:
            carried = {
                objective
                for objective in self._scope.runtime_state.get_turret_config().values()
                if objective is not None
            }
            if not named.issubset(carried):
                self._refuse(
                    reason='turret_objectives_unassigned',
                    title='Turret Configuration Required',
                    message=(
                        'This protocol uses objectives that are not assigned to turret '
                        f'positions.\n\nIt needs: {self._render(named)}\n'
                        f'The turret carries: {self._render(carried) or "None"}\n\n'
                        'Assign the missing objectives in Objective Control > Turret.'
                    ),
                )
        elif len(named) > 1:
            self._refuse(
                reason='objectives_require_turret',
                title='One Objective At A Time',
                message=(
                    'This protocol changes objective between steps, and this scope has '
                    'no way to change it: the objective in the light path is the one '
                    f'that is mounted.\n\nIt names: {self._render(named)}\n\n'
                    'Use a protocol that names a single objective.'
                ),
            )

    @staticmethod
    def _render(objective_ids: set[object]) -> str:
        """Objective ids as a stable, readable list for a refusal message.

        Sorted AS STRINGS on purpose. Objective validation does not run on
        the load path, so a hand-edited file can put a blank cell or a
        float NaN in this set, and sorting those against real ids raises
        TypeError -- which would leave the API boundary as a crash in
        place of the refusal the caller is owed.
        """
        return ', '.join(sorted(map(str, objective_ids)))

    def _refuse(self, reason: str, title: str, message: str) -> typing.NoReturn:
        """Log, notify once, and raise the typed refusal.

        The single funnel this surface's refusals route through, so one
        refusal is always one log line, one notification and one typed
        exception. Each caller passes its reason as a LITERAL rather than
        through a variable: the refusal vocabulary is censused by reading
        that argument out of the source, and a name in its place makes the
        refusal invisible to the census, which is how a reason ships with
        no coverage.

        WARNING, not ERROR: a refusal is a designed outcome, and an error
        log is where someone goes to find what went wrong.
        """
        from modules.notification_center import REFUSAL_OPERATION_KEY, notifications

        _api_log.warning(f'[API] Protocol refused ({reason}): {message}')
        # Solicited: a refusal answers something the caller just asked
        # for, so it must reach the user even while a run is in flight.
        notifications.warning(
            'Protocol',
            title,
            message,
            solicited=True,
            operation_key=REFUSAL_OPERATION_KEY,
        )
        raise ProtocolRunRefusedError(reason=reason, title=title, message=message)
