# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""ProtocolsAPI -- protocol-author surface: construct Protocol objects.

The protocol constructors need one thing the Protocol data class cannot
resolve for itself: where `data/tiling.json` lives. That path is a
property of the running INSTALLATION, not of any protocol, so the
constructors resolve it here and callers never pass
`tiling_configs_file_loc` by hand.

The data folder is the scope's own, given at its construction; the
tiling config and the labware catalogue a loaded protocol is judged
against both come from it, so a protocol is never built against another
installation's files.

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

import dataclasses
import logging
import math
import pathlib
import typing
from collections.abc import Iterable
from typing import TYPE_CHECKING

import modules.common_utils as common_utils
import modules.path_utils as path_utils
from modules.finite_number import refuse_unless_finite_number
from modules.coord_transformations import AxisLimits, PlateLocation, StageOffset
from modules.exceptions import (
    ArgumentRefusedError,
    PositionOutOfRangeError,
    ProtocolRunRefusedError,
    ProtocolStepsInvalidNotice,
    unknown_positions_sentence,
)
from modules.lumascope_api.imaging import camera_range_words
from modules.lumascope_api.runtime_state import slots_holding
from modules.api_surface import api, api_fields

if TYPE_CHECKING:
    import pandas as pd

    from modules.lumascope_api._lumascope import Lumascope
    from modules.protocol import Protocol
    from modules.tiling_config import TilingConfig

_api_log = logging.getLogger('LVP.api')


@api_fields('turret_slot', 'x', 'y', 'z')
@dataclasses.dataclass(frozen=True)
class StepTargets:
    """Where a protocol step puts the scope, in the frames the motors take.

    ``turret_slot`` is None on a scope with no turret. ``x`` and ``y`` are
    stage micrometres on the protocol's own plate; ``z`` is the step's. An
    axis the scope has no motor for is None (``stage_targets``).
    """

    turret_slot: int | None
    x: float | None
    y: float | None
    z: float | None


@api_fields('objective_id', 'steps', 'reassigned', 'choices')
@dataclasses.dataclass(frozen=True)
class TurretSlotQuestion:
    """Which turret slot some steps of a protocol mean: what to ask the person.

    Returned by ``ProtocolsAPI.turret_slot_questions``, one per objective
    whose steps cannot say which slot they mean: the objective is in two
    or more slots and the steps recorded none (a protocol from before steps
    recorded their slot), or the slot a step recorded now holds another
    objective. The answer goes back through
    ``ProtocolsAPI.point_steps_at_turret_slot``; until then a run and going
    to one of these steps are refused.

    Attributes:
        objective_id: The objective the steps name.
        steps: The indices of the steps the answer points, ascending.
        reassigned: Those of ``steps`` whose recorded slot now holds
            another objective; the others recorded no slot.
        choices: The slots that hold ``objective_id`` now, ascending.
    """

    objective_id: str
    steps: tuple[int, ...]
    reassigned: tuple[int, ...]
    choices: tuple[int, ...]


class ProtocolsAPI:
    """Protocol-construction sub-API on a Lumascope.

    Hosts the two public constructors (`load_protocol`, `create_protocol`)
    and the installation data root both resolve against.
    """

    def __init__(self, scope: Lumascope) -> None:
        self._scope = scope

    def tiling_configs_path(self) -> pathlib.Path:
        """Resolve data/tiling.json from the scope's data folder.

        The one owner of that path: the protocol constructors resolve it
        here, and the run engine takes it from here at run start for the
        post-run composite merge and hyperstack build, so a headless run
        reads the same tiling config the session was built with instead
        of whatever the process's script root holds. An engine seam, not
        part of the L2 API surface: a caller never needs the path itself.
        """
        return pathlib.Path(self._scope.source_path) / 'data' / 'tiling.json'

    @api(reads=True)
    def tiling_config(self) -> TilingConfig:
        """The tiling grids this installation offers: the scope's copy, read at bring-up.

        What a caller needs to choose a grid: ``available`` lists the
        labels ``create_protocol`` and ``Protocol.apply_tiling`` accept,
        and ``default`` is the one to preselect; ``Protocol.tiling()``
        names the grid a protocol's steps already carry. Read once with the
        installation's other catalogues, so a tiling.json edited while the
        program runs is seen at the next start; a file no reader can use
        refuses the bring-up (``InstallationFileError``, naming it).
        """
        return self._scope._tiling_config

    def load_protocol(self, file_path: str | pathlib.Path) -> Protocol:
        """Load a Protocol from disk.

        Wraps ``Protocol.from_file(...)`` and resolves
        ``data/tiling.json`` from the scope's data folder.

        Args:
            file_path: Path to the protocol file.

        Returns:
            Protocol: The loaded Protocol instance.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                ``file_path``; ``'not_a_file'``, a folder is.
            ProtocolNotLoadedError: The file is there and cannot be read.
            ProtocolFormatError: On format issues (same surface as
                Protocol.from_file), or when the file names a plate this
                installation's labware catalogue does not have. Refused
                here, by name, before any object exists: a protocol whose
                plate the scope cannot be on must never be adopted.
            ProtocolRunRefusedError: The file names glass this scope
                cannot put in the light path, or has a step on a layer this
                scope does not have, with the reason a run would give for
                the same file. Refused for the same reason the plate is: a
                caller must not be handed a protocol it can edit, navigate
                and save but never perform.
        """
        from modules.protocol import Protocol

        protocol = Protocol.from_file(
            file_path=path_utils.existing_file(file_path),
            tiling_configs_file_loc=self.tiling_configs_path(),
            wellplate_loader=self._scope.wellplate_loader,
        )
        # After the parse, so a file that is not a protocol at all is
        # answered as that rather than as a turret problem, and before the
        # return, so no caller ever holds an inadmissible protocol.
        self.refuse_unaddressable_objectives(protocol.steps()['Objective'].to_list())
        self.refuse_absent_layers(protocol)
        self._fill_turret_slots(protocol)
        # Unsolicited, as the loader's duplicate-filename notice is: a load
        # may be the startup adoption, which nobody asked for.
        self._report_invalid_steps(protocol, solicited=False)
        return protocol

    def _report_invalid_steps(self, protocol: Protocol, *, solicited: bool) -> None:
        """Tell the caller once when the protocol holds a step the run will refuse.

        A loaded file may carry a field the run gate rejects, and an edit
        composed from live settings may too; neither is refused here, so
        the step can be fixed in the app, and the run start is where the
        refusal lives. A notice, like the duplicate-filename one the loader
        reports: reported, never raised.
        """
        from modules.notification_center import notifications

        errors = protocol.validate_steps(
            self._scope.objectives, led_max_ma=self._scope.capabilities.led_max_ma
        )
        if errors:
            notifications.report_outcome(
                ProtocolStepsInvalidNotice(errors=errors), solicited=solicited, category='Protocol'
            )

    @api
    def create_protocol(
        self,
        *,
        input_config: dict | None = None,
        empty_config: dict | None = None,
    ) -> Protocol:
        """Construct a Protocol in-memory.

        Two sources (pass exactly one):
          - input_config={...}: positions, layer_configs, the z-stack, the
            tiling, the schedule and the frame; routed through
            Protocol.from_config.
          - empty_config={...}: labware, period, duration, frame_dimensions
            and binning_size for an empty-steps protocol, which needs no
            objective; routed through Protocol.create_empty.
        tiling_configs_file_loc is resolved internally from the scope's data
        folder. The configuration is judged whole before anything is built:
        every key the build reads, present and of its kind.

        Args:
            input_config: A protocol's configuration, or None.
            empty_config: An empty protocol's configuration, or None.

        Returns:
            Protocol: Newly constructed Protocol instance.

        Raises:
            ArgumentRefusedError: ``'protocol_source_ambiguous'``, not
                exactly one source was given; ``'missing_key'``,
                ``'wrong_kind'``, ``'not_a_number'``,
                ``'zstack_reference_unknown'``, ``'acquire_mode_unknown'``
                or ``'layer_unknown'``, a key of the configuration, named by
                its path in ``argument``.
            CatalogueNameRefusedError: the configuration names a plate or an
                objective the installation's catalogues do not hold.
            ProtocolRunRefusedError: the configuration cannot be built --
                ``objective_not_given``, ``focus_not_given``,
                ``overlap_out_of_range``, ``tiling_unknown`` -- or the stack
                it asks for cannot be, as ``apply_zstacking`` refuses it
                (``zstack_not_configured``, ``positions_unreachable``,
                ``zslices_outside_travel``); reported once.

        A configuration with ``use_zstacking`` is built flat and then
        stacked by ``Protocol.apply_zstacking``, the one z-stack build, so
        a New protocol's stacks are grouped, focused and refused as Apply's
        and Add's are.
        """
        from modules.protocol import Protocol, refuse_unless_buildable_config

        sources = {'input_config': input_config, 'empty_config': empty_config}
        given = [name for name, value in sources.items() if value is not None]
        if len(given) != 1:
            raise ArgumentRefusedError(
                'protocol_source_ambiguous',
                argument='input_config',
                value=len(given),
                offered=tuple(sources),
            )
        refuse_unless_buildable_config(
            input_config if input_config is not None else empty_config,
            given[0],
            empty=empty_config is not None,
        )
        tcfg = self.tiling_configs_path()
        if input_config is not None:
            protocol = Protocol.from_config(
                input_config=input_config,
                tiling_configs_file_loc=tcfg,
                capabilities=self._scope.capabilities,
                objective_helper=self._scope.objective_helper,
                wellplate_loader=self._scope.wellplate_loader,
            )
            if input_config['use_zstacking']:
                protocol.apply_zstacking(input_config['zstack_params'], self._travel_limits())
            self._stamp_turret_slots(protocol)
            return protocol
        return Protocol.create_empty(
            config=empty_config,
            tiling_configs_file_loc=tcfg,
            capabilities=self._scope.capabilities,
            objective_helper=self._scope.objective_helper,
            wellplate_loader=self._scope.wellplate_loader,
        )

    def add_step(
        self,
        protocol: Protocol,
        *,
        layer_configs: dict,
        stim_configs: dict,
        plate_position: PlateLocation,
        objective_id: str | None,
        channel_order: list[str] | None = None,
        before_step: int | None = None,
        after_step: int | None = None,
        zstack_params: dict | None = None,
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
        reads in that order. The steps of one add share one channel group
        id, and that, not their order, is what makes them one composite:
        two adds at one position are two composites.

        ``objective_id`` is the active objective, or None when no one can say
        which objective is in the light path.

        ``before_step`` or ``after_step`` places the steps; with neither
        they follow the last step, which is what a script building a
        protocol one add at a time means by "add".

        ``zstack_params`` (``range``, ``step_size``, ``z_reference``) makes
        each added step a z-stack around the live Z, built as
        ``apply_zstacking`` builds one; the protocol's other steps are left as
        they are, where they are. The stack is judged before anything is
        added, so a stack that cannot be built adds nothing.

        Returns the inserted step names, in protocol order: each stack's
        slices when stacked.

        Raises:
            ProtocolRunRefusedError: an axis does not know its position, no
                layer acquires, the turret's current slot has no objective,
                or the active objective is unknown. Logged and notified once.
                With ``zstack_params``, also what ``apply_zstacking`` refuses.
            ArgumentRefusedError: with ``zstack_params``, as
                ``apply_zstacking``.
            ProtocolError: an impossible ``before_step`` / ``after_step``,
                or both given (raised by the protocol).
        """
        from modules.protocol import Protocol

        self._refuse_unrecordable_step(verb='add', objective_id=objective_id)
        self.refuse_no_acquiring_layer(layer_configs)

        ordered = [layer for layer in (channel_order or []) if layer in layer_configs]
        ordered += [layer for layer in layer_configs if layer not in ordered]
        stim_configs = self._stim_configs_with_invalid_channels_disabled(stim_configs)
        if before_step is None and after_step is None:
            after_step = protocol.num_steps() - 1
        acquiring = [layer for layer in ordered if Protocol.layer_acquires(layer_configs[layer])]
        if zstack_params is not None:
            Protocol.refuse_unbuildable_zstack(
                zstack_params,
                self._travel_limits(),
                [(f'the new {layer} step', plate_position.z_um) for layer in acquiring],
            )
        first_added = before_step if before_step is not None else after_step + 1

        names: list[str] = []
        channel_group_id = protocol.new_channel_group_id()
        turret_slot = self._live_turret_slot()
        for layer in acquiring:
            layer_config = layer_configs[layer]
            name = protocol.insert_step(
                step_name=None,
                layer=layer,
                layer_config=layer_config,
                stim_configs=stim_configs,
                plate_position=plate_position,
                objective_id=objective_id,
                channel_group_id=channel_group_id,
                turret_slot=turret_slot,
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
        if zstack_params is not None:
            steps_before = protocol.num_steps()
            protocol.apply_zstacking(
                zstack_params,
                self._travel_limits(),
                only=range(first_added, first_added + len(names)),
            )
            added = len(names) + protocol.num_steps() - steps_before
            names = [
                protocol.step(idx=idx)['Name'] for idx in range(first_added, first_added + added)
            ]
        self._report_invalid_steps(protocol, solicited=True)
        return names

    def _refuse_channel_already_in_add(self, protocol: Protocol, step_idx: int, layer: str) -> None:
        """Refuse giving step ``step_idx`` a channel another step of its add takes.

        An add's steps are one composite with one image per channel, so a
        second step of one channel would leave the composite two images for
        one plane, refused only after the run has captured both. A step
        recorded before adds were (group -1) has no add to check.
        """
        group = protocol.step(idx=step_idx)['Channel Group ID']
        if group == -1:
            return
        steps = protocol.steps().reset_index(drop=True)
        same_add = steps[(steps['Channel Group ID'] == group) & (steps.index != step_idx)]
        if layer in set(same_add['Color']):
            self._refuse(
                reason='channel_already_in_add',
                message=(
                    f'Another step added with this one already takes {layer}, and steps '
                    f'added together are one composite with one image per channel. '
                    f'Open the drawer of the channel this step takes, or add {layer} '
                    'as a new step.'
                ),
            )

    def update_step(
        self,
        protocol: Protocol,
        step_idx: int,
        *,
        layer: str,
        layer_configs: dict,
        stim_configs: dict,
        plate_position: PlateLocation,
        objective_id: str | None,
        label: str | None = None,
    ) -> str:
        """Rewrite step ``step_idx`` of ``protocol`` from ``layer`` at ``plate_position``.

        The GUI's Update Step and a script's update are this one call, and
        it is refused for the same reasons an add is: a step is a saved
        position and the objective it was taken with, whichever button
        saved it.

        ``layer`` is the channel the caller is editing. When that layer's
        stim config is enabled the edit is a stim edit, not a channel
        change, so the step keeps the channel it already acquires.
        ``label`` renames the step; None keeps its label.

        On an axis this scope has no motor for, the step keeps the position
        it has: the scope measured nothing there, so a loaded step's X/Y
        survive an update on a scope without a stage.

        Returns the step's name after the update.

        Raises:
            ProtocolRunRefusedError: an axis does not know its position, the
                turret's current slot has no objective, the active objective
                is unknown, or the layer the step takes is not set to
                acquire (reason ``no_acquiring_layer``, as an add), or
                another step of the same add already takes that channel
                (reason ``channel_already_in_add``). Logged and notified
                once; the step is unchanged.
            ProtocolError: ``step_idx`` is not a step of ``protocol``
                (raised by the protocol), judged before anything else: an
                empty protocol has no step to update, whatever the layer.
        """
        from modules.protocol import Protocol

        step = protocol.step(idx=step_idx)
        self._refuse_unrecordable_step(verb='update', objective_id=objective_id)

        stim_config = layer_configs[layer].get('stim_config')
        if stim_config is not None and stim_config['enabled']:
            layer = step['Color']

        if not Protocol.layer_acquires(layer_configs[layer]):
            self._refuse(
                reason='no_acquiring_layer',
                message=(
                    f'{layer} is not set to acquire, so there is no step to make. '
                    f'Set {layer} to Image or Video first.'
                ),
            )

        self._refuse_channel_already_in_add(protocol, step_idx, layer)

        axes = self._scope.capabilities.axes
        has_xy = 'X' in axes and 'Y' in axes
        plate_position = PlateLocation(
            x_mm=plate_position.x_mm if has_xy else step['X'],
            y_mm=plate_position.y_mm if has_xy else step['Y'],
            z_um=plate_position.z_um if 'Z' in axes else step['Z'],
        )

        protocol.modify_step(
            step_idx=step_idx,
            label=label,
            layer=layer,
            layer_config=layer_configs[layer],
            stim_configs=self._stim_configs_with_invalid_channels_disabled(stim_configs),
            plate_position=plate_position,
            objective_id=objective_id,
            turret_slot=self._live_turret_slot(),
        )
        self._report_invalid_steps(protocol, solicited=True)
        return protocol.step(idx=step_idx)['Name']

    @api(reads=True)
    def focus_z(self, *, then: str) -> float:
        """The live Z, as a focus to save into a layer or a step.

        Args:
            then: What the user does once Z knows its position, ending the
                refusal (e.g. ``'save the focus'``).

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable`` -- this scope
                has no Z axis, so there is no focus to save; the position
                cache would answer 0. Logged and notified once.
            AxisStateUnknownError: Z lost its reference, so the number it
                answers is the last one it reported. Reported once by the
                motion API.
        """
        self._refuse_no_focus_axis(then)
        self._scope.motion.refuse_unknown_positions(('Z',), recording=True, then=then)
        return self._scope.motion.get_current_position('Z')

    def check_focus_z(self, z_um: float, *, then: str) -> float:
        """A given Z, as a focus to save into a layer: one this scope's Z can reach.

        A layer's focus is where every new step of the layer is born, so a
        number no move could reach would be refused at every step made from
        it. This says only that the number is such a Z; where it came from is
        the caller's.

        Args:
            z_um: The Z, in micrometres.
            then: What the user does once the scope has a Z, ending the
                refusal (e.g. ``'save the focus'``).

        Returns:
            ``z_um`` as a float.

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable`` -- this scope
                has no Z axis. Logged and notified once.
            ArgumentRefusedError: ``'not_a_number'``, ``z_um`` is not a
                finite number.
            PositionOutOfRangeError: ``z_um`` lies outside Z's travel.
        """
        self._refuse_no_focus_axis(then)
        refuse_unless_finite_number(z_um, 'z_um')
        z = float(z_um)
        limits = self._scope.motion.get_axis_limits('Z')
        low, high = (-math.inf, math.inf) if limits is None else (limits.min_um, limits.max_um)
        if not low <= z <= high:
            raise PositionOutOfRangeError('Z', z, low, high)
        return z

    def _refuse_no_focus_axis(self, then: str) -> None:
        """Refuse ``positions_unreachable`` on a scope with no Z axis."""
        if not self._scope.capabilities.has_focus:
            self._refuse(
                reason='positions_unreachable',
                message=f'This scope has no motor for Z, so it cannot {then}.',
            )

    @api
    def set_step_z(self, protocol: Protocol, step_idx: int, z: float) -> None:
        """Write ``z`` as step ``step_idx``'s Z.

        Raises:
            ProtocolRunRefusedError: ``'positions_unreachable'``, this scope
                has no Z axis, so no Z is a place it can go. Nothing is written.
            ProtocolError: ``step_idx`` is not a step of ``protocol``, or
                ``z`` is not a number (raised by the protocol).
        """
        self._refuse_no_focus_axis(then="set a step's Z")
        protocol.modify_step_z_height(step_idx=step_idx, z=z)

    def apply_focus_to_layer_steps(self, protocol: Protocol, layer: str, z: float) -> int:
        """Write ``z`` as the Z of every step of ``layer``; returns how many.

        Raises:
            ProtocolError: ``z`` is not a number (raised by the protocol).
        """
        return protocol.apply_focus_all_layer_steps(layer=layer, z=z)

    def delete_step(self, protocol: Protocol, step_idx: int) -> None:
        """Remove step ``step_idx`` from ``protocol``; the steps after it move up one.

        Raises:
            ProtocolError: ``step_idx`` is not a step of ``protocol``
                (raised by the protocol). Nothing is removed.
        """
        protocol.delete_step(step_idx=step_idx)
        self._report_invalid_steps(protocol, solicited=True)

    def rename_step(self, protocol: Protocol, step_idx: int, name: str) -> str:
        """Give step ``step_idx`` the label ``name``, kept through later channel changes.

        Characters a filename cannot carry are removed from ``name``.
        Returns the step's name after the rename. A rename to the label the
        step already holds as its own changes nothing, so it gives no
        invalid-step notice: a name field sends the name it shows each time
        it loses focus.

        Raises:
            ProtocolError: ``step_idx`` is not a step of ``protocol``, or
                ``name`` has no letter, digit, dash or underscore (raised
                by the protocol). The step keeps its name.
        """
        before = protocol.step(idx=step_idx)[['Label', 'Auto_Named']].tolist()
        protocol.modify_name(step_idx=step_idx, step_name=name)
        after = protocol.step(idx=step_idx)
        if after[['Label', 'Auto_Named']].tolist() != before:
            self._report_invalid_steps(protocol, solicited=True)
        return after['Name']

    def set_labware(self, protocol: Protocol, plate_key: str) -> str:
        """Put ``protocol`` on the plate ``plate_key``; returns the key it took.

        Every position a step holds is stated against the protocol's plate.
        On a scope with no XY stage there is no plate to move over, so the
        protocol takes Center Plate whatever was asked, and a different plate
        asked for is logged as replaced.

        Raises:
            CatalogueNameRefusedError: ``'labware_unknown'``, ``plate_key``
                is not a plate the catalogue has. The protocol keeps its
                plate.
        """
        from modules.labware_loader import CENTER_PLATE

        key = self._scope.wellplate_loader.resolve_plate_key(plate_key)
        if not self._scope.capabilities.has_xy_stage and key != CENTER_PLATE:
            _api_log.info(
                f'[API] protocol plate {key!r} replaced by {CENTER_PLATE!r}: '
                'this scope has no XY stage'
            )
            key = CENTER_PLATE
        protocol.modify_labware(labware_id=key)
        return key

    def apply_tiling(
        self,
        protocol: Protocol,
        tiling: str,
        *,
        frame_dimensions: dict,
        binning_size: int,
        overlap_percent: float,
    ) -> None:
        """Expand every step of ``protocol`` into the tile grid ``tiling``.

        The tiles are laid out on the protocol's own plate, at the scope's
        stage offset, and ordered as a run visits them.

        Raises:
            ProtocolRunRefusedError: ``tiling`` is not a grid this
                installation offers, the protocol is already tiled, a step's
                objective is not in the catalogue, the scope has no X/Y
                motor, a step has no position on an axis this scope moves
                (``'step_position_missing'``), or a tile falls outside the
                stage's travel. Nothing changes.
            CatalogueNameRefusedError: the protocol's plate is not in the
                catalogue (``'labware_unknown'``). Nothing changes.
            ConfigError: no session has bound the scope, so it has no stage
                offset. Nothing changes.
        """
        self.refuse_missing_positions(protocol.steps(), then='tile them')
        protocol.apply_tiling(
            tiling=tiling,
            frame_dimensions=frame_dimensions,
            binning_size=binning_size,
            axis_limits=self._travel_limits(),
            labware=self._scope.wellplate_loader.get_plate(plate_key=protocol.labware()),
            stage_offset=self._scope.runtime_state.get_stage_offset(),
            overlap_percent=overlap_percent,
            capabilities=self._scope.capabilities,
            objective_helper=self._scope.objective_helper,
        )
        self._report_invalid_steps(protocol, solicited=True)

    def apply_zstacking(
        self,
        protocol: Protocol,
        *,
        range_um: float,
        step_size_um: float,
        z_reference: str,
    ) -> None:
        """Expand every step of ``protocol`` not already in a stack into a z-stack.

        ``z_reference`` says where each step's Z sits in its stack:
        ``'top'``, ``'center'`` or ``'bottom'``. The slices are ordered as a
        run visits them.

        Raises:
            ProtocolRunRefusedError: ``range_um`` or ``step_size_um`` is not
                greater than zero, the scope has no Z motor, a step has no
                position on an axis this scope moves
                (``'step_position_missing'``), or a slice falls outside the Z
                travel. Nothing changes.
            ArgumentRefusedError: ``'not_a_number'``, ``range_um`` or
                ``step_size_um`` is not a finite number. Nothing changes.
            ArgumentRefusedError: ``'zstack_reference_unknown'``,
                ``z_reference`` is not one of the three. Nothing changes.
        """
        self.refuse_missing_positions(protocol.steps(), then='stack them')
        protocol.apply_zstacking(
            zstack_params={
                'range': range_um,
                'step_size': step_size_um,
                'z_reference': z_reference,
            },
            axis_limits=self._travel_limits(),
        )
        self._report_invalid_steps(protocol, solicited=True)

    def stage_targets(
        self,
        protocol: Protocol,
        px: float | None,
        py: float | None,
        z: float | None,
        *,
        stage_offset: StageOffset | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """A plate position and Z of ``protocol`` as the stage targets this scope drives.

        The one conversion every step move takes its targets from -- a run's
        steps, its return between scans and after a z-stack, a person's Go
        To Step -- so none drives an axis the scope lacks: an axis with no
        motor is None, and so are X and Y when ``px`` or ``py`` is. X and Y
        are converted on the plate the PROTOCOL stores, not the one the
        session has selected: a step is driven in the frame it was written
        in, whatever plate the operator has since picked. ``stage_offset``
        is the live one unless a caller supplies the one it took (a run
        keeps the offset it started with).

        Returns:
            ``(x, y, z)`` in stage micrometres, each None where nothing is
            driven.

        Raises:
            CatalogueNameRefusedError: the protocol's plate is not in the
                catalogue (``'labware_unknown'``).
        """
        axes = self._scope.capabilities.axes
        x = y = None
        if 'X' in axes and 'Y' in axes and px is not None and py is not None:
            stage = self._scope.runtime_state.plate_to_stage(
                px, py, labware_id=protocol.labware(), stage_offset=stage_offset
            )
            x, y = stage.x_um, stage.y_um
        return x, y, (z if 'Z' in axes else None)

    @api(reads=True)
    def step_targets(
        self, protocol: Protocol, step_idx: int, *, stage_offset: StageOffset | None = None
    ) -> StepTargets:
        """The motor targets of step ``step_idx``: the one conversion a run and a person's navigation share.

        The turret slot is the one the step recorded, so navigating to a
        step and running it look through the same glass, and two steps
        naming one objective in two slots each go to their own. A step that
        recorded no slot means the only slot holding its objective. X, Y and
        Z are ``stage_targets``'s: None for an axis the scope lacks.

        Raises:
            StepNotFoundError: ``step_idx`` is not a step of ``protocol``.
            ProtocolRunRefusedError: ``'step_position_missing'``, the step has
                no position on an axis this scope moves
                (``refuse_missing_positions``). Or this scope cannot put the step's
                objective in the light path (``refuse_unaddressable_objectives``):
                ``'turret_objectives_unassigned'``, no turret slot carries it;
                ``'objective_not_mounted'``, without a turret, it is not the
                mounted objective. Asked here, so a step whose slot was cleared
                after its protocol loaded is refused, never moved to through
                the wrong glass. ``'turret_slot_ambiguous'`` or
                ``'turret_slot_reassigned'``, the step cannot say which slot
                it means (``turret_slot_questions``).
            CatalogueNameRefusedError: as ``stage_targets``.
        """
        step = protocol.step(idx=step_idx)
        # This step's own fields, not the protocol's frame: a run asks it at every step.
        lacking = [axis for axis in self._driven_axes() if step[axis] is None]
        if lacking:
            self._refuse_missing_positions([step['Name']], lacking, then='go to it')
        self.refuse_unaddressable_objectives([step['Objective']])
        turret_slot = None
        if self._scope.capabilities.has_turret:
            turret_slot = self._slot_meant(
                step['Objective'],
                step['Turret Slot'],
                self._scope.runtime_state.get_turret_config(),
            )
            if turret_slot is None:
                self.refuse_unchosen_turret_slots(protocol, steps=[step_idx])
                # The rule above found the objective in a slot; none now
                # means one was cleared in between, and a step without its
                # slot would be imaged through other glass.
                self._refuse_unassigned({step['Objective']})
        x, y, z = self.stage_targets(
            protocol, step['X'], step['Y'], step['Z'], stage_offset=stage_offset
        )
        return StepTargets(turret_slot=turret_slot, x=x, y=y, z=z)

    def _travel_limits(self) -> dict[str, AxisLimits]:
        """The travel limits of each axis this scope has a motor for, by axis.

        The one read the run gate and the tile and z-stack builds judge
        positions against. Only the axes in ``capabilities.axes``: an axis the
        scope lacks has no travel, and ``get_axis_limits`` refuses it. An
        axis without software-enforced bounds (the turret's T) answers None
        and is left out.
        """
        return {
            axis: limits
            for axis in self._scope.capabilities.axes
            if (limits := self._scope.motion.get_axis_limits(axis)) is not None
        }

    def _refuse_unrecordable_step(self, *, verb: str, objective_id: str | None) -> None:
        """Refuse to save a step the scope cannot vouch for.

        The one rule behind adding and updating a step, so the two cannot
        drift apart: a step records where the scope is and which objective
        it is looking through, and either can be unknown.
        """
        # First, because a step is a saved position and ``plate_position``
        # is only as good as the axes it was read from: an axis that lost
        # its reference keeps answering the last number it reported, so the
        # step would save a real-looking place the scope no longer vouches
        # for. Every axis, Z and T included, even when the step's Z comes
        # from the layer's saved focus -- a step is where the scope will be
        # sent. And ahead of the turret check, whose advice (set the slot's
        # objective) is wrong when the slot itself is what is unknown.
        unknown_axes = self._scope.motion.axes_without_position()
        if unknown_axes:
            self._refuse(
                reason='step_position_unknown',
                message=(
                    f'Cannot {verb} the step. '
                    + unknown_positions_sentence(unknown_axes, then=f'{verb} the step')
                ),
            )
        if self._scope.capabilities.has_turret and (
            not self._scope.motion.is_current_turret_position_objective_set()
        ):
            self._refuse(
                reason='turret_objective_unset',
                message=(
                    f'Cannot {verb} the step. Please set objective for current turret position.'
                ),
            )
        if objective_id is None:
            self._refuse(
                reason='objective_unknown',
                message=(
                    f'Cannot {verb} the step: the objective in the light path is unknown, so '
                    'the step could not say which objective it was taken with.'
                ),
            )

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

    def refuse_no_acquiring_layer(self, layer_configs: dict) -> None:
        """Refuse a step add or a protocol build when no layer is set to acquire.

        A layer whose ``acquire`` is neither image nor video contributes no
        step, so with none acquiring an add adds nothing and a build gives
        an empty protocol; each used to do so silently, the click looking
        dead. The GUI's Add and New and a script's call share this one
        refusal.

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this refusal through ``ScopeSession.new_protocol`` and ``add_step``.

        Raises:
            ProtocolRunRefusedError: reason ``no_acquiring_layer``. Logged
                and notified once.
        """
        from modules.protocol import Protocol

        if not any(Protocol.layer_acquires(cfg) for cfg in layer_configs.values()):
            self._refuse(
                reason='no_acquiring_layer',
                message=(
                    'No channel is set to acquire, so there is no step to make. '
                    'Set a channel to Image or Video first.'
                ),
            )

    def refuse_absent_layers(self, protocol: Protocol) -> None:
        """Refuse a protocol with a step on a layer this scope does not have.

        Asked where steps arrive that this scope's settings did not build: a
        loaded file, and a run about to start. A step this session builds
        (New, Add, Update, a composite) comes from layers set to acquire,
        and a layer this scope lacks is never set to acquire. A scope whose
        layers could not be resolved has none, and the words say so rather
        than naming a layer it lacks.

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this refusal through ``ScopeSession.load_protocol`` and a run's start.

        Raises:
            ProtocolRunRefusedError: reason ``layer_not_on_scope``, naming
                the layers and the steps. Logged and notified once.
        """
        identity = self._scope.layer_identity
        present = {record.key_name for record in identity.layers}
        steps = protocol.steps()
        absent = steps.loc[~steps['Color'].isin(present)]
        if absent.empty:
            return
        missing = sorted(set(absent['Color']))
        layers = ', '.join(missing) + (' layers' if len(missing) > 1 else ' layer')
        names = common_utils.first_few(list(absent['Name']), separator=', ')
        if identity.layers:
            scope = f'This scope ({identity.model}) has no {layers}'
        else:
            scope = f"This scope's layers could not be resolved (model {identity.model})"
        self._refuse(
            reason='layer_not_on_scope',
            message=(
                f'{scope}, so {len(absent)} step(s) cannot be run: {names}. '
                f'Delete those steps, or open the protocol on a scope that has the layer.'
            ),
        )

    def refuse_unaddressable_objectives(self, objective_ids: Iterable[str]) -> None:
        """Refuse unless this scope can put every objective named here in the light path.

        The one rule behind every moment that admits a protocol: starting
        a run, loading a file, and navigating to a single step. A new
        protocol needs no ask: it names only the objective in the light
        path, and its build is refused while that one is unknown. Each of
        those carried its own version of the rule and the versions
        disagreed, so a protocol the run refused could still be loaded,
        displayed and navigated -- and the step pointer moved before
        anything had been asked at all.

        With a turret, an objective is addressable when a slot is assigned
        to it. An unassigned turret therefore addresses NOTHING and
        refuses everything. That is deliberate and it is the part that
        changed: the exemption this replaces existed to keep a fresh
        install runnable, but the startup objective question already does
        that -- it assigns the current position before any protocol loads,
        behind a popup that cannot be dismissed. A scope arriving here
        with every slot empty is one whose slots were CLEARED.

        Without a turret, the objective in the light path is the one
        mounted, which is the one selected, and nothing can change it, so a
        protocol may name only that one. With none selected nothing vouches
        for the glass, so every objective named is refused.

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
            ProtocolRunRefusedError: This scope cannot address them. Raised,
                not reported: the caller's boundary reports it.
        """
        named = set(objective_ids)

        if self._scope.capabilities.has_turret:
            if not named.issubset(self._turret_carries()):
                self._refuse_unassigned(named)
        elif len(named) > 1:
            self._refuse(
                reason='objectives_require_turret',
                message=(
                    'This protocol changes objective between steps, and this scope has '
                    'no way to change it: the objective in the light path is the one '
                    f'that is mounted.\n\nIt names: {self._render(named)}\n\n'
                    'Use a protocol that names a single objective.'
                ),
            )
        else:
            # The glass is the selected objective, and a protocol for any
            # other would still run: its images would carry the mounted
            # objective's scale while autofocus and every post-processed
            # output took the protocol's.
            selected = self._scope.runtime_state.get_current_objective_id()
            if named and named != {selected}:
                mounted = (
                    f'this scope has {selected} mounted'
                    if selected is not None
                    else 'no objective is selected on this scope'
                )
                self._refuse(
                    reason='objective_not_mounted',
                    message=(
                        f'This protocol was made for {self._render(named)}, and {mounted}.'
                        '\n\nSelect the mounted objective, or use a protocol made for it.'
                    ),
                )

    def refuse_unreachable_positions(self, steps: pd.DataFrame) -> None:
        """Refuse a protocol whose steps need motion this scope has no motor for.

        A run drives only the axes the scope has (``stage_targets``), so a
        run whose steps sit at different places on a missing axis would image
        one place and save each image under its step's name and coordinates.
        A manual scope (no motor board) lacks every axis; a Z-only scope
        lacks X and Y. Steps that all sit at one place on a missing axis ask
        for no motion there and are admitted: a single-location time lapse on
        a manual scope is the case this keeps. Autofocus moves Z, so it needs
        a Z axis. Permanent, so it is asked before whether the scope knows
        its position.

        A step with no position on a missing axis asks for no motion there.
        Without X/Y a step's well and tile name a place too: steps with no X/Y
        that name different wells or tiles would image one field under each
        name, so they ask for motion as positions do.

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this rule by starting a run, which asks it here.

        Args:
            steps: The protocol's steps table (``Protocol.steps()``): each
                position a finite number, or NaN where the step has none.

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable``. It has been
                logged and shown before it is raised.
        """
        present = set(self._scope.capabilities.axes)
        needed = []
        if not {'X', 'Y'} <= present:
            stated = len(steps[['X', 'Y']].dropna().round(3).drop_duplicates())
            unplaced = steps[steps['X'].isna()]
            named = unplaced[(unplaced['Well'] != '') | (unplaced['Tile'] != '')]
            named = len(named[['Well', 'Tile']].drop_duplicates())
            if stated > 1:
                needed.append('the steps are at different X/Y positions')
            elif stated + named > 1:
                needed.append('the steps name different wells or tiles')
        if 'Z' not in present:
            if steps['Z'].round(3).nunique() > 1:
                needed.append('the steps are at different Z positions')
            if steps['Auto_Focus'].astype(bool).any():
                needed.append('autofocus moves Z')
        if needed:
            missing = ', '.join(axis for axis in ('X', 'Y', 'Z') if axis not in present)
            self._refuse(
                reason='positions_unreachable',
                message=(
                    f'This protocol needs the scope to move, and this scope has no motor for '
                    f'{missing}: {"; ".join(needed)}.\n\nUse a protocol whose steps are all at '
                    'one position on those axes, without autofocus if there is no Z motor.'
                ),
            )

    def refuse_missing_positions(self, steps: pd.DataFrame, *, then: str = 'run it') -> None:
        """Refuse steps that have no position on an axis this scope drives.

        A step recorded on a scope without an axis has no position on it, and
        on a scope that has the axis it says nothing about where to go there:
        driven, it would image wherever the last move left the stage under
        the step's name. One rule for the run, Go To Step and the tile and
        z-stack builds, each asking it before it reads a position.

        A consult seam, not part of the L2 API surface.

        Args:
            steps: The steps judged (``Protocol.steps()`` or a selection of it).
            then: What the person was doing, ending the refusal's words.

        Raises:
            ProtocolRunRefusedError: ``step_position_missing``, naming the
                steps and the axes.
        """
        lacking = [axis for axis in self._driven_axes() if steps[axis].isna().any()]
        if not lacking:
            return
        missing = steps[steps[lacking].isna().any(axis=1)]
        self._refuse_missing_positions(list(missing['Name']), lacking, then=then)

    def _driven_axes(self) -> list[str]:
        """The position axes this scope has a motor for, in X, Y, Z order."""
        return [axis for axis in ('X', 'Y', 'Z') if axis in self._scope.capabilities.axes]

    def _refuse_missing_positions(self, names: list[str], lacking: list[str], *, then: str) -> None:
        """Refuse ``step_position_missing`` for the steps ``names``, which have no ``lacking`` position."""
        axes = ', '.join(lacking)
        self._refuse(
            reason='step_position_missing',
            message=(
                f'{len(names)} step(s) have no {axes} position, and this scope moves '
                f'{axes}: {common_utils.first_few(names, separator=", ")}. They were '
                'recorded on a scope without that axis.\n\n'
                f'Update those steps here, or delete them, to {then}.'
            ),
        )

    def refuse_positions_outside_travel(self, steps: pd.DataFrame, labware_key: str) -> None:
        """Refuse a protocol whose steps lie outside this scope's stage travel.

        A step outside an axis's limits would drive the stage to the end of
        travel and stop the run there. Judged on the axes the scope has, at
        the scope's stage offset -- the one the run converts plate positions
        with -- on the protocol's own plate. Asked once the scope knows its
        position: an unhomed scope is told to home, not that its steps are
        outside travel.

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this rule by starting a run, which asks it here.

        Args:
            steps: The protocol's steps table (``Protocol.steps()``), its
                positions already validated as numbers.
            labware_key: The plate the protocol's X/Y are measured on
                (``Protocol.labware()``), already validated as known.

        Raises:
            ProtocolRunRefusedError: ``positions_outside_travel``. It has
                been logged and shown before it is raised.
            ConfigError: No session has bound the scope, so it has no
                stage offset to judge X/Y with.
        """
        from modules.protocol import axes_outside_travel

        axis_limits = self._travel_limits()
        plate = {}
        if {'X', 'Y'} & set(axis_limits):
            plate = {
                'labware': self._scope.wellplate_loader.get_plate(plate_key=labware_key),
                'stage_offset': self._scope.runtime_state.get_stage_offset(),
            }
        outside = []
        for idx, step in steps.iterrows():
            axes = axes_outside_travel(
                {axis: step[axis] for axis in ('X', 'Y', 'Z')}, axis_limits, **plate
            )
            if axes:
                outside.append(f'step {idx + 1} ({step["Name"]}): {", ".join(axes)}')
        if outside:
            self._refuse(
                reason='positions_outside_travel',
                message=(
                    f"{len(outside)} of the {len(steps)} steps fall outside the stage's travel "
                    f'({common_utils.first_few(outside, separator="; ")}).\n\nMove those steps '
                    "inside the stage's travel."
                ),
            )

    def refuse_camera_values_out_of_range(self, steps: pd.DataFrame) -> None:
        """Refuse a protocol whose steps ask the camera for a value it cannot take.

        A step stores a gain and an exposure, and the run hands them to the
        camera at every capture. A stored value outside this camera's range --
        one saved on a camera that could reach it, or the old 48 dB Lumi
        default a ``current.json`` still carries -- is refused by the camera
        setter at every step, and the run would abandon scan after scan. So
        the run is refused before it starts, naming each step, and the stored
        values are left alone: the same value may be right on the next,
        larger camera.

        A limit the camera does not declare is not checked (the API caches it
        as None): a missing floor is not a floor of zero.

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this rule by starting a run, which asks it here.

        Args:
            steps: The protocol's steps table (``Protocol.steps()``).

        Raises:
            ProtocolRunRefusedError: A step's gain or exposure is outside the
                camera's range. It has been logged and shown before it is
                raised.
        """
        limits = self._scope.capabilities.camera_limits
        checks = (
            (
                'Gain',
                'gain',
                'dB',
                limits.gain_db_min if limits else None,
                limits.gain_db_max if limits else None,
            ),
            (
                'Exposure',
                'exposure',
                'ms',
                limits.exposure_ms_min if limits else None,
                limits.exposure_ms_max if limits else None,
            ),
        )
        problems = []
        for _, step in steps.iterrows():
            for column, word, unit, low, high in checks:
                value = float(step[column])
                # A blank cell reads as NaN; it asks the camera for nothing.
                if math.isnan(value):
                    continue
                if (low is not None and value < low) or (high is not None and value > high):
                    problems.append(
                        f'Step "{step["Name"]}" ({step["Color"]}): {word} {value:g} {unit} is '
                        f"outside this camera's range, {camera_range_words(low, high, unit)}."
                    )
        if problems:
            self._refuse(
                reason='camera_setting_out_of_range',
                message=(
                    common_utils.first_few(problems, separator='\n')
                    + '\n\nEdit these steps to values this camera can take, then run again.'
                ),
            )

    def _live_turret_slot(self) -> int:
        """The slot in the light path, which a step made now records; -1 without a turret.

        Asked after ``_refuse_unrecordable_step``, which refuses a step
        while the slot is unknown.
        """
        if not self._scope.capabilities.has_turret:
            return -1
        return self._scope.motion.get_turret_slot()

    @classmethod
    def _slot_meant(cls, objective_id: object, slot: int, turret_config: dict) -> int | None:
        """The slot a step means, or None when the step cannot say.

        A recorded slot is meant while it holds the step's objective. A
        step that recorded none means the only slot holding its objective:
        that is no guess. With two or more, it cannot say, and choosing one
        -- the current slot, the lowest -- would image through glass the
        step may not have meant.
        """
        if slot != -1:
            return slot if turret_config.get(slot) == objective_id else None
        holding = slots_holding(objective_id, turret_config)
        return holding[0] if len(holding) == 1 else None

    def _fill_turret_slots(self, protocol: Protocol) -> None:
        """Record the slot of each step that recorded none, where only one slot can be meant.

        A protocol from before steps recorded their slot; the steps whose
        objective is in two or more slots are left for the person
        (``turret_slot_questions``).
        """
        if not self._scope.capabilities.has_turret:
            return
        turret_config = self._scope.runtime_state.get_turret_config()
        steps = protocol.steps()
        by_slot: dict[int, list[int]] = {}
        for idx, (objective_id, slot) in enumerate(
            zip(steps['Objective'], steps['Turret Slot'], strict=True)
        ):
            if slot != -1:
                continue
            meant = self._slot_meant(objective_id, slot, turret_config)
            if meant is not None:
                by_slot.setdefault(meant, []).append(idx)
        for slot, indices in by_slot.items():
            protocol.set_turret_slot(indices, slot)
            _api_log.info(
                f'[Protocol ] {len(indices)} step(s) recorded no turret slot; slot {slot} is the '
                'only one holding their objective, and they now record it'
            )

    def _stamp_turret_slots(self, protocol: Protocol) -> None:
        """Record the slot of each step of a protocol just built.

        The slot in the light path where it holds the step's objective --
        the slot the person stands at -- and otherwise the only slot holding
        it. A caller naming an objective two other slots hold leaves its
        steps to ``turret_slot_questions``.
        """
        if not self._scope.capabilities.has_turret:
            return
        live = self._scope.motion.get_turret_slot()
        turret_config = self._scope.runtime_state.get_turret_config()
        by_slot: dict[int, list[int]] = {}
        for idx, objective_id in enumerate(protocol.steps()['Objective']):
            if live is not None and turret_config.get(live) == objective_id:
                meant = live
            else:
                meant = self._slot_meant(objective_id, -1, turret_config)
            if meant is not None:
                by_slot.setdefault(meant, []).append(idx)
        for slot, indices in by_slot.items():
            protocol.set_turret_slot(indices, slot)

    @api(reads=True)
    def turret_slot_questions(self, protocol: Protocol) -> list[TurretSlotQuestion]:
        """What to ask about ``protocol``'s turret slots: one question per objective, or none.

        A step that cannot say which slot it means is one whose objective is
        in two or more slots and that recorded none, or whose recorded slot
        now holds another objective. Each is refused by a run and by going
        to it until ``point_steps_at_turret_slot`` answers its question.
        LumaViewPro asks when a protocol is opened. Empty on a scope with no
        turret, and for steps whose objective no slot holds, which a load
        and a run refuse as ``turret_objectives_unassigned``.
        """
        if not self._scope.capabilities.has_turret:
            return []
        turret_config = self._scope.runtime_state.get_turret_config()
        steps = protocol.steps()
        asked: dict[object, tuple[list[int], list[int]]] = {}
        for idx, (objective_id, slot) in enumerate(
            zip(steps['Objective'], steps['Turret Slot'], strict=True)
        ):
            if self._slot_meant(objective_id, slot, turret_config) is not None:
                continue
            if not slots_holding(objective_id, turret_config):
                continue
            unchosen, reassigned = asked.setdefault(objective_id, ([], []))
            unchosen.append(idx)
            if slot != -1:
                reassigned.append(idx)
        return [
            TurretSlotQuestion(
                objective_id=objective_id,
                steps=tuple(unchosen),
                reassigned=tuple(reassigned),
                choices=slots_holding(objective_id, turret_config),
            )
            for objective_id, (unchosen, reassigned) in asked.items()
        ]

    @api
    def point_steps_at_turret_slot(
        self, protocol: Protocol, objective_id: str, slot: int
    ) -> tuple[int, ...]:
        """Answer ``objective_id``'s turret slot question: its steps now record ``slot``.

        Points exactly the steps ``turret_slot_questions`` names for
        ``objective_id``; every other step keeps its slot. Returns the
        indices pointed, which are empty when no question names it.

        Raises:
            ArgumentRefusedError: ``'slot_not_holding_objective'``, ``slot``
                does not hold ``objective_id`` now (``offered``: the slots
                that do). Nothing is changed.
        """
        holding = slots_holding(objective_id, self._scope.runtime_state.get_turret_config())
        if slot not in holding:
            raise ArgumentRefusedError(
                'slot_not_holding_objective', argument='slot', value=slot, offered=holding
            )
        asked = [q for q in self.turret_slot_questions(protocol) if q.objective_id == objective_id]
        if not asked:
            return ()
        protocol.set_turret_slot(list(asked[0].steps), slot)
        return asked[0].steps

    def refuse_unchosen_turret_slots(
        self, protocol: Protocol, *, steps: Iterable[int] | None = None
    ) -> None:
        """Refuse when a step (of ``steps``, or any) cannot say which turret slot it means.

        Asked by a run's ``prepare()`` and by going to a step, after the
        objective rule; a load admits the protocol and asks instead
        (``turret_slot_questions``), so the protocol can be answered and
        opened.

        Raises:
            ProtocolRunRefusedError: ``'turret_slot_reassigned'``, a step's
                recorded slot now holds another objective;
                ``'turret_slot_ambiguous'``, a step recorded no slot and its
                objective is in two or more. Raised, not reported: the
                caller's boundary reports it.
        """
        wanted = None if steps is None else set(steps)
        for question in self.turret_slot_questions(protocol):
            named = [idx for idx in question.steps if wanted is None or idx in wanted]
            if not named:
                continue
            which = ', '.join(str(idx + 1) for idx in named)
            several = len(named) > 1
            steps = f'Steps {which} use' if several else f'Step {which} uses'
            slots = ' and '.join(str(slot) for slot in question.choices)
            reassigned = [idx for idx in named if idx in question.reassigned]
            if reassigned:
                self._refuse(
                    reason='turret_slot_reassigned',
                    message=(
                        f'{steps} {question.objective_id}, and the turret slot recorded for {"them" if several else "it"} '
                        f'now holds another objective. {question.objective_id} is in slot '
                        f'{slots}. Choose the slot; LumaViewPro asks when the protocol is opened.'
                    ),
                )
            self._refuse(
                reason='turret_slot_ambiguous',
                message=(
                    f'{steps} {question.objective_id}, which is in slots {slots}, and which of '
                    'them is meant is not recorded. Choose the slot; LumaViewPro asks when the '
                    'protocol is opened.'
                ),
            )

    def _turret_carries(self) -> set[str]:
        """The objectives a turret slot is assigned, read from the turret configuration."""
        return {
            objective
            for objective in self._scope.runtime_state.get_turret_config().values()
            if objective is not None
        }

    def _refuse_unassigned(self, named: set[object]) -> typing.NoReturn:
        """Refuse objectives no turret slot carries, naming what the turret does carry."""
        self._refuse(
            reason='turret_objectives_unassigned',
            message=(
                'This protocol uses objectives that are not assigned to turret '
                f'positions.\n\nIt needs: {self._render(named)}\n'
                f'The turret carries: {self._render(self._turret_carries()) or "None"}\n\n'
                'Assign the missing objectives in Objective Control > Turret.'
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

    def _refuse(self, reason: str, message: str) -> typing.NoReturn:
        """Raise the typed refusal.

        The single funnel this surface's refusals route through, so one
        refusal is always one typed exception. Raised, not reported: the
        boundary the caller sits behind reports it, so a REST caller is
        answered and not also shown it. Each caller passes its reason
        as a LITERAL rather than through a variable: the refusal vocabulary
        is censused by reading that argument out of the source, and a name
        in its place makes the refusal invisible to the census, which is how
        a reason ships with no coverage.
        """
        raise ProtocolRunRefusedError(reason=reason, message=message)
