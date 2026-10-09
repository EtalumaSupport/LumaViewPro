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
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import modules.common_utils as common_utils
from modules.finite_number import refuse_unless_finite_number
from modules.coord_transformations import CoordinateTransformer
from modules.exceptions import (
    ArgumentRefusedError,
    PositionOutOfRangeError,
    ProtocolRunRefusedError,
    ProtocolStepsInvalidNotice,
    unknown_positions_sentence,
)
from modules.lumascope_api.imaging import camera_range_words
from modules.api_surface import api, api_fields

if TYPE_CHECKING:
    import pandas as pd

    from modules.lumascope_api._lumascope import Lumascope
    from modules.protocol import Protocol
    from modules.tiling_config import TilingConfig

_api_log = logging.getLogger('LVP.api')
_coordinate_transformer = CoordinateTransformer()


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

    @api
    def tiling_config(self) -> TilingConfig:
        """The tiling grids this installation offers, from the scope's data folder.

        What a caller needs to choose a grid: ``available`` lists the
        labels ``create_protocol`` and ``Protocol.apply_tiling`` accept,
        and ``default`` is the one to preselect; ``Protocol.tiling()``
        names the grid a protocol's steps already carry. Read from the file on
        each call, so there is no copy to fall out of step with it.

        Raises:
            RuntimeError: tiling.json is missing or is not valid JSON.
            ValueError: tiling.json does not have the expected structure.
        """
        from modules.tiling_config import TilingConfig

        return TilingConfig(tiling_configs_file_loc=self.tiling_configs_path())

    def load_protocol(self, file_path: str | pathlib.Path) -> Protocol:
        """Load a Protocol from disk.

        Wraps ``Protocol.from_file(...)`` and resolves
        ``data/tiling.json`` from the scope's data folder.

        Args:
            file_path: Path to the protocol file.

        Returns:
            Protocol: The loaded Protocol instance.

        Raises:
            ProtocolNotLoadedError: The file cannot be read.
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
            file_path=file_path,
            tiling_configs_file_loc=self.tiling_configs_path(),
            wellplate_loader=self._scope.wellplate_loader,
        )
        # After the parse, so a file that is not a protocol at all is
        # answered as that rather than as a turret problem, and before the
        # return, so no caller ever holds an inadmissible protocol.
        self.refuse_unaddressable_objectives(protocol.steps()['Objective'].to_list())
        self.refuse_absent_layers(protocol)
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
            self._scope.objective_helper, led_max_ma=self._scope.capabilities.led_max_ma
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
                ``overlap_out_of_range``, ``tiling_unknown``,
                ``zstack_not_configured`` -- reported once.
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
            return Protocol.from_config(
                input_config=input_config,
                tiling_configs_file_loc=tcfg,
                capabilities=self._scope.capabilities,
                objective_helper=self._scope.objective_helper,
                wellplate_loader=self._scope.wellplate_loader,
            )
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

        ``before_step`` or ``after_step`` places the steps; with neither
        they follow the last step, which is what a script building a
        protocol one add at a time means by "add".

        Returns the inserted step names, in protocol order.

        Raises:
            ProtocolRunRefusedError: an axis does not know its position, no
                layer acquires, the turret's current slot has no objective,
                or the active objective is unknown. Logged and notified once.
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

        names: list[str] = []
        for layer in ordered:
            layer_config = layer_configs[layer]
            if not Protocol.layer_acquires(layer_config):
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
        self._report_invalid_steps(protocol, solicited=True)
        return names

    def update_step(
        self,
        protocol: Protocol,
        step_idx: int,
        *,
        layer: str,
        layer_configs: dict,
        stim_configs: dict,
        plate_position: dict,
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

        Returns the step's name after the update.

        Raises:
            ProtocolRunRefusedError: an axis does not know its position, the
                turret's current slot has no objective, the active objective
                is unknown, or the layer the step takes is not set to
                acquire (reason ``no_acquiring_layer``, as an add). Logged
                and notified once; the step is unchanged.
            ProtocolError: ``step_idx`` is not a step of ``protocol``
                (raised by the protocol).
        """
        from modules.protocol import Protocol

        self._refuse_unrecordable_step(verb='update', objective_id=objective_id)

        stim_config = layer_configs[layer].get('stim_config')
        if stim_config is not None and stim_config['enabled']:
            layer = protocol.step(idx=step_idx)['Color']

        if not Protocol.layer_acquires(layer_configs[layer]):
            self._refuse(
                reason='no_acquiring_layer',
                title='No Channel Set to Acquire',
                message=(
                    f'{layer} is not set to acquire, so there is no step to make. '
                    f'Set {layer} to Image or Video first.'
                ),
            )

        protocol.modify_step(
            step_idx=step_idx,
            label=label,
            layer=layer,
            layer_config=layer_configs[layer],
            stim_configs=self._stim_configs_with_invalid_channels_disabled(stim_configs),
            plate_position=plate_position,
            objective_id=objective_id,
        )
        self._report_invalid_steps(protocol, solicited=True)
        return protocol.step(idx=step_idx)['Name']

    @api
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
        low, high = (-math.inf, math.inf) if limits is None else (limits['min'], limits['max'])
        if not low <= z <= high:
            raise PositionOutOfRangeError('Z', z, low, high)
        return z

    def _refuse_no_focus_axis(self, then: str) -> None:
        """Refuse ``positions_unreachable`` on a scope with no Z axis."""
        if not self._scope.capabilities.has_focus:
            self._refuse(
                reason='positions_unreachable',
                title='Position Not Reachable',
                message=f'This scope has no motor for Z, so it cannot {then}.',
            )

    @api
    def set_step_z(self, protocol: Protocol, step_idx: int, z: float) -> None:
        """Write ``z`` as step ``step_idx``'s Z.

        Raises:
            ProtocolError: ``step_idx`` is not a step of ``protocol``, or
                ``z`` is not a number (raised by the protocol).
        """
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
                motor, or a tile falls outside the stage's travel. Nothing
                changes.
            CatalogueNameRefusedError: the protocol's plate is not in the
                catalogue (``'labware_unknown'``). Nothing changes.
            ConfigError: no session has bound the scope, so it has no stage
                offset. Nothing changes.
        """
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
                greater than zero, the scope has no Z motor, or a slice falls
                outside the Z travel. Nothing changes.
            ArgumentRefusedError: ``'not_a_number'``, ``range_um`` or
                ``step_size_um`` is not a finite number. Nothing changes.
            ArgumentRefusedError: ``'zstack_reference_unknown'``,
                ``z_reference`` is not one of the three. Nothing changes.
        """
        protocol.apply_zstacking(
            zstack_params={
                'range': range_um,
                'step_size': step_size_um,
                'z_reference': z_reference,
            },
            axis_limits=self._travel_limits(),
        )
        self._report_invalid_steps(protocol, solicited=True)

    @api
    def plate_to_stage(
        self, protocol: Protocol, px: float, py: float, *, stage_offset: dict | None = None
    ) -> tuple[float, float]:
        """A plate position of ``protocol``, as stage micrometres.

        Converted against the plate the PROTOCOL stores, not the one the
        session has selected: a step is driven in the frame it was written
        in, whatever plate the operator has since picked. ``stage_offset``
        is the live one unless a caller supplies the snapshot it took (a
        run keeps the offset it started with).

        Raises:
            CatalogueNameRefusedError: the protocol's plate is not in the
                catalogue (``'labware_unknown'``).
            ConfigError: no session has bound the scope, so it has no stage
                offset.
        """
        if stage_offset is None:
            stage_offset = self._scope.runtime_state.get_stage_offset()
        return _coordinate_transformer.plate_to_stage(
            labware=self._scope.wellplate_loader.get_plate(plate_key=protocol.labware()),
            stage_offset=stage_offset,
            px=px,
            py=py,
        )

    @api(in_process=True)
    def plate_transform(
        self, protocol: Protocol, *, stage_offset: dict | None
    ) -> Callable[[float, float], tuple[float, float]] | None:
        """The stage-to-plate transform of ``protocol``'s frame: ``plate_to_stage`` inverted.

        What a run states a frame's position in: the plate the protocol
        stores and the offset the run started with, the frame its steps
        were driven in. The selected plate and the live offset are the
        frame of a capture made outside a run
        (``runtime_state.plate_transform``); a run converted through them
        would state its positions in a frame it never moved in once the
        operator picked another plate. Both are bound when this is called,
        so a later change moves no position it converts, and it cannot
        raise.

        Args:
            protocol: The protocol whose plate the positions are stated on.
            stage_offset: The offset the run started with; None on a scope
                with no X/Y stage, which has no plate position to state.

        Returns:
            A function of ``(sx_um, sy_um)`` answering ``(px_mm, py_mm)``,
            or None when ``stage_offset`` is.

        Raises:
            CatalogueNameRefusedError: the protocol's plate is not in the
                catalogue (``'labware_unknown'``).
        """
        if stage_offset is None:
            return None
        labware = self._scope.wellplate_loader.get_plate(plate_key=protocol.labware())
        stage_offset = dict(stage_offset)

        def to_plate(sx: float, sy: float) -> tuple[float, float]:
            return _coordinate_transformer.stage_to_plate(
                labware=labware, stage_offset=stage_offset, sx=sx, sy=sy
            )

        return to_plate

    @api
    def stage_targets(
        self,
        protocol: Protocol,
        px: float | None,
        py: float | None,
        z: float | None,
        *,
        stage_offset: dict | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """A plate position and Z of ``protocol`` as the stage targets this scope drives.

        The one conversion every step move takes its targets from -- a run's
        steps, its return between scans and after a z-stack, a person's Go
        To Step -- so none drives an axis the scope lacks: an axis with no
        motor is None, and so are X and Y when ``px`` or ``py`` is. X and Y
        are converted as ``plate_to_stage`` converts them.

        Returns:
            ``(x, y, z)`` in stage micrometres, each None where nothing is
            driven.

        Raises:
            CatalogueNameRefusedError: as ``plate_to_stage``.
        """
        axes = self._scope.capabilities.axes
        x = y = None
        if 'X' in axes and 'Y' in axes and px is not None and py is not None:
            x, y = self.plate_to_stage(protocol, px, py, stage_offset=stage_offset)
        return x, y, (z if 'Z' in axes else None)

    @api
    def step_targets(
        self, protocol: Protocol, step_idx: int, *, stage_offset: dict | None = None
    ) -> StepTargets:
        """The motor targets of step ``step_idx``: the one conversion a run and a person's navigation share.

        The turret slot is the one carrying the step's objective, chosen as
        ``motion.get_turret_position_for_objective_id`` chooses it, so
        navigating to a step and running it look through the same glass. X,
        Y and Z are ``stage_targets``'s: None for an axis the scope lacks.

        Raises:
            StepNotFoundError: ``step_idx`` is not a step of ``protocol``.
            CatalogueNameRefusedError: as ``stage_targets``.
            RuntimeError: this scope has a turret and no slot carries the
                step's objective. The admissibility rule
                (``refuse_unaddressable_objectives``) reads the same turret
                configuration, so a step it admitted has a slot; the two
                disagreeing is a defect, and what this replaced moved X, Y
                and Z anyway, capturing through the wrong glass under the
                right file name.
        """
        step = protocol.step(idx=step_idx)
        turret_slot = None
        if self._scope.capabilities.has_turret:
            turret_slot = self._scope.motion.get_turret_position_for_objective_id(
                objective_id=step['Objective']
            )
            if turret_slot is None:
                raise RuntimeError(
                    f'no turret slot carries {step["Objective"]!r} for step {step_idx}, '
                    'though the admissibility rule accepted it from the same turret '
                    'configuration'
                )
        x, y, z = self.stage_targets(
            protocol, step['X'], step['Y'], step['Z'], stage_offset=stage_offset
        )
        return StepTargets(turret_slot=turret_slot, x=x, y=y, z=z)

    def _travel_limits(self) -> dict:
        """The travel limits of each axis this scope has a motor for, by axis.

        The one read the run gate and the tile and z-stack builds judge
        positions against. Only the axes in ``capabilities.axes``: the
        motion board answers limits for X and Y on a Z-only scope too, and a
        build judged against them would lay out tiles no motor can reach. An
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
        title = f'Protocol {verb.capitalize()} Step Error'
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
                title=title,
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
                title=title,
                message=(
                    f'Cannot {verb} the step. Please set objective for current turret position.'
                ),
            )
        if objective_id is None:
            self._refuse(
                reason='objective_unknown',
                title=title,
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
                title='No Channel Set to Acquire',
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
            title='Layer Not on This Scope',
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
                    title='Objective Not Mounted',
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

        A consult seam, not part of the L2 API surface: an L2 caller meets
        this rule by starting a run, which asks it here.

        Args:
            steps: The protocol's steps table (``Protocol.steps()``), its
                positions already validated as numbers.

        Raises:
            ProtocolRunRefusedError: ``positions_unreachable``. It has been
                logged and shown before it is raised.
        """
        present = set(self._scope.capabilities.axes)
        needed = []
        if not {'X', 'Y'} <= present and len(steps[['X', 'Y']].round(3).drop_duplicates()) > 1:
            needed.append('the steps are at different X/Y positions')
        if 'Z' not in present:
            if steps['Z'].round(3).nunique() > 1:
                needed.append('the steps are at different Z positions')
            if steps['Auto_Focus'].astype(bool).any():
                needed.append('autofocus moves Z')
        if needed:
            missing = ', '.join(axis for axis in ('X', 'Y', 'Z') if axis not in present)
            self._refuse(
                reason='positions_unreachable',
                title='Position Not Reachable',
                message=(
                    f'This protocol needs the scope to move, and this scope has no motor for '
                    f'{missing}: {"; ".join(needed)}.\n\nUse a protocol whose steps are all at '
                    'one position on those axes, without autofocus if there is no Z motor.'
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
                title='Positions Outside Stage Travel',
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
        imaging = self._scope.imaging
        checks = (
            ('Gain', 'gain', 'dB', imaging.min_gain_db_cached, imaging.max_gain_db_cached),
            (
                'Exposure',
                'exposure',
                'ms',
                imaging.min_exposure_ms_cached,
                imaging.max_exposure_ms_cached,
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
                title='Camera Setting Out of Range',
                message=(
                    common_utils.first_few(problems, separator='\n')
                    + '\n\nEdit these steps to values this camera can take, then run again.'
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
        """Report once, and raise, the typed refusal.

        The single funnel this surface's refusals route through, so one
        refusal is always one report -- a WARNING line naming the reason and
        one warning -- and one typed exception. Each caller passes its reason
        as a LITERAL rather than through a variable: the refusal vocabulary
        is censused by reading that argument out of the source, and a name
        in its place makes the refusal invisible to the census, which is how
        a reason ships with no coverage.
        """
        from modules.notification_center import notifications

        refusal = ProtocolRunRefusedError(reason=reason, title=title, message=message)
        # Solicited: a refusal answers something the caller just asked
        # for, so it must reach the user even while a run is in flight.
        notifications.report_outcome(refusal, solicited=True, category='Protocol')
        raise refusal
