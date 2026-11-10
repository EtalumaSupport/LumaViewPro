# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import csv
import ast
import dataclasses
import json
import datetime
import io
import numpy as np
import pandas as pd
import os
import pathlib
import re
import contextlib
import copy
import enum
import hashlib
from collections.abc import Collection, Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, NoReturn

from _lumascope.lvp_logger import logger
from _lumascope.modules.exceptions import (
    ConfigError,
    DuplicateCaptureFilenamesNotice,
    FocusNotWrittenError,
    ProtocolError,
    ProtocolNotLoadedError,
    ProtocolNotSavedError,
    ProtocolRunRefusedError,
    Refusal,
    RefusalCause,
    SettingRefusedError,
)
from _lumascope.modules.notification_center import notifications

import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.path_utils as path_utils
from _lumascope.modules.finite_number import is_finite_number, refuse_unless_finite_number
from _lumascope.modules.python_values import python_values
from _lumascope.modules.channel_record import refuse_unknown_channel
from _lumascope.modules.request_fields import key_path, refuse_unless_kind, required
import _lumascope.modules.labware_loader as labware_loader
from _lumascope.modules.tiling_config import TilingGrids
from _lumascope.modules.zstack_config import ZStackConfig, ZStackReference
from _lumascope.modules.coord_transformations import CoordinateTransformer
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.axes import StageAxis
from _lumascope.modules.closed_sets import member_of, refuse_unless_offered
from _lumascope.modules.api_surface import FilePath, api, api_fields

if TYPE_CHECKING:
    import _lumascope.modules.labware as labware_module
    from _lumascope.modules.channel_record import ChannelIdentity
    from _lumascope.modules.coord_transformations import AxisLimits, PlateLocation, StageOffset
    from _lumascope.modules.objectives_loader import Objective, ObjectiveLoader
    from _lumascope.modules.scope_capabilities import ScopeCapabilities

coordinate_transformer = CoordinateTransformer()


def to_python_scalars(step: pd.Series) -> pd.Series:
    """Return a step row whose values are Python scalars, not numpy ones.

    The step frame is declared with an explicit numpy dtype, which is what
    makes the vectorised column work possible -- so every value sitting in
    it is a numpy scalar. Those must not leave: callers copy step values
    straight into the channel settings, and a Kivy NumericProperty does
    EXACT type checks, so it rejects `np.float64` and takes the whole
    application down with it.

    No caller can defend itself against this. `isinstance(np.float64(5.0),
    float)` is True, so a type guard cannot see it, and the sibling
    `BoundedNumericProperty` accepts numpy silently -- one widget dies,
    the next stores a value nothing else can read back.

    `Series.to_dict()` already boxes to natives, which is why only the
    callers reading by subscript were affected, and why the leak stayed
    invisible for as long as it did.
    """
    return step.map(lambda v: v.item() if isinstance(v, np.generic) else v)


# The step columns whose cells are dicts. A frame copy shares them, so they
# are stored read-only: a copy of the frame can then share them safely.
_DICT_COLUMNS = ('Video Config',)


class ReadOnlyDict(dict):
    """A dict a protocol stores in a step cell, which refuses every change.

    A dict subclass, not a mapping proxy: readers ask ``isinstance(cell,
    dict)`` and a proxy would read to them as no config at all. A write
    raises TypeError, so a caller holding a cell from ``steps()``, ``step()``
    or a run's copy cannot change the protocol, or another copy, behind its
    writers. What it is built from is frozen as it is built (``_read_only``):
    a dict inside it is one, a list a tuple, a numpy scalar its Python
    value, so nothing a protocol's file can hold in it can change, and a
    cell can be shared by every step built from it. A deep copy is a plain
    dict, all the way down, its tuples lists again, for the caller that
    means to edit its own.
    """

    def __init__(self, items=()):
        dict.__init__(self, {key: _read_only(value) for key, value in dict(items).items()})

    def _refuse(self, *args, **kwargs):
        raise TypeError(
            "a protocol's step data is read-only: change a step through the "
            "protocol's writers, or edit a copy.deepcopy() of it"
        )

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __deepcopy__(self, memo):
        return {key: _editable(value) for key, value in self.items()}

    def __copy__(self):
        return dict(self)

    def __reduce__(self):
        return (ReadOnlyDict, (dict(self),))


def _read_only(value):
    """``value`` with every dict and list in it, at any depth, made unchangeable: a dict a ``ReadOnlyDict``, a list a tuple; a numpy scalar is made its Python value.

    Each dict and list is built anew, so nothing the caller holds is shared
    with the protocol; a ``ReadOnlyDict`` is returned as it is, since it was
    frozen when it was built and nothing can change it. A tuple is written
    to a protocol's file as JSON writes a list, so the file is unchanged.
    """
    if isinstance(value, ReadOnlyDict):
        return value
    if isinstance(value, dict):
        return ReadOnlyDict(value)
    if isinstance(value, list | tuple):
        return tuple(_read_only(item) for item in value)
    return python_values(value)


def _editable(value):
    """A frozen cell's value as plain, editable data: the inverse of ``_read_only``."""
    if isinstance(value, dict):
        return {key: _editable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_editable(item) for item in value]
    return copy.deepcopy(value)


def _refuse_build(*, reason: str, message: str) -> NoReturn:
    """Raise a refused protocol build (a tile grid, a z-stack).

    Refused where the protocol is built, so the GUI, a script and REST are all
    told the same thing. Raised, not reported: the boundary the caller sits
    behind reports it, so a REST caller is answered and not also shown it.
    Callers pass the reason as a literal, which is how the refusal census
    reads it.
    """
    raise ProtocolRunRefusedError(reason=reason, message=message)


def _refuse_unless_tiling_offered(tiling_grids: TilingGrids, tiling: str) -> None:
    """Refuse a tile grid this installation's tiling.json does not offer.

    Both builds ask, so a new protocol and a grid over an existing one are
    refused alike; the build from a config used to raise KeyError from the
    grid lookup instead.
    """
    if tiling not in tiling_grids.config.available:
        _refuse_build(
            reason='tiling_unknown',
            message=f'"{tiling}" is not one of the tiling grids this installation offers.',
        )


# What a channel's settings must hold for the build to make its step; a channel
# that acquires nothing is read for its acquire mode only.
_ACQUIRING_CHANNEL_KINDS: tuple[tuple[str, str, bool], ...] = (
    ('focus', 'number', True),
    ('autofocus', 'bool', False),
    ('false_color', 'bool', False),
    ('illumination_ma', 'number', False),
    ('sum', 'whole', False),
    ('gain_db', 'number', False),
    ('auto_gain', 'bool', False),
    ('exposure_ms', 'number', False),
    ('video_config', 'dict', False),
)


def refuse_unless_buildable_config(config: object, argument: str, *, empty: bool) -> None:
    """Refuse a client's protocol configuration the build cannot read, before anything is built.

    Every key the build reads is judged here, present and of its kind,
    whichever of them this build happens to reach: a one-tile or empty
    build never reads the frame or the binning, so a check at the read
    would leave them unjudged there. ``empty`` judges the five keys an
    empty protocol takes. The schedule (``period``, ``duration``) is judged
    by the protocol itself, as it is for a file.

    Raises:
        ArgumentRefusedError: ``'missing_key'``, ``'wrong_kind'``,
            ``'not_a_number'``, ``'not_offered'`` (the z-stack's reference
            or a channel's acquire mode), ``'channel_unknown'``; each names the
            key's path inside ``argument``.
    """
    refuse_unless_kind(config, 'dict', argument)

    def read(mapping, key, kind, path, *, nullable=False):
        value = required(mapping, key, path)
        refuse_unless_kind(value, kind, key_path(path, key), nullable=nullable)
        return value

    read(config, 'labware_id', 'str', argument)
    required(config, 'period', argument)
    required(config, 'duration', argument)
    frame = read(config, 'frame_dimensions', 'dict', argument)
    for side in ('width', 'height'):
        read(frame, side, 'number', key_path(argument, 'frame_dimensions'))
    read(config, 'binning_size', 'whole', argument)
    if empty:
        return

    read(config, 'objective_id', 'str', argument, nullable=True)
    read(config, 'tiling', 'str', argument)
    if 'tiling_overlap_percent' in config:
        refuse_unless_kind(
            config['tiling_overlap_percent'], 'number', key_path(argument, 'tiling_overlap_percent')
        )
    if config.get('current_z') is not None:
        refuse_unless_kind(config['current_z'], 'number', key_path(argument, 'current_z'))
    if config.get('previous_well_z') is not None:
        tuned_path = key_path(argument, 'previous_well_z')
        tuned = config['previous_well_z']
        refuse_unless_kind(tuned, 'dict', tuned_path)
        for well, z in tuned.items():
            refuse_unless_kind(z, 'number', key_path(tuned_path, well), nullable=True)

    zstack = read(config, 'zstack_params', 'dict', argument)
    if read(config, 'use_zstacking', 'bool', argument):
        zstack_path = key_path(argument, 'zstack_params')
        read(zstack, 'range', 'number', zstack_path)
        read(zstack, 'step_size', 'number', zstack_path)
        refuse_unless_offered(
            ZStackReference,
            required(zstack, 'z_reference', zstack_path),
            key_path(zstack_path, 'z_reference'),
        )

    channel_configs = read(config, 'channel_configs', 'dict', argument)
    channel_configs_path = key_path(argument, 'channel_configs')
    for channel, channel_config in channel_configs.items():
        refuse_unknown_channel(channel, channel_configs_path)
        channel_path = key_path(channel_configs_path, channel)
        refuse_unless_kind(channel_config, 'dict', channel_path)
        acquire = required(channel_config, 'acquire', channel_path)
        if acquire is None:
            continue
        refuse_unless_offered(AcquireMode, acquire, key_path(channel_path, 'acquire'))
        for key, kind, nullable in _ACQUIRING_CHANNEL_KINDS:
            read(channel_config, key, kind, channel_path, nullable=nullable)

    positions = config.get('positions')
    if positions is not None:
        positions_path = key_path(argument, 'positions')
        refuse_unless_kind(positions, 'list', positions_path)
        for index, position in enumerate(positions):
            position_path = key_path(positions_path, index)
            refuse_unless_kind(position, 'dict', position_path)
            # None where the scope has no motor for the axis: no position.
            read(position, 'x', 'number', position_path, nullable=True)
            read(position, 'y', 'number', position_path, nullable=True)
            read(position, 'z', 'number', position_path, nullable=True)
            read(position, 'name', 'str', position_path)


def _fill_factor_or_refuse(overlap_percent: object) -> float:
    """The tile fill factor for ``overlap_percent``, or a refusal of the build.

    One answer for both builds, a new protocol's grid and a grid over an
    existing one. NaN passed the range test and built, and text or ``True``
    was converted; each is ``not_a_number`` now.

    Raises:
        ArgumentRefusedError: ``'not_a_number'``.
        ProtocolRunRefusedError: ``'overlap_out_of_range'``, outside 0-50.
    """
    refuse_unless_finite_number(overlap_percent, 'tiling_overlap_percent')
    if not 0.0 <= overlap_percent <= 50.0:
        _refuse_build(
            reason='overlap_out_of_range',
            message=f'The tile overlap must be 0 to 50 percent; {overlap_percent:g} is outside it.',
        )
    return TilingGrids.fill_factor_from_overlap_percent(overlap_percent)


def _refuse_unless_zstack_has_extent(zstack_params: dict) -> None:
    """Refuse a z-stack whose range or step size is not greater than zero.

    Such a stack has no planes, and building it used to produce a protocol
    unchanged (or one plane per position) reported as success: the caller
    asked for a stack and got a photograph.

    Raises:
        ArgumentRefusedError: ``'not_a_number'``, the range or the step size
            is not a finite number, which a comparison with zero would pass.
    """
    for key in ('range', 'step_size'):
        refuse_unless_finite_number(zstack_params[key], f'z-stack {key}')
    if zstack_params['range'] <= 0 or zstack_params['step_size'] <= 0:
        _refuse_build(
            reason='zstack_not_configured',
            message=(
                f'Z-stack range ({zstack_params["range"]}) and step size '
                f'({zstack_params["step_size"]}) must both be greater than zero.'
            ),
        )


def axes_outside_travel(
    position: dict,
    axis_limits: 'Mapping[str, AxisLimits]',
    *,
    labware: 'labware_module.Labware | None' = None,
    stage_offset: 'StageOffset | None' = None,
) -> list[str]:
    """The axes on which one position lies outside the stage's travel.

    The one travel judgment: the run gate and both builders ask it, so a
    position one of them admits is never one another refuses.

    Args:
        position: Any of 'X' and 'Y' in plate millimetres, and 'Z' in stage
            micrometres -- the frames a step stores them in.
        axis_limits: Axis name to its travel in stage micrometres. Only
            the axes present in both this and *position* are judged.
        labware: The plate X/Y are measured on; required when X or Y is judged.
        stage_offset: Where the plate is mounted; required when X or Y is
            judged.

    Returns:
        The axes outside their limits, in X, Y, Z order; empty when inside.
    """
    stage = {}
    if 'Z' in position:
        stage['Z'] = float(position['Z'])
    if ({'X', 'Y'} & set(axis_limits)) and {'X', 'Y'} <= set(position):
        converted = coordinate_transformer.plate_to_stage(
            labware=labware,
            stage_offset=stage_offset,
            x_mm=float(position['X']),
            y_mm=float(position['Y']),
        )
        stage['X'], stage['Y'] = converted.x_um, converted.y_um
    return [
        axis
        for axis in StageAxis
        if axis in stage
        and axis in axis_limits
        and not (axis_limits[axis].min_um <= stage[axis] <= axis_limits[axis].max_um)
    ]


def _refuse_positions_outside_travel(
    *, reason: str, what: str, outside: dict[str, int], total: int, remedy: str
) -> None:
    """Refuse a build if any of its new positions lie outside the stage's travel.

    Dropping them and building the rest made a grid with holes, or a stack
    missing its ends, that stitched and projected as if it were whole; only a
    caller that read a returned count knew. *outside* maps each step's name to
    how many of its positions fall outside, in the protocol's step order.
    """
    if not outside:
        return
    count = sum(outside.values())
    names = common_utils.first_few(list(outside), separator=', ')
    _refuse_build(
        reason=reason,
        message=(
            f"{count} of the {total} {what} fall outside the stage's travel "
            f'(steps: {names}).\n\n{remedy}'
        ),
    )


def _axis_limits_or_refuse(
    axis_limits: 'Mapping[str, AxisLimits]', axes: tuple[str, ...], *, what: str
) -> 'dict[str, AxisLimits]':
    """The travel limits of each of *axes*, or a refusal naming the missing ones.

    *axis_limits* maps each axis the scope has a motor for to its travel
    limits; an axis absent from it has no motor.

    A z-stack or a tile grid is built by moving an axis. On a scope without
    that axis's motor there are no limits to build against, and building
    nothing while reporting nothing read as success: the caller asked for a
    stack and got a photograph. Refused here, where the protocol is built,
    so every caller -- the GUI, a script, REST -- is told the same thing.

    Raises:
        ProtocolRunRefusedError: An axis in *axes* has no limits.
    """
    _refuse_without_motors(axis_limits, axes, what=what)
    return {axis: axis_limits[axis] for axis in axes}


def _refuse_without_motors(present: 'Collection[str]', axes: tuple[str, ...], *, what: str) -> None:
    """Refuse building *what* when an axis in *axes* is not among the *present* ones.

    Raises:
        ProtocolRunRefusedError: ``positions_unreachable``, naming the axes
            with no motor.
    """
    missing = [axis for axis in axes if axis not in present]
    if missing:
        _refuse_build(
            reason='positions_unreachable',
            message=(
                f'This scope has no motor for {", ".join(missing)}, so {what} cannot be built.'
            ),
        )


class ProtocolFormatError(Refusal, ProtocolError):
    """A protocol LumaViewPro cannot take: a file it cannot parse, or steps that do not validate.

    A refusal: the file or the configuration is the person's, nothing broke,
    and the answer is to fix or choose another. The words say what is wrong
    and, for a file, name it, so the one report -- a warning in these words,
    one log line -- needs no log line of its own beside each check.

    Attributes:
        reason: ``'protocol_invalid'``.
        file: The file refused, or None for a protocol built in memory.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'protocol_invalid': RefusalCause.REQUEST}
    title = 'Protocol Refused'

    def __init__(self, problem: str, file=None):
        super().__init__(problem if file is None else f'{file} was not loaded: {problem}')
        self.file = file


class StepNotFoundError(Refusal, ProtocolError):
    """A step index the protocol does not have, refused before anything is read or written.

    A refusal: the caller named a step, and the words say which steps there
    are. Unmarked, a Delete on an empty protocol was reported as a fault --
    "Operation failed", an ERROR and a traceback for a designed answer.

    The words name no index: the index is the API's, zero-based, and a
    person who typed a step number into the panel counts from 1. A caller
    that wants it reads it here.

    Attributes:
        reason: ``'no_such_step'``.
        index: The index refused, as the caller gave it.
        num_steps: How many steps the protocol has.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'no_such_step': RefusalCause.REQUEST}
    title = 'No Such Step'

    def __init__(self, index: int, num_steps: int):
        if num_steps == 0:
            words = 'The protocol has no steps.'
        elif num_steps == 1:
            words = 'The protocol has 1 step.'
        else:
            words = f'The protocol has {num_steps} steps.'
        super().__init__(words)
        self.index = index
        self.num_steps = num_steps


class StepEditRefusedError(Refusal, ProtocolError):
    """A value a step writer cannot take; the protocol is unchanged.

    A name with nothing left once cleaned, a value not of its column's type,
    an insert that names no place or two, or another channel for a slice of
    a z-stack. A refusal, as a missing step is.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {'step_value_refused': RefusalCause.REQUEST}
    title = 'Step Not Changed'


class ProtocolScheduleRefusedError(ProtocolFormatError):
    """A period or duration the scope cannot run, refused rather than changed.

    Attributes:
        reason: ``'schedule_not_runnable'``.
        key: ``'period'`` or ``'duration'``.
        value: What was given, as given.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'schedule_not_runnable': RefusalCause.REQUEST
    }
    # A schedule the scope cannot run is a protocol refused, and says so.
    title = ProtocolFormatError.title

    _RULES: ClassVar[dict] = {
        'period': 'a period is 0 (one scan) or at least 1 second',
        'duration': 'a duration is 0 (one scan) or more',
    }

    def __init__(self, key: str, value: object, file=None):
        shown = (
            f'{value.total_seconds():g} s' if isinstance(value, datetime.timedelta) else repr(value)
        )
        super().__init__(f'the {key} {shown} cannot be run: {self._RULES[key]}', file=file)
        self.key = key
        self.value = value


class ProtocolAsksForStimulationError(ProtocolFormatError):
    """A protocol file that asks for stimulation, which this release does not perform.

    A file written before the protocol's stimulation was removed can ask for
    it, and under the release that wrote it a run stimulated. Loaded, it
    would run every step and stimulate none, saying nothing, so it is
    refused: the words name the first step or Layer Settings channel that
    asks, as the file spells it, and how to put the file right.

    Attributes:
        reason: ``'stimulation_not_performed'``.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'stimulation_not_performed': RefusalCause.REQUEST
    }
    title = ProtocolFormatError.title


def _refuse_asked_stimulation(
    protocol_df: pd.DataFrame, block_rows: dict | None, file_path: pathlib.Path
) -> None:
    """Refuse a file whose steps or Layer Settings ask for stimulation.

    Read only from a file that carries the columns, one written before v12:
    a step's Stim_Config holds an entry per channel, and an entry asks when
    its ``enabled`` is true; a Layer Settings row asks when its Stim_Enabled
    is. Each is read by the bool cell reader, so a value that is neither
    True nor False refuses the file as a cell, never taken for on or off.
    Entries that are all off state nothing and are dropped unread, a channel
    this release lacks included.

    Raises:
        ProtocolAsksForStimulationError: a step or a channel asks; the first
            is named, with how many steps ask.
        ProtocolFormatError: a Stim_Config cell is not a config of entries,
            or an ``enabled`` or a Stim_Enabled is not True or False.
    """
    asking = []
    if 'Stim_Config' in protocol_df.columns:
        names = protocol_df['Name'].to_numpy()
        for position, cell in enumerate(protocol_df['Stim_Config'].to_numpy()):
            where = f'step {position + 1} ({names[position]})'
            try:
                entries = _dict_cell(cell)
            except ValueError as e:
                raise ProtocolFormatError(
                    f'{where} has Stim_Config {cell!r}, which {e}.', file=file_path
                ) from None
            for channel, entry in entries.items():
                try:
                    if not isinstance(entry, dict):
                        raise ValueError('is not a config')
                    enabled = _bool_cell(entry.get('enabled', False))
                except ValueError as e:
                    raise ProtocolFormatError(
                        f'{where} has a Stim_Config entry for {channel}, {entry!r}, which {e}.',
                        file=file_path,
                    ) from None
                if enabled:
                    asking.append((where, channel))
                    break
    if asking:
        where, channel = asking[0]
        others = f' {len(asking)} steps ask for it, this one first.' if len(asking) > 1 else ''
        raise ProtocolAsksForStimulationError(
            f'{where} asks for stimulation on {channel}, which this release of LumaViewPro '
            f'does not perform.{others} To load it, set "enabled" to false in every entry of '
            'its Stim_Config column, or delete the column.',
            file=file_path,
        )
    for channel, row in (block_rows or {}).items():
        cell = row.get('Stim_Enabled')
        cell = cell.strip() if isinstance(cell, str) else cell
        if cell in (None, ''):
            continue
        try:
            enabled = _bool_cell(cell)
        except ValueError as e:
            raise ProtocolFormatError(
                f'the Layer Settings Stim_Enabled of {channel}, {cell!r}, {e}', file=file_path
            ) from None
        if enabled:
            raise ProtocolAsksForStimulationError(
                f'its Layer Settings turn stimulation on for {channel}, which this release of '
                "LumaViewPro does not perform. To load it, set that row's Stim_Enabled to "
                'False, or delete the column.',
                file=file_path,
            )


# The largest protocol file a load reads. A bound on how much memory one file
# may ask for, not a judgement about the file: the whole file is read into
# memory, several times over, before any of it is parsed. Step COUNT is
# deliberately not bounded: the count is known only after the parse has spent
# the memory, and a ceiling on it sized below what this app's own writer
# produces made saved protocols unreadable. Bytes bound the memory across file
# versions where they do not bound the steps: a load holds about 33 times the
# file's bytes for a v12 file (~150 B and ~5 KB a step) and about 43 times for
# a v2 file (~80 B and ~3.4 KB a step). 16 MiB holds about 109,000 steps of
# the current shape, a 96-well plate of 4 channels and 200 Z slices (76,800
# steps, 10.6 MiB) with room, for about 0.7 GB held at the ceiling.
PROTOCOL_FILE_CEILING_BYTES = 16 * 1024 * 1024

# Below a second, a period is a schedule the run loop cannot keep: it would
# start each scan before the last one's captures are taken.
_MIN_PERIOD = datetime.timedelta(seconds=1)
_UNITS = {'period': 'minutes', 'duration': 'hours'}


def _refuse_unless_runnable(
    key: str, value: object, file: pathlib.Path | str | None = None
) -> None:
    """The one range rule for a protocol's period and duration.

    None and 0 are one scan; a composite or standalone capture is built with
    None.
    """
    if value is None:
        return
    if not isinstance(value, datetime.timedelta):
        raise ProtocolScheduleRefusedError(key, value, file=file)
    floor = _MIN_PERIOD if key == 'period' else datetime.timedelta(0)
    if value != datetime.timedelta(0) and value < floor:
        raise ProtocolScheduleRefusedError(key, value, file=file)


def _bool_cell(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('is not True or False')


def _float_cell(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('is not a number')
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('is not a number') from None
    if not is_finite_number(number):
        raise ValueError('is not a number')
    return number


def _position_cell(value: object) -> float:
    """A step's X, Y or Z: a finite number, or NaN where the step has no position on that axis.

    A scope records no position on an axis it lacks, so a blank, a None or a
    NaN is that, not a cell of the wrong type. NaN is how the frame holds it:
    a float column keeps its arithmetic, and ``step`` answers None for it.
    """
    if (
        value is None
        or (isinstance(value, str) and value == '')
        or (isinstance(value, float) and np.isnan(value))
    ):
        return np.nan
    return _float_cell(value)


def _int_cell(value: object) -> int:
    number = _float_cell(value)
    if not number.is_integer():
        raise ValueError('is not a whole number')
    return int(number)


def _acquire_cell(value: object) -> str:
    if member_of(AcquireMode, value) is None:
        raise ValueError(f'is not {" or ".join(repr(mode.value) for mode in AcquireMode)}')
    return value


def _str_cell(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError('is not text')
    return value


def _dict_cell(value: object) -> dict:
    """A config cell: a dict, or the text of one.

    The current writer saves JSON; files written before it hold Python's
    repr of the dict (single quotes, True/False), read by literal_eval. A
    dict is read as a copy whose numpy scalars are the Python values they
    hold, so a step's config is copied into the settings and written to the
    protocol's file as Python values; the caller's dict is not changed. A
    ``ReadOnlyDict`` is returned as it is: it was read when it was made, and
    nothing can change it, so a step built from another shares its cell.
    """
    if isinstance(value, ReadOnlyDict):
        return value
    if isinstance(value, dict):
        return python_values(copy.deepcopy(value))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                raise ValueError('is not a config') from None
        if isinstance(parsed, dict):
            return parsed
    raise ValueError('is not a config')


# Each step column with the type its cell is read as: the one reader for a
# file's steps, a caller's frame and every in-place write. Only a position
# may be blank, where the step has none on that axis; an empty number or
# True/False is a cell of the wrong type.
_POSITION_COLUMNS = ('X', 'Y', 'Z')
_STEP_CELLS = {
    'Name': _str_cell,
    'X': _position_cell,
    'Y': _position_cell,
    'Z': _position_cell,
    'Auto_Focus': _bool_cell,
    'channel': _str_cell,
    'False_Color': _bool_cell,
    'Illumination': _float_cell,
    'Gain': _float_cell,
    'Auto_Gain': _bool_cell,
    'Exposure': _float_cell,
    'Sum': _int_cell,
    'Objective': _str_cell,
    'Well': _str_cell,
    'Tile': _str_cell,
    'Z-Slice': _int_cell,
    'Custom Step': _bool_cell,
    'Tile Group ID': _int_cell,
    'Z-Stack Group ID': _int_cell,
    'Acquire': _acquire_cell,
    'Video Config': _dict_cell,
    'Auto_Named': _bool_cell,
    'Label': _str_cell,
    'Channel Group ID': _int_cell,
    'Turret Slot': _int_cell,
    'Z-Stack Offset': _float_cell,
}


def _cells_differ(now: np.ndarray, then: np.ndarray) -> np.ndarray:
    """Per step, whether two columns' cells differ: neither equal nor both missing."""
    equal = np.asarray(now == then, dtype=bool)
    return ~(equal | (pd.isna(now) & pd.isna(then)))


def _read_step_column(df: pd.DataFrame, column: str, *, blank_kept: bool = False) -> list:
    """Every step's ``column`` cell, read as its column's type, in step order.

    The column is taken from the frame once and walked as the frame holds
    it: fetching each cell by position costs a pandas lookup per cell, about
    25 times the readers' own work on a large protocol. The walk is over the
    column's numpy values, so a refused cell is worded as the frame holds it.

    Args:
        blank_kept: a blank text cell stays blank instead of being read, as
            a file's cell pass keeps it; a position is read either way.

    Raises:
        ProtocolFormatError: a cell is not of its column's type; the words
            name the step and the column.
    """
    reader = _STEP_CELLS[column]
    blank_kept = blank_kept and column not in _POSITION_COLUMNS
    cells = []
    for position, cell in enumerate(df[column].to_numpy()):
        if blank_kept and cell == '':
            cells.append(cell)
            continue
        try:
            cells.append(reader(cell))
        except ValueError as e:
            name = df['Name'].iloc[position]
            raise ProtocolFormatError(
                f'step {position + 1} ({name}) has {column} {cell!r}, which {e}.'
            ) from None
    return cells


@api_fields('duration_s')
@dataclasses.dataclass(frozen=True)
class VideoConfig:
    """What a video step records.

    A step carries no rate: every recording keeps every frame the camera
    delivers, or samples down to the one global limit (``video.max_fps``).

    Attributes:
        duration_s: How long the step records, in seconds. None where the
            step's config does not say; a step that records no video may not
            say.
    """

    duration_s: float | None


@api_fields(
    'name',
    'x_mm',
    'y_mm',
    'z_um',
    'auto_focus',
    'channel',
    'false_color',
    'illumination_ma',
    'gain_db',
    'auto_gain',
    'exposure_ms',
    'frames_summed',
    'objective',
    'well',
    'tile',
    'z_slice',
    'custom_step',
    'tile_group_id',
    'z_stack_group_id',
    'acquire',
    'video_config',
    'auto_named',
    'label',
    'channel_group_id',
    'turret_slot',
    'z_stack_offset_um',
)
@dataclasses.dataclass(frozen=True)
class StepRecord:
    """One step of a protocol, a field per column of its steps table.

    Attributes:
        name: The step's name, rendered from its other fields.
        x_mm: X in mm on the protocol's plate; None where the step has no
            position on X.
        y_mm: Y in mm on the protocol's plate; None as for X.
        z_um: Z in um; None as for X.
        auto_focus: The step is autofocused before its capture.
        channel: The channel the step captures.
        false_color: The channel's image is false-coloured.
        illumination_ma: The LED current, in mA.
        gain_db: The camera gain, in dB.
        auto_gain: The camera's auto-gain is on for the step.
        exposure_ms: The camera exposure, in ms.
        frames_summed: How many frames the capture sums.
        objective: The objective's catalogue key.
        well: The plate's label for the well; '' for a step made at a
            position of the caller's.
        tile: The step's tile in its grid; '' for no tile.
        z_slice: The step's slice in its z-stack; None for none.
        custom_step: The step was made at a position of the caller's, not
            at a well of the plate.
        tile_group_id: The tile grid the step is one tile of; None for none.
        z_stack_group_id: The z-stack the step is one slice of; None for
            none.
        acquire: What the step captures.
        video_config: What the step records as video; None where the step
            carries no video config.
        auto_named: The name is rendered from the step's fields.
        label: The caller's label for the step; '' for none.
        channel_group_id: The steps captured together as one composite
            share it; None for a step recorded before groups were.
        turret_slot: The turret slot the step uses; None for none.
        z_stack_offset_um: The slice's distance in um from its channel's
            reference slice in its z-stack, the slice built at the focus
            the stack was built around; 0 for a step in no z-stack.
    """

    name: str
    x_mm: float | None
    y_mm: float | None
    z_um: float | None
    auto_focus: bool
    channel: str
    false_color: bool
    illumination_ma: float
    gain_db: float
    auto_gain: bool
    exposure_ms: float
    frames_summed: int
    objective: str
    well: str
    tile: str
    z_slice: int | None
    custom_step: bool
    tile_group_id: int | None
    z_stack_group_id: int | None
    acquire: AcquireMode
    video_config: VideoConfig | None
    auto_named: bool
    label: str
    channel_group_id: int | None
    turret_slot: int | None
    z_stack_offset_um: float


# Each steps-table column with the step record's field for it: the one place
# a column maps to a field. A column with no row fails the record's build.
_STEP_FIELDS = {
    'Name': 'name',
    'X': 'x_mm',
    'Y': 'y_mm',
    'Z': 'z_um',
    'Auto_Focus': 'auto_focus',
    'channel': 'channel',
    'False_Color': 'false_color',
    'Illumination': 'illumination_ma',
    'Gain': 'gain_db',
    'Auto_Gain': 'auto_gain',
    'Exposure': 'exposure_ms',
    'Sum': 'frames_summed',
    'Objective': 'objective',
    'Well': 'well',
    'Tile': 'tile',
    'Z-Slice': 'z_slice',
    'Custom Step': 'custom_step',
    'Tile Group ID': 'tile_group_id',
    'Z-Stack Group ID': 'z_stack_group_id',
    'Acquire': 'acquire',
    'Video Config': 'video_config',
    'Auto_Named': 'auto_named',
    'Label': 'label',
    'Channel Group ID': 'channel_group_id',
    'Turret Slot': 'turret_slot',
    'Z-Stack Offset': 'z_stack_offset_um',
}

# The columns whose -1 says the step has none, answered as None.
_NONE_AT_MINUS_ONE = (
    'Z-Slice',
    'Tile Group ID',
    'Z-Stack Group ID',
    'Channel Group ID',
    'Turret Slot',
)


def _no_position_as_none(row: 'pd.Series | dict') -> None:
    """Set None, in place, where ``row`` holds the frame's NaN for an axis position.

    The frame's NaN does not cross the wire, and None is what every record
    says for no position; a single step and the whole table answer it here.
    """
    for column in _POSITION_COLUMNS:
        if pd.isna(row[column]):
            row[column] = None


def _step_record(row: Mapping) -> StepRecord:
    """A step's row, as ``Protocol.step_row`` answers it, as its record: every column one field."""
    fields = {_STEP_FIELDS[column]: row[column] for column in _STEP_CELLS}
    for column in _NONE_AT_MINUS_ONE:
        if row[column] == -1:
            fields[_STEP_FIELDS[column]] = None
    video = row['Video Config']
    fields['video_config'] = (
        None if video is None else VideoConfig(duration_s=video.get('duration'))
    )
    return StepRecord(**fields)


@api_fields(
    'acquire',
    'illumination_ma',
    'gain_db',
    'auto_gain',
    'exposure_ms',
    'false_color',
    'frames_summed',
)
@dataclasses.dataclass(frozen=True)
class ChannelSettings:
    """One channel's row of a protocol's Layer Settings, each cell typed.

    Every field but ``acquire`` is None where the cell is blank, or where a
    row inferred from the channel's first step cannot supply it: applied,
    the channel's control keeps what it holds.

    Attributes:
        acquire: What the channel captures.
        illumination_ma: The LED current, in mA.
        gain_db: The camera gain, in dB.
        auto_gain: The camera's auto-gain is on.
        exposure_ms: The camera exposure, in ms.
        false_color: The channel's image is false-coloured.
        frames_summed: How many frames a capture sums.
    """

    acquire: AcquireMode
    illumination_ma: float | None
    gain_db: float | None
    auto_gain: bool | None
    exposure_ms: float | None
    false_color: bool | None
    frames_summed: int | None


# Each Layer Settings column after 'Layer', with the channel settings' field
# for it and the type its cell is read as: the one place a column maps to a
# field. Every cell but Acquire may be blank, read as None: the channel's
# control keeps what it holds.
_CHANNEL_SETTINGS_CELLS = (
    ('Acquire', 'acquire', _acquire_cell),
    ('Illumination', 'illumination_ma', _float_cell),
    ('Gain', 'gain_db', _float_cell),
    ('Auto_Gain', 'auto_gain', _bool_cell),
    ('Exposure', 'exposure_ms', _float_cell),
    ('False_Color', 'false_color', _bool_cell),
    ('Sum', 'frames_summed', _int_cell),
)


def _channel_settings_cells(record: ChannelSettings) -> list[str]:
    """A record's Layer Settings cells after 'Layer', in column order; None is blank.

    The writer's half of ``_CHANNEL_SETTINGS_CELLS``: what it writes,
    ``_channel_settings`` reads back as the same record.
    """
    return [
        '' if getattr(record, field) is None else str(getattr(record, field))
        for _, field, _ in _CHANNEL_SETTINGS_CELLS
    ]


def _channel_settings(row: dict) -> ChannelSettings:
    """One Layer Settings row as its record, each cell read as its column's type.

    The one reader for a saved block and an inferred one, so every caller
    of ``channel_settings()`` gets the same types and none casts a cell
    itself.

    Raises:
        ProtocolFormatError: A cell is not of its column's type; the words
            name the channel, the column and the cell.
    """
    channel = row.get('Layer')
    if not isinstance(channel, str) or channel.strip() == '':
        raise ProtocolFormatError(f'a Layer Settings row names no channel: {row!r}')
    channel = channel.strip()
    typed = {}
    for column, field, read in _CHANNEL_SETTINGS_CELLS:
        cell = row.get(column)
        if isinstance(cell, str):
            cell = cell.strip()
        if column != 'Acquire' and cell in (None, ''):
            typed[field] = None
            continue
        try:
            typed[field] = read(cell)
        except ValueError as e:
            raise ProtocolFormatError(
                f'the Layer Settings {column} of {channel}, {cell!r}, {e}'
            ) from None
    return ChannelSettings(**typed)


# Each step column a channel range holds, and the channel setting it is stored as.
_STEP_CHANNEL_VALUES = (
    ('Exposure', 'exposure_ms'),
    ('Illumination', 'illumination_ma'),
    ('Gain', 'gain_db'),
    ('Sum', 'sum'),
)

_VIDEO_KEYS = ('duration',)


def step_channel_values_refused(step: Mapping) -> list[tuple[str, SettingRefusedError]]:
    """Each of a step's values the channel ranges refuse, with the column that holds it.

    The rules a write of them to the channel's settings holds
    (``settings_paths.refused_channel_values``): the step's exposure,
    illumination, gain and sum, and its video's length (required when it
    records video).
    """
    # Deferred: settings_paths imports this module.
    from _lumascope.modules.settings_paths import refused_channel_values

    columns = {key: column for column, key in _STEP_CHANNEL_VALUES}
    values = {key: step[column] for column, key in _STEP_CHANNEL_VALUES}
    video = step.get('Video Config')
    if isinstance(video, dict) or step['Acquire'] == AcquireMode.VIDEO:
        video = video if isinstance(video, dict) else {}
        for part in _VIDEO_KEYS:
            if part in video or step['Acquire'] == AcquireMode.VIDEO:
                values[f'video_config.{part}'] = video.get(part)
                columns[f'video_config.{part}'] = f'Video Config {part}'
    return [
        (columns[r.path.partition('.')[2]], r)
        for r in refused_channel_values(step['channel'], values)
    ]


def channel_settings_values_refused(
    rows: Mapping[str, ChannelSettings],
) -> list[tuple[str, SettingRefusedError]]:
    """Each Layer Settings value the channel ranges refuse, with the channel and column holding it.

    ``rows`` are ``Protocol.channel_settings()``' records, or those of them
    a caller will apply. A blank cell holds no value and is not judged.
    """
    # Deferred: settings_paths imports this module.
    from _lumascope.modules.settings_paths import refused_channel_values

    columns = {key: column for column, key in _STEP_CHANNEL_VALUES}
    fields = {column: field for column, field, _ in _CHANNEL_SETTINGS_CELLS}
    refused = []
    for name, row in rows.items():
        values = {
            key: getattr(row, fields[column])
            for column, key in _STEP_CHANNEL_VALUES
            if getattr(row, fields[column]) is not None
        }
        refused.extend(
            (f'{name} {columns[r.path.partition(".")[2]]}', r)
            for r in refused_channel_values(name, values)
        )
    return refused


def refuse_values_out_of_range(refused: list[tuple[str, SettingRefusedError]], what: str) -> None:
    """Refuse a call that would store ``refused``, naming each value; nothing has changed.

    ``refused`` is what ``step_channel_values_refused`` or
    ``channel_settings_values_refused`` found; ``what`` names what holds the
    values, as a person reads it.

    Raises:
        ProtocolRunRefusedError: ``'values_out_of_range'``, when ``refused``
            is not empty.
    """
    if refused:
        raise ProtocolRunRefusedError(
            reason='values_out_of_range',
            message=f'{what} holds values no channel setting can take, so nothing was changed: '
            + '; '.join(f'{where}: {r.detail}' for where, r in refused)
            + '.',
        )


def _protocol_lines(file_path: pathlib.Path) -> Iterator[str]:
    """The lines of a protocol file, each read once, a blank one skipped.

    The one source both the header block and the step table read, so the
    file's text is never held whole; the file is open while it is read and
    closed when the source is. A line of whitespace alone is skipped
    wherever it falls: the shipped v2 example has one before its 'Steps'
    row. A byte that is not UTF-8 is refused wherever it falls, and a file
    that cannot be opened, or fails part way through, is not loaded.
    """
    try:
        with open(file_path, encoding='utf-8') as fp:
            for line in fp:
                if line.strip():
                    yield line
    except UnicodeDecodeError as e:
        raise ProtocolFormatError('it is not a text file.', file=file_path) from e
    except OSError as e:
        raise ProtocolNotLoadedError(file=file_path, cause=e) from e


def _header_rows(lines: Iterator[str], file_path: object) -> Iterator[list[str]]:
    """The rows of a protocol file's header block, read one at a time.

    Read as ``common_utils.read_table`` reads the step table, strictly, so
    a row the csv module cannot read (a cell over its field limit, text
    after a closing quote) is refused naming the file instead of escaping
    as a raw ``csv.Error``. Lazy, so ``lines`` stands at the step table once
    the 'Steps' row is taken.
    """
    try:
        yield from csv.reader(lines, delimiter='\t', strict=True)
    except csv.Error as e:
        raise ProtocolFormatError(
            f'its header block is not well-formed ({e}).', file=file_path
        ) from None


def _refuse_a_header_row_with_no_value(row: list[str], file_path: object) -> None:
    """Refuse a protocol header row that holds its name and no value cell.

    A row cut down to its name (a hand edit, a truncated save) reads as a
    one-cell row, and indexing its value would raise IndexError out of the
    reader instead of saying which row is wrong.
    """
    if len(row) < 2:
        raise ProtocolFormatError(f"The '{row[0]}' row has no value.", file=file_path)


def schedule_from_units(
    key: str,
    value: object,
    *,
    file: pathlib.Path | str | None = None,
    judged: bool = True,
    unit: str | None = None,
) -> datetime.timedelta | None:
    """A period in minutes or a duration in hours, as the protocol takes it.

    The units the protocol file, the settings store and the GUI's fields
    all hold; ``unit='seconds'`` takes the API's. Anything that is not a
    finite number, or (when ``judged``) is a number the protocol cannot
    run, is refused here in the range rule's words.

    Raises:
        ProtocolScheduleRefusedError: ``value`` is not a runnable ``key``.
    """
    if value is None:
        return None
    try:
        number = float(value)
        schedule = datetime.timedelta(**{unit or _UNITS[key]: number})
    except (TypeError, ValueError, OverflowError):
        raise ProtocolScheduleRefusedError(key, value, file=file) from None
    if judged:
        _refuse_unless_runnable(key, schedule, file)
    return schedule


@api_fields('message', 'num_steps', 'projected_mb')
@dataclasses.dataclass(frozen=True)
class ProtocolSizeAdvisory:
    """A protocol big enough that the user should be told before running it.

    Carries the numbers alongside the sentence so a caller that wants to
    render something other than the sentence -- REST, a log line -- does not
    have to parse it back out.

    Attributes:
        num_steps: How many steps the protocol has.
        projected_mb: About how many megabytes of images the protocol's
            steps write. When its video steps cannot be sized (the camera
            delivering no frames and no rate limit set), the still steps only.
        message: The advisory in words a person reads: the step count, the
            size, any video steps left unsized, and to check free disk.
    """

    num_steps: int
    projected_mb: float
    message: str


class StoredNames(enum.Enum):
    """What ``Protocol.from_file`` takes in place of a unit: the file's names, kept.

    For post-processing's read of a run's protocol, where no unit is known:
    the run's folders and files are named as its protocol names its
    channels, so nothing is renamed; and the protocol is never run or
    edited, so it is not judged as one that will be.
    """

    KEPT = 'kept'


def _in_unit_keys(named: dict, unit: 'ChannelIdentity', what: str, file_path: object) -> dict:
    """``named`` keyed by ``unit``'s key for each name (``ChannelIdentity.unit_key``).

    Raises:
        ProtocolFormatError: two of its names are one channel on ``unit`` --
            an old name and its key, written beside each other by hand --
            so neither value can be taken as the channel's.
    """
    renamed: dict = {}
    taken: dict = {}
    for name, value in named.items():
        key = unit.unit_key(name)
        if key in renamed:
            raise ProtocolFormatError(
                f'{what} names both {taken[key]} and {name}, which are one channel ({key}) '
                'on this scope.',
                file=file_path,
            )
        renamed[key], taken[key] = value, name
    return renamed


@dataclasses.dataclass(frozen=True)
class _SavedAs:
    """A protocol's record of what it was when last built, loaded or saved, and its file."""

    fingerprint: str
    file_path: pathlib.Path | None


@api_fields('CURRENT_COLUMNS', 'CURRENT_VERSION', 'PROTOCOL_FILE_HEADER')
class Protocol:
    """A protocol: its steps, its plate and its schedule, and the file it is read from and written to.

    Attributes:
        CURRENT_COLUMNS: The columns of the steps table a protocol file of
            ``CURRENT_VERSION`` holds, in the order they are written.
        CURRENT_VERSION: The version of the protocol file this release
            writes; a file of an earlier version is converted when it is read.
        PROTOCOL_FILE_HEADER: The first cell of a protocol file's first row,
            which a read checks before anything else.
    """

    PROTOCOL_FILE_HEADER: ClassVar[str] = 'LumaViewPro Protocol'
    COLUMNS: ClassVar[dict] = {
        1: [
            'Name',
            'X',
            'Y',
            'Z',
            'Auto_Focus',
            'Channel',
            'False_Color',
            'Illumination',
            'Gain',
            'Auto_Gain',
            'Exposure',
            'Objective',
        ],
        2: [
            'Name',
            'X',
            'Y',
            'Z',
            'Auto_Focus',
            'Color',
            'False_Color',
            'Illumination',
            'Gain',
            'Auto_Gain',
            'Exposure',
            'Objective',
            'Well',
            'Tile',
            'Z-Slice',
            'Custom Step',
            'Tile Group ID',
            'Z-Stack Group ID',
        ],
        3: [
            'Name',
            'X',
            'Y',
            'Z',
            'Auto_Focus',
            'Color',
            'False_Color',
            'Illumination',
            'Gain',
            'Auto_Gain',
            'Exposure',
            'Objective',
            'Well',
            'Tile',
            'Z-Slice',
            'Custom Step',
            'Tile Group ID',
            'Z-Stack Group ID',
            'Acquire',
            'Video Config',
        ],
        4: [
            'Name',
            'X',
            'Y',
            'Z',
            'Auto_Focus',
            'Color',
            'False_Color',
            'Illumination',
            'Gain',
            'Auto_Gain',
            'Exposure',
            'Sum',
            'Objective',
            'Well',
            'Tile',
            'Z-Slice',
            'Custom Step',
            'Tile Group ID',
            'Z-Stack Group ID',
            'Acquire',
            'Video Config',
        ],
        5: [
            'Name',
            'X',
            'Y',
            'Z',
            'Auto_Focus',
            'Color',
            'False_Color',
            'Illumination',
            'Gain',
            'Auto_Gain',
            'Exposure',
            'Sum',
            'Objective',
            'Well',
            'Tile',
            'Z-Slice',
            'Custom Step',
            'Tile Group ID',
            'Z-Stack Group ID',
            'Acquire',
            'Video Config',
            'Stim_Config',
        ],
        # v6: step columns unchanged from v5; the format-version bump
        # signals a new 'Layer Settings' block in the header preamble
        # that persists per-channel UI state (acquire mode + illumination
        # + gain + exposure + false_color + sum + stim-enabled) so a
        # protocol round-trips through the UI without losing per-channel
        # configuration. v5 files without the block fall back to
        # inference from the steps Color column (see channel_settings).
        6: [
            'Name',
            'X',
            'Y',
            'Z',
            'Auto_Focus',
            'Color',
            'False_Color',
            'Illumination',
            'Gain',
            'Auto_Gain',
            'Exposure',
            'Sum',
            'Objective',
            'Well',
            'Tile',
            'Z-Slice',
            'Custom Step',
            'Tile Group ID',
            'Z-Stack Group ID',
            'Acquire',
            'Video Config',
            'Stim_Config',
        ],
    }
    # v7 adds Auto_Named: an explicit flag for whether a step still carries its
    # auto-generated name (vs a user-typed one). It replaces inferring that from
    # the name string, which could not tell a user name that happened to match
    # the auto pattern from a real auto name.
    COLUMNS[7] = COLUMNS[6] + ['Auto_Named']
    # v8 adds Label: the step's base text as a first-class column -- a
    # user-typed name kept verbatim, or the 'custom<NNNN>' prefix of an added
    # step; empty means the base is the Well. Name becomes a DERIVED display
    # column rendered from the structured columns. Persisting the label is
    # what keeps a rename in every filename: recovering it by parsing the
    # rendered Name truncated any user text with a token-shaped segment
    # ('Treatment_10x' -> 'Treatment') and dropped renames of well-anchored
    # steps entirely.
    COLUMNS[8] = COLUMNS[7] + ['Label']
    # v9 adds Channel Group ID: which Add (or, for a generated protocol,
    # which position) a step's channel belongs to. Every Added step at one
    # position is otherwise identical in every column a builder groups on,
    # so separate Adds merged into one composite and all but one image per
    # channel were dropped. -1 is a step recorded before the column, whose
    # Add is not known. It also adds Turret Slot: the slot a step was made
    # at. A step that named only its objective could not say which of two
    # slots holding that objective it meant. -1: none (no turret, or a step
    # recorded before the column).
    COLUMNS[9] = COLUMNS[8] + ['Channel Group ID', 'Turret Slot']
    # v10 adds Z-Stack Offset: a slice's distance from its channel's
    # reference slice, the one built at the focus the stack was built
    # around. A run places every slice at the focus it finds plus this
    # offset. The reference was marked by Auto_Focus on that one slice,
    # which every other writer of Auto_Focus erased, and the found focus then
    # landed on the stack's floor. The version moves with the column so a
    # build that cannot place by it refuses the file instead of loading it.
    COLUMNS[10] = COLUMNS[9] + ['Z-Stack Offset']
    # v11 changes no column: a step's Video Config holds its duration and no
    # longer its fps. A step's fps capped its recording below the one global
    # limit with no control to see or change it; a build that still read it
    # would record a v11 step at its old default, so it refuses the file.
    COLUMNS[11] = COLUMNS[10]
    # v12 drops Stim_Config, and the Layer Settings drop Stim_Enabled. Every
    # step copied every channel's stimulation, off, and no run performed a
    # step's: the column was most of every file and of every load, and put
    # large protocols past the file ceiling. A protocol states what it does,
    # so stimulation returns only with a step that stimulates. An older file
    # that asks for it is refused at load (_refuse_asked_stimulation).
    COLUMNS[12] = COLUMNS[11].copy()
    COLUMNS[12].remove('Stim_Config')
    CURRENT_VERSION: ClassVar[int] = 12
    CURRENT_COLUMNS: ClassVar[tuple[str, ...]] = tuple(COLUMNS[CURRENT_VERSION])

    # Header columns for the v6 'Layer Settings' block. Order is the
    # written-out order; rows that omit trailing columns are still valid as
    # long as the leading required columns are present. The cells after 'Layer' are
    # _CHANNEL_SETTINGS_CELLS', the one map of a column to its field.
    CHANNEL_SETTINGS_COLUMNS: ClassVar[list] = [
        'Layer',
        *(column for column, _, _ in _CHANNEL_SETTINGS_CELLS),
    ]

    # Counts the changes to which steps this protocol holds and in what
    # order: a step added or deleted, or the frame replaced. A change to a
    # step's values does not count, since the step keeps its index. A step
    # index means something only against the revision it was read at, so a
    # caller holding one across a wait compares this before using it.
    _step_list_revision: int = 0

    # What this protocol was when it was last built, loaded or saved, and its
    # file (``record_saved``). None on a protocol no build, load or save has
    # returned: a run's copy, the post-processing reader's.
    _saved_as: _SavedAs | None = None

    def __init__(
        self,
        tiling_grids: TilingGrids,
        config: dict | None = None,
        *,
        runnable: bool = True,
    ):
        """A protocol over ``config``.

        Args:
            tiling_grids: The installation's tiling grids, the scope's one
                copy (``ProtocolsAPI.tiling_grids``): the grids this protocol
                accepts, lays its tiles out on and names its grid from.
            runnable: False only for a reader that never runs or edits the
                protocol (``from_file``'s, which says when): its period and
                duration are kept as given and not held to the range, and
                its step cells are kept as read.

        Raises:
            ProtocolScheduleRefusedError: the config's period or duration
                cannot be run, and ``runnable`` is True.
            ProtocolFormatError: a step cell is not of its column's type, and
                ``runnable`` is True.
        """
        self._tiling_grids = tiling_grids

        if config is None:
            self._config = {}
        else:
            self._config = config

        self._runnable = runnable
        for key in ('period', 'duration'):
            if runnable and key in self._config:
                _refuse_unless_runnable(key, self._config[key])

        # Cache for num_steps() -- invalidated by _step_list_changed().
        # num_steps was called 35x per step during real-HW protocol runs, so
        # caching len(self._config['steps']) is a measurable win (M14 follow-up).
        self._num_steps_cache: int | None = None

        # Construction is a frame replacement like any other, so it takes
        # the same path: a caller's empty or column-less frame becomes the
        # canonical typed empty frame here, not at the first consumer that
        # asks it for a column.
        if 'steps' in self._config:
            self._set_steps(self._config['steps'])

    @staticmethod
    def _build_z_height_map(values) -> dict:
        z_height_map = {}

        z_heights = sorted(set(values.astype('float').tolist()))
        if len(z_heights) >= 2:
            z_height_map = {z_height: idx for idx, z_height in enumerate(z_heights)}

        return z_height_map

    @api(reads=True)
    def period_s(self) -> float | None:
        """How often the protocol scans, in seconds; 0 or None for one scan.

        Raises:
            Nothing.
        """
        period = self._config['period']
        return None if period is None else period.total_seconds()

    @api(reads=True)
    def duration_s(self) -> float | None:
        """How long the protocol runs its scans, in seconds; 0 or None for one scan.

        Raises:
            Nothing.
        """
        duration = self._config['duration']
        return None if duration is None else duration.total_seconds()

    @api(reads=True)
    def num_scans(self) -> int:
        """How many scans a full run of the protocol takes: its duration over its period, whole.

        A period of 0 or None is one scan, and so is a duration shorter than
        one period: a run captures its first scan before any period elapses.
        Divided as timedeltas, which is exact where float seconds drop a scan
        on some ratios.

        Raises:
            Nothing.
        """
        period, duration = self._config['period'], self._config['duration']
        if not period:
            return 1
        return max(1, int((duration or datetime.timedelta(0)) / period))

    @api(reads=True)
    def labware_id(self) -> str:
        """The catalogue key of the plate the protocol's steps are placed on.

        Raises:
            Nothing.
        """
        return self._config['labware_id']

    @api(reads=True)
    def tiling(self) -> str | None:
        """The tile grid this protocol's steps carry.

        The no-tiling label when no step carries a tile; the grid's label
        when the steps' tiles form a grid this installation offers; None
        when they are tiled in a layout it does not offer, which only a
        hand-edited file produces, since every offered grid is square.

        Raises:
            Nothing.
        """
        tiles = [t for t in self._config['steps']['Tile'] if t not in (None, '')]
        if not tiles:
            return self._tiling_grids.no_tiling_label()
        return self._tiling_grids.determine_tiling_label_from_tiles(tiles)

    @staticmethod
    def sanitize_step_name(input: str) -> str:
        return re.sub(r'[^a-zA-Z0-9-_]', '', input)

    @staticmethod
    def _sanitized_label(step_name: str) -> str:
        """Sanitize a user-typed step name, refusing one that sanitizes away.

        Silently storing an empty label would revert the step to its machine
        base and collapse same-channel custom steps onto one filename; fail
        loud instead so the caller keeps the existing name.
        """
        label = Protocol.sanitize_step_name(step_name)
        if label == '':
            raise StepEditRefusedError(
                'Step name must contain at least one letter, digit, dash, or underscore.'
            )
        return label

    @api(reads=True)
    def capture_root(self) -> str:
        """The capture root as it was typed or loaded.

        Raises:
            Nothing.
        """
        return self._config.get('capture_root', '')

    @api(reads=True)
    def capture_prefix(self) -> str:
        """The capture root as the prefix of a saved file's name.

        The one derivation, read by the image writer and by post-processing:
        two derivations named the images ``exp202610_A1_BF`` and looked for
        ``exp/2026:10_A1_BF``. Empty when there is no root.

        Raises:
            Nothing.
        """
        return self.sanitize_step_name(self.capture_root() or '')

    @api(reads=True)
    def channel_settings(self) -> dict[str, ChannelSettings]:
        """Each channel's Layer Settings row as its record, keyed by channel.

        A protocol loaded from a file with a 'Layer Settings' block returns
        that block. One without (a v5 file, or a protocol built in memory)
        has it inferred from its steps: each channel with steps yields one
        record seeded from its first step, ``acquire`` 'video' when any of
        its steps records video, else 'image'.

        Every cell is its column's type (``ChannelSettings``), read once
        when the file was read; a blank cell but Acquire is None, and so is
        an inferred one its step cannot supply (see
        _channel_settings_inferred_from_steps).

        Raises:
            Nothing.
        """
        records = self._config.get('channel_settings')
        if records is None:
            return self._channel_settings_inferred_from_steps()
        return dict(records)

    def has_saved_channel_settings(self) -> bool:
        """Whether the protocol carries its own Layer Settings block.

        False for a v5 file or a protocol built in memory, whose
        ``channel_settings()`` are inferred from its steps.
        """
        return self._config.get('channel_settings') is not None

    def _channel_settings_inferred_from_steps(self) -> dict[str, ChannelSettings]:
        """Each channel's record, typed, from its steps.

        A channel with steps acquires: 'video' when any of its steps records
        video, else 'image'. Every other field is read from the channel's
        first step by its column's reader. A runnable protocol's step cells
        were typed when it was built, so each reader takes its cell; a cell
        a reader cannot take, in a protocol kept as read, has no value to
        infer, so it is None and the channel's control keeps what it holds.
        """
        steps = self._config.get('steps')
        if steps is None or len(steps) == 0 or 'channel' not in steps.columns:
            return {}
        out = {}
        for color, group in steps.groupby('channel'):
            if not color:
                continue
            first = group.iloc[0]
            video = 'Acquire' in group.columns and (group['Acquire'] == AcquireMode.VIDEO).any()
            fields = {'acquire': AcquireMode.VIDEO if video else AcquireMode.IMAGE}
            for column, field, read in _CHANNEL_SETTINGS_CELLS:
                if field in fields:
                    continue
                cell = first.get(column)
                try:
                    fields[field] = read(cell.strip() if isinstance(cell, str) else cell)
                except ValueError:
                    fields[field] = None
            out[str(color)] = ChannelSettings(**fields)
        return out

    def copy_for_execution(self) -> 'Protocol':
        """Lightweight copy for protocol execution.

        Copies the config dict and the steps DataFrame (which get mutated by
        autofocus Z updates), so nothing the run writes reaches the caller's
        protocol; shares the read-only dicts in the step cells, the frozen
        Layer Settings records and the read-only TilingGrids. Much cheaper
        than copy.deepcopy() for large protocols with many steps.
        """
        new_config = dict(self._config)  # shallow copy of config dict
        if 'steps' in new_config:
            new_config['steps'] = new_config['steps'].copy()
        new = Protocol.__new__(Protocol)
        new._config = new_config
        new._tiling_grids = self._tiling_grids  # shared, read-only
        new._num_steps_cache = None
        new._runnable = self._runnable
        return new

    def _header_text(self) -> str:
        """The file's header rows, as ``to_file`` writes them."""
        # Manual Z-Stack + single-shot capture pass period=None /
        # duration=None into the Protocol config (ui/zstack.py:187-188).
        # The loader accepts 0 as the single-scan marker (Period < 0 and
        # Duration < 0 are hard-rejected); emit 0 for None so a Manual
        # Z-Stack TSV round-trips cleanly. Pre-fix, Apply-Z-Projection on
        # any Manual Z-Stack aborted with "Invalid 'Period' value... must
        # be >= 0" because the writer emitted -1 (issue #669).
        # Period (Interval) needs the same sub-minute precision as Duration:
        # at 2 decimals a 1-second interval (0.016667 min) quantizes to 0.02
        # min on disk, corrupting the stored value. Match Duration's 6 decimals.
        period_s, duration_s = self.period_s(), self.duration_s()
        period_minutes = 0 if period_s is None else round(period_s / 60.0, 6)
        duration_hours = 0 if duration_s is None else round(duration_s / 3600.0, 6)

        out = io.StringIO()
        csvwriter = csv.writer(out, delimiter='\t', lineterminator='\n')
        csvwriter.writerow(['LumaViewPro Protocol'])
        csvwriter.writerow(['Version', self._config['version']])
        csvwriter.writerow(['Period', period_minutes])
        csvwriter.writerow(['Duration', duration_hours])
        csvwriter.writerow(['Labware', self._config['labware_id']])
        csvwriter.writerow(
            [
                'Capture Root',
                self.capture_root()
                if self.capture_root() is not None or self.capture_root() != ''
                else '',
            ]
        )
        return out.getvalue()

    def _channel_settings_text(self, channel_settings: Mapping[str, ChannelSettings] | None) -> str:
        """The file's Layer Settings block, as ``to_file`` writes it; empty with no record."""
        # v6 Layer Settings block -- emitted only when there is a
        # record to write; every record acquires, so each is a row.
        # Older readers tolerate the block (the 'find Steps' loop
        # in from_file already skips unknown rows until it hits
        # 'Steps'); this lets v6 files round-trip cleanly without
        # breaking any v5-aware downstream tooling.
        if not channel_settings:
            return ''
        out = io.StringIO()
        out.write('\nLayer Settings\n')
        csvwriter = csv.writer(out, delimiter='\t', lineterminator='\n')
        csvwriter.writerow(self.CHANNEL_SETTINGS_COLUMNS)
        for name, record in channel_settings.items():
            csvwriter.writerow([name, *_channel_settings_cells(record)])
        return out.getvalue()

    def _steps_text(self) -> str:
        """The file's steps table, as ``to_file`` writes it."""
        # Serialize dict columns as JSON strings before writing to CSV.
        # Without this, pandas uses Python repr (single quotes) which
        # fails json.loads() on reload.
        steps_df = self._config['steps'].rename(columns={'channel': 'Color'})
        for col in _DICT_COLUMNS:
            if col in steps_df.columns:
                steps_df[col] = steps_df[col].apply(
                    lambda v: json.dumps(v) if isinstance(v, dict) else v
                )
        return '\nSteps\n' + steps_df.to_csv(sep='\t', lineterminator='\n', index=False)

    @staticmethod
    def _fingerprint_of(header: str, steps: str) -> str:
        # The header and the steps, never the Layer Settings block: the block
        # is the session's channel controls at the save, which a person
        # changes without editing the protocol.
        return hashlib.sha256((header + steps).encode('utf-8')).hexdigest()

    def fingerprint(self) -> str:
        """The fingerprint of what ``to_file`` would write now, its Layer Settings block apart.

        Not published: what a build, a load and a save record
        (``record_saved``), and what ``has_unsaved_changes`` compares. Each
        read writes the protocol out in memory.

        Raises:
            Nothing.
        """
        return self._fingerprint_of(self._header_text(), self._steps_text())

    def record_saved(self, fingerprint: str, file_path: pathlib.Path | None) -> None:
        """Record what this protocol is as built, loaded or saved, and its file.

        Not published: called where a build, a load or a save returns whole
        (``ProtocolsAPI.create_protocol``, ``ScopeSession.load_protocol``,
        ``ScopeSession.save_protocol``), never by a client.

        Args:
            fingerprint: The fingerprint of the protocol as it then was
                (``fingerprint``, or what ``to_file`` returned).
            file_path: The file it was loaded from or saved to; None for one
                built.

        Raises:
            Nothing.
        """
        self._saved_as = _SavedAs(fingerprint, file_path)

    @api
    @property
    def has_unsaved_changes(self) -> bool:
        """Whether the protocol has changed since it was last built, loaded or saved.

        Every edit that changes what a save writes counts, whichever member
        made it; the session's channel controls, which a save writes as the
        Layer Settings block, do not. True for a protocol no build, load or
        save returned, such as a run's copy. Each read writes the protocol
        out in memory, so it is asked at a decision, not polled.

        Raises:
            Nothing.
        """
        return self._saved_as is None or self._saved_as.fingerprint != self.fingerprint()

    @api
    @property
    def file_path(self) -> pathlib.Path | None:
        """The file this protocol was last loaded from or saved to; None when it has none.

        Raises:
            Nothing.
        """
        return None if self._saved_as is None else self._saved_as.file_path

    def to_file(
        self,
        file_path: FilePath,
        channel_settings: Mapping[str, ChannelSettings] | None = None,
    ) -> str:
        """Write the protocol to a TSV file, whole or not at all.

        Not published: ``ScopeSession.save_protocol`` is the one way a
        caller writes a protocol file, so the write is the session's to
        admit; a run writes its own protocol's file through this.

        The file is written beside the target and replaces it only once it is
        complete, so a write that fails part-way leaves a file already there
        as it was: opening the target itself empties it first, and a failure
        then left a stub that no longer loaded.

        Args:
            file_path: Destination path.
            channel_settings: Each channel's record to write as its Layer
                Settings row, keyed by channel; a None field is a blank
                cell. When None, the records stored on the Protocol (set
                by load) are written; when both are absent the file is
                written without a 'Layer Settings' block (channel-enable
                state will be inferred from steps on reload, matching the
                v5 fallback path).

        A missing folder is made where ``file_path`` sends it.

        Returns:
            The fingerprint of what was written, its Layer Settings block
            apart (``fingerprint``). Nothing is recorded: a run's write of
            its own copy is not a save, so the caller that saves records it
            (``record_saved``).

        Raises:
            LiveFolderPathRefusedError: ``'not_a_file'``, a folder is at
                ``file_path``; ``'wrong_extension'``, its name does not end
                in ``.tsv`` (in any case), which is not added; ``'not_a_folder'``,
                a file is where its folder would be. Nothing is written.
            ProtocolNotSavedError: the write failed (chained from its
                ``OSError``); nothing at ``file_path`` changed.
        """
        file_path = path_utils.file_to_write(file_path, extension='.tsv')
        tmp_loc = file_path.with_name(file_path.name + '.tmp')
        if channel_settings is None:
            channel_settings = self._config.get('channel_settings')
        header, steps = self._header_text(), self._steps_text()
        block = self._channel_settings_text(channel_settings)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_loc, 'w', encoding='utf-8') as fp:
                fp.write(header + block + steps)
            os.replace(tmp_loc, file_path)
        except Exception as e:
            with contextlib.suppress(OSError):
                tmp_loc.unlink(missing_ok=True)
            if isinstance(e, OSError):
                raise ProtocolNotSavedError(file=file_path, cause=e) from e
            raise
        return self._fingerprint_of(header, steps)

    def optimize_step_ordering(self) -> None:
        """Order the steps by place, then by objective and z-stack slice at each place.

        A place's steps of one objective keep the order they were made in at
        every slice, which for a build is the channels' step order: Z orders
        only a stack's slices, lowest first, never one channel against another.
        Ordered by each channel's own Z, a place ran its channels in the
        order of their saved focus, so the same channels ran differently
        by how the protocol was built.

        Raises:
            Nothing.
        """
        steps = self._config['steps']

        if len(steps) == 0:
            return

        # Each X/Y location numbered in the order it first appears; steps with
        # no X/Y are one place, kept.
        place = steps.groupby(by=['X', 'Y'], sort=False, dropna=False).ngroup()

        # One sort over every step, not one per place: a built protocol has
        # thousands of places, and sorting each and joining them cost over a
        # second on a 1536-well tile build.
        # The keys sit in a frame of their own, indexed by position, so the
        # order takes any index the steps carry and no column of theirs.
        # Stable, so the channels at one slice keep the order they were made
        # in. A step that is no slice (-1) keeps its place among its own.
        keys = pd.DataFrame(
            {
                'place': place.to_numpy(),
                'Objective': steps['Objective'].to_numpy(),
                'Z-Slice': steps['Z-Slice'].to_numpy(),
            }
        )
        order = keys.sort_values(by=['place', 'Objective', 'Z-Slice'], kind='stable').index
        self._reorder_steps(order)

    @staticmethod
    def _create_empty_steps_df() -> pd.DataFrame:
        dtypes = np.dtype(
            [
                ('Name', str),
                ('X', float),
                ('Y', float),
                ('Z', float),
                ('Auto_Focus', bool),
                ('channel', str),
                ('False_Color', bool),
                ('Illumination', float),
                ('Gain', float),
                ('Auto_Gain', bool),
                ('Exposure', float),
                ('Sum', int),
                ('Objective', str),
                ('Well', str),
                ('Tile', str),
                ('Z-Slice', int),
                ('Custom Step', bool),
                ('Tile Group ID', int),
                ('Z-Stack Group ID', int),
                ('Acquire', str),
                ('Video Config', object),
                ('Auto_Named', bool),
                ('Label', str),
                ('Channel Group ID', int),
                ('Turret Slot', int),
                ('Z-Stack Offset', float),
            ]
        )
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
        return df

    # Valid values for field validation
    @staticmethod
    def valid_colors() -> set:
        """The channel names a protocol Color column may carry.

        Derived at call time from the release catalogue, so a channel added
        to the catalogue validates without touching this class -- an
        import-time constant would freeze the vocabulary at whatever the
        catalogue said when this module first loaded.
        """
        from _lumascope.modules.common_utils import get_channels

        return set(get_channels())

    @staticmethod
    def channel_acquires(channel_config: dict) -> bool:
        """Whether a channel's settings make it a step: acquire is image or video.

        The one answer for the builder, the step add and the refusal that
        guards both; a second spelling (``is not None``) once let a config
        with an empty-string acquire pass the add and build no step.
        """
        return member_of(AcquireMode, channel_config['acquire']) is not None

    @staticmethod
    def _capture_base_names(steps_df: pd.DataFrame) -> pd.Series:
        """The filename base each step's captures render, one per step.

        A video step renders with its 'video' suffix, so an image step and a
        video step may legitimately share a Name without their files
        colliding. The Objective column is deliberately NOT part of the
        base: the writer stamps the objective short name per step, but only
        on turret builds -- collision checks pair this base with the
        Objective column (see _capture_collision_key) instead of baking a
        token into the base that non-turret filenames never carry.
        """
        return steps_df.apply(
            lambda row: common_utils.build_step_name(
                common_utils.step_components(
                    row, post=('video',) if row.get('Acquire') == AcquireMode.VIDEO else ()
                )
            ),
            axis=1,
        )

    @classmethod
    def _capture_collision_key(cls, steps_df: pd.DataFrame) -> pd.DataFrame:
        """The per-step key on which capture filenames collide.

        Pairs the rendered base with the Objective column: the writer stamps
        the objective short name per step on turret builds, so steps
        differing only in objective save distinct files there. On a
        non-turret scope that pair genuinely collides on disk -- that
        residual case falls to the write-time rename suffix (with a warning)
        rather than refusing turret protocols that are legitimate.
        """
        return pd.DataFrame(
            {
                'base': cls._capture_base_names(steps_df),
                'objective': steps_df['Objective'] if 'Objective' in steps_df.columns else '',
            }
        )

    @api(in_process=True, reads=True)
    def validate_steps(
        self, objectives: 'Mapping[str, Objective]', *, led_max_ma: int | None
    ) -> list:
        """Validate all step fields and return a list of error strings.

        In-process: a wire client needs no call of its own. It is told of a
        loaded protocol's invalid steps when the protocol loads, and a run
        start refuses them.

        Args:
            objectives: The scope's objectives (``scope.objectives``); a step
                naming an objective they lack is an error.
            led_max_ma: The connected scope's LED current cap
                (``capabilities.led_max_ma``). Asked of the caller, not
                carried on the protocol: a copy made for a run would not
                carry it, and the gate would admit a step the LED refuses.
                None on a scope that came up without its LED board: there
                is no cap to judge against, and a run on it is refused for
                the missing LED controller.

        Returns an empty list if all steps are valid.

        Raises:
            Nothing.
        """
        errors = []
        steps = self._config['steps']
        if steps is None or len(steps) == 0:
            return errors

        # The caller's catalogue -- the scope's, read from its data folder. A
        # loader built here used to answer a failed load by skipping the
        # objective check for every step, and an empty catalogue skipped it
        # the same way -- so the one protocol that most needed refusing
        # validated clean.
        valid_objectives = set(objectives)
        valid_colors = self.valid_colors()

        # Each column read once and each step a dict of them: a Series per
        # step (iterrows) cost most of the judge on a built protocol of tens
        # of thousands of steps. A step's index is its position (_set_steps).
        columns = {column: steps[column].tolist() for column in steps.columns}
        for idx in range(len(steps)):
            step = {column: values[idx] for column, values in columns.items()}
            label = f'Step {idx + 1} ({step.get("Name", "?")})'

            # Color
            color = step.get('channel', '')
            if color not in valid_colors:
                errors.append(
                    f"{label}: Color '{color}' is not valid. "
                    f'Must be one of: {", ".join(sorted(valid_colors))}'
                )

            # Objective
            obj = step.get('Objective', '')
            if obj not in valid_objectives:
                errors.append(f"{label}: Objective '{obj}' not found in objectives.json")

            # Every cell is already its column's type (_set_steps); what is
            # judged here is whether the value is one the run can use, by the
            # channel ranges the settings hold. The camera's own floor is
            # checked when the run is prepared, against the connected body.
            for column, refused in step_channel_values_refused(step):
                errors.append(f'{label}: {column}: {refused.detail}')

            # The LED cap is the connected board's, not a range.
            illum = step['Illumination']
            if led_max_ma is not None and illum > led_max_ma:
                errors.append(f'{label}: Illumination must be 0-{led_max_ma} mA, got {illum}')

            sum_count = step['Sum']
            if sum_count > 100:
                logger.warning(
                    f'[PROTOCOL] {label}: Sum count {sum_count} is unusually high (> 100). This may cause long capture times.'
                )

            # Name length
            name = step.get('Name', '')
            if len(str(name)) > 200:
                errors.append(f'{label}: Name exceeds 200 characters ({len(str(name))})')

        return errors

    def validate_for_run(
        self,
        *,
        objective_helper: 'ObjectiveLoader',
        wellplate_loader: 'labware_loader.WellPlateLoader',
        led_max_ma: int | None,
    ) -> list:
        """Validate that the protocol's steps are well-formed enough to run.

        validate_steps() checks the camera and objective fields; this adds the
        checks a run needs on top: unique capture filenames and a known plate.
        Whether the positions lie inside the stage's travel is the scope's
        question, asked by the run gate
        (``ProtocolsAPI.refuse_positions_outside_travel``).

        Args:
            objective_helper: The scope's objective catalogue.
            wellplate_loader: The scope's labware catalogue.
            led_max_ma: The scope's LED current cap (``capabilities.led_max_ma``);
                None with no LED board (see ``validate_steps``).

        Returns:
            List of error strings. Empty list if all checks pass.
        """
        errors = []
        steps = self._config['steps']
        if steps is None or len(steps) == 0:
            return errors

        # Validate step field values first
        errors.extend(self.validate_steps(objective_helper.objectives, led_max_ma=led_max_ma))

        # Refuse a run whose steps would write identical files. Collisions
        # are caught here, loudly, before any hardware moves -- never left
        # to the write-time rename suffix, which preserves pixels but makes
        # which file belongs to which step undeterminable afterwards.
        if len(steps) > 1:
            collision_key = self._capture_collision_key(steps)
            dup_mask = collision_key.duplicated(keep=False)
            if dup_mask.any():
                groups = collision_key.loc[dup_mask].groupby(['base', 'objective']).groups
                for (base, _objective), indices in sorted(groups.items()):
                    rows = ', '.join(str(idx + 1) for idx in indices)
                    errors.append(
                        f"Steps {rows}: each would save captures as '{base}_...'. "
                        f'Rename these steps so each produces a unique filename.'
                    )

        # Use is_known_plate() rather than plate_list membership so
        # legacy/alias names (e.g. "384 well Corning Spheroid Microplate" ->
        # "384 well microplate") are accepted here exactly as they are at
        # runtime in get_plate(). Without this, validation hard-fails on
        # names that would have run fine.
        loader = wellplate_loader
        labware_key = self.labware_id()
        if labware_key:
            try:
                if not loader.is_known_plate(labware_key):
                    plate_list = loader.get_plate_list()
                    errors.append(
                        f"Labware '{labware_key}' not found. Available: {', '.join(plate_list)}"
                    )
            except Exception as ex:
                # A loader failure leaves labware unvalidated; surface it so
                # the gap is visible rather than a silently-skipped net.
                logger.warning(f'[Protocol] Labware validation skipped for {labware_key!r}: {ex}')

        return errors

    @api(reads=True)
    def num_steps(self) -> int:
        """How many steps the protocol holds.

        Raises:
            Nothing.
        """
        if self._num_steps_cache is None:
            self._num_steps_cache = len(self._config['steps'])
        return self._num_steps_cache

    def _refuse_unless_in_range(self, idx: int, *, lowest: int = 0, past_end: int = 0) -> None:
        """Refuse an index that names no step, before anything is written.

        A step index is 0..num_steps-1. insert_step's places widen it by one
        at an end: before_step_index may be num_steps (at the end) and after_step_index
        -1 (after no step, which is how an empty protocol takes its first).
        Checked here and not left to pandas, whose .at answers an index past
        the end, or -1, by appending a row nobody added.

        Raises:
            StepNotFoundError: idx is outside the range.
        """
        num_steps = self.num_steps()
        highest = num_steps - 1 + past_end
        if lowest <= idx <= highest:
            return
        raise StepNotFoundError(idx, num_steps)

    @api
    @property
    def step_list_revision(self) -> int:
        """A count that changes whenever the step list does, for a reader to tell it changed.

        Raises:
            Nothing.
        """
        return self._step_list_revision

    def _step_list_changed(self) -> None:
        self._num_steps_cache = None
        self._step_list_revision += 1

    def _reorder_steps(self, order: Collection[int]) -> None:
        """Store the steps in ``order``, a permutation of their positions, and record the change.

        A reorder changes no cell, so no cell is read again: each was read
        when ``_set_steps`` stored it. The frame is renumbered 0..n-1, as
        ``_set_steps`` numbers it.
        """
        self._config['steps'] = self._config['steps'].iloc[order].reset_index(drop=True)
        self._step_list_changed()

    def _set_steps(self, df: pd.DataFrame) -> None:
        """Assign the steps DataFrame and record that the step list changed.

        All code paths that replace self._config['steps'] should go through
        this helper so the cache and the revision stay consistent; a
        reorder, which changes no cell, goes through ``_reorder_steps``. In-place
        row mutations (e.g. delete_step's drop(inplace=True)) must call
        _step_list_changed() explicitly.

        A steps frame always carries the full column schema, at any row
        count. The expansions build their replacement with
        pd.DataFrame.from_dict(rows), and pandas gives that NO columns at
        all when rows is empty -- which every input step being skipped
        produces -- so a consumer's df[['X', 'Y']] raises KeyError rather
        than returning an empty selection. Restoring the schema here, at
        the one place the frame is replaced, keeps a protocol with no steps
        a queryable protocol for every consumer instead of each one
        carrying its own guard.

        A frame WITH rows must already carry every current column. Run code
        indexes columns by name mid-scan (the capture path reads 'Label' for
        the filename, tiling reads 'Auto_Named'), so a frame missing one
        would pass construction and fail every scan as a bare KeyError that
        the run loop cannot tell from a hardware fault. Extra columns are
        allowed: the z-stack marking adds and removes its own.

        Every cell of a runnable protocol is read by its column's reader
        (_STEP_CELLS), and each column takes the schema's dtype
        (_create_empty_steps_df): a cell of the wrong type refuses the frame,
        so a reader never meets one and nothing is guessed for it. A
        protocol that is not runnable keeps its cells as given.

        The dicts in its Video Config cells are stored read-only (ReadOnlyDict), so a copy of the frame can share them.

        A step's index is its position: the frame is renumbered 0..n-1 here.
        The writers address a step by label (.at) and step() by position
        (iloc), and a caller's frame may carry any index, so without this a
        write to an index in range lands on a new row instead of that step.

        Raises:
            ProtocolFormatError: a non-empty frame is missing current columns.
            ProtocolFormatError: a cell of a runnable protocol is not of its
                column's type; the words name the step and the column.
        """
        if df.empty:
            df = self._create_empty_steps_df()
        else:
            missing = [c for c in _STEP_CELLS if c not in df.columns]
            if missing:
                named = ', '.join(repr(column) for column in missing)
                raise ProtocolFormatError(
                    f"The protocol's steps lack the columns {named}, which every step needs."
                )
            if self._runnable:
                df = self._typed_steps(df)
            df = df.assign(
                **{
                    column: pd.Series(
                        [_read_only(cell) for cell in df[column]], index=df.index, dtype=object
                    )
                    for column in _DICT_COLUMNS
                    if column in df.columns
                }
            ).reset_index(drop=True)
        self._config['steps'] = df
        self._step_list_changed()

    @classmethod
    def _typed_steps(cls, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` with every step column read by its reader and given its dtype.

        Raises:
            ProtocolFormatError: a cell is not of its column's type, or a step
                has one of X and Y without the other.
        """
        dtypes = cls._create_empty_steps_df().dtypes
        columns = {
            column: pd.Series(
                _read_step_column(df, column),
                index=df.index,
                dtype=dtypes[column],
            )
            for column in _STEP_CELLS
        }
        # X and Y are one stage: a step has both or neither, since no scope
        # drives one without the other and a move needs the pair.
        half = (columns['X'].isna() != columns['Y'].isna()).to_numpy().nonzero()[0]
        if len(half):
            position = int(half[0])
            raise ProtocolFormatError(
                f'step {position + 1} ({df["Name"].iloc[position]}) has one of X and Y '
                'without the other: a step has both or neither.'
            )
        return df.assign(**columns)

    @staticmethod
    def _typed_value(column: str, value: object) -> object:
        """``value`` read as a ``column`` cell, for a write in place.

        A frame does not hold its schema: pandas admits None into a number or
        text column and anything into a dict column, so each in-place writer
        reads its values here before it writes.

        Raises:
            StepEditRefusedError: ``value`` is not of the column's type.
        """
        try:
            return _STEP_CELLS[column](value)
        except ValueError as e:
            raise StepEditRefusedError(f'a step {column} of {value!r} {e}') from None

    @staticmethod
    def _measured_z(value: object) -> float:
        """``value`` read as a focus Z: a number, never no position.

        The focus writers write a Z the stage or an autofocus measured; a None
        there would erase the step's focus, not record that it has none.

        Raises:
            StepEditRefusedError: ``value`` is not a finite number.
        """
        try:
            return _float_cell(value)
        except ValueError as e:
            raise StepEditRefusedError(f'a step Z of {value!r} {e}') from None

    def steps(self) -> pd.DataFrame:
        """A copy of the steps frame.

        A write to it does not reach the protocol, and its dict cells are
        read-only, so a write into one raises: the protocol changes only
        through its writers. Readers inside this class use the stored frame.
        A step with no position on an axis holds NaN in that column, the
        frame's own missing value; ``step`` answers None for it.

        Raises:
            Nothing.
        """
        return self._config['steps'].copy()

    def estimate_write_mb(self, *, video_as_frames: bool = False, video_fps: float | None) -> float:
        """Estimate the disk this whole protocol will write, in MB.

        One owner for "how big is this protocol", so the pre-run free-space
        guard and anything that wants to tell the user up front cannot drift
        apart. Lives on Protocol because the estimate has to know the steps
        frame schema, and Protocol is what guarantees that schema at any row
        count.

        Stateless by design -- no cached total. A cache here has to be
        invalidated by every mutator, and the in-place ones (modify_step
        flipping Acquire, delete_step dropping rows) do not go through the
        one place that would clear it; a stale total silently under-reserves
        disk for a run. Recomputing costs single-digit milliseconds on a
        protocol large enough to care about.

        Returns the estimate only, NOT floored at the run's minimum free-disk
        figure. That floor is the disk guard's refusal policy, not a property
        of the protocol -- folding it in would project gigabytes for a
        three-step protocol.

        Args:
            video_as_frames: Run-level flag -- video saved as individual
                frames rather than a compressed MP4.
            video_fps: The rate a video step is sized at
                (``video_cadence.video_sizing_fps``); None when it is not
                known, which only a protocol with no video step may pass.

        Returns:
            Estimated megabytes the whole protocol will write.

        Raises:
            ValueError: the protocol has a video step and ``video_fps`` is
                None.
        """
        steps = self._config['steps']
        n_steps = len(steps)
        if n_steps == 0:
            return 0.0

        # Every non-video step costs the same constant, so they are counted
        # with one mask and multiplied; only video rows have to be visited.
        # The mask matches the per-step estimator's own `!= 'video'` test --
        # NaN, missing and 'image' all compare False.
        video_mask = (steps['Acquire'] == AcquireMode.VIDEO).to_numpy()
        n_video = int(video_mask.sum())
        image_mb = common_utils.estimate_step_write_mb(
            None, video_as_frames=video_as_frames, video_fps=video_fps
        )
        total = float((n_steps - n_video) * image_mb)
        # Positional, matching step()'s own .iloc, so this total and a
        # per-step sum over the same protocol agree exactly.
        for pos in np.flatnonzero(video_mask):
            total += common_utils.estimate_step_write_mb(
                steps.iloc[pos],
                video_as_frames=video_as_frames,
                video_fps=video_fps,
            )
        return total

    def size_advisory(
        self, *, video_as_frames: bool = False, video_fps: float | None
    ) -> ProtocolSizeAdvisory | None:
        """Describe this protocol if it is large enough to warn about, else None.

        The threshold and the sentence live here rather than in the GUI, and
        a client asks through ``ScopeSession.protocol_size_advisory``, which
        passes the two run settings this takes: the GUI and a REST caller
        get the same answer, and the GUI's job is to render what comes back.

        Advisory only. Nothing here refuses anything -- a legitimate large
        protocol runs untouched, and the run-start disk guard is unchanged.

        Args:
            video_as_frames: True when the run saves a video step as individual frames
                rather than one compressed file, which writes more.
            video_fps: The rate a video step is sized at
                (``video_cadence.video_sizing_fps``); None when it is not
                known (no limit set and the camera delivering nothing), and
                the advisory then sizes the still steps and says the video
                steps are not sized.

        Raises:
            Nothing.
        """
        num_steps = self.num_steps()
        if num_steps <= common_utils.PROTOCOL_SIZE_ADVISORY_STEPS:
            return None

        n_video = int((self._config['steps']['Acquire'] == AcquireMode.VIDEO).sum())
        if video_fps is None and n_video:
            projected_mb = float(
                (num_steps - n_video)
                * common_utils.estimate_step_write_mb(
                    None, video_as_frames=video_as_frames, video_fps=None
                )
            )
            unsized = (
                f' Its {n_video:,} video step{"" if n_video == 1 else "s"} '
                f'{"is" if n_video == 1 else "are"} not sized while the camera '
                'is not delivering frames.'
            )
        else:
            projected_mb = self.estimate_write_mb(
                video_as_frames=video_as_frames, video_fps=video_fps
            )
            unsized = ''
        return ProtocolSizeAdvisory(
            num_steps=num_steps,
            projected_mb=projected_mb,
            message=(
                f'{num_steps:,} steps, about '
                f'{common_utils.format_disk_size_mb(projected_mb)} of images.{unsized} '
                f'Check free disk before running.'
            ),
        )

    def modify_autofocus(self, step_index: int, enabled: bool) -> None:
        """Turn autofocus on or off for the step at ``step_index``.

        For a slice of a z-stack, the flag is its channel's in that stack,
        so every slice of the channel there takes it.

        Raises:
            StepNotFoundError: ``step_index`` is not a step of this protocol.
                Nothing changes.
        """
        self._refuse_unless_in_range(step_index)
        self._config['steps'].loc[self._stack_channel_slices(step_index), 'Auto_Focus'] = (
            self._typed_value('Auto_Focus', enabled)
        )

    # A step's columns that one channel's slices of a z-stack hold as one,
    # and those every slice of the stack holds as one (``modify_step``).
    _STACK_CHANNEL_COLUMNS = (
        'Auto_Focus',
        'False_Color',
        'Illumination',
        'Gain',
        'Auto_Gain',
        'Exposure',
        'Sum',
        'Acquire',
        'Video Config',
    )
    _STACK_PLACE_COLUMNS = ('X', 'Y', 'Objective', 'Turret Slot')

    def _stack_slices(self, step_index: int) -> pd.Index:
        """The steps of ``step_index``'s z-stack, every channel; the step alone outside one."""
        steps = self._config['steps']
        group_id = steps.at[step_index, 'Z-Stack Group ID']
        if group_id == -1:
            return pd.Index([step_index])
        return steps.index[steps['Z-Stack Group ID'] == group_id]

    def _stack_channel_slices(self, step_index: int) -> pd.Index:
        """The steps of ``step_index``'s channel in its z-stack; the step alone outside one."""
        steps = self._config['steps']
        group_id = steps.at[step_index, 'Z-Stack Group ID']
        if group_id == -1:
            return pd.Index([step_index])
        return steps.index[
            (steps['Z-Stack Group ID'] == group_id)
            & (steps['channel'] == steps.at[step_index, 'channel'])
        ]

    @staticmethod
    def _channel_reference(steps: pd.DataFrame, slices: pd.Index) -> int:
        """A channel's reference slice in a stack, from its ``slices``.

        The slice at offset 0, or, when it was deleted, the surviving slice
        nearest it (the lower of two equally near).
        """
        offsets = steps.loc[slices, 'Z-Stack Offset']
        return int(min(slices, key=lambda idx: (abs(offsets[idx]), offsets[idx])))

    @classmethod
    def _channel_plane(cls, steps: pd.DataFrame, slices: pd.Index) -> float:
        """Where a channel's reference slice in a stack sits: its reference's Z less that slice's offset."""
        reference = cls._channel_reference(steps, slices)
        return float(steps.at[reference, 'Z'] - steps.at[reference, 'Z-Stack Offset'])

    def modify_autofocus_all_steps(self, enabled: bool) -> None:
        """Turn autofocus on or off for every step.

        Raises:
            Nothing.
        """
        # The whole column: every step is set, so the z-stack rule of
        # modify_autofocus (a channel's slices hold one flag) holds already.
        self._config['steps'].loc[:, 'Auto_Focus'] = self._typed_value('Auto_Focus', enabled)

    def delete_step(self, step_index: int) -> None:
        self._refuse_unless_in_range(step_index)
        self._config['steps'].drop(index=step_index, axis=0, inplace=True)
        self._config['steps'].reset_index(drop=True, inplace=True)
        self._step_list_changed()

    def modify_labware(
        self,
        labware_id: str,
    ):
        self._config['labware_id'] = labware_id

    def modify_time_params(
        self,
        period: datetime.timedelta | None,
        duration: datetime.timedelta | None,
    ) -> None:
        """Give the protocol a new period and duration, both or neither.

        Raises:
            ProtocolScheduleRefusedError: either cannot be run; the protocol keeps
                the timing it had.
        """
        _refuse_unless_runnable('period', period)
        _refuse_unless_runnable('duration', duration)
        self._config['period'] = period
        self._config['duration'] = duration

    def modify_step_z_height(self, step_index: int, z: float) -> None:
        self._refuse_unless_in_range(step_index)
        self._config['steps'].at[step_index, 'Z'] = self._measured_z(z)

    # What makes a step the one a focus was found for. Not the step-list
    # revision: an Update moves a step without changing the list.
    _FOCUS_IDENTITY = (
        ('X', 'position'),
        ('Y', 'position'),
        ('channel', 'channel'),
        ('Objective', 'objective'),
    )

    def adopt_focus_from(self, scanned: 'Protocol') -> None:
        """Take the Z of every step from *scanned*, the copy a scan focused.

        All or nothing: the steps must be the ones the scan focused -- the
        same number, each at the same position, channel and objective --
        or no Z is written.

        Raises:
            FocusNotWrittenError: The steps differ from the scanned copy's.
        """
        mine = self._config['steps']
        theirs = scanned._config['steps']
        if len(mine) != len(theirs):
            raise FocusNotWrittenError(
                f'the protocol has {len(mine)} steps, not the {len(theirs)} the scan focused'
            )
        # Compared a column at a time: a row fetched per step costs a pandas
        # row build each, over a second on a large protocol, on the run's
        # thread. The refusal names the first step that differs and, within
        # it, the first identity in _FOCUS_IDENTITY's order.
        differs = np.column_stack(
            [
                _cells_differ(mine[column].to_numpy(), theirs[column].to_numpy())
                for column, _ in self._FOCUS_IDENTITY
            ]
        )
        changed = differs.any(axis=1).nonzero()[0]
        if len(changed):
            step = int(changed[0])
            words = self._FOCUS_IDENTITY[int(differs[step].argmax())][1]
            raise FocusNotWrittenError(f"step {step + 1}'s {words} changed")
        mine['Z'] = theirs['Z'].to_numpy()

    def focus_sweep_step(self, step_index: int) -> int | None:
        """The step whose light the focus sweep at ``step_index`` runs in; None where none runs.

        A step in no z-stack sweeps in its own light when its Auto_Focus is
        on. A z-stack sweeps once, at its first slice, because every slice
        must be placed before any is captured and they are captured in
        order. It sweeps in the light of the first channel there whose
        Auto_Focus is on, at that channel's reference slice (or the slice
        nearest it, when the reference slice was deleted); no other slice
        sweeps. A run sweeps here and then hands the answer and the found
        focus to ``apply_zstack_group_focus``, and an estimate counts a sweep
        at each step this answers for.

        Raises:
            StepNotFoundError: ``step_index`` is not a step of this protocol.
        """
        self._refuse_unless_in_range(step_index)
        steps = self._config['steps']
        group_id = steps.at[step_index, 'Z-Stack Group ID']
        if group_id == -1:
            return step_index if bool(steps.at[step_index, 'Auto_Focus']) else None
        members = steps.index[steps['Z-Stack Group ID'] == group_id]
        if members[0] != step_index:
            return None
        focused = [idx for idx in members if bool(steps.at[idx, 'Auto_Focus'])]
        if not focused:
            return None
        return self._channel_reference(steps, self._stack_channel_slices(focused[0]))

    def apply_zstack_group_focus(self, reference_step_idx: int, z: float) -> int:
        """Place a whole z-stack around the focus ``z`` found at ``reference_step_idx``.

        ``reference_step_idx`` is the step ``focus_sweep_step`` answered for
        the stack. Its channel's reference slice is placed at ``z``, and
        every channel moves by the same amount, so the Z between channels is
        kept. Each slice is then placed at its channel's reference plus its
        own offset: the ladder is rigid, so a slice whose Z was written by
        hand goes back onto it, and a reference slice whose Z was written by
        hand moves its channel's plane. Writing the stored steps is what
        makes the stage move at all: each step's move is issued from this
        frame, not from the row object the run is holding.

        Returns the number of slices placed.

        Raises:
            StepNotFoundError: ``reference_step_idx`` is not a step of this
                protocol. Nothing changes.
        """
        self._refuse_unless_in_range(reference_step_idx)
        z = self._measured_z(z)
        steps = self._config['steps']
        group_id = steps.at[reference_step_idx, 'Z-Stack Group ID']
        members = steps.index[steps['Z-Stack Group ID'] == group_id]
        planes = {
            color: self._channel_plane(steps, members[steps.loc[members, 'channel'] == color])
            for color in steps.loc[members, 'channel'].unique()
        }
        shift = z - planes[steps.at[reference_step_idx, 'channel']]
        for idx in members:
            steps.at[idx, 'Z'] = self._measured_z(
                planes[steps.at[idx, 'channel']] + shift + steps.at[idx, 'Z-Stack Offset']
            )
        return len(members)

    def apply_focus_all_channel_steps(self, channel: str, z: float) -> int:
        """Set Z on EVERY step of the channel, unconditionally.

        A slice of a z-stack is set to ``z`` plus its offset from its
        channel's reference slice, so the channel's stacks are placed with
        their reference at ``z`` and keep their spacing.

        The explicit bulk counterpart to the selected-step Save Focus
        write: covers the fresh-protocol workflow where the user focuses
        a channel once and wants every step of that channel to start
        there. Takes no previous-Z parameter by design -- a selective
        "only steps that still look untouched" update is deliberately
        not representable, because sibling steps are all born at the
        identical channel focus, so matching a prior value is the default
        state and never evidence of user intent.

        Args:
            channel: Color/channel name (BF, Ex405, Ex488, Ex589, Lumi, etc.).
            z: The Z position to write to every step of the channel.

        Returns:
            Number of steps updated.
        """
        z = self._measured_z(z)
        steps_df = self._config['steps']
        if steps_df is None or len(steps_df) == 0:
            return 0
        mask = steps_df['channel'] == channel
        count = int(mask.sum())
        if count > 0:
            steps_df.loc[mask, 'Z'] = [
                self._measured_z(z + offset) for offset in steps_df.loc[mask, 'Z-Stack Offset']
            ]
        return count

    def modify_capture_root(self, capture_root: str) -> None:
        """Set the capture root, which prefixes the run's saved file names (``capture_prefix``).

        Raises:
            Nothing.
        """
        self._config['capture_root'] = capture_root

    def _render_step_names(self) -> None:
        """Render every step's derived Name from its structured columns."""
        steps = self._config['steps']
        steps['Name'] = pd.Series(
            [
                common_utils.build_step_name(common_utils.step_components(row))
                for _, row in steps.iterrows()
            ],
            index=steps.index,
            dtype=self._create_empty_steps_df().dtypes['Name'],
        )

    def _regenerate_step_name(self, step_index: int) -> str:
        """Re-render the step's derived Name from its structured columns.

        Name is a display rendering of (Label, Well, channel, Tile, Z-Slice);
        every mutation of those columns re-renders it here so the stored
        string can never drift from the fields it encodes.
        """
        step = self._config['steps'].loc[step_index]
        name = common_utils.build_step_name(common_utils.step_components(step))
        self._config['steps'].at[step_index, 'Name'] = name
        return name

    def modify_name(
        self,
        step_index: int,
        step_name: str,
    ) -> None:
        self._refuse_unless_in_range(step_index)
        self._config['steps'].at[step_index, 'Label'] = Protocol._sanitized_label(step_name)
        # A user typing a name makes it theirs: clear the auto flag so a later
        # channel change does not regenerate over it.
        self._config['steps'].at[step_index, 'Auto_Named'] = False
        self._regenerate_step_name(step_index=step_index)

    def modify_step(
        self,
        step_index: int,
        channel: str,
        channel_config: dict,
        plate_position: 'PlateLocation',
        objective_id: str,
        turret_slot: int,
        label: str | None = None,
    ) -> None:
        """Update a step in place; label=None keeps the step's existing label.

        A non-None label is a user rename (clears the auto flag). The derived
        Name is re-rendered from the updated columns either way, so a channel
        change updates exactly the channel token while the label -- user text
        or auto base -- rides along untouched.

        For a slice of a z-stack, the light and camera written are its
        channel's in that stack, so every slice of the channel there takes
        them: the stack is one acquisition of the channel at several heights.
        The X, Y and objective written are the stack's place, so every slice
        of every channel there takes them: a run visits a stack as one place,
        and a slice elsewhere would move the stage or the turret mid-stack.
        The Z and the label are the slice's own. Its channel cannot change:
        the slice's offset is measured from its channel's reference slice, so
        another channel would place it from a plane it was never built
        against.

        Raises:
            StepNotFoundError: ``step_index`` is not a step of this protocol.
            StepEditRefusedError: a value is not of its column's type, or
                ``channel`` is not the channel of a slice of a z-stack. Nothing
                changes.
        """
        self._refuse_unless_in_range(step_index)
        steps = self._config['steps']
        if (
            steps.at[step_index, 'Z-Stack Group ID'] != -1
            and channel != steps.at[step_index, 'channel']
        ):
            raise StepEditRefusedError(
                f'step {step_index + 1} is a slice of a z-stack of {steps.at[step_index, "channel"]}, '
                f'so it cannot become {channel}. Change the channel of the step before it is '
                'stacked, then stack it again.'
            )

        # Every value is read before any is written, so a refused one leaves
        # the step as it was.
        values = {
            'X': plate_position.x_mm,
            'Y': plate_position.y_mm,
            'Z': plate_position.z_um,
            'Auto_Focus': channel_config['autofocus'],
            'channel': channel,
            'False_Color': channel_config['false_color'],
            'Illumination': channel_config['illumination_ma'],
            'Gain': channel_config['gain_db'],
            'Auto_Gain': channel_config['auto_gain'],
            'Exposure': channel_config['exposure_ms'],
            'Sum': channel_config['sum'],
            'Objective': objective_id,
            'Turret Slot': turret_slot,
            'Acquire': channel_config['acquire'],
            'Video Config': channel_config['video_config'],
        }
        typed = {column: self._typed_value(column, value) for column, value in values.items()}
        if label is not None:
            typed['Label'] = Protocol._sanitized_label(label)
            typed['Auto_Named'] = False
        written_to = {
            **dict.fromkeys(
                Protocol._STACK_CHANNEL_COLUMNS, self._stack_channel_slices(step_index)
            ),
            **dict.fromkeys(Protocol._STACK_PLACE_COLUMNS, self._stack_slices(step_index)),
        }
        for column, value in typed.items():
            for idx in written_to.get(column, (step_index,)):
                steps.at[idx, column] = _read_only(value)
        self._regenerate_step_name(step_index=step_index)

    def new_channel_group_id(self) -> int:
        """A channel group id no step of this protocol carries yet."""
        steps = self._config['steps']
        if steps.empty:
            return 0
        return max(int(steps['Channel Group ID'].max()), -1) + 1

    def set_turret_slot(self, step_indexes: list[int], slot: int) -> None:
        """Point steps ``step_indexes`` at turret slot ``slot``; their other columns are kept."""
        for step_index in step_indexes:
            self._refuse_unless_in_range(step_index)
        typed = self._typed_value('Turret Slot', slot)
        for step_index in step_indexes:
            self._config['steps'].at[step_index, 'Turret Slot'] = typed

    def insert_step(
        self,
        step_name: str | None,
        channel: str,
        channel_config: dict,
        plate_position: 'PlateLocation',
        objective_id: str,
        channel_group_id: int,
        turret_slot: int,
        before_step_index: int | None = None,
        after_step_index: int | None = None,
    ) -> str:
        """Insert one channel's step; ``channel_group_id`` names the Add it belongs to.

        An Add makes one step per channel, all at one position and so
        identical in every other column a builder groups on: the group id is
        what keeps two Adds at one position two composites. The caller mints
        it once per Add (``new_channel_group_id``) and passes it to every
        step of that Add.
        """

        def _validate_inputs():
            if (before_step_index is None) and (after_step_index is None):
                raise StepEditRefusedError('Must specify after_step_index or before_step_index')

            if (before_step_index is not None) and (after_step_index is not None):
                raise StepEditRefusedError(
                    'Must specify only after_step_index or before_step_index, not both'
                )

            if before_step_index is not None:
                self._refuse_unless_in_range(before_step_index, past_end=1)
            else:
                self._refuse_unless_in_range(after_step_index, lowest=-1)

        _validate_inputs()

        auto_named = step_name is None
        if step_name is None:
            CUSTOM_INDEX_WIDTH = 4
            label = f'custom{self._config["custom_step_count"]:0{CUSTOM_INDEX_WIDTH}d}'
            self._config['custom_step_count'] += 1
        else:
            label = Protocol._sanitized_label(step_name)

        well = ''
        tile = ''  # Manually inserted step is not a tile
        zslice = -1
        custom_step = True
        tile_group_id = -1
        zstack_group_id = -1

        # Every channel's step takes the stage's current Z, not the channel's
        # saved focus: the focus of the channel being viewed is a better
        # guess for every acquiring channel than their older saved focus.
        # The saved focus is left as it is; it is where a protocol built for
        # every well (from_config) starts, the closest to the average focus
        # for a later autofocus over all steps.
        step_dict = self._create_step_dict(
            label=label,
            x=plate_position.x_mm,
            y=plate_position.y_mm,
            z=plate_position.z_um,
            af=channel_config['autofocus'],
            color=channel,
            fc=channel_config['false_color'],
            ill=channel_config['illumination_ma'],
            gain=channel_config['gain_db'],
            auto_gain=channel_config['auto_gain'],
            exp=channel_config['exposure_ms'],
            sum=channel_config['sum'],
            objective=objective_id,
            well=well,
            tile=tile,
            zslice=zslice,
            custom_step=custom_step,
            tile_group_id=tile_group_id,
            zstack_group_id=zstack_group_id,
            acquire=channel_config['acquire'],
            video_config=channel_config['video_config'],
            auto_named=auto_named,
            channel_group_id=channel_group_id,
            turret_slot=turret_slot,
            zstack_offset=0.0,
        )

        if before_step_index is not None:
            pos_index = before_step_index - 0.5
        else:
            pos_index = after_step_index + 0.5

        line = pd.DataFrame(data=step_dict, index=[pos_index])
        line = line.astype({'Video Config': 'object'})
        line.at[pos_index, 'Video Config'] = step_dict['Video Config']

        self._set_steps(
            pd.concat([self._config['steps'], line], ignore_index=False, axis=0)
            .sort_index()
            .reset_index(drop=True)
        )

        return step_dict['Name']

    @api(reads=True)
    def step_records(self) -> tuple[StepRecord, ...]:
        """Every step, in order, as its record: the step table in one read.

        Raises:
            Nothing.
        """
        return tuple(_step_record(row) for row in self._step_rows())

    def _step_rows(self) -> list[dict]:
        """Every step, in order, as a plain dict keyed by the steps table's columns.

        What ``step_row`` answers for each step -- Python scalars, None for
        no position -- read a column at a time: building one row at a time
        costs about ten times as much on a large protocol, and a run's
        start reads every step.

        ``.tolist()`` boxes a numeric column to Python scalars, but it leaves
        a numpy scalar sitting in an object column as it is. No stored frame
        holds one: ``_set_steps`` reads every cell by its column's reader.
        """
        steps = self._config['steps']
        columns = {column: steps[column].tolist() for column in steps.columns}
        rows = [
            dict(zip(columns, values, strict=True))
            for values in zip(*columns.values(), strict=True)
        ]
        for row in rows:
            _no_position_as_none(row)
        return rows

    def step_row(self, step_index: int) -> pd.Series:
        """One step, as a DETACHED copy of its row, keyed by the steps table's columns.

        The returned Series is built fresh, so it does NOT track later writes
        to the steps frame. A caller that holds a row across something that
        can rewrite the frame -- an autofocus result, a z-stack group being
        placed -- is holding the pre-write values, and re-reads have to come
        back through here. That row is also what the image writer records its
        position from, so a stale hold does not merely misinform the caller,
        it is written into the saved image.

        X, Y or Z is None where the step has no position on that axis: the
        frame's NaN does not cross the wire, and None is what every record
        says for no position.

        Raises:
            StepNotFoundError: ``step_index`` is not a step of this protocol.
        """
        self._refuse_unless_in_range(step_index)
        row = to_python_scalars(self._config['steps'].iloc[step_index]).astype(object)
        _no_position_as_none(row)
        return row

    def apply_tiling(
        self,
        tiling: str,
        frame_dimensions: dict,
        binning_size: int,
        axis_limits: 'Mapping[str, AxisLimits]',
        labware: 'labware_module.Labware',
        stage_offset: 'StageOffset',
        overlap_percent: float = 0.0,
        *,
        capabilities: 'ScopeCapabilities',
        objective_helper: 'ObjectiveLoader',
    ) -> None:
        """Expand every step into a tile grid, in the order a run visits them.

        The tile spacing derives from each step's objective and the scope's
        optics, so the caller that owns the scope hands its capabilities and
        its objective catalogue in; a protocol is a data object handed a
        scale, never one that finds a scope. *axis_limits* holds the travel
        of each axis the scope has a motor for, and only those.

        Raises:
            ProtocolRunRefusedError: the grid is not one this installation
                offers, the protocol is already tiled, a step's objective is
                not in the catalogue, the scope has no X/Y motor, or a tile
                falls outside the stage's travel. Each is refused before any
                step changes, and reported once.
        """

        # Every refusal comes before anything below touches the steps: a
        # refused build leaves the protocol as it was.
        _refuse_unless_tiling_offered(self._tiling_grids, tiling)

        # Tiled steps are carried over as they are, so a second grid over a
        # tiled protocol would change nothing while reporting success. There is
        # no un-tile path; the untiled protocol has to be reloaded.
        no_tiling = self._tiling_grids.no_tiling_label()
        current_tiling = self.tiling()
        if current_tiling != no_tiling:
            grid = f' ({current_tiling})' if current_tiling else ''
            _refuse_build(
                reason='already_tiled',
                message=(
                    f'This protocol is already tiled{grid}. Reload the '
                    f'original (untiled) protocol before changing the tiling.'
                ),
            )

        # Judged before the one-tile return: an overlap no grid can take is
        # refused whatever grid it came with.
        fill_factor = _fill_factor_or_refuse(overlap_percent)
        if tiling == no_tiling:
            return

        limits = _axis_limits_or_refuse(axis_limits, ('X', 'Y'), what='a tile grid')

        # A copy: the focal-length column is working data for this build, and
        # the stored frame is the protocol's own.
        orig_steps_df = self._config['steps'].copy()
        focal_lengths = {o.id: o.focal_length_mm for o in objective_helper.objectives.values()}
        orig_steps_df['focal_length'] = orig_steps_df['Objective'].map(focal_lengths)

        # A step whose objective the catalogue does not know has no focal
        # length, and its tiles would land at NaN positions.
        unknown = sorted(
            {str(o) for o in orig_steps_df.loc[orig_steps_df['focal_length'].isna(), 'Objective']}
        )
        if unknown:
            _refuse_build(
                reason='objective_not_in_catalogue',
                message=(
                    f'The protocol names {", ".join(unknown)}, which is not in this '
                    f"scope's objective catalogue, so its tile spacing cannot be worked out."
                ),
            )

        existing_max_tile_group_id = orig_steps_df['Tile Group ID'].max()
        tile_group_id = existing_max_tile_group_id + 1

        # A z-stack group is the slices acquired at ONE XY position. Tiling
        # multiplies the XY positions, so each (parent group, tile) pair has to
        # become a group of its own; carrying the parent's id across the tiles
        # tells every consumer that several XY positions are one stack. The
        # LED-hold decision reads exactly that and would then hold illumination
        # on the sample across the move from tile to tile.
        existing_max_zstack_group_id = orig_steps_df['Z-Stack Group ID'].max()
        next_zstack_group_id = existing_max_zstack_group_id + 1
        tiled_zstack_group_ids: dict[tuple[int, str], int] = {}

        new_steps = []
        outside: dict[str, int] = {}
        new_tiles = 0

        # Each column read once, in the two forms a step is read in here: the
        # copied step's Python scalars, and the numpy scalars the tile
        # arithmetic rounds. They are not interchangeable: numpy's round and
        # Python's settle a tie differently, so reading X and Y as Python
        # floats would move tile coordinates by a last digit.
        python_columns = {column: orig_steps_df[column].tolist() for column in orig_steps_df}
        numpy_columns = {column: orig_steps_df[column].to_numpy() for column in orig_steps_df}

        for idx in range(len(orig_steps_df)):
            orig_step = {column: values[idx] for column, values in numpy_columns.items()}
            orig_step_dict = {column: values[idx] for column, values in python_columns.items()}

            # A step whose grid cannot be computed (ConfigError when the scale
            # is unknown, which no step can tile around) fails the whole build:
            # the steps are only replaced at the end, so the protocol is left
            # as it was rather than silently missing that step.
            tiles = self._tiling_grids.get_tile_centers(
                config_label=tiling,
                focal_length=orig_step_dict['focal_length'],
                frame_size=frame_dimensions,
                fill_factor=fill_factor,
                binning_size=binning_size,
                capabilities=capabilities,
            )

            if len(tiles) == 1:  # No tiles generated
                new_steps.append(orig_step_dict)
                continue

            # If already a tile, copy it over to the new protocol
            if orig_step['Tile'] not in (None, ''):
                new_steps.append(orig_step_dict)
                continue

            x = orig_step['X']
            y = orig_step['Y']
            new_tiles += len(tiles)

            for tile_label, tile_position in tiles.items():
                x_tile = round(
                    x + tile_position['x'] / 1000, common_utils.max_decimal_precision('x')
                )  # in 'plate' coordinates
                y_tile = round(
                    y + tile_position['y'] / 1000, common_utils.max_decimal_precision('y')
                )  # in 'plate' coordinates

                if axes_outside_travel(
                    {'X': x_tile, 'Y': y_tile},
                    limits,
                    labware=labware,
                    stage_offset=stage_offset,
                ):
                    name = orig_step['Name']
                    outside[name] = outside.get(name, 0) + 1
                    continue

                # -1 is the not-part-of-a-stack sentinel this column uses
                # throughout; those steps stay ungrouped.
                parent_zstack_group_id = orig_step['Z-Stack Group ID']
                if parent_zstack_group_id == -1:
                    tile_zstack_group_id = -1
                else:
                    group_key = (int(parent_zstack_group_id), tile_label)
                    if group_key not in tiled_zstack_group_ids:
                        tiled_zstack_group_ids[group_key] = int(next_zstack_group_id)
                        next_zstack_group_id += 1
                    tile_zstack_group_id = tiled_zstack_group_ids[group_key]

                new_step_dict = self._create_step_dict(
                    label=orig_step['Label'],
                    x=x_tile,
                    y=y_tile,
                    z=orig_step['Z'],
                    af=orig_step['Auto_Focus'],
                    color=orig_step['channel'],
                    fc=orig_step['False_Color'],
                    ill=orig_step['Illumination'],
                    gain=orig_step['Gain'],
                    auto_gain=orig_step['Auto_Gain'],
                    exp=orig_step['Exposure'],
                    sum=orig_step['Sum'],
                    objective=orig_step['Objective'],
                    well=orig_step['Well'],
                    tile=tile_label,
                    zslice=orig_step['Z-Slice'],
                    custom_step=orig_step['Custom Step'],
                    tile_group_id=tile_group_id,
                    zstack_group_id=tile_zstack_group_id,
                    acquire=orig_step['Acquire'],
                    video_config=orig_step['Video Config'],
                    auto_named=orig_step['Auto_Named'],
                    channel_group_id=orig_step['Channel Group ID'],
                    turret_slot=orig_step['Turret Slot'],
                    zstack_offset=orig_step['Z-Stack Offset'],
                )

                new_steps.append(new_step_dict)

            tile_group_id += 1

        _refuse_positions_outside_travel(
            reason='tiles_outside_travel',
            what='tiles',
            outside=outside,
            total=new_tiles,
            remedy='Choose a smaller grid, or move those steps away from the edge of the stage.',
        )
        self._set_steps(pd.DataFrame.from_dict(new_steps))
        self.optimize_step_ordering()
        logger.info(
            f'[Protocol] Tile grid {tiling} applied: {len(orig_steps_df)} -> '
            f'{self.num_steps()} steps'
        )

    @staticmethod
    def refuse_unbuildable_zstack(
        zstack_params: dict,
        axis_limits: 'Mapping[str, AxisLimits]',
        stacks: list[tuple[str, float]],
    ) -> None:
        """Refuse a z-stack around each of *stacks* that cannot be built whole.

        *stacks* holds, for each step to be stacked, a name the refusal can
        show and the Z it is stacked around. The one judgment of a stack's
        buildability: ``apply_zstacking`` asks it of the steps it is about to
        expand, and an Add asks it of the steps it is about to insert, so the
        Add refuses before it adds anything.

        Raises:
            ProtocolRunRefusedError: the range or step size is not greater
                than zero, the scope has no Z motor, or a slice falls outside
                the Z travel. Reported once.
            ArgumentRefusedError: ``'not_offered'``, the reference is not a
                ``ZStackReference``.
        """
        refuse_unless_offered(ZStackReference, zstack_params['z_reference'], 'z_reference')
        _refuse_unless_zstack_has_extent(zstack_params)

        z_limits = _axis_limits_or_refuse(axis_limits, ('Z',), what='a z-stack')

        outside: dict[str, int] = {}
        total = 0
        for name, z in stacks:
            positions = ZStackConfig(
                range=zstack_params['range'],
                step_size=zstack_params['step_size'],
                current_z_reference=zstack_params['z_reference'],
                current_z_value=z,
            ).step_positions()
            total += len(positions)
            # A slice past the Z travel would drive the stage to the end of
            # travel and stop the run there.
            beyond = sum(
                1
                for position in positions.values()
                if axes_outside_travel({'Z': position}, z_limits)
            )
            if beyond:
                outside[name] = beyond
        _refuse_positions_outside_travel(
            reason='zslices_outside_travel',
            what='z-slices',
            outside=outside,
            total=total,
            remedy='Reduce the range, or move the focus of those steps away from the end of travel.',
        )

    def apply_zstacking(
        self,
        zstack_params: dict,
        axis_limits: 'Mapping[str, AxisLimits]',
        *,
        only: range | None = None,
    ) -> None:
        """Expand every step not already in a stack into a z-stack, in the order a run visits them.

        The slices of the steps one Add made at one place -- one New position
        and tile, or one Add -- are one z-stack group, every channel in it:
        a run focuses a group once and moves all its slices by what it found
        (Eric, 2026-10-14). A step that belongs to no Add (a protocol written
        before Adds were recorded) is a group of its own.

        *axis_limits* holds the travel of each axis the scope has a motor
        for, and only those.

        *only* limits the expansion to the steps at those indices, which sit
        together (one Add's steps), and leaves every other step where it is:
        their slices take the steps' place as one block, ordered by objective
        and then Z, as a run visits one position. Without it the whole
        protocol is reordered as a run visits it.

        Raises:
            ProtocolRunRefusedError: the range or step size is not greater
                than zero, the scope has no Z motor, or a slice falls outside
                the Z travel. Each is refused before any step changes, and
                reported once.
            ArgumentRefusedError: ``'not_offered'``, the reference is not a
                ``ZStackReference``, refused before any
                step changes -- on a protocol with no steps too, which never
                reaches the slice positions that would read it.
            StepNotFoundError: *only* names an index that is no step.
        """
        if only is not None and len(only) > 0:
            self._refuse_unless_in_range(only.start)
            self._refuse_unless_in_range(only[-1])
        rows = self._step_rows()
        expanded = range(len(rows)) if only is None else only
        unstacked = [idx for idx in expanded if rows[idx]['Z-Slice'] in (None, '', -1)]
        Protocol.refuse_unbuildable_zstack(
            zstack_params,
            axis_limits,
            [(rows[idx]['Name'], rows[idx]['Z']) for idx in unstacked],
        )

        to_stack = set(unstacked)
        steps = self._config['steps']
        next_group_id = steps['Z-Stack Group ID'].max() + 1
        group_ids: dict[tuple, int] = {}

        num_steps = self.num_steps()
        new_steps = []
        block: list[dict] = []
        block_at = None

        for row_idx, orig_step in enumerate(rows):
            orig_step_dict = dict(orig_step)

            # If already part of a Z-Stack, or not asked for, copy it over to the new protocol
            if row_idx not in to_stack:
                new_steps.append(orig_step_dict)
                continue
            if block_at is None:
                block_at = len(new_steps)

            zstack_config = ZStackConfig(
                range=zstack_params['range'],
                step_size=zstack_params['step_size'],
                current_z_reference=zstack_params['z_reference'],
                current_z_value=orig_step['Z'],
            )

            if zstack_config.number_of_steps() == 0:
                continue

            zstack_positions = zstack_config.step_positions()

            add = orig_step['Channel Group ID']
            place = (add, orig_step['X'], orig_step['Y']) if add != -1 else ('step', row_idx)
            if place not in group_ids:
                group_ids[place] = next_group_id
                next_group_id += 1
            zstack_group_id = group_ids[place]

            # Every slice records its distance from the reference slice, the
            # one built at the step's Z (the focus the stack is built around),
            # and every slice carries the step's Auto_Focus: the flag says the
            # channel is focused, the offset where each slice sits from the
            # focus found. A run sweeps once per stack, so the flag on every
            # slice does not walk them onto one plane.
            reference_position = zstack_positions[zstack_config.reference_slice()]
            max_precision = common_utils.max_decimal_precision(parameter='z')
            group_steps: list[dict] = []
            for zstack_slice, zstack_position in zstack_positions.items():
                new_step_dict = self._create_step_dict(
                    label=orig_step['Label'],
                    x=orig_step['X'],
                    y=orig_step['Y'],
                    z=zstack_position,
                    af=orig_step['Auto_Focus'],
                    color=orig_step['channel'],
                    fc=orig_step['False_Color'],
                    ill=orig_step['Illumination'],
                    gain=orig_step['Gain'],
                    auto_gain=orig_step['Auto_Gain'],
                    exp=orig_step['Exposure'],
                    sum=orig_step['Sum'],
                    objective=orig_step['Objective'],
                    well=orig_step['Well'],
                    tile=orig_step['Tile'],
                    zslice=zstack_slice,
                    custom_step=orig_step['Custom Step'],
                    tile_group_id=orig_step['Tile Group ID'],
                    zstack_group_id=zstack_group_id,
                    acquire=orig_step['Acquire'],
                    video_config=orig_step['Video Config'],
                    auto_named=orig_step['Auto_Named'],
                    channel_group_id=orig_step['Channel Group ID'],
                    turret_slot=orig_step['Turret Slot'],
                    zstack_offset=round(zstack_position - reference_position, max_precision),
                )

                group_steps.append(new_step_dict)

            (new_steps if only is None else block).extend(group_steps)

        if only is None:
            self._set_steps(pd.DataFrame.from_dict(new_steps))
            self.optimize_step_ordering()
        else:
            # Stable, so every slice keeps the Add's channel order.
            block.sort(key=lambda step_dict: (step_dict['Objective'], step_dict['Z-Slice']))
            if block_at is not None:
                new_steps[block_at:block_at] = block
            self._set_steps(pd.DataFrame.from_dict(new_steps))
        logger.info(
            f'[Protocol] Z-stack applied (range {zstack_params["range"]} um, step '
            f'{zstack_params["step_size"]} um): {num_steps} -> {self.num_steps()} steps'
        )

    @classmethod
    def from_config(
        cls,
        input_config: dict,
        tiling_grids: TilingGrids,
        *,
        capabilities: 'ScopeCapabilities',
        objective_helper: 'ObjectiveLoader',
        wellplate_loader: 'labware_loader.WellPlateLoader',
    ) -> 'Protocol':
        positions = input_config.get('positions')

        # Optional per-(well, channel) z map: when provided it overrides the
        # channel focus for matching steps. New Protocol does NOT populate it by
        # default -- the carry-over was removed (see
        # ProtocolSettings.new_protocol); this path is kept dormant for a
        # future opt-in. Empty / missing falls back to channel_config['focus'].
        previous_well_z = input_config.get('previous_well_z') or {}

        # Stored under the catalogue's key: a plate renamed since the caller
        # named it is found under the old spelling, and the protocol then
        # carries the name every later reader resolves.
        labware_id = wellplate_loader.resolve_plate_key(input_config['labware_id'])
        objective_id = input_config['objective_id']
        channel_configs = input_config['channel_configs']
        period = input_config['period']
        duration = input_config['duration']
        # Where a channel whose focus was never saved is imaged: the stage's Z
        # when the config was built. Read only for such a channel.
        current_z = input_config.get('current_z')

        # The objective is stamped into every step; a protocol with none --
        # an empty one -- is built with no objective (None), since there may
        # be none in the light path to name. A named one is the catalogue's.
        acquiring = any(cls.channel_acquires(cfg) for cfg in channel_configs.values())
        if objective_id is None:
            if acquiring:
                _refuse_build(
                    reason='objective_not_given',
                    message='Protocol steps need an objective, and the configuration names none.',
                )
        else:
            objective_helper.get_objective_info(objective_id=objective_id)

        # A scope records only what it measures: without an X/Y stage a step
        # has no X/Y (the plate's well centres are places it cannot reach),
        # and without Z no Z, so no focus is asked for.
        has_xy = {'X', 'Y'} <= set(capabilities.axes)
        has_z = 'Z' in capabilities.axes

        config = {
            'version': cls.CURRENT_VERSION,
            'period': period,
            'duration': duration,
            'positions': positions,
            'labware_id': labware_id,
            'custom_step_count': 0,
            'capture_root': '',
        }

        if positions is not None:
            actual_positions = positions
            position_source = 'from_manual'
        else:
            well_positions = wellplate_loader.get_plate(
                labware_id=labware_id
            ).get_positions_with_labels()

            tmp = []
            for well_position in well_positions:
                well_x, well_y, well_label = well_position
                tmp.append(
                    {
                        'x': well_x if has_xy else None,
                        'y': well_y if has_xy else None,
                        # Per-(well, channel) tuned z is resolved in the channel
                        # loop below; do not collapse it to one z per well here.
                        'z': None,
                        'name': well_label,
                    }
                )

            actual_positions = tmp
            position_source = 'from_labware'

        # Every channel at one position is one group: the tiles and z-slices
        # the protocols API builds from a position's steps keep it, and the
        # builders separate those by Tile and Z-Slice.
        channel_group_id = 0
        steps = []

        # One step per acquiring channel per position; the grid and the stack
        # are built over these by apply_tiling and apply_zstacking.
        for pos in actual_positions:
            for channel_name, channel_config in channel_configs.items():
                if not cls.channel_acquires(channel_config):
                    continue

                # Each axis as given: one of X and Y without the other is
                # the steps table's to refuse, naming the step.
                x = y = None
                if pos['x'] is not None:
                    x = round(pos['x'], common_utils.max_decimal_precision('x'))
                if pos['y'] is not None:
                    y = round(pos['y'], common_utils.max_decimal_precision('y'))

                # Resolve Z per (well, channel): a tuned carry-over for
                # this exact channel wins (so New keeps per-channel
                # focus, not one channel's z for all); else an explicit
                # manual-position z; else this channel's saved focus;
                # else, never saved, the current Z.
                tuned_z = previous_well_z.get((pos.get('name'), channel_name))
                if not has_z:
                    z = None
                elif tuned_z is not None:
                    z = tuned_z
                elif pos['z'] is not None:
                    z = pos['z']
                elif channel_config['focus'] is not None:
                    z = channel_config['focus']
                elif current_z is not None:
                    z = current_z
                else:
                    _refuse_build(
                        reason='focus_not_given',
                        message=(
                            f'No focus is saved for {channel_name} and the '
                            'configuration names no current Z to image it at.'
                        ),
                    )

                if z is not None:
                    z = round(z, common_utils.max_decimal_precision('z'))

                autofocus = channel_config['autofocus']
                false_color = channel_config['false_color']
                illumination = round(
                    channel_config['illumination_ma'],
                    common_utils.max_decimal_precision('illumination'),
                )
                sum = int(channel_config['sum'])
                gain = round(channel_config['gain_db'], common_utils.max_decimal_precision('gain'))
                auto_gain = common_utils.to_bool(channel_config['auto_gain'])
                exposure = round(
                    channel_config['exposure_ms'],
                    common_utils.max_decimal_precision('exposure'),
                )
                video_config = channel_config['video_config']

                well_label = pos['name']
                if position_source == 'from_labware':
                    custom_step = False
                else:
                    custom_step = True

                step_dict = cls._create_step_dict(
                    # A manual position's name is its label (there is
                    # no well anchor); a labware well needs none.
                    label=well_label if custom_step else '',
                    x=x,
                    y=y,
                    z=z,
                    af=autofocus,
                    color=channel_name,
                    fc=false_color,
                    ill=illumination,
                    gain=gain,
                    auto_gain=auto_gain,
                    exp=exposure,
                    sum=sum,
                    objective=objective_id,
                    well=well_label,
                    tile='',
                    zslice=-1,
                    custom_step=custom_step,
                    tile_group_id=-1,
                    zstack_group_id=-1,
                    acquire=channel_config['acquire'],
                    video_config=video_config,
                    auto_named=True,
                    channel_group_id=channel_group_id,
                    # The protocols API stamps the slot once the
                    # protocol is built: this build has no scope.
                    turret_slot=-1,
                    zstack_offset=0.0,
                )
                steps.append(step_dict)

            channel_group_id += 1

        steps_df = cls._create_empty_steps_df()
        steps_df = cls._add_steps_to_steps_df(steps_df=steps_df, new_steps=steps)

        config['steps'] = steps_df

        protocol = cls(
            tiling_grids=tiling_grids,
            config=config,
        )

        # Validate step fields -- reject protocol if any errors found
        validation_errors = protocol.validate_steps(
            objective_helper.objectives, led_max_ma=capabilities.led_max_ma
        )
        if validation_errors:
            raise ProtocolFormatError(
                f'Protocol has {len(validation_errors)} validation error(s): {validation_errors[0]}'
            )

        return protocol

    @classmethod
    def create_empty(
        cls,
        config: dict,
        tiling_grids: TilingGrids,
        *,
        capabilities: 'ScopeCapabilities',
        objective_helper: 'ObjectiveLoader',
        wellplate_loader: 'labware_loader.WellPlateLoader',
    ) -> 'Protocol':
        labware_id = config['labware_id']
        # No steps and one tile: nothing to stamp an objective into, so none
        # is read -- the protocol can be created before the objective is known.
        objective_id = None
        zstack_params = {'range': 0, 'step_size': 0}
        use_zstacking = False
        tiling = TilingGrids.no_tiling_label()
        channel_configs = {}
        period = config['period']
        duration = config['duration']
        frame_dimensions = config['frame_dimensions']

        input_config = {
            'labware_id': labware_id,
            'objective_id': objective_id,
            'zstack_params': zstack_params,
            'use_zstacking': use_zstacking,
            'tiling': tiling,
            'channel_configs': channel_configs,
            'period': period,
            'duration': duration,
            'frame_dimensions': frame_dimensions,
            'binning_size': config['binning_size'],
        }

        return cls.from_config(
            input_config=input_config,
            tiling_grids=tiling_grids,
            capabilities=capabilities,
            objective_helper=objective_helper,
            wellplate_loader=wellplate_loader,
        )

    @staticmethod
    def _create_step_dict(
        label,
        x,
        y,
        z,
        af,
        color,
        fc,
        ill,
        gain,
        auto_gain,
        exp,
        sum,
        objective,
        well,
        tile,
        zslice,
        custom_step,
        tile_group_id,
        zstack_group_id,
        acquire,
        video_config,
        auto_named,
        channel_group_id,
        turret_slot,
        zstack_offset,
    ):
        """Build a step dict; the derived Name is rendered here, never passed.

        Deriving Name from the structured fields at the single creation
        point means no caller can hand in a Name that disagrees with the
        columns it encodes.
        """
        step = {
            'Name': '',
            'X': x,
            'Y': y,
            'Z': z,
            'Auto_Focus': af,
            'channel': color,
            'False_Color': fc,
            'Illumination': ill,
            'Gain': gain,
            'Auto_Gain': auto_gain,
            'Exposure': exp,
            'Sum': sum,
            'Objective': objective,
            'Well': well,
            'Tile': tile,
            'Z-Slice': zslice,
            'Custom Step': custom_step,
            'Tile Group ID': tile_group_id,
            'Z-Stack Group ID': zstack_group_id,
            'Acquire': acquire,
            'Video Config': video_config,
            'Auto_Named': auto_named,
            'Label': label,
            'Channel Group ID': channel_group_id,
            'Turret Slot': turret_slot,
            'Z-Stack Offset': zstack_offset,
        }
        step['Name'] = common_utils.build_step_name(common_utils.step_components(step))
        return step

    @staticmethod
    def _add_steps_to_steps_df(steps_df: pd.DataFrame, new_steps: list[dict]):
        new_steps_df = pd.DataFrame(new_steps)
        steps_df = pd.concat([steps_df, new_steps_df], ignore_index=True).reset_index(drop=True)
        return steps_df

    @staticmethod
    def _with_one_tile_per_zstack_group(
        protocol_df: pd.DataFrame, file_path: pathlib.Path
    ) -> pd.DataFrame:
        """``protocol_df`` with each tile of a z-stack group in a group of its own.

        A z-stack group is the slices taken at one place. Tiling a protocol
        already stacked once copied the stack's group onto every tile, and a
        file saved then still says so: read as written, its LED is held lit
        while the stage travels between tiles, and one focus places every
        tile. Each (group, tile) of a group holding several tiles takes a new
        id, minted as ``apply_tiling`` mints them, and each file so read is
        warned of once. The tile, not the position, says which slices are one
        place: a scope without X and Y records no position, and a slice
        updated at a nudged stage sits apart from its stack. A blank group
        cell is left for the constructor to refuse.
        """
        groups = protocol_df['Z-Stack Group ID']
        stacked = protocol_df.index[groups.map(lambda group: group != '' and group != -1)]
        tiles_per_group = protocol_df.loc[stacked].groupby('Z-Stack Group ID', sort=False)['Tile']
        spanning = [group for group, tiles in tiles_per_group if tiles.nunique() > 1]
        if not spanning:
            return protocol_df

        next_group_id = int(groups[stacked].max()) + 1
        minted: dict[tuple[int, str], int] = {}

        def one_place(group, tile):
            if group not in spanning:
                return group
            key = (int(group), tile)
            if key not in minted:
                nonlocal next_group_id
                minted[key] = next_group_id
                next_group_id += 1
            return minted[key]

        # A whole column, as the offset inference writes its own: the
        # constructor types every cell of the frame the read hands it.
        protocol_df['Z-Stack Group ID'] = pd.Series(
            [
                one_place(group, tile)
                for group, tile in zip(groups, protocol_df['Tile'], strict=True)
            ],
            index=protocol_df.index,
            dtype=object,
        )
        logger.warning(
            f'Protocol {file_path.name}: z-stack group(s) '
            f'{", ".join(str(group) for group in spanning)} each spanned several tiles, '
            f'as tiling an already-stacked protocol saved them; each tile is now a z-stack '
            f'of its own ({len(minted)} in all), so each is lit and focused at its own place.'
        )
        return protocol_df

    @staticmethod
    def _with_inferred_zstack_offsets(
        protocol_df: pd.DataFrame, file_path: pathlib.Path
    ) -> pd.DataFrame:
        """``protocol_df``, from a file before Z-Stack Offset, with each slice's offset recovered.

        Such a file marked a stack's reference by Auto_Focus on one slice, a
        mark any other Auto_Focus write erased. For each channel in each
        stack, the one flagged slice is its reference; with none or several
        flagged, the slice nearest the middle of the channel's span is (the
        lower of two equally near), and each stack recovered that way is
        logged once. Each slice's offset is its Z less its reference's, and a
        channel with any slice flagged is flagged on all of them, as a stack
        built now is. A step in no stack takes 0.
        """
        # The cells were read by their readers, a blank kept blank for the
        # constructor to refuse, so a flag is True only where the file says so.
        offsets = pd.Series(0.0, index=protocol_df.index, dtype=float)
        flags = protocol_df['Auto_Focus'].copy()
        stacked = protocol_df.index[protocol_df['Z-Stack Group ID'] != -1]
        max_precision = common_utils.max_decimal_precision(parameter='z')
        inferred = set()
        for (group_id, _color), slices in (
            protocol_df.loc[stacked].groupby(['Z-Stack Group ID', 'channel']).groups.items()
        ):
            z = protocol_df.loc[slices, 'Z'].astype(float)
            flagged = [idx for idx in slices if flags[idx] != '' and bool(flags[idx])]
            if len(flagged) == 1:
                reference = flagged[0]
            else:
                middle = (z.min() + z.max()) / 2
                reference = min(slices, key=lambda idx: (abs(z[idx] - middle), z[idx]))
                inferred.add(group_id)
            for idx in slices:
                offsets[idx] = round(z[idx] - z[reference], max_precision)
            if flagged:
                flags[slices] = True
        for group_id in sorted(inferred, key=str):
            logger.info(
                f'[Protocol] {file_path.name}: z-stack {group_id} recorded no reference slice; '
                'its middle slice is taken as the reference.'
            )
        return protocol_df.assign(**{'Z-Stack Offset': offsets, 'Auto_Focus': flags})

    @classmethod
    def from_file(
        cls,
        file_path: pathlib.Path,
        tiling_grids: TilingGrids,
        *,
        unit: 'ChannelIdentity | StoredNames',
        wellplate_loader: 'labware_loader.WellPlateLoader | None' = None,
    ) -> 'Protocol':
        """
        Returns Protocol object loaded from file on success
        Raises ProtocolNotLoadedError when the file cannot be read, and
        ProtocolFormatError on format issues and, when ``wellplate_loader``
        (the installation's catalogue) is given, on a Labware row naming a
        plate that catalogue does not have

        ``unit`` is the identity of the scope the protocol is read to run on,
        and each channel name the file holds -- a step's Color, a Layer
        Settings row -- is read as that unit's key for it
        (``ChannelIdentity.unit_key``): an old name meant an LED, and on this
        unit that LED's channel may be another key. Such a protocol is
        runnable: a period or duration no protocol can run raises
        ProtocolScheduleRefusedError, naming the file, a step cell or a
        Layer Settings cell that is not of its column's type raises
        ProtocolFormatError, naming the file, the step or channel, and the
        column, and a file from before v12 that asks for stimulation raises
        ProtocolAsksForStimulationError, naming the step or channel.

        ``StoredNames.KEPT`` is for a reader that never runs or edits the
        protocol -- post-processing a finished run reads its record and never
        its period -- so its names are kept as the run's files carry them,
        and a file saved before the period had a floor, or with a blank step
        cell a run would refuse, stays readable there; a step cell or a
        Layer Settings cell that is neither blank nor its column's type
        refuses the file either way.
        """
        runnable = unit is not StoredNames.KEPT

        config = {}

        try:
            file_size = os.path.getsize(file_path)
            if file_size > PROTOCOL_FILE_CEILING_BYTES:
                raise ProtocolFormatError(
                    f'it is larger than the {PROTOCOL_FILE_CEILING_BYTES // (1024 * 1024)} MB a protocol '
                    f'may be ({file_size:,} bytes) and cannot be loaded without exhausting memory.',
                    file=file_path,
                )
        except OSError as e:
            raise ProtocolNotLoadedError(file=file_path, cause=e) from e

        with contextlib.closing(_protocol_lines(file_path)) as lines:
            csvreader = _header_rows(lines, file_path)

            try:
                verify = next(csvreader)
            except StopIteration:
                raise ProtocolFormatError(
                    'Protocol file is empty or invalid.', file=file_path
                ) from None

            if not (verify[0] == cls.PROTOCOL_FILE_HEADER):
                raise ProtocolFormatError('Not a valid LumaViewPro Protocol', file=file_path)

            try:
                version_row = next(csvreader)
            except StopIteration:
                raise ProtocolFormatError(
                    'Protocol file is missing version information.', file=file_path
                ) from None

            if version_row[0] != 'Version':
                raise ProtocolFormatError(
                    "Protocol format is missing 'Version' row.", file=file_path
                )
            _refuse_a_header_row_with_no_value(version_row, file_path)

            try:
                config['version'] = int(version_row[1])
            except ValueError as ve:
                raise ProtocolFormatError(
                    "Invalid 'Version' value in protocol file: must be a whole number",
                    file=file_path,
                ) from ve

            allowed = False
            if config['version'] == cls.CURRENT_VERSION:
                allowed = True

            elif 2 <= config['version'] < cls.CURRENT_VERSION:
                # v6 introduced the Layer Settings header block; v7 added the
                # Auto_Named column; v8 the Label column; v9 the Channel Group
                # ID and Turret Slot columns; v10 the Z-Stack Offset column; v11
                # dropped the fps from each step's Video Config; v12 the
                # stimulation columns. Older versions load with the missing
                # columns recovered or defaulted below, and a stored fps and the
                # stimulation columns dropped.
                allowed = True

            if not allowed:
                raise ProtocolFormatError(
                    f'protocol version {config["version"]} is not supported; create a new protocol '
                    'in this version of LumaViewPro.',
                    file=file_path,
                )

            # Read Period
            try:
                period_row = next(csvreader)
                if period_row[0] != 'Period':
                    raise ProtocolFormatError(
                        "Missing 'Period' row in protocol file", file=file_path
                    )
                _refuse_a_header_row_with_no_value(period_row, file_path)

                # Period == 0 is a valid single-scan / non-periodic marker
                # (Z-stack, single-shot capture, autofocus characterization);
                # downstream consumers in protocol_time_estimator already treat
                # period_s == 0 as one scan rather than dividing by zero.
                config['period'] = schedule_from_units(
                    'period', period_row[1], file=file_path, judged=runnable
                )

            except StopIteration:
                raise ProtocolFormatError(
                    "Missing 'Period' row in protocol file", file=file_path
                ) from None

            except ProtocolFormatError as pfe:
                raise pfe

            # Read Duration
            try:
                duration = next(csvreader)
                if duration[0] != 'Duration':
                    raise ProtocolFormatError(
                        "Missing 'Duration' row in protocol file", file=file_path
                    )
                _refuse_a_header_row_with_no_value(duration, file_path)

                # Duration == 0 mirrors Period == 0 -- valid single-scan
                # marker for Manual Z-Stack / single-shot capture (where the
                # config carries duration=None). Refusing 0 here once kept
                # Apply-Z-Projection broken: the writer saves None as 0.
                config['duration'] = schedule_from_units(
                    'duration', duration[1], file=file_path, judged=runnable
                )

            except StopIteration:
                raise ProtocolFormatError(
                    "Missing 'Duration' row in protocol file", file=file_path
                ) from None

            except ProtocolFormatError as pfe:
                raise pfe

            # Read Labware
            try:
                labware = next(csvreader)
                if labware[0] != 'Labware':
                    raise ProtocolFormatError(
                        "Invalid 'Labware' row in protocol file", file=file_path
                    )
                _refuse_a_header_row_with_no_value(labware, file_path)

                # Stored in the catalogue's spelling whatever the file carried:
                # a plate renamed since the file was saved is translated here,
                # once, and nothing downstream sees the old name. Whether the
                # plate EXISTS is judged only when the caller handed over the
                # installation's catalogue -- post-processing reads a run's
                # protocol on whatever install it is on and never uses the
                # plate, so it has no standing to refuse for it.
                if wellplate_loader is None:
                    config['labware_id'] = labware_loader.canonical_plate_name(labware[1])
                else:
                    try:
                        config['labware_id'] = wellplate_loader.resolve_plate_key(labware[1])
                    except ConfigError as e:
                        raise ProtocolFormatError(str(e), file=file_path) from e

            except StopIteration:
                raise ProtocolFormatError(
                    "Missing 'Labware' row in protocol file", file=file_path
                ) from None

            except ProtocolFormatError as pfe:
                raise pfe

            # Optional fields not present in older/newer files
            # Ensure defaults to avoid KeyErrors
            try:
                next_row = next(csvreader)
                if next_row[0] == 'Capture Root':
                    # The shipped data/new_default_protocol.tsv writes
                    # "Capture Root" as a single-cell row (no tab + value),
                    # so next_row[1] raises IndexError. Treat a missing or
                    # empty value column as the empty-capture-root default.
                    config['capture_root'] = next_row[1] if len(next_row) > 1 else ''
                else:
                    config['capture_root'] = ''
            except StopIteration:
                raise ProtocolFormatError('Protocol file is incomplete.', file=file_path) from None

            # Search for "Steps" to indicate start of steps. Along the way,
            # optionally capture a v6 'Layer Settings' block -- a header row
            # followed by one row per enabled channel. Block ends at the first
            # blank row or at the 'Steps' marker. Its rows are typed into
            # records once the channel names are read as the unit's.
            block_rows = None
            if next_row[0] != 'Steps':  # Check to ensure compatibility with no capture_root field
                found_steps = False
                try:
                    while not found_steps:
                        tmp = next(csvreader)
                        if len(tmp) == 0:
                            continue

                        if tmp[0] == 'Steps':
                            found_steps = True
                            break

                        if tmp[0] == 'Layer Settings':
                            # Parse the block in place. The first non-blank row is
                            # the column header; subsequent non-blank rows are
                            # per-channel entries; block ends at the first blank
                            # row or at the 'Steps' marker. A header with no
                            # 'Layer' column is refused: an inference from the
                            # steps would put back settings the file never held.
                            ls_header = None
                            block_rows = {}
                            for sub_row in csvreader:
                                if len(sub_row) == 0 or sub_row[0] == '':
                                    if ls_header is not None:
                                        break  # blank row ends the block
                                    continue  # tolerate blank rows before header
                                if sub_row[0] == 'Steps':
                                    found_steps = True
                                    break
                                if ls_header is None:
                                    ls_header = sub_row
                                    if 'Layer' not in ls_header:
                                        raise ProtocolFormatError(
                                            "its Layer Settings block has no 'Layer' column.",
                                            file=file_path,
                                        )
                                    continue
                                row_dict = dict(zip(ls_header, sub_row, strict=False))
                                channel_name = row_dict.get('Layer', '').strip()
                                if not channel_name:
                                    raise ProtocolFormatError(
                                        'a row of its Layer Settings block names no channel.',
                                        file=file_path,
                                    )
                                block_rows[channel_name] = row_dict
                except StopIteration:
                    raise ProtocolFormatError(
                        "Missing 'Steps' section in protocol file", file=file_path
                    ) from None
                # The writer never writes a block with no rows; one that has none
                # would turn every channel off at a restore.
                if block_rows == {}:
                    raise ProtocolFormatError(
                        'its Layer Settings block has no rows.', file=file_path
                    )

            try:
                header, rows = common_utils.read_table(lines, sep='\t')
            except ValueError as e:
                raise ProtocolFormatError(f'{e}.', file=file_path) from None
        protocol_df = pd.DataFrame(rows, columns=header, dtype=object)
        # The frame holds the cells; their row lists would stay to the end of the load.
        del rows

        # M19: Validate required columns before processing.
        # Old versions use 'Channel' instead of 'Color' -- accept either.
        actual_cols = set(protocol_df.columns)
        required_columns = {
            'Name',
            'X',
            'Y',
            'Z',
            'Illumination',
            'Gain',
            'Exposure',
            'Well',
            'Tile',
            'Z-Slice',
        }
        if 'Color' not in actual_cols and 'Channel' not in actual_cols:
            required_columns.add('Color')  # will trigger error
        missing = required_columns - actual_cols
        if missing:
            raise ProtocolFormatError(
                f'Protocol missing required columns: {missing}', file=file_path
            )

        # The file names the channel column 'Color', the oldest files
        # 'Channel'; in memory it is 'channel', named once here at the read
        # boundary and written back as 'Color' by to_file.
        file_column = 'Color' if 'Color' in protocol_df.columns else 'Channel'
        protocol_df = protocol_df.rename(columns={file_column: 'channel'})

        # Each step cell is read by its column's reader, so a name or label of
        # 'NA' or '0600' stays that text and a number is the number written.
        # A blank cell stays blank: a runnable protocol refuses it at
        # construction, and post-processing's reader keeps it. A blank
        # position is the step's having none on that axis, read by its reader.
        # Any other cell its reader refuses refuses the file, naming the step
        # and column. The config columns are parsed below, with their legacy
        # repairs.
        for column in _STEP_CELLS:
            if column in protocol_df.columns and column not in _DICT_COLUMNS:
                try:
                    protocol_df[column] = pd.Series(
                        _read_step_column(protocol_df, column, blank_kept=True),
                        index=protocol_df.index,
                        dtype=object,
                    ).infer_objects()
                except ProtocolFormatError as e:
                    raise ProtocolFormatError(str(e), file=file_path) from None

        # A runnable protocol's Z-Slice is read with its other cells, and a
        # blank one refuses the file (no shipped writer left one blank). The
        # post-processing reader keeps the -1 sentinel for a blank, so a z
        # index never renders as 'Z3.0' in the names it matches files by.
        if not runnable:
            protocol_df['Z-Slice'] = protocol_df['Z-Slice'].apply(common_utils.to_int)

        # Added in v3
        DEFAULT_VIDEO_CONFIG = {'duration': 5}

        # Added in v4
        DEFAULT_SUM_CONFIG = 1

        if config['version'] < cls.CURRENT_VERSION:
            logger.info(
                f'Converting loaded protocol from {config["version"]} to {cls.CURRENT_VERSION}'
            )

        def _parse_config_per_row(position, row, column):
            """One row's config cell, read by the step reader.

            Parsed here rather than in the constructor because the legacy
            repairs below need the dict. A cell that is not a config refuses
            a runnable file, naming the step; post-processing's reader keeps
            it as read.
            """
            try:
                return _dict_cell(row[column])
            except ValueError as e:
                if not runnable:
                    return row[column]
                raise ProtocolFormatError(
                    f'step {position + 1} ({row["Name"]}) has {column} {row[column]!r}, which {e}.',
                    file=file_path,
                ) from None

        def _with_video_defaults(video_config):
            """A step's Video Config as this version holds it: its duration,
            the default where a legacy save stored none. A pre-v11 save's
            fps is dropped: a recording's one rate limit is the global one."""
            if not isinstance(video_config, dict):
                return video_config
            return {**DEFAULT_VIDEO_CONFIG, **{k: v for k, v in video_config.items() if k != 'fps'}}

        def _typed_column(values, dtype) -> pd.Series:
            """A per-row column for the steps frame, typed explicitly.

            A bare list assignment takes its dtype from its contents, and
            DataFrame.apply(axis=1) invokes the function once on a phantom
            all-NaN row to infer a result type. At zero rows the first
            silently loses the column's type and the second logs errors for
            a step that does not exist, so every per-row column is built
            here: iterate the rows, then pin the dtype.
            """
            return pd.Series(list(values), dtype=dtype, index=protocol_df.index)

        if (config['version'] == 2) and (cls.CURRENT_VERSION >= 4):
            protocol_df['Acquire'] = AcquireMode.IMAGE.value
            # Each row gets its own independent dict
            protocol_df['Video Config'] = _typed_column(
                (copy.deepcopy(DEFAULT_VIDEO_CONFIG) for _ in protocol_df.index), object
            )
        else:
            protocol_df['Video Config'] = _typed_column(
                (
                    _with_video_defaults(_parse_config_per_row(position, row, 'Video Config'))
                    for position, (_, row) in enumerate(protocol_df.iterrows())
                ),
                object,
            )

        if (
            config['version']
            in (
                2,
                3,
            )
        ) and (cls.CURRENT_VERSION >= 4):
            protocol_df['Sum'] = DEFAULT_SUM_CONFIG

        # A file from before v12 carries the step's Stim_Config and the Layer
        # Settings' Stim_Enabled. A runnable read refuses one that asks for
        # stimulation, naming it as the file spells it; post-processing's
        # never runs the protocol. Both drop the columns unread after that.
        if runnable:
            _refuse_asked_stimulation(protocol_df, block_rows, file_path)
        protocol_df = protocol_df.drop(columns='Stim_Config', errors='ignore')

        # Renamed before the label recovery and the Name re-render read the
        # channels, so both see the unit's keys.
        if runnable:
            protocol_df['channel'] = protocol_df['channel'].map(unit.unit_key)
            if block_rows is not None:
                block_rows = {
                    key: {**row, 'Layer': key}
                    for key, row in _in_unit_keys(
                        block_rows, unit, 'its Layer Settings block', file_path
                    ).items()
                }

        if config['version'] in (2, 3, 4, 5):
            protocol_df['Step Index'] = protocol_df.index

        if 'Label' not in protocol_df.columns:
            # Pre-v8 files never persisted the label; recover it per row from
            # the Name's shape (machine-rendered or machine-anchored names
            # yield their machine base, anything else is user text kept
            # verbatim). Files that also predate the auto/user flag take the
            # flag from the same classification; a stored flag is kept as-is.
            recovered = [common_utils.recover_step_label(row) for _, row in protocol_df.iterrows()]
            protocol_df['Label'] = _typed_column((label for label, _ in recovered), str)
            if 'Auto_Named' not in protocol_df.columns:
                protocol_df['Auto_Named'] = _typed_column(
                    (is_auto for _, is_auto in recovered), bool
                )

        if 'Channel Group ID' not in protocol_df.columns:
            # Filled by the column's absence, not the file's version: a file
            # from before the column never recorded which Add a step came
            # from, and the boundaries cannot be recovered from the other
            # columns (two Adds at one position share every one of them).
            protocol_df['Channel Group ID'] = _typed_column((-1 for _ in protocol_df.index), int)
        if 'Turret Slot' not in protocol_df.columns:
            # A file from before the column named only each step's objective;
            # the protocols API fills the slot where only one slot can mean it.
            protocol_df['Turret Slot'] = _typed_column((-1 for _ in protocol_df.index), int)
        if runnable:
            protocol_df = cls._with_one_tile_per_zstack_group(protocol_df, file_path)
        if 'Z-Stack Offset' not in protocol_df.columns:
            protocol_df = cls._with_inferred_zstack_offsets(protocol_df, file_path)

        # The rename entry points store labels sanitized, but a file edited
        # outside the app may carry characters the writer strips at save
        # time -- letting two labels that differ only in stripped characters
        # pass the collision checks yet collide on disk. Normalize at the
        # read boundary, loudly: a silently different filename would read as
        # a lost rename.
        sanitized_labels = protocol_df['Label'].map(cls.sanitize_step_name)
        changed_labels = sanitized_labels != protocol_df['Label']
        if changed_labels.any():
            originals = ', '.join(repr(v) for v in protocol_df.loc[changed_labels, 'Label'].head(5))
            logger.warning(
                f'Protocol load: removed unsupported characters from '
                f'{int(changed_labels.sum())} step label(s); letters, digits, '
                f'dash, and underscore are kept. Affected: {originals}'
            )
        protocol_df['Label'] = sanitized_labels

        # The in-memory steps now carry every current column; stamp the
        # current version so a save writes the format the file actually
        # holds instead of an older version label over new columns.
        config['version'] = cls.CURRENT_VERSION

        config['steps'] = protocol_df
        # Resume the custom-step counter past the highest machine label
        # already in the file; restarting at zero would hand a newly added
        # step a 'custom<NNNN>' label an existing step already owns, and
        # their captures would collide.
        custom_indices = protocol_df['Label'].str.extract(r'^custom(\d+)$', expand=False).dropna()
        config['custom_step_count'] = (
            int(custom_indices.astype(int).max()) + 1 if len(custom_indices) else 0
        )

        try:
            protocol = cls(
                tiling_grids=tiling_grids,
                config=config,
                runnable=runnable,
            )
        except ProtocolFormatError as e:
            raise ProtocolFormatError(str(e), file=file_path) from None

        # Name is a derived rendering of the structured columns; re-render it
        # so a stale or hand-edited Name cannot disagree with the fields it
        # encodes. After the cells are typed, so it renders what they hold.
        # For auto-named rows this reproduces the stored Name byte-identically.
        protocol._render_step_names()
        protocol_df = protocol._config['steps']

        # The Layer Settings rows are typed into the protocol's records here,
        # once, and only the records are kept. A cell the reader cannot type
        # refuses the file, naming it, as a step cell does, on every read:
        # post-processing's included.
        if block_rows is not None:
            try:
                protocol._config['channel_settings'] = {
                    name: _channel_settings(row) for name, row in block_rows.items()
                }
            except ProtocolFormatError as e:
                raise ProtocolFormatError(str(e), file=file_path) from None

        # Warn -- do not reject -- when steps would render the same capture
        # filename (#636's harm class). The file must stay loadable so the
        # user can rename the colliding steps IN the app; the run itself is
        # refused at start (validate_for_run), which is where the data-loss
        # gate lives. A load-time rejection would block its own remedy.
        if len(protocol_df) > 1:
            collision_key = cls._capture_collision_key(protocol_df)
            dup_mask = collision_key.duplicated(keep=False)
            if dup_mask.any():
                notifications.report_outcome(
                    DuplicateCaptureFilenamesNotice(
                        colliding_steps=int(dup_mask.sum()),
                        shared_names=len(collision_key.loc[dup_mask].drop_duplicates()),
                    ),
                    solicited=False,
                    category='Protocol',
                )

        return protocol
