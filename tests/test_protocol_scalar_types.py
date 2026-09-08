# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Protocol hands out PYTHON scalars, never numpy ones.

The step table is declared with an explicit numpy dtype, so every value
sitting in the frame is a numpy scalar -- and that is correct, it is what
makes the vectorised column work possible. What is not correct is letting
those escape to callers.

They escaped, and the app died of it: a numpy scalar copied out of a step
and into the layer settings reached a Kivy NumericProperty, which does
EXACT type checks and rejects `np.float64` even though it is a `float`
subclass. Ending any protocol run terminated the application.

The trap that makes this class of bug expensive to find:

- `isinstance(np.float64(5.0), float)` is True, so no isinstance guard
  anywhere in the stack can screen for it.
- `BoundedNumericProperty` ACCEPTS numpy and stores it, so a sibling
  widget propagates the value instead of raising -- the failure is loud
  at one widget and silent at the next.
- The test suite stubs Kivy with a MagicMock, so no test can observe the
  rejection. That is exactly why the contract is asserted HERE, at the
  boundary that owns the type, where it needs no Kivy at all.

Anything the protocol hands to a caller as a scalar is a Python scalar.
The frame itself stays a frame -- `steps()` is used for vectorised work
and must keep its dtypes.
"""

import datetime
import pathlib

import numpy as np
import pandas as pd

from modules.config_helpers import find_nearest_step
from modules.protocol import Protocol

TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


def _step_row(name='S1', x=1.5, y=2.5, z=3.5, color='BF', illumination=5.0):
    """One step, shaped like the production writers shape it."""
    return {
        'Name': name,
        'X': x,
        'Y': y,
        'Z': z,
        'Auto_Focus': False,
        'Color': color,
        'False_Color': False,
        'Illumination': illumination,
        'Gain': 0.0,
        'Auto_Gain': False,
        'Exposure': 4.47,
        'Sum': 1,
        'Objective': '20x Oly',
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': -1,
        'Custom Step': True,
        'Tile Group ID': -1,
        'Z-Stack Group ID': -1,
        'Acquire': 'image',
        'Video Config': {},
        'Stim_Config': {},
        'Step Index': 0,
        'Auto_Named': False,
        'Label': '',
    }


def _protocol_from_declared_schema(rows):
    """Build through the production empty-frame builder.

    This is the path the New button takes: the frame is created from the
    module's own `np.dtype([...])` declaration and filled in memory, with
    no file anywhere. Two of the three field crashes came in this way, so
    a fix that only cleaned up file parsing would have missed them.
    """
    df = Protocol._create_empty_steps_df()
    for i, row in enumerate(rows):
        df.loc[i] = row
    return _wrap(df)


def _protocol_from_dataframe(rows):
    """Build the way the existing protocol tests build: dicts -> DataFrame."""
    return _wrap(pd.DataFrame(rows))


def _wrap(df):
    return Protocol(
        tiling_configs_file_loc=TILING_CONFIGS,
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': df,
            'period': datetime.timedelta(minutes=1),
            'duration': datetime.timedelta(hours=1),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _numpy_valued(step):
    """Every field that reads back as a numpy scalar, by DIRECT SUBSCRIPT.

    Subscript is the access the GUI actually uses -- `step['Illumination']`
    in ui/step_navigation.py. It must be the access this test measures:
    `Series.to_dict()` boxes numpy scalars to Python natives on the way
    out, so a to_dict()-based assertion passes on a frame that is still
    fully numpy and would have missed the shipped bug entirely.
    """
    return {k: type(step[k]).__name__ for k in step.index if isinstance(step[k], np.generic)}


class TestProtocolHandsOutPythonScalars:
    def test_step_from_declared_schema_is_python(self):
        protocol = _protocol_from_declared_schema([_step_row()])
        offenders = _numpy_valued(protocol.step(0))
        assert not offenders, (
            'Protocol.step() handed out numpy scalars: '
            f'{offenders}. These reach the layer settings verbatim and a '
            'Kivy NumericProperty rejects them, terminating the app at the '
            'end of every protocol run.'
        )

    def test_step_from_dataframe_is_python(self):
        protocol = _protocol_from_dataframe([_step_row()])
        offenders = _numpy_valued(protocol.step(0))
        assert not offenders, f'Protocol.step() handed out numpy scalars: {offenders}.'

    def test_step_from_file_is_python(self, tmp_path):
        """The file path must be clean too -- one of the three crashes came in this way."""
        protocol = _protocol_from_declared_schema([_step_row()])
        filepath = tmp_path / 'p.tsv'
        assert protocol.to_file(filepath) is None
        reloaded = Protocol.from_file(file_path=filepath, tiling_configs_file_loc=TILING_CONFIGS)
        offenders = _numpy_valued(reloaded.step(0))
        assert not offenders, f'Protocol.from_file(...).step() handed out numpy: {offenders}.'

    def test_step_keeps_its_series_behaviour(self):
        """Coercion must not cost the callers what they already rely on."""
        protocol = _protocol_from_declared_schema([_step_row(illumination=5.0)])
        step = protocol.step(0)
        assert step['Illumination'] == 5.0
        assert step.get('Illumination') == 5.0
        assert dict(step)['Color'] == 'BF'
        assert step.to_dict()['Color'] == 'BF'
        assert not pd.isna(step['Illumination'])
        # The step runner copies before mutating; that must still work.
        copied = dict(step)
        copied['Auto_Focus'] = False
        assert step['Color'] == 'BF'

    def test_find_nearest_step_returns_a_python_int(self):
        """The annotation says int; it used to return np.int64.

        The index is handed around as a step number, so a numpy one leaks
        into anything that stores or displays it.
        """
        protocol = _protocol_from_declared_schema(
            [_step_row(name='S1', x=0.0, y=0.0), _step_row(name='S2', x=10.0, y=10.0)]
        )
        idx = find_nearest_step(x=9.0, y=9.0, protocol=protocol)
        assert idx == 1
        assert type(idx) is int, f'find_nearest_step returned {type(idx).__name__}, not int.'

    def test_has_zstacks_returns_a_python_bool(self):
        """Same shape as above: annotated bool, returned np.bool_."""
        protocol = _protocol_from_declared_schema([_step_row()])
        result = protocol.has_zstacks()
        assert type(result) is bool, f'has_zstacks returned {type(result).__name__}, not bool.'

    def test_the_frame_itself_keeps_its_dtypes(self):
        """steps() is for vectorised work and must NOT be coerced.

        Twenty-five callers use it as a frame -- column subtraction,
        idxmin, max, column assignment, tolist. Flattening it to objects
        to fix the scalar leak would break all of them, which is why the
        coercion belongs at the scalar exits instead.
        """
        protocol = _protocol_from_declared_schema([_step_row()])
        assert protocol.steps()['Illumination'].dtype == np.float64
