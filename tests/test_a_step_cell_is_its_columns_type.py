# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's step cells are each their column's type, or the protocol is refused.

The steps were read with ``pd.read_csv`` and no column was typed. One
``maybe`` in the Auto_Focus column made the column text, a saved ``False``
became the string ``'False'``, and the run autofocused that step anyway
(``bool('False')`` is True) -- no notice, no refusal. A Video or Stim config
that did not parse ran as the defaults; a Z-Slice that was not a number
raised a bare ``ValueError``; an empty X made the stage's right-click raise.

Now every cell is read by its column's reader where the frame is taken, and
a cell of the wrong type refuses the load in one message naming the file,
the step and the column. The in-place writers read their values the same
way, since a pandas frame does not hold its own schema. Post-processing,
which reads a finished run's record and never runs it, keeps its cells as
read.
"""

import ast
import pathlib

import pytest

from modules.exceptions import ProtocolError
from modules.protocol import Protocol, ProtocolFormatError
from tests.ast_seams import parse_module, walk_defs
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step


def _saved(tmp_path, steps) -> pathlib.Path:
    path = tmp_path / 'protocol.tsv'
    _build_protocol(steps).to_file(path)
    return path


def _with_cell(path: pathlib.Path, step: int, column: str, text: str) -> pathlib.Path:
    """The saved file with one step cell replaced by ``text``, as a hand edit would."""
    lines = path.read_text().splitlines()
    header_at = next(i for i, line in enumerate(lines) if line.split('\t')[0] == 'Steps') + 1
    header = lines[header_at].split('\t')
    row = lines[header_at + 1 + step].split('\t')
    row[header.index(column)] = text
    lines[header_at + 1 + step] = '\t'.join(row)
    path.write_text('\n'.join(lines) + '\n')
    return path


def _load(path, **kwargs):
    return Protocol.from_file(file_path=path, tiling_configs_file_loc=TILING_CONFIGS, **kwargs)


TWO_STEPS = [
    _make_step(name='A1_BF', well='A1', auto_focus=True),
    _make_step(name='A2_BF', well='A2', auto_focus=False),
]


@pytest.mark.parametrize(
    ('column', 'text'),
    [
        ('X', 'left'),
        ('X', ''),
        ('Y', ''),
        ('Z', ''),
        ('Exposure', '10 ms'),
        ('Sum', '2.5'),
        ('Z-Slice', 'top'),
        ('Z-Slice', ''),
        ('Tile Group ID', 'first'),
        ('Auto_Focus', 'maybe'),
        ('Auto_Gain', ''),
        ('Acquire', 'photo'),
        ('Video Config', '{fps: 5'),
        ('Stim_Config', '[1, 2]'),
    ],
)
def test_a_file_with_one_mistyped_cell_is_refused_naming_file_step_and_column(
    tmp_path, column, text
):
    path = _with_cell(_saved(tmp_path, TWO_STEPS), 1, column, text)

    with pytest.raises(ProtocolFormatError) as refused:
        _load(path)

    words = str(refused.value)
    assert str(path) in words
    assert 'step 2' in words
    assert column in words
    assert refused.value.file == path


def test_a_saved_false_autofocus_is_false_once_the_file_is_typed(tmp_path):
    # The A2 file: one 'maybe' made step 2's saved False the string 'False'.
    path = _with_cell(_saved(tmp_path, TWO_STEPS), 0, 'Auto_Focus', 'maybe')
    with pytest.raises(ProtocolFormatError):
        _load(path)

    fixed = _with_cell(path, 0, 'Auto_Focus', 'True')
    protocol = _load(fixed)

    assert protocol.step(idx=0)['Auto_Focus'] is True
    assert protocol.step(idx=1)['Auto_Focus'] is False


def test_a_typed_file_loads_with_the_schemas_dtypes(tmp_path):
    schema = Protocol._create_empty_steps_df().dtypes
    loaded = _load(_saved(tmp_path, TWO_STEPS)).steps().dtypes
    assert {column: loaded[column] for column in schema.index} == schema.to_dict()


@pytest.mark.parametrize(
    ('column', 'value'),
    [('X', None), ('Exposure', 'ten'), ('Auto_Focus', 'True?'), ('Objective', None)],
)
def test_a_protocol_built_from_a_mistyped_frame_is_refused(column, value):
    with pytest.raises(ProtocolFormatError) as refused:
        _build_protocol([_make_step(), {**_make_step(name='A2_BF', well='A2'), column: value}])

    assert 'step 2' in str(refused.value)
    assert column in str(refused.value)


WRITES = {
    'modify_autofocus': (
        'Auto_Focus',
        lambda p, v: p.modify_autofocus(step_idx=0, enabled=v),
    ),
    'modify_step_z_height': ('Z', lambda p, v: p.modify_step_z_height(step_idx=0, z=v)),
    'apply_zstack_group_focus': (
        'Z',
        lambda p, v: p.apply_zstack_group_focus(reference_step_idx=0, z=v),
    ),
    'apply_focus_all_layer_steps': (
        'Z',
        lambda p, v: p.apply_focus_all_layer_steps(layer='BF', z=v),
    ),
    'modify_step': (
        'Exposure',
        lambda p, v: p.modify_step(
            step_idx=0,
            layer='BF',
            layer_config={
                'autofocus': False,
                'false_color': False,
                'illumination_ma': 50.0,
                'gain_db': 1.0,
                'auto_gain': False,
                'exposure_ms': v,
                'sum': 1,
                'acquire': 'image',
                'video_config': {'fps': 5, 'duration': 5},
            },
            plate_position={'x': 1.0, 'y': 2.0, 'z': 3.0},
            objective_id='10x Oly',
            stim_configs={},
        ),
    ),
}


@pytest.mark.parametrize('value', [None, 'tall'])
@pytest.mark.parametrize('writer', WRITES)
def test_every_in_place_writer_refuses_a_value_not_of_its_columns_type(writer, value):
    column, write = WRITES[writer]
    protocol = _build_protocol(TWO_STEPS)
    before = protocol.steps()

    with pytest.raises(ProtocolError) as refused:
        write(protocol, value)

    assert column in str(refused.value)
    assert protocol.steps().equals(before)


def test_no_step_frame_write_sits_outside_the_typed_writers():
    # The frame does not hold its schema, so every .at / .loc write in the
    # protocol module is in a writer that reads its values first.
    # insert_step writes a one-row frame that _set_steps then types;
    # modify_name and _regenerate_step_name write text they build.
    typed_writers = {
        'Protocol.modify_autofocus',
        'Protocol.modify_step_z_height',
        'Protocol.apply_zstack_group_focus',
        'Protocol.apply_focus_all_layer_steps',
        'Protocol.modify_step',
        'Protocol.modify_name',
        'Protocol._regenerate_step_name',
        'Protocol.insert_step',
    }
    writing = set()
    for qualname, node in walk_defs(parse_module('modules/protocol.py').body):
        for statement in ast.walk(node):
            targets = getattr(statement, 'targets', None) or [getattr(statement, 'target', None)]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr in ('at', 'loc', 'iat', 'iloc')
                ):
                    writing.add(qualname)

    assert writing == typed_writers


def test_post_processing_reads_a_runs_record_with_its_cells_as_read(tmp_path):
    path = _with_cell(_saved(tmp_path, TWO_STEPS), 0, 'Auto_Focus', 'maybe')

    protocol = _load(path, runnable=False)

    assert protocol.steps()['Auto_Focus'].tolist() == ['maybe', 'False']
    assert protocol.step(idx=0)['X'] == 10.0


def test_the_post_processing_reader_is_not_runnable():
    node = next(
        node
        for _, node in walk_defs(parse_module('modules/protocol_post_processing_helper.py').body)
        for node in ast.walk(node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'from_file'
        and getattr(node.func.value, 'id', None) == 'Protocol'
    )
    assert {(k.arg, getattr(k.value, 'value', None)) for k in node.keywords} >= {
        ('runnable', False)
    }


def test_a_runs_copy_reads_its_cells_as_its_source_does():
    # The copy is built without the constructor, so it carries the flag
    # itself; a frame replacement on it types the new frame as the source's
    # would.
    copy = _build_protocol(TWO_STEPS).copy_for_execution()

    with pytest.raises(ProtocolFormatError, match='Auto_Focus'):
        copy.insert_step(
            step_name='added',
            layer='BF',
            layer_config={
                'autofocus': 'maybe',
                'false_color': False,
                'illumination_ma': 50.0,
                'gain_db': 1.0,
                'auto_gain': False,
                'exposure_ms': 10.0,
                'sum': 1,
                'acquire': 'image',
                'video_config': {'fps': 5, 'duration': 5},
                'focus': None,
            },
            plate_position={'x': 1.0, 'y': 2.0, 'z': 3.0},
            objective_id='10x Oly',
            stim_configs={},
            after_step=1,
        )
