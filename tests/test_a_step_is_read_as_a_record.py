# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step is read as a record, one step or the whole table in one read.

``step(step_index)`` answered a pandas row, which crossed the wire as an object of
no declared shape keyed by the steps table's column names, and a caller
wanting every step asked once per step. A step is now a ``StepRecord``, a
field per column, its video and stimulation configs records of their own,
and ``step_records()`` answers every step at once.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from _lumascope.modules.coord_transformations import PlateLocation
from _lumascope.modules.protocol import (
    _STEP_CELLS,
    _STEP_FIELDS,
    ProtocolFormatError,
    Stimulation,
    StepRecord,
    VideoConfig,
)
from _lumascope.rest.app import build_app
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture
from tests.stock_channels import LED0


def _bf_and_blue(session):
    protocol = session.create_empty_protocol()
    session.add_step(
        protocol, channels=('BF', LED0), at=PlateLocation(x_mm=20.0, y_mm=15.0, z_um=6000.0)
    )
    return protocol


def test_the_record_has_a_field_for_every_column_of_the_steps_table():
    fields = [field.name for field in dataclasses.fields(StepRecord)]
    assert list(_STEP_FIELDS) == list(_STEP_CELLS)
    assert list(_STEP_FIELDS.values()) == fields


def test_a_step_reads_as_its_record_with_its_configs_as_records(session):
    protocol = _bf_and_blue(session)
    row = protocol.step_row(step_index=1)

    step = protocol.step_records()[1]

    assert isinstance(step, StepRecord)
    assert (step.name, step.channel, step.x_mm, step.y_mm, step.z_um) == (
        row['Name'],
        LED0,
        20.0,
        15.0,
        6000.0,
    )
    assert step.video_config == VideoConfig(duration_s=row['Video Config']['duration'])
    assert step.stim_config == (
        None
        if row['Stim_Config'] is None
        else {
            layer: Stimulation(
                enabled=entry.get('enabled'),
                illumination_ma=entry.get('illumination_ma'),
                frequency_hz=entry.get('frequency'),
                pulse_width_ms=entry.get('pulse_width'),
                pulse_count=entry.get('pulse_count'),
            )
            for layer, entry in row['Stim_Config'].items()
        }
    )


def test_the_step_table_is_every_step_in_order(session):
    protocol = _bf_and_blue(session)

    table = protocol.step_records()

    assert len(table) == protocol.num_steps()
    assert [step.channel for step in table] == ['BF', LED0]


def test_a_step_with_no_position_on_an_axis_reads_none_there(session):
    protocol = session.create_empty_protocol()
    session.add_step(
        protocol, channels=('BF',), at=PlateLocation(x_mm=None, y_mm=None, z_um=6000.0)
    )

    step = protocol.step_records()[0]

    assert (step.x_mm, step.y_mm, step.z_um) == (None, None, 6000.0)


def test_the_table_crosses_the_wire_as_records_keyed_by_field(session):
    with TestClient(build_app(session)) as client:
        handle = client.post('/api/v1/create_empty_protocol').json()['handle']
        added = client.post('/api/v1/add_step', json={'protocol': handle, 'channels': ['BF', LED0]})
        answer = client.get(f'/api/v1/handles/Protocol/{handle}/step_records')

    assert added.status_code == 200, added.text
    assert answer.status_code == 200, answer.text
    table = answer.json()
    assert [step['channel'] for step in table] == ['BF', LED0]
    assert set(table[0]) == {field.name for field in dataclasses.fields(StepRecord)}


def test_a_step_with_no_slice_group_or_slot_reads_none_where_the_table_holds_minus_one(session):
    protocol = session.create_empty_protocol()
    session.add_step(protocol, channels=('BF',))
    row = protocol.step_row(step_index=0)
    columns = ('Z-Slice', 'Tile Group ID', 'Z-Stack Group ID', 'Turret Slot')
    assert [row[column] for column in columns] == [-1, -1, -1, -1]

    step = protocol.step_records()[0]

    assert (step.z_slice, step.tile_group_id, step.z_stack_group_id, step.turret_slot) == (
        None,
        None,
        None,
        None,
    )


def test_every_value_of_a_record_is_a_python_value_not_a_numpy_one(session):
    # The steps table holds numpy scalars; a record handed one leaks it to
    # every caller that copies a value into a Kivy property or a JSON body.
    protocol = _bf_and_blue(session)
    session.add_step(
        protocol, channels=('BF',), at=PlateLocation(x_mm=None, y_mm=None, z_um=6000.0)
    )

    numpy_values = [
        (step.name, field.name, type(value).__name__)
        for step in protocol.step_records()
        for field in dataclasses.fields(step)
        if isinstance(value := getattr(step, field.name), np.generic)
    ]

    assert numpy_values == []


def test_a_stimulation_entry_that_is_not_a_config_is_refused_naming_its_step(session, tmp_path):
    protocol = _bf_and_blue(session)
    saved = tmp_path / 'protocol.tsv'
    protocol.to_file(saved)
    first = protocol.step_row(step_index=0)

    # The cell as the writer saves it: JSON, quoted for the TSV, its quotes doubled.
    def cell(config):
        return '"' + json.dumps(config).replace('"', '""') + '"'

    stims = cell(dict(first['Stim_Config']))
    assert stims in saved.read_text()
    saved.write_text(saved.read_text().replace(stims, cell({'BF': 5}), 1))
    loaded = session.load_protocol(saved)

    with pytest.raises(ProtocolFormatError) as refused:
        loaded.step_records()

    assert str(refused.value) == (
        f'step {first["Name"]} has a Stim_Config entry for BF that is not a config: 5.'
    )
