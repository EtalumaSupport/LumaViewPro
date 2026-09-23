# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved frame records the objective it was taken with, not the one in the
light path when the file is written.

A run's saves go to the file writer and land later, and the next step's
turret move does not wait for them. The objective behind a file's scale was
read at save time, so a step whose write was still queued when the turret
turned was stamped with the NEXT step's objective: a wrong scale, measured
off the file forever. The frame now carries its objective from capture, the
way it carries its payload depth.

The race is built rather than hoped for: the first step's save is held on
the file writer until the second step's turret move has landed, then let
through to the real writer.
"""

import datetime
import pathlib
import threading
import time

import pandas as pd

import modules.common_utils as common_utils
import modules.protocol_image_writer as protocol_image_writer
from modules.image_utils import read_pixel_size_um
from modules.protocol import Protocol
from tests.scope_fakes import TEST_TURRET_OBJECTIVES
from tests.test_composite_run_e2e import headless_settings, open_composite_session


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

FIRST, SECOND = '10x Oly', '4x Oly'
SECOND_SLOT = next(slot for slot, obj in TEST_TURRET_OBJECTIVES.items() if obj == SECOND)
MOVE_TIMEOUT_S = 15.0


def _step(name, objective, index):
    return {
        'Name': name,
        'X': 10.0,
        'Y': 20.0,
        'Z': 5000.0,
        'Auto_Focus': False,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': 50.0,
        'Gain': 1.0,
        'Auto_Gain': False,
        'Exposure': 10.0,
        'Sum': 1,
        'Objective': objective,
        'Well': 'A1',
        'Tile': '',
        'Z-Slice': 0,
        'Custom Step': True,
        'Tile Group ID': 0,
        'Z-Stack Group ID': 0,
        'Acquire': 'image',
        'Video Config': {'duration': 1.0, 'fps': 5},
        'Stim_Config': {},
        'Step Index': index,
        'Auto_Named': False,
        'Label': '',
    }


def _token(objective_id):
    """The objective as a saved file's name spells it."""
    return objective_id.replace(' ', '')


def _two_objective_protocol():
    return Protocol(
        tiling_configs_file_loc=REPO_ROOT / 'data' / 'tiling.json',
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame([_step('first', FIRST, 0), _step('second', SECOND, 1)]),
            'period': datetime.timedelta(minutes=1.0),
            'duration': datetime.timedelta(hours=1.0),
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def test_a_save_after_the_next_turret_move_keeps_its_own_objective(tmp_path, monkeypatch):
    run_parent = tmp_path / 'runs'
    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        motion = session.scope.motion
        real_save = protocol_image_writer.save_image
        second_turret_move_returned = threading.Event()
        held = []

        real_turret_move = motion._move_turret_impl

        def _turret_move(position, restore_z=True):
            # The run's own turret move, observed: the engine turns the
            # turret before each step's capture.
            real_turret_move(position=position, restore_z=restore_z)
            if position == SECOND_SLOT:
                second_turret_move_returned.set()

        monkeypatch.setattr(motion, '_move_turret_impl', _turret_move)

        def _held_until_the_turret_moves(scope, **kwargs):
            if not held:
                held.append(kwargs['append'])
                second_turret_move_returned.wait(MOVE_TIMEOUT_S)
            return real_save(scope, **kwargs)

        monkeypatch.setattr(protocol_image_writer, 'save_image', _held_until_the_turret_moves)

        outcome = runner.run_single_scan(
            protocol=_two_objective_protocol(),
            parent_dir=str(run_parent),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )
        result = outcome.wait(timeout_s=60.0)
        assert result is not None and result.status == 'completed', result

        # A run's outcome does not wait for its still writes to land.
        deadline = time.monotonic() + MOVE_TIMEOUT_S
        while len(list(run_parent.rglob('*.tiff'))) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)

        binning = session.scope.imaging.get_binning_size()
        capabilities = session.scope.capabilities

    assert len(held) == 1 and _token(FIRST) in held[0], held
    assert second_turret_move_returned.is_set(), (
        "the second step's turret move never returned, so the first step's save "
        'was not held across it and this run did not build the race'
    )

    def _expected(objective_id):
        focal_length = session.objective_helper.get_objective_info(objective_id)['focal_length']
        return round(
            common_utils.get_pixel_size(
                focal_length=focal_length, binning_size=binning, capabilities=capabilities
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )

    assert _expected(FIRST) != _expected(SECOND), 'the two objectives must differ in scale'
    files = {path.name: path for path in run_parent.rglob('*.tiff')}
    for objective_id in (FIRST, SECOND):
        [path] = [p for name, p in files.items() if _token(objective_id) in name]
        assert read_pixel_size_um(path) == _expected(objective_id), (
            f'{path.name} records the scale of the wrong objective'
        )
