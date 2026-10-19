"""P10: a run turns the turret to each step's objective itself, and each file records its own scale.

Two BF steps at the stage's present position, the first on an objective in
another slot than the light path's and the second back on the one in the
light path, so the run must turn the turret before each. The run's turret
moves are observed at the motion API's public member, and each saved file's
pixel size is read back and compared with its own objective's. On hardware
the turret is the only axis that moves, and the operator can watch it turn
twice.

    python tests/capability/motion_p10_run_turns_the_turret.py             # a simulated LS850T
    python tests/capability/motion_p10_run_turns_the_turret.py --hardware  # the connected scope

The hardware run needs two turret slots assigned to objectives of different
scale in the installation's current.json.
"""

import datetime
import pathlib
import threading
import traceback

import pandas as pd

from harness import HARDWARE, SCRATCH, check, figure, hardware_session, headless_session, report

from modules.lumascope_api.illumination import LedState

RUN_TIMEOUT_S = 120.0


def _step(name, objective, slot, index, position, layer):
    # A step as Add Step builds one: no tile, no z-stack (-1), so the run
    # treats each as its own step and does not hold the LED across them,
    # and the turret slot it was made at.
    return {
        'Name': name,
        'X': position.x_mm,
        'Y': position.y_mm,
        'Z': position.z_um,
        'Auto_Focus': False,
        'Color': 'BF',
        'False_Color': False,
        'Illumination': layer['illumination_ma'],
        'Gain': layer['gain_db'],
        'Auto_Gain': False,
        'Exposure': layer['exposure_ms'],
        'Sum': 1,
        'Objective': objective,
        'Well': '',
        'Tile': '',
        'Z-Slice': -1,
        'Custom Step': True,
        'Tile Group ID': -1,
        'Z-Stack Group ID': -1,
        'Acquire': 'image',
        'Video Config': {'duration': 1.0, 'fps': 5},
        'Stim_Config': {},
        'Step Index': index,
        'Auto_Named': False,
        'Label': '',
        'Channel Group ID': 0,
        'Turret Slot': slot,
    }


def _expected_pixel_size(session, objective_id):
    import modules.common_utils as common_utils

    focal_length = session.scope.objectives[objective_id].focal_length_mm
    return round(
        common_utils.get_pixel_size(
            focal_length=focal_length,
            binning_size=session.scope.camera.read_binning(),
            capabilities=session.scope.capabilities,
        ),
        common_utils.max_decimal_precision('pixel_size'),
    )


def main():
    from modules.image_utils import read_pixel_size_um
    from modules.protocol import Protocol
    from modules.run_events import RunEvents

    repo = pathlib.Path(__file__).resolve().parents[2]
    session_cm = (
        hardware_session()
        if HARDWARE
        else headless_session(SCRATCH / 'p10_live', microscope='LS850T')
    )
    with session_cm as (session, runner):
        motion = session.scope.motion
        if not HARDWARE:
            # The hardware bring-up turns the turret to slot 1; the simulated
            # session's home leaves it in no known slot.
            motion.home('T')
            motion.move_turret(1)
        slots = {
            slot: objective
            for slot, objective in session.scope.runtime_state.get_turret_config().items()
            if objective
        }
        start_slot = motion.get_turret_slot()
        figure('turret slots assigned', slots)
        figure('turret slot at start', start_slot)
        here = slots.get(start_slot)
        others = (
            [
                slot
                for slot, objective in slots.items()
                if _expected_pixel_size(session, objective) != _expected_pixel_size(session, here)
            ]
            if here
            else []
        )
        if not check(
            'the light path holds an assigned objective', here is not None, f'slot {start_slot}'
        ):
            return
        if not check(
            'another slot holds an objective of a different scale', bool(others), str(slots)
        ):
            return
        away_slot = others[0]
        away = slots[away_slot]

        labware = session.settings['protocol']['labware']
        position = session.get_current_plate_position()
        figure('step position (plate mm, z um)', position)
        layer = session.settings['BF']
        protocol = Protocol(
            tiling_configs_file_loc=repo / 'data' / 'tiling.json',
            config={
                'version': Protocol.CURRENT_VERSION,
                'steps': pd.DataFrame(
                    [
                        _step('away', away, away_slot, 0, position, layer),
                        _step('back', here, start_slot, 1, position, layer),
                    ]
                ),
                'period': datetime.timedelta(minutes=1.0),
                'duration': datetime.timedelta(hours=1.0),
                'labware_id': labware,
                'capture_root': '',
                'tiling': '1x1',
            },
        )

        turned_to = []
        real_move_turret = motion.move_turret

        lit_at_turn = []
        illumination = session.scope.illumination

        def _observed_move_turret(position, restore_z=True):
            lit_at_turn.append(
                sorted(
                    str(r.name) for r in illumination.get_led_states() if r.state is LedState.LIT
                )
            )
            real_move_turret(position=position, restore_z=restore_z)
            turned_to.append((position, motion.get_turret_slot()))

        motion.move_turret = _observed_move_turret
        run_parent = SCRATCH / 'p10_runs'
        files_written = threading.Event()
        try:
            outcome = runner.start_single_scan(
                protocol=protocol,
                parent_dir=str(run_parent),
                events=RunEvents(files_written=lambda *written: files_written.set()),
            )
            result = outcome.result(timeout=RUN_TIMEOUT_S)
        finally:
            del motion.move_turret
        figure('run result', None if result is None else (result.status, result.reason))
        check('the run completed', result is not None and result.status == 'completed')
        check("the run's files were written", files_written.wait(15.0))

        figure('turret moves (asked, landed)', turned_to)
        expected_slots = [away_slot, start_slot]
        check(
            "the run turned the turret to each step's slot, in order",
            [asked for asked, _ in turned_to] == expected_slots,
            f'expected {expected_slots}',
        )
        check(
            'each turret move landed on the slot it asked for',
            all(asked == landed for asked, landed in turned_to),
        )
        check('the turret ends on the slot it started on', motion.get_turret_slot() == start_slot)
        figure('channels lit as each turret move starts', lit_at_turn)
        check('no LED is lit as a turret move starts', lit_at_turn and not any(lit_at_turn))

        files = sorted(run_parent.rglob('*.tiff'))
        figure('files', [f.name for f in files])
        for objective in (away, here):
            token = objective.replace(' ', '')
            named = [f for f in files if token in f.name]
            if not check(
                f'one file names {objective}', len(named) == 1, str([f.name for f in named])
            ):
                continue
            recorded = read_pixel_size_um(named[0])
            expected = _expected_pixel_size(session, objective)
            figure(f'{objective} pixel size (recorded, expected) um', (recorded, expected))
            check(f'the {objective} file records its own scale', recorded == expected)


if __name__ == '__main__':
    import sys

    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
