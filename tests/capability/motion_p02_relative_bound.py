"""P02 -- does a relative jog past the travel limit raise, clamp, or go through?

The GUI's coarse/fine Z and XY jogs are all relative moves
(vertical_control.py:142 _z_jog -> move_relative; motion_settings.py:428
_xy_jog -> move_relative).
"""

from harness import check, run, void
from modules.exceptions import PositionOutOfRangeError
from modules.lumascope_api.motion import MOTOR_POSITION_LIMIT


def body(s):
    m = s.scope.motion
    m.home('ALL')
    print('MOTOR_POSITION_LIMIT =', MOTOR_POSITION_LIMIT, flush=True)
    print('Z travel limits =', m.get_axis_limits('Z'), flush=True)

    m.move_absolute('Z', 2000.0, wait_until_complete=True)
    zmax = m.get_axis_limits('Z')['max']

    # a jog well past the travel ceiling but inside the safety limit
    over = zmax + 5000.0
    outcome = None
    try:
        m.move_relative('Z', over - 2000.0, wait_until_complete=True)
        outcome = 'accepted'
    except PositionOutOfRangeError as e:
        outcome = f'raised: {e}'
    landed = m.get_current_position('Z')
    print(
        f'relative jog of {over - 2000.0} um from 2000: outcome={outcome} landed={landed}',
        flush=True,
    )
    void(
        'relative jog past the travel ceiling is refused (raises)',
        outcome != 'accepted',
        f'outcome={outcome} landed={landed}',
    )
    void(
        'relative jog is refused rather than silently clamped to the ceiling',
        not (outcome == 'accepted' and abs(landed - zmax) < 1.0),
        f'landed={landed} ceiling={zmax}',
    )

    # beyond the SAFETY limit the relative path does raise
    try:
        m.move_relative('Z', MOTOR_POSITION_LIMIT + 1.0, wait_until_complete=True)
        check('relative jog past MOTOR_POSITION_LIMIT raises', False, 'NO RAISE')
    except PositionOutOfRangeError as e:
        check('relative jog past MOTOR_POSITION_LIMIT raises', True, str(e)[:90])


run(body)
