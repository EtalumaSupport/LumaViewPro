"""P01 -- home, XY/Z absolute + relative moves, readback, out-of-range refusal.

GUI capabilities covered: Home XY (motion_settings.py:605), Home Z
(vertical_control.py:264), typed X/Y plate coordinate (motion_settings.py:483,
504), stage click to move (stage.py:147), Z slider / Z text box
(vertical_control.py:201, 207), Z coarse/fine jog (vertical_control.py:165-177),
XY coarse/fine jog (motion_settings.py:452-480).
"""

from harness import check, run, void

from modules.exceptions import PositionOutOfRangeError, AxisStateUnknownError


def body(s):
    m = s.scope.motion

    # --- an un-homed axis REFUSES to move (it does not guess) ---
    try:
        m.move_absolute('Z', 1000.0, wait_until_complete=True)
        check('un-homed move refused', False, 'moved without homing')
    except AxisStateUnknownError as e:
        check('un-homed move raises AxisStateUnknownError', True, str(e)[:70])

    # --- HOME (the Home XY button and the Home Z button) ---
    check('home ALL returns True', m.home('ALL') is True)
    check('Z position known after home', m.position_is_known('Z'))
    check('X position known after home', m.position_is_known('X'))
    check('home Z alone returns True', m.home('Z') is True)

    # --- Z absolute (slider / text box commit) ---
    m.move_absolute('Z', 2500.0, wait_until_complete=True)
    z = m.get_current_position('Z')
    check('Z absolute move readback', abs(z - 2500.0) < 5.0, f'Z={z}')

    # --- Z relative (coarse/fine jog) ---
    m.move_relative('Z', -500.0, wait_until_complete=True)
    z2 = m.get_current_position('Z')
    check('Z relative jog readback', abs(z2 - 2000.0) < 5.0, f'Z={z2}')

    # --- XY absolute, stage frame ---
    m.move_absolute('X', 30000.0, wait_until_complete=True)
    m.move_absolute('Y', 20000.0, wait_until_complete=True)
    x, y = m.get_current_position('X'), m.get_current_position('Y')
    check('XY absolute move readback', abs(x - 30000) < 5 and abs(y - 20000) < 5, f'X={x} Y={y}')

    # --- XY relative (jog) ---
    m.move_relative('X', 1000.0, wait_until_complete=True)
    xj = m.get_current_position('X')
    check('X relative jog readback', abs(xj - 31000) < 5, f'X={xj}')

    # --- PLATE frame: what the typed coordinate box and the stage click send ---
    s.select_labware('96 well microplate')
    before = s.get_current_plate_position()
    m.move_absolute('X', 60.0, frame='plate', wait_until_complete=True)
    m.move_absolute('Y', 40.0, frame='plate', wait_until_complete=True)
    after = s.get_current_plate_position()
    check(
        'plate-frame move changed the plate position',
        abs(after['x'] - before['x']) > 0.5 or abs(after['y'] - before['y']) > 0.5,
        f'{before["x"]:.2f},{before["y"]:.2f} -> {after["x"]:.2f},{after["y"]:.2f}',
    )
    check(
        'plate-frame move landed on the typed coordinate',
        abs(after['x'] - 60.0) < 0.05 and abs(after['y'] - 40.0) < 0.05,
        f'{after["x"]:.3f},{after["y"]:.3f}',
    )

    # --- out of range RAISES rather than clamping ---
    for axis, bad in (('Z', 99999.0), ('X', 999999.0), ('Y', -5000.0)):
        pos_before = m.get_current_position(axis)
        try:
            m.move_absolute(axis, bad, wait_until_complete=True)
            check(f'{axis} out-of-range raises', False, 'NO RAISE -- silently accepted/clamped')
        except PositionOutOfRangeError:
            unchanged = abs(m.get_current_position(axis) - pos_before) < 5.0
            check(
                f'{axis} out-of-range raises and does not move',
                unchanged,
                f'before={pos_before} after={m.get_current_position(axis)}',
            )

    try:
        m.move_relative('Z', 99999.0, wait_until_complete=True)
        void('Z relative out-of-range raises', False, 'accepted without raising')
    except PositionOutOfRangeError:
        void('Z relative out-of-range raises', True, 'it raises now')

    try:
        m.move_absolute('X', 9999.0, frame='plate', wait_until_complete=True)
        check('plate-frame out-of-range raises', False, 'NO RAISE')
    except PositionOutOfRangeError as e:
        check('plate-frame out-of-range raises', True, str(e)[:90])


run(body)
