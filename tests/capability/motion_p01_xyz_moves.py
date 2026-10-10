"""P01 -- home, XY/Z absolute + relative moves, readback, out-of-range refusal.

GUI capabilities covered: Home XY (motion_settings.py:605), Home Z
(vertical_control.py:264), typed X/Y plate coordinate (motion_settings.py:483,
504), stage click to move (stage.py:147), Z slider / Z text box
(vertical_control.py:201, 207), Z coarse/fine jog (vertical_control.py:165-177),
XY coarse/fine jog (motion_settings.py:452-480).
"""

from harness import check, run

from modules.exceptions import HomingFailedError, PositionOutOfRangeError, AxisStateUnknownError


def body(s):
    m = s.scope.motion

    # --- an un-homed axis REFUSES to move (it does not guess) ---
    try:
        m.move_absolute('Z', 1000.0)
        check('un-homed move refused', False, 'moved without homing')
    except AxisStateUnknownError as e:
        check('un-homed move raises AxisStateUnknownError', True, str(e)[:70])

    # --- HOME (the Home XY button and the Home Z button) ---
    try:
        m.home('ALL')
        check('home ALL returns', True)
    except HomingFailedError as e:
        check('home ALL returns', False, str(e))
    check('Z position known after home', m.position_is_known('Z'))
    check('X position known after home', m.position_is_known('X'))
    try:
        m.home('Z')
        check('home Z alone returns', True)
    except HomingFailedError as e:
        check('home Z alone returns', False, str(e))

    # --- Z absolute (slider / text box commit) ---
    m.move_absolute('Z', 2500.0)
    z = m.get_current_position('Z')
    check('Z absolute move readback', abs(z - 2500.0) < 5.0, f'Z={z}')

    # --- Z relative (coarse/fine jog) ---
    m.move_relative('Z', -500.0)
    z2 = m.get_current_position('Z')
    check('Z relative jog readback', abs(z2 - 2000.0) < 5.0, f'Z={z2}')

    # --- XY absolute, stage frame ---
    m.move_absolute('X', 30000.0)
    m.move_absolute('Y', 20000.0)
    x, y = m.get_current_position('X'), m.get_current_position('Y')
    check('XY absolute move readback', abs(x - 30000) < 5 and abs(y - 20000) < 5, f'X={x} Y={y}')

    # --- XY relative (jog) ---
    m.move_relative('X', 1000.0)
    xj = m.get_current_position('X')
    check('X relative jog readback', abs(xj - 31000) < 5, f'X={xj}')

    # --- PLATE frame: what the typed coordinate box and the stage click send ---
    s.select_labware('96 well microplate')
    m.move_absolute('X', 60.0, frame='plate')
    m.move_absolute('Y', 40.0, frame='plate')
    after = s.get_current_plate_position()
    check(
        'plate-frame move landed on the typed coordinate',
        abs(after['x'] - 60.0) < 0.05 and abs(after['y'] - 40.0) < 0.05,
        f'{after["x"]:.3f},{after["y"]:.3f}',
    )

    # --- out of range RAISES rather than clamping ---
    for axis, bad in (('Z', 99999.0), ('X', 999999.0), ('Y', -5000.0)):
        pos_before = m.get_current_position(axis)
        try:
            m.move_absolute(axis, bad)
            check(f'{axis} out-of-range raises', False, 'NO RAISE -- silently accepted/clamped')
        except PositionOutOfRangeError:
            unchanged = abs(m.get_current_position(axis) - pos_before) < 5.0
            check(
                f'{axis} out-of-range raises and does not move',
                unchanged,
                f'before={pos_before} after={m.get_current_position(axis)}',
            )

    try:
        m.move_relative('Z', 99999.0)
        check('Z relative out-of-range raises', False, 'accepted without raising')
    except PositionOutOfRangeError:
        check('Z relative out-of-range raises', True)

    try:
        m.move_absolute('X', 9999.0, frame='plate')
        check('plate-frame out-of-range raises', False, 'NO RAISE')
    except PositionOutOfRangeError as e:
        check(
            'plate-frame out-of-range raises in the plate frame',
            e.bound == 'reachable range' and e.quantity == 'plate position',
            str(e)[:90],
        )


run(body)
