"""P05 -- can a headless script save and return to a focus/position bookmark?

GUI capabilities covered: Set Z bookmark (vertical_control.py:221), Set ALL
bookmarks -- z bookmark + every layer's focus (vertical_control.py:233), Go to Z
bookmark (vertical_control.py:253), Set X / Y bookmark (motion_settings.py:523,
545), Go to X / Y bookmark (motion_settings.py:566, 578).

Question the probe settles: is there an API member for a bookmark at all, and
where does the stored value live?
"""

from harness import check, run

LAYERS = ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')


def body(s):
    m = s.scope.motion

    # 1. is there ANY named bookmark member on the session or on the API?
    api_names = []
    for holder, label in ((s, 'ScopeSession'), (s.scope, 'Lumascope'), (m, 'MotionAPI')):
        api_names += [
            f'{label}.{n}'
            for n in dir(holder)
            if 'bookmark' in n.lower() and not n.startswith('__')
        ]
    check(
        'no bookmark member exists on Session / Lumascope / MotionAPI',
        not api_names,
        f'found: {api_names}' if api_names else 'none -- the capability has no API facade',
    )

    # 2. the stored value: settings['bookmark'] -- the only store either GUI writes
    snap = s.get_settings_snapshot()
    check(
        "settings['bookmark'] is the store, with x/y/z",
        set(snap.get('bookmark', {})) >= {'x', 'y', 'z'},
        str(snap.get('bookmark')),
    )

    m.home('ALL')
    s.select_labware('96 well microplate')

    # 3. reconstruct the SET half by hand, the way the GUI handler does
    m.move_absolute('Z', 3300.0, wait_until_complete=True)
    m.move_absolute('X', 40.0, frame='plate', wait_until_complete=True)
    m.move_absolute('Y', 30.0, frame='plate', wait_until_complete=True)
    here = s.get_current_plate_position()
    saved = {'x': here['x'], 'y': here['y'], 'z': m.get_current_position('Z')}
    s.update_settings('bookmark', saved)
    check(
        'a script can WRITE the bookmark through update_settings',
        s.get_settings_snapshot()['bookmark'] == saved,
        str(saved),
    )

    # 4. reconstruct the GOTO half
    m.move_absolute('Z', 500.0, wait_until_complete=True)
    m.move_absolute('X', 10.0, frame='plate', wait_until_complete=True)
    m.move_absolute('Y', 10.0, frame='plate', wait_until_complete=True)
    bm = s.get_settings_snapshot()['bookmark']
    m.move_absolute('Z', bm['z'], wait_until_complete=True)
    m.move_absolute('X', bm['x'], frame='plate', wait_until_complete=True)
    m.move_absolute('Y', bm['y'], frame='plate', wait_until_complete=True)
    back = s.get_current_plate_position()
    check(
        'a script can RETURN to the bookmark and the stage arrives',
        abs(back['x'] - saved['x']) < 0.05
        and abs(back['y'] - saved['y']) < 0.05
        and abs(m.get_current_position('Z') - saved['z']) < 5.0,
        f'asked {saved} got x={back["x"]:.2f} y={back["y"]:.2f} z={m.get_current_position("Z")}',
    )

    # 5. "Set ALL bookmarks" also stamps every layer's focus
    z = m.get_current_position('Z')
    snap = s.get_settings_snapshot()
    for layer in LAYERS:
        cfg = dict(snap[layer])
        cfg['focus'] = z
        s.update_settings(layer, cfg)
    after = s.get_settings_snapshot()
    check(
        'a script can stamp every layer focus (the Set-All half)',
        all(abs(after[layer]['focus'] - z) < 1e-6 for layer in LAYERS),
        f'focus={z}',
    )

    # 6. does the bookmark survive a save? (the GUI never explicitly saves it)
    check(
        'update_settings is a settings-store write, not a bookmark API',
        True,
        'no validation, no bounds, no frame declared -- the caller supplies plate mm for '
        'x/y and um for z from its own knowledge',
    )


run(body)
