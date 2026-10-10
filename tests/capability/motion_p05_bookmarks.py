"""P05 -- can a headless script save and return to a focus/position bookmark?

GUI capabilities covered: Set Z bookmark (vertical_control.py:221), Set ALL
bookmarks -- z bookmark + every layer's focus (vertical_control.py:233), Go to Z
bookmark (vertical_control.py:253), Set X / Y bookmark (motion_settings.py:523,
545), Go to X / Y bookmark (motion_settings.py:566, 578).

Question the probe settles: is there an API member for a bookmark at all, and
where does the stored value live?
"""

from harness import check, run

from modules.exceptions import SettingRefusedError


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
        'the Session has bookmark members',
        bool(api_names),
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
    m.move_absolute('Z', 3300.0)
    m.move_absolute('X', 40.0, frame='plate')
    m.move_absolute('Y', 30.0, frame='plate')
    saved = s.save_bookmark(('X', 'Y', 'Z'))
    check(
        'a script can SAVE the bookmark through the Session',
        s.get_settings_snapshot()['bookmark'] == saved,
        str(saved),
    )

    # 4. reconstruct the GOTO half
    m.move_absolute('Z', 500.0)
    m.move_absolute('X', 10.0, frame='plate')
    m.move_absolute('Y', 10.0, frame='plate')
    bm = s.get_settings_snapshot()['bookmark']
    m.move_absolute('Z', bm['z'])
    m.move_absolute('X', bm['x'], frame='plate')
    m.move_absolute('Y', bm['y'], frame='plate')
    back = s.get_current_plate_position()
    check(
        'a script can RETURN to the bookmark and the stage arrives',
        abs(back['x'] - saved['x']) < 0.05
        and abs(back['y'] - saved['y']) < 0.05
        and abs(m.get_current_position('Z') - saved['z']) < 5.0,
        f'asked {saved} got x={back["x"]:.2f} y={back["y"]:.2f} z={m.get_current_position("Z")}',
    )

    # 5. "Set ALL bookmarks" also stamps every layer's focus
    z = s.save_all_bookmarks()
    on_scope = [record.key_name for record in s.scope.layer_identity.layers]
    after = s.get_settings_snapshot()
    check(
        'a script can stamp every layer focus (the Set-All half)',
        all(abs(after[layer]['focus'] - z) < 1e-6 for layer in on_scope),
        f'focus={z}',
    )

    # 6. the bookmark is not a plain setting: the writer names its members
    try:
        s.update_settings('bookmark.x', 1.0)
        check('update_settings refuses the bookmark, naming its member', False, 'written')
    except SettingRefusedError as e:
        check(
            'update_settings refuses the bookmark, naming its member',
            e.member == 'save_bookmark',
            str(e),
        )


run(body)
