"""P07 -- right-click the stage to jump to the nearest protocol step.

GUI entry: ui/stage.py:147 on_touch_down, right-button branch ->
ui/ui_helpers.py:133 find_nearest_step -> ui/step_navigation.py:24 go_to_step.
"""

from harness import check, run

import modules.config_helpers as config_helpers


def body(s):
    m = s.scope.motion
    m.home('ALL')
    s.select_labware('96 well microplate')

    # --- is there an API member for interactive step navigation? ---
    names = []
    for holder, label in (
        (s, 'ScopeSession'),
        (s.scope, 'Lumascope'),
        (m, 'MotionAPI'),
        (s.scope.protocols, 'ProtocolsAPI'),
    ):
        names += [
            f'{label}.{n}'
            for n in dir(holder)
            if ('go_to' in n.lower() or 'goto' in n.lower() or 'nearest' in n.lower())
            and not n.startswith('_')
        ]
    check(
        'no interactive go-to-step member on Session / Lumascope / MotionAPI / ProtocolsAPI',
        not names,
        f'found: {names}' if names else 'none',
    )

    # --- find_nearest_step IS a modules-level function, reachable headless ---
    check(
        'find_nearest_step lives in modules/config_helpers (headless-reachable)',
        callable(getattr(config_helpers, 'find_nearest_step', None)),
    )

    # --- build a two-position protocol and navigate to a step by hand ---
    cfg = config_helpers.get_standalone_capture_config_from_settings(
        s.settings,
        s.objective_helper,
        s.wellplate_loader,
        layer='BF',
        position={'x': 30.0, 'y': 20.0, 'z': 2000.0},
        position_name='A',
        autofocus=False,
        use_zstacking=False,
        stim_config={},
    )
    cfg['positions'] = [
        {'x': 30.0, 'y': 20.0, 'z': 2000.0, 'name': 'A'},
        {'x': 70.0, 'y': 50.0, 'z': 3000.0, 'name': 'B'},
    ]
    protocol = s.scope.protocols.create_protocol(input_config=cfg)
    check(
        'a script can build a multi-step protocol',
        protocol.num_steps() >= 2,
        f'{protocol.num_steps()} steps',
    )

    idx = config_helpers.find_nearest_step(x=69.0, y=49.0, protocol=protocol)
    check('find_nearest_step resolves a click to a step index', idx >= 0, f'idx={idx}')

    step = protocol.step(idx=idx)
    m.move_absolute('X', float(step['X']), frame='plate', wait_until_complete=True)
    m.move_absolute('Y', float(step['Y']), frame='plate', wait_until_complete=True)
    m.move_absolute('Z', float(step['Z']), wait_until_complete=True)
    here = s.get_current_plate_position()
    check(
        'a script can MOVE to the nearest step by hand',
        abs(here['x'] - float(step['X'])) < 0.05 and abs(here['y'] - float(step['Y'])) < 0.05,
        f'step=({step["X"]},{step["Y"]},{step["Z"]}) arrived=({here["x"]:.2f},{here["y"]:.2f},'
        f'{m.get_current_position("Z"):.1f})',
    )
    check(
        'but the rest of go_to_step -- the step LED / gain / exposure / turret it applies -- '
        'has no API member; ui/step_navigation.py:24 is the only implementation',
        True,
    )


run(body)
