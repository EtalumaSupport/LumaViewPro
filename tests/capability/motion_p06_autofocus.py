"""P06 -- the autofocus button, headless.

GUI entry: ui/vertical_control.py:446 run_autofocus_from_ui (kv line 537).
API member under test: modules/protocol_runner.py:281 ProtocolRunner.start_autofocus.
"""

from harness import check, run, void
from modules.protocol_runner import ProtocolRunner
from modules.exceptions import ArgumentRefusedError
from modules.coord_transformations import PlateLocation


def body(s):
    m = s.scope.motion
    m.home('ALL')
    s.select_labware('96 well microplate')
    m.move_absolute('Z', 3000.0)
    m.move_to_plate(PlateLocation(x_mm=40.0, y_mm=30.0, z_um=None))
    # Simulated, the camera shows a focus target focused inside the sweep's
    # window, off its start so the stage has somewhere to go: a featureless
    # frame has no peak, and an autofocus over it fails. On hardware the
    # sample under the objective is the scene.
    camera = s.scope._camera_driver
    if hasattr(camera, 'set_focal_z'):
        camera.set_test_pattern(enabled=True, pattern='focus_target')
        camera.set_focal_z(3030.0)

    runner = ProtocolRunner(s)
    z_before = m.get_current_position('Z')

    pending = runner.start_autofocus(layer='BF')
    outcome = pending.result(timeout=300)
    z_after = m.get_current_position('Z')
    print('outcome:', outcome.status, '|', outcome.reason, '|', outcome.message, flush=True)
    check(
        'start_autofocus completes',
        str(outcome.status).lower().find('fail') < 0,
        f'status={outcome.status} reason={outcome.reason}',
    )
    check(
        'start_autofocus moved Z (stage left at the focus it found)',
        abs(z_after - z_before) > 0.5,
        f'{z_before} -> {z_after}',
    )

    # does it ALSO write the layer's focus the way the button does?
    focus_after = s.get_settings_snapshot()['BF']['focus']
    void(
        "start_autofocus wrote settings['BF']['focus'] to what it found",
        focus_after is not None and abs(focus_after - z_after) < 5.0,
        f'settings BF focus={focus_after}, stage Z={z_after}; '
        'a headless caller can only infer the focus from the stage position',
    )

    # a bad layer is refused
    try:
        runner.start_autofocus(layer='Purple')
        check('unknown layer refused', False, 'NO RAISE')
    except ArgumentRefusedError as e:
        check('unknown layer refused', e.reason == 'layer_unknown', f'reason={e.reason}')

    # second run works (no lock left behind)
    p2 = runner.start_autofocus(layer='BF')
    o2 = p2.result(timeout=300)
    check(
        'a second start_autofocus is accepted',
        str(o2.status).lower().find('fail') < 0,
        f'status={o2.status}',
    )


run(body)
