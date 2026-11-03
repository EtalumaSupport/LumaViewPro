"""Camera P14: how bright brightfield renders, by objective, at two lights.

The simulator's brightfield is matched to the bench by objective; this
measures the bench it is matched to. On the mounted objective, the field is
read at two settings: brightfield's own stored settings (the shipped
defaults, unless the installation changed them) and the transition stage's
start, 25 mA, 25 ms and 0 dB. Each reading is a frame-valid still at the
camera's own depth and in 8 bits: its mean, its median, and the fraction of
its pixels at the frame's top value. A dark frame at each setting gives the
black level the means sit on.

    python tests/capability/camera_p14_brightfield_level.py --objectives '4x Oly,20x Oly'                   # a simulated LS850
    python tests/capability/camera_p14_brightfield_level.py --objectives '4x Oly,40x w/collar' --hardware  # the connected scope

`--objectives` are catalogue keys, read in that order, each selected the
way any host selects one: on a turret scope the turret turns to the slot
the installation has it assigned to (Z parked and restored); on a scope
without one, the objective is the glass the operator has mounted. The stage
homes at bring-up; nothing else moves. Brightfield is lit only while
its frames are taken. With no `--objectives`, the mounted objective is read
and nothing is selected.
"""

import sys
import traceback

import numpy as np

from harness import HARDWARE, check, figure, hardware_session, make_session, report

LAYER = 'BF'
# The transition stage's stated start, given for the 40x on a Ronchi.
STAGE_START = {'illumination_ma': 25.0, 'exposure_ms': 25.0, 'gain_db': 0.0}
OBJECTIVES = (
    tuple(sys.argv[sys.argv.index('--objectives') + 1].split(','))
    if '--objectives' in sys.argv
    else ()
)


def _levels(image):
    flat = np.asarray(image).reshape(-1)
    top = int(flat.max())
    return {
        'mean': round(float(flat.mean()), 3),
        'median': float(np.median(flat)),
        'max': top,
        'frac_at_max': round(float((flat == top).mean()), 5),
    }


def _read(session, label, light):
    scope = session.scope
    scope.camera.set_gain_db(light['gain_db'])
    scope.camera.set_exposure_ms(light['exposure_ms'])
    figure(
        f'{label}: gain dB, exposure ms',
        (scope.camera.read_gain_db(), scope.camera.read_exposure_ms()),
    )
    for lit in (False, True):
        state = 'lit' if lit else 'dark'
        try:
            if lit:
                scope.illumination.led_on(LAYER, light['illumination_ma'])
            native = scope.imaging.capture_still(force_to_8bit=False, accept_dark=True)
            eight = scope.imaging.capture_still(force_to_8bit=True, accept_dark=True)
        finally:
            scope.illumination.leds_off()
        figure(f'{label} {state}: native {native.image.dtype}', _levels(native.image))
        figure(f'{label} {state}: 8-bit', _levels(eight.image))


def main():
    session_cm = hardware_session() if HARDWARE else _sim()
    with session_cm as (session, _runner):
        scope = session.scope
        figure('has turret', scope.capabilities.has_turret)
        figure('camera pixel format', scope.camera.read_pixel_format())
        stored = session.settings[LAYER]
        defaults = {k: float(stored[k]) for k in ('illumination_ma', 'exposure_ms', 'gain_db')}
        figure(f'{LAYER} stored settings', defaults)
        for objective in OBJECTIVES or (scope.runtime_state.get_current_objective_id(),):
            if OBJECTIVES:
                session.select_objective(objective)
            active = scope.runtime_state.get_current_objective_id()
            if not check(
                f'{objective} is the active objective', active == objective, f'{active!r}'
            ):
                continue
            _read(session, f'{objective} stored', defaults)
            _read(session, f'{objective} stage start', STAGE_START)


def _sim():
    from contextlib import contextmanager

    @contextmanager
    def _session():
        session, _live = make_session('probe_bf_level', home=True)
        try:
            yield session, None
        finally:
            session.shutdown()

    return _session()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
