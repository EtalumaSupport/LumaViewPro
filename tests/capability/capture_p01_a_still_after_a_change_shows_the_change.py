"""A still taken right after an LED change or a Z move shows the scene after it.

The capture readiness row: a capture waits until every change is met by its
source's rule (``capabilities.frame_rules``, the camera's timing record when
its profile names one) and keeps only a frame the judge counts. The judge is
tracked state; whether the count it waits is enough is the camera's, so on a
scope this reads the pixels as the falsifier: the still kept right after the
change against later stills of the same scene, with nothing changed between
them.

1. The rules: each source's count and the run that measured it
   (``FrameRule.provenance``); a source whose count was measured carries one.
2. LED off -> on, ``--trials`` times: a dark still, the shown channel lit at
   its stored current, one still at once (``session.capture_still``), then
   ``--later`` stills of the lit scene.
3. Z by ``--dz`` and back, ``--trials`` times each way: a still before,
   ``move_z_by`` (it returns once Z has arrived), one still at once, then
   ``--later`` stills.

Each trial's figures are the distance from the kept still to the last later
still, the distance between the last two later stills (the scene's own
frame-to-frame change), and the change's own size (the still before against
the last later still), in two measures: the mean absolute pixel difference,
and the largest difference of one of eight horizontal bands' means (a frame
read out partly before the change, the rolling shutter's stale case, moves
one band), the band for the LED alone. A kept still that predates the change
sits about the change's whole size from the scene after it; one that shows
the change sits within the scene's own frame-to-frame change.

Bring-up homes the stage, so a scope run names the sample's plate X and Y
and its focus Z, and the channel's levels if the stored ones are not the
sample's (written on the session in memory, never saved):

    python tests/capability/capture_p01_a_still_after_a_change_shows_the_change.py
    python tests/capability/capture_p01_a_still_after_a_change_shows_the_change.py --hardware \\
        --x 61.93 --y 43.75 --z 5000.97 --ma 25 --exposure-ms 23.22 --gain-db 8.7 --dz 30
"""

import argparse
import sys

import numpy as np

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report

from _lumascope.modules.coord_transformations import PlateLocation
from _lumascope.modules.frame_validity import FrameValiditySource

_BANDS = 8

# The kept still is called stale above this share of the change's own size:
# a frame exposed before the change sits at about 1 of it, a band read
# before it at about 1 in that band.
_STALE_SHARE = 0.1


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--channel', default='BF')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--later', type=int, default=5)
    parser.add_argument('--dz', type=float, default=100.0, help='um')
    parser.add_argument('--z', type=float, default=None, help='the focus Z, um; bring-up homes it')
    parser.add_argument(
        '--x', type=float, default=None, help='the sample, plate mm; bring-up homes it'
    )
    parser.add_argument('--y', type=float, default=None, help='the sample, plate mm')
    parser.add_argument('--ma', type=float, default=None, help="the channel's current; else stored")
    parser.add_argument('--exposure-ms', type=float, default=None, help="else the channel's stored")
    parser.add_argument('--gain-db', type=float, default=None, help="else the channel's stored")
    return parser.parse_args()


def _distances(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    pixel = float(np.mean(np.abs(a - b)))
    band = max(
        abs(float(x.mean()) - float(y.mean()))
        for x, y in zip(
            np.array_split(a, _BANDS, axis=0), np.array_split(b, _BANDS, axis=0), strict=True
        )
    )
    return {'pixel': pixel, 'band': band}


def _judgment(still, source):
    for judgment in still.record.judgments or ():
        if judgment.change.source is source:
            return judgment
    return None


def _trial(name, session, channel, before, change, source, later, measures):
    change()
    kept = session.capture_still(channel)
    after = [session.capture_still(channel).image for _ in range(later)]
    judgment = _judgment(kept, source)
    figure(f'{name}.frames_after', judgment.frames_after if judgment else None)
    check(
        f'{name}: the kept still counts for the {source.value} change',
        judgment is not None and judgment.counts,
        repr(judgment),
    )
    size = _distances(before, after[-1])
    scene = _distances(after[-2], after[-1])
    kept_d = _distances(kept.image, after[-1])
    figure(f'{name}.size', size)
    figure(f'{name}.scene', scene)
    figure(f'{name}.kept', kept_d)
    figure(f'{name}.later_pixel', [_distances(img, after[-1])['pixel'] for img in after[:-1]])
    for measure in measures:
        check(
            f"{name}: the change is visible by {measure} (larger than the scene's own)",
            size[measure] > 3 * scene[measure],
            f'size {size[measure]:.3f}, scene {scene[measure]:.3f}',
        )
    check(
        f'{name}: the kept still shows the scene after the change',
        all(
            kept_d[measure] <= max(2 * scene[measure], _STALE_SHARE * size[measure])
            for measure in measures
        ),
        f'kept {kept_d}, scene {scene}, size {size}',
    )


def main():
    args = _arguments()
    live = probe_dir('capture_p01')
    session_cm = (
        hardware_session() if HARDWARE else headless_session(live, acquiring=(args.channel,))
    )
    with session_cm as (session, _runner):
        scope = session.scope
        if not HARDWARE:
            scope._camera_driver.set_timing_mode('realistic')
            scope._led_driver.set_timing_mode('realistic')
            scope._motion_driver.set_timing_mode('realistic')
        rules = scope.capabilities.frame_rules
        for source in (
            FrameValiditySource.EXPOSURE,
            FrameValiditySource.GAIN,
            FrameValiditySource.LED,
            FrameValiditySource.Z_MOVE,
            FrameValiditySource.FRAME_SIZE,
        ):
            rule = rules[source]
            figure(f'rule.{source.value}', {'count': rule.count, 'provenance': rule.provenance})
        # The simulator's default camera has no timing record, so its counts
        # are the shipped ones; a scope's camera is the one this asks about.
        for source in (FrameValiditySource.LED, FrameValiditySource.Z_MOVE) if HARDWARE else ():
            check(
                f'the {source.value} count is a measured one',
                rules[source].provenance is not None,
                repr(rules[source]),
            )

        channel = args.channel
        illumination = scope.illumination
        # Written on the session in memory, never saved.
        if args.ma is not None:
            session.update_settings(f'{channel}.illumination_ma', args.ma)
        if args.exposure_ms is not None:
            session.update_settings(f'{channel}.exposure_ms', args.exposure_ms)
        if args.gain_db is not None:
            session.update_settings(f'{channel}.gain_db', args.gain_db)
        stored = session.get_settings_snapshot()[channel]
        current_ma = stored['illumination_ma']
        figure('illumination_ma', current_ma)
        figure('exposure_ms', stored['exposure_ms'])
        session.show_channel(channel)
        illumination.led_on(channel, current_ma)
        if args.x is not None or args.y is not None:
            scope.motion.move_to_plate(PlateLocation(x_mm=args.x, y_mm=args.y, z_um=None))
        if args.z is not None:
            scope.motion.move_z(args.z)
        figure('position', scope.motion.get_current_positions())
        figure('lit_mean_dn', float(session.capture_still(channel).image.mean()))

        for trial in range(args.trials):
            illumination.led_off(channel)
            dark = session.capture_still(channel, accept_dark=True).image
            _trial(
                f'led_on.{trial}',
                session,
                channel,
                dark,
                lambda: illumination.led_on(channel, current_ma),
                FrameValiditySource.LED,
                args.later,
                ('pixel', 'band'),
            )

        for trial in range(args.trials):
            for sign in (1, -1):
                before = session.capture_still(channel).image
                _trial(
                    f'z_move.{trial}.{"up" if sign > 0 else "down"}',
                    session,
                    channel,
                    before,
                    lambda dz=sign * args.dz: scope.motion.move_z_by(dz),
                    FrameValiditySource.Z_MOVE,
                    args.later,
                    # A move shifts detail, not a band's light: on the LS720
                    # a 30 um step moved the pixels 71 DN and a band's mean
                    # under 1 DN, so the band measure reads only the light.
                    ('pixel',),
                )
        illumination.led_off(channel)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
