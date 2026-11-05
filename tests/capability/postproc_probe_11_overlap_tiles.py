"""Overlapping tiled sets on one field, kept for the stitch-registration instrument.

#838 and #860 (``STITCH_REGISTRATION_PLAN_2026-10-12.md``, the #838 section)
are designed from real overlapping tiles, and none existed. This captures
one tiled scan per overlap at the field the scope is on, through the member
a protocol run reaches (``runner.start_single_scan`` on the session's own
sequenced capture config), one layer acquiring, autofocus as stored, and
keeps each run folder whole. Each set is then stitched in every
``StitchMode`` and what the build answered is recorded: the stitch's
correctness is the instrument's to judge, not this probe's, so a stitch
outcome is a figure and never a check.

The overlap is written through ``update_settings`` and the other layers set
to acquire nothing through ``set_channel_acquire``, both in memory only;
both are put back before the session closes and the installation's
``current.json`` is checked unchanged by the harness.

    python tests/capability/postproc_probe_11_overlap_tiles.py
    python tests/capability/postproc_probe_11_overlap_tiles.py --hardware --specimen usaf --z 7116
"""

import argparse
import pathlib
import sys

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report
from runfolder import run_protocol_folder


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument(
        '--specimen', default='sim', help='names the sets: usaf, glass, plate, ronchi'
    )
    parser.add_argument('--layer', default='BF')
    parser.add_argument('--tiling', default='3x3')
    parser.add_argument('--overlaps', type=float, nargs='+', default=[10.0, 20.0])
    parser.add_argument(
        '--z', type=float, default=None, help='the focus Z the sets are taken at, um'
    )
    parser.add_argument(
        '--at-bookmark',
        action='store_true',
        help="the sets are taken at the installation's X, Y and Z bookmark",
    )
    parser.add_argument(
        '--objective', default=None, help='a catalogue key; selected before the sets are taken'
    )
    parser.add_argument(
        '--out',
        type=pathlib.Path,
        default=None,
        help='where the sets are kept; a temp folder if not given',
    )
    return parser.parse_args()


def _tiles(folder):
    return sorted(p for p in folder.rglob('*.tif*') if 'stitched' not in p.name.lower())


def _one_set(session, runner, live, args, overlap, here):
    from _lumascope.modules.exceptions import CaptureError

    name = f'{args.specimen}_{args.tiling}_{overlap:g}pct'
    session.update_settings('tiling_overlap_percent', overlap)
    # Each set under its own parent: the run folder is found as the first
    # protocol record beneath it.
    parent = live / name
    parent.mkdir()
    outcome, folder = run_protocol_folder(
        parent,
        session,
        runner,
        tiling=args.tiling,
        sequence_name=name,
        positions=[{'x': here.x_mm, 'y': here.y_mm, 'z': here.z_um, 'name': 'A1'}],
    )
    check(
        f'{name}: the run completed',
        outcome.status == 'completed',
        f'{outcome.status} {outcome.reason}',
    )
    check(f'{name}: a run folder was written', folder is not None)
    if folder is None:
        return
    figure(f'{name}.folder', str(folder))
    columns, rows = (int(n) for n in args.tiling.split('x'))
    tiles = _tiles(folder)
    check(
        f'{name}: one frame per tile',
        len(tiles) == columns * rows,
        f'{len(tiles)} frames for a {args.tiling} grid',
    )
    for mode in session.post_processing.stitch_modes():
        try:
            result = session.post_processing.stitch(folder, mode=mode)
        except CaptureError as error:
            figure(f'{name}.stitch.{mode}', f'{type(error).__name__}: {error}')
            continue
        figure(
            f'{name}.stitch.{mode}',
            {
                'message': result.message,
                'made': [str(p) for p in result.artifact_paths],
                'degraded': [str(p) for p in result.degraded_outputs],
            },
        )


def main():
    args = _arguments()
    live = probe_dir('overlap_tiles') if args.out is None else args.out.resolve()
    live.mkdir(parents=True, exist_ok=True)
    # Simulated, a second layer acquires so the probe sets it off and back.
    session_cm = (
        hardware_session(objective=args.objective)
        if HARDWARE
        else headless_session(live, acquiring=(args.layer, harness.LED0))
    )
    with session_cm as (session, runner):
        from _lumascope.modules.acquire_mode import AcquireMode

        motion = session.scope.motion
        if args.objective is not None and not HARDWARE:
            session.select_objective(args.objective)
        if args.at_bookmark:
            # Bring-up homes every axis, so the field set in LumaViewPro is
            # handed over as its bookmark.
            figure('bookmark', str(session.bookmark))
            session.go_to_bookmark(('X', 'Y', 'Z'))
        elif not HARDWARE:
            motion.move_xy(x_um=40000.0)
            motion.move_xy(y_um=30000.0)
        if args.z is not None or (not HARDWARE and not args.at_bookmark):
            motion.move_z(args.z if args.z is not None else 1000.0)
        motion.wait_until_finished_moving(timeout_s=60)
        before = session.get_settings_snapshot()
        overlap_before = before['tiling_overlap_percent']
        acquiring_before = {
            layer: stored['acquire']
            for layer, stored in before.items()
            if isinstance(stored, dict) and stored.get('acquire') and layer != args.layer
        }
        here = session.get_current_plate_position()
        figure('specimen', args.specimen)
        figure('objective', session.scope.runtime_state.get_current_objective_id())
        figure('layer', args.layer)
        figure('at', {'x_mm': here.x_mm, 'y_mm': here.y_mm, 'z_um': here.z_um})
        figure('other_layers_set_off', sorted(acquiring_before))
        try:
            for layer in acquiring_before:
                session.set_channel_acquire(layer, None)
            for overlap in args.overlaps:
                _one_set(session, runner, live, args, overlap, here)
        finally:
            session.update_settings('tiling_overlap_percent', overlap_before)
            for layer, mode in acquiring_before.items():
                session.set_channel_acquire(layer, AcquireMode(mode))
        after = session.get_settings_snapshot()
        check(
            'the overlap and every layer acquiring are put back',
            after['tiling_overlap_percent'] == overlap_before
            and all(after[layer]['acquire'] == mode for layer, mode in acquiring_before.items()),
        )
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
