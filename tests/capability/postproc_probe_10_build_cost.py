"""How long does each post-processing build take on this host?

The number a build's "Slow task" budget is set from. The builds' time scales
with the folder, so the probe captures folders of a stated size through the
production runner on the simulator and times each build through its
``session.post_processing`` member, as a REST or SDK caller reaches it:

  - a tiled scan, one position, in several channels: stitch, composite,
    quick enhance of the folder, cell count;
  - a Z-stack at one position: z-project;
  - a time-lapse at one position: video.

The frames are the simulator's at the shipped frame size (1900 x 1900); the
build reads and writes them on this host's CPU and disk, so the figures are
this host's. Run it on the field-class host: a fast development machine says
nothing about what a field build costs. The figures are printed for a person
to judge; the checks are only that every build answered.

    python tests/capability/postproc_probe_10_build_cost.py
    python tests/capability/postproc_probe_10_build_cost.py --tiles 3 --channels 3 \\
        --slices 10 --frames 50
"""

import argparse
import os
import platform
import shutil
import sys
import time

from harness import check, figure, headless_session, probe_dir, report
from postproc_probe_06_cell_count import METHOD
from runfolder import run_protocol_folder

CHANNELS = ('BF', 'Blue', 'Green', 'Red')
ZSTACK_STEP_UM = 5.0


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles', type=int, default=3, help='tiles per side of the scan')
    parser.add_argument('--channels', type=int, default=3, choices=range(1, len(CHANNELS) + 1))
    parser.add_argument('--slices', type=int, default=10, help='Z-stack slices')
    parser.add_argument('--frames', type=int, default=50, help='time points of the time-lapse')
    return parser.parse_args()


def _host():
    import psutil

    figure('host', f'{platform.system()} {platform.release()} {platform.machine()}')
    figure('host_cpu', f'{platform.processor() or "unknown"}, {os.cpu_count()} logical cores')
    figure('host_ram_gb', round(psutil.virtual_memory().total / 2**30, 1))


def _over_a_well(session):
    """Park over a plate position clear of the lower Z limit, and return it."""
    session.scope.motion.move_absolute('X', 40000.0)
    session.scope.motion.move_absolute('Y', 30000.0)
    session.scope.motion.move_absolute('Z', 1000.0)
    session.scope.motion.wait_until_finished_moving(timeout_s=60)
    here = session.get_current_plate_position()
    return [{'x': here['x'], 'y': here['y'], 'z': here['z'], 'name': 'A1'}]


def _images(folder):
    return sum(1 for _ in folder.rglob('*.tif*'))


def _files(folder):
    return sum(1 for p in folder.rglob('*') if p.is_file())


def _timed(name, build, folder):
    """Run one build on its own copy of *folder* and report its wall time.

    Each build gets the run's folder as the run left it: a build writes its
    outputs into the folder, and the next build would read them as input.
    The copy is not timed, and its name leads with the build's: a folder
    whose name ends in ``_video`` is read as a video recording.
    """
    own = folder.with_name(f'{name}_{folder.name}')
    shutil.copytree(folder, own)
    figure(f'{name}_input_images', _images(own))
    before = _files(own)
    started = time.monotonic()
    try:
        result = build(own)
    except Exception as outcome:
        check(f'{name} answered', False, f'{type(outcome).__name__}: {outcome}')
        return
    elapsed = time.monotonic() - started
    check(f'{name} answered', result is not None, getattr(result, 'message', ''))
    figure(f'{name}_s', round(elapsed, 2))
    figure(f'{name}_files_written', _files(own) - before)


def _tiled(args):
    live = probe_dir('build_cost_tiled')
    acquiring = CHANNELS[: args.channels]
    with headless_session(live, acquiring=acquiring) as (session, runner):
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            tiling=f'{args.tiles}x{args.tiles}',
            sequence_name='probe_build_cost_tiled',
            positions=_over_a_well(session),
        )
        check('the tiled scan made a folder', folder is not None, str(outcome))
        if folder is None:
            return
        post = session.post_processing
        _timed('stitch', post.stitch, folder)
        _timed('composite', post.composite, folder)
        _timed('enhance', post.enhance, folder)
        _timed('count_cells', lambda own: post.count_cells(own, method=METHOD), folder)


def _zstack(args):
    live = probe_dir('build_cost_zstack')
    zstack = {
        'step_size': ZSTACK_STEP_UM,
        'range': ZSTACK_STEP_UM * (args.slices - 1),
        'position': 'Current Position at Center',
    }
    with headless_session(live, acquiring=('BF',), zstack=zstack) as (session, runner):
        _over_a_well(session)
        pending = runner.start_zstack(layer='BF', sequence_name='probe_build_cost_zstack')
        pending.result(timeout=600)
        check("the Z-stack's files were written", pending.files.result(timeout=120) is not None)
        records = sorted(live.rglob('protocol_record.tsv'))
        check('the Z-stack made a folder', bool(records))
        if not records:
            return
        folder = records[0].parent
        _timed('zproject', lambda own: session.post_processing.zproject(own, method='Max'), folder)


def _timelapse(args):
    live = probe_dir('build_cost_video')
    period_s = 1.0
    # Minutes and hours, as the protocol settings hold them.
    timelapse = {
        'period': period_s / 60.0,
        'duration': args.frames * period_s / 3600.0,
        'labware': '96 well microplate',
    }
    with headless_session(live, acquiring=('BF',), protocol=timelapse) as (session, runner):
        outcome, folder = run_protocol_folder(
            live,
            session,
            runner,
            sequence_name='probe_build_cost_video',
            positions=_over_a_well(session),
            single_scan=False,
        )
        check('the time-lapse made a folder', folder is not None, str(outcome))
        if folder is None:
            return
        _timed('video', session.post_processing.video, folder)


def main() -> int:
    args = _arguments()
    _host()
    _tiled(args)
    _zstack(args)
    _timelapse(args)
    return report()


if __name__ == '__main__':
    sys.exit(main())
