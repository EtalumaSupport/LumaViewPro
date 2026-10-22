"""Does a run's own post-run build harm the next run? (#812, W4 part E0)

Three runs, in order. A is a time-series run saved as a hyperstack, whose
per-well stacks build on the post-processing lane once its images land (or,
with ``--first composite``, a composite run, whose merge builds the same way). B,
the same time series, starts the first moment the runner admits it: A's
claim released and its images written, which is before A's build ends, so
B overlaps the build the way a next run in a queue does. C runs alone once
every build has ended. A and C bracket B, so B's difference is the build and
not warm-up or drift. Each run reports its outcome, its files against its
record, and the interval between captures from the record's own timestamps,
taken at the grab.

A run's builds take no claim and nothing in ``prepare()`` reads them, so B
is admitted under A's build today; the overlap is the case the one
post-processing path's next-run rule decides (W4 part E3). The figures --
the overlap, A's build's length after B started, the intervals -- are
printed for a person to judge; the checks are that every run completed with
every file written, and that B was refused only while A held the scope or
its images drained.

What it measures is a headless host: no live-view display, no Kivy. A run
unharmed here says the build does not starve the capture path; it says
nothing about the GUI's live view. The simulator's frames are near-flat and
compress to almost nothing, so its builds take milliseconds: on the
simulator this checks the instrument, and the load is the bench's.

    python tests/capability/postproc_probe_09_build_during_run.py            # simulator
    python tests/capability/postproc_probe_09_build_during_run.py --hardware \\
        --labware "96 well microplate" --still BF --scans 4                  # this scope
"""

import argparse
import datetime
import itertools
import statistics
import sys
import time

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report

# B asks to start until the runner admits it; these are the refusals of a
# run that is still letting go, and any other is the probe's failure.
_LETTING_GO = frozenset({'already_running', 'files_writing'})


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument(
        '--labware',
        default=None if HARDWARE else '6 well microplate',
        help="the plate to plan over (default: the scope's own selection on hardware)",
    )
    parser.add_argument('--tiling', default='1x1')
    parser.add_argument(
        '--first',
        choices=('hyperstack', 'composite'),
        default='hyperstack',
        help="A's own build: a time series' hyperstacks, or a composite run's merge",
    )
    parser.add_argument('--scans', type=int, default=3, help='scans in each time series')
    parser.add_argument(
        '--period',
        type=float,
        default=1.0,
        help='seconds between scans; shorter than a scan runs them back to back',
    )
    parser.add_argument(
        '--still',
        default='',
        help='layers to capture as still images for these runs, comma-separated (e.g. BF)',
    )
    return parser.parse_args()


def _capture_still(session, layers):
    """Capture *layers* as still images, for this session only.

    Through the Session member the GUI's acquire toggle uses, on the
    session's settings in memory. Nothing saves them: the installation's
    current.json is left as it was, which `hardware_session()` checks.
    """
    if not layers:
        return
    for layer in layers:
        session.set_layer_acquire(layer, 'image')
    print(f'STILL: {", ".join(layers)} captured as still images for these runs', flush=True)


def _refuse_video_layers(session):
    """A video step has one record row and its frames sit where a cell count
    does not look, so neither the timing nor the build load would be real."""
    import modules.common_utils as common_utils

    settings = session.get_settings_snapshot()
    video = [
        layer for layer in common_utils.get_layers() if settings[layer].get('acquire') == 'video'
    ]
    if video:
        print(
            f'REFUSED: {", ".join(video)} set to video. Pass --still {",".join(video)} to '
            'capture them as still images for these runs.',
            flush=True,
        )
        sys.exit(2)


def _start(runner, protocol, parent, name):
    parent.mkdir(parents=True, exist_ok=True)
    return runner.start_protocol(
        protocol, sequence_name=name, parent_dir=str(parent), enable_image_saving=True
    )


def _finish(pending, started, parent):
    outcome = pending.result(timeout=3600)
    ended = time.monotonic()
    # The outcome answers before the run's files drain, and its files before
    # its own build ends; the folder is read once both have.
    assert pending.files.result(timeout=3600) is not None, "the run's files never finished"
    records = sorted(parent.rglob('protocol_record.tsv'))
    return outcome, started, ended, (records[0] if records else None)


def _start_when_admitted(runner, protocol, parent, name):
    """Start the run the first moment the runner admits it.

    Returns the operation, when it started, and the refusals met first.
    """
    from modules.exceptions import ProtocolRunRefusedError

    refused = []
    while True:
        try:
            pending = _start(runner, protocol, parent, name)
        except ProtocolRunRefusedError as refusal:
            if refusal.reason not in _LETTING_GO:
                raise
            refused.append(refusal.reason)
            time.sleep(0.05)
            continue
        return pending, time.monotonic(), refused


def _post_run_steps(session):
    from modules import live_work

    lane = (live_work.POST_PROCESSING, live_work.POST_PROCESSING_QUEUED)
    return [item.name for item in session.live_work.work if item.kind in lane]


def _record_rows(record):
    lines = record.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('Filename\t'))
    columns = lines[header].split('\t')
    return [
        dict(zip(columns, line.split('\t'), strict=True))
        for line in lines[header + 1 :]
        if line.strip()
    ]


def _report_run(name, outcome, started, ended, record):
    check(f'{name}: the run completed', outcome.status == 'completed', str(outcome.status))
    if record is None:
        check(f'{name}: the run wrote a record', False)
        return
    rows = _record_rows(record)
    missing = [r['Filename'] for r in rows if not (record.parent / r['Filename']).is_file()]
    check(f'{name}: every recorded file is on disk', not missing and bool(rows), str(missing[:5]))
    figure(f'{name}.captures', len(rows))
    figure(f'{name}.duration_s', round(ended - started, 2))
    stamps = [datetime.datetime.fromisoformat(r['Timestamp']) for r in rows]
    gaps = [(b - a).total_seconds() for a, b in itertools.pairwise(stamps)]
    if gaps:
        gaps.sort()
        figure(f'{name}.interval_median_s', round(statistics.median(gaps), 3))
        figure(f'{name}.interval_p95_s', round(gaps[int(0.95 * (len(gaps) - 1))], 3))
        figure(f'{name}.interval_max_s', round(gaps[-1], 3))


def main():
    args = _arguments()
    live = probe_dir('build_during_run')
    session_cm = (
        hardware_session() if HARDWARE else headless_session(live, acquiring=('BF', 'Blue'))
    )
    with session_cm as (session, runner):
        if args.labware is not None:
            session.select_labware(args.labware)
        _capture_still(session, [layer for layer in args.still.split(',') if layer])
        _refuse_video_layers(session)
        if args.first == 'hyperstack':
            from modules import image_mode

            session.update_settings(
                'image_output_format.sequenced', image_mode.OUTPUT_FORMAT_HYPERSTACK
            )
        protocol = session.new_protocol(tiling=args.tiling)
        period = datetime.timedelta(seconds=args.period)
        session.set_protocol_schedule(protocol, period=period, duration=period * args.scans)
        figure('first', args.first)
        figure('protocol.steps', protocol.num_steps())
        figure('protocol.scans', args.scans)
        figure('labware', args.labware or "the scope's own selection")

        start_a = time.monotonic()
        if args.first == 'hyperstack':
            pending_a = _start(runner, protocol, live / 'A', 'probe_A')
            # The files settle when the run's hyperstack build ends.
            build_of_a = pending_a.files
        else:
            (live / 'A').mkdir(parents=True, exist_ok=True)
            pending_a = runner.start_composite(sequence_name='probe_A', parent_dir=str(live / 'A'))
            # A composite run's outcome settles when its merge ends.
            build_of_a = pending_a
        built = {}
        build_of_a.add_done_callback(lambda _op: built.setdefault('a', time.monotonic()))

        pending_b, start_b, refused = _start_when_admitted(runner, protocol, live / 'B', 'probe_B')
        steps_at_b = _post_run_steps(session)
        figure('B.refusals_before_admitted', len(refused))
        figure('B.admitted_after_A_started_s', round(start_b - start_a, 2))
        figure('B.post_run_steps_at_start', ', '.join(steps_at_b) or 'none')

        outcome_a, _, end_a, record_a = _finish(pending_a, start_a, live / 'A')
        if args.first == 'hyperstack':
            _report_run('A', outcome_a, start_a, end_a, record_a)
        else:
            check(
                'A: the composite run completed', outcome_a.status == 'completed', outcome_a.status
            )
            check('A: the composite merged', outcome_a.merged, outcome_a.merge_reason)
        outcome_b, _, end_b, record_b = _finish(pending_b, start_b, live / 'B')
        _report_run('B', outcome_b, start_b, end_b, record_b)
        figure('A.build_after_B_started_s', round(built['a'] - start_b, 3))
        figure('B.overlap_with_A_build_s', round(max(0.0, min(built['a'], end_b) - start_b), 3))

        start_c = time.monotonic()
        pending_c = _start(runner, protocol, live / 'C', 'probe_C')
        outcome_c, _, end_c, record_c = _finish(pending_c, start_c, live / 'C')
        _report_run('C', outcome_c, start_c, end_c, record_c)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
