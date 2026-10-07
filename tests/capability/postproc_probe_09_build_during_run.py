"""Does a cell-count build alongside a protocol run harm the run? (#812)

Three runs of one protocol, in order: A alone; B with cell-count builds on A's
folder running back to back on the post-processing lane, the first started
before B; C alone. A and C bracket B, so B's difference is the build and not
warm-up or drift. Each run reports its outcome, its files against its record,
and the interval between captures from the record's own timestamps, taken at
the grab. The figures are printed for a person to judge; the checks are only
that every run completed with every file written and that completed builds
covered all of B.

What it measures is a headless host: no live-view display, no Kivy. A run
unharmed here says the build does not starve the capture path; it says
nothing about the GUI's live view.

    python tests/capability/postproc_probe_09_build_during_run.py            # simulator
    python tests/capability/postproc_probe_09_build_during_run.py --hardware \\
        --labware "96 well microplate"                                      # this scope
"""

import argparse
import datetime
import itertools
import statistics
import sys
import threading
import time

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report
from postproc_probe_06_cell_count import METHOD


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


def _run(session, runner, protocol, parent, name):
    parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pending = runner.run_single_scan(
        protocol=protocol,
        sequence_name=name,
        parent_dir=str(parent),
        enable_image_saving=True,
    )
    outcome = pending.wait(timeout_s=3600)
    ended = time.monotonic()
    # The outcome answers before the run's files drain; the folder is read
    # once the run has finished writing it.
    assert pending.wait_for_files(timeout_s=120) is not None, "the run's files never finished"
    records = sorted(parent.rglob('protocol_record.tsv'))
    return outcome, started, ended, (records[0] if records else None)


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


class _Builds:
    """Cell counts on one folder, back to back, until told to stop."""

    def __init__(self, session, folder):
        self._session = session
        self._folder = folder
        self._stop = threading.Event()
        self.first_started = threading.Event()
        self.completed: list[tuple[float, float]] = []
        self.failed: list[str] = []
        self._thread = threading.Thread(target=self._loop, name='probe-builds', daemon=True)

    def start(self):
        self._thread.start()
        self.first_started.wait(timeout=30)

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _loop(self):
        while not self._stop.is_set():
            began = time.monotonic()
            self.first_started.set()
            try:
                self._session.post_processing.count_cells(self._folder, method=METHOD)
            except Exception as outcome:
                # A refused or failed build is not cover for the run; it is
                # recorded and the loop ends, since every next one would be
                # the same refusal.
                self.failed.append(f'{type(outcome).__name__}: {outcome}')
                return
            self.completed.append((began, time.monotonic()))


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
        protocol = session.new_protocol(tiling=args.tiling)
        figure('protocol.steps', protocol.num_steps())
        figure('labware', args.labware or "the scope's own selection")

        outcome_a, start_a, end_a, record_a = _run(session, runner, protocol, live / 'A', 'probe_A')
        _report_run('A', outcome_a, start_a, end_a, record_a)
        if record_a is None:
            sys.exit(report())

        builds = _Builds(session, record_a.parent)
        builds.start()
        outcome_b, start_b, end_b, record_b = _run(session, runner, protocol, live / 'B', 'probe_B')
        builds.stop()
        _report_run('B', outcome_b, start_b, end_b, record_b)
        figure('B.builds_completed', len(builds.completed))
        if builds.failed:
            figure('B.build_failed', builds.failed[0])
        covered = (
            not builds.failed
            and bool(builds.completed)
            and builds.completed[0][0] <= start_b
            and builds.completed[-1][1] >= end_b
        )
        check('B: completed builds covered the whole run', covered)

        outcome_c, start_c, end_c, record_c = _run(session, runner, protocol, live / 'C', 'probe_C')
        _report_run('C', outcome_c, start_c, end_c, record_c)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
