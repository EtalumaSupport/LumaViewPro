"""Stack S3/S10: a run's per-step breakdown from the production traces -- the
IOTask trace (queue wait and execution per lane task) and the frame-validity
trace (the drains) -- over one single-scan run on a 6-well plate, two layers.

The fast tier: no serial transit, no travel time, so what is left is the
product's own hops, drains and writes.

    python tests/capability/stack_p06_run_breakdown.py            # simulator
    python tests/capability/stack_p06_run_breakdown.py --hardware
"""

import collections
import csv
import datetime
import itertools
import statistics
import sys
import time

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report
from lib import profile_trace


def _run(session, runner, protocol, parent, name):
    parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pending = runner.run_single_scan(
        protocol=protocol, sequence_name=name, parent_dir=str(parent), enable_image_saving=True
    )
    outcome = pending.wait(timeout_s=3600)
    ended = time.monotonic()
    pending.wait_for_files(timeout_s=300)
    files_done = time.monotonic()
    records = sorted(parent.rglob('protocol_record.tsv'))
    return outcome, started, ended, files_done, (records[0] if records else None)


def _record_rows(record):
    lines = record.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('Filename\t'))
    columns = lines[header].split('\t')
    return [
        dict(zip(columns, line.split('\t'), strict=True))
        for line in lines[header + 1 :]
        if line.strip()
    ]


def _iotask_summary(run_dir):
    f = next(iter(run_dir.rglob('iotask_trace.csv')), None)
    if f is None:
        return {'iotask_trace.csv': 'absent'}
    by = collections.defaultdict(lambda: {'n': 0, 'wait': [], 'exec': []})
    with open(f) as fh:
        for row in csv.DictReader(fh):
            k = (row['executor'], row['action'])
            by[k]['n'] += 1
            by[k]['wait'].append(float(row['queue_wait_ms'] or 0))
            by[k]['exec'].append(float(row['exec_ms'] or 0))
    out = {}
    for (ex, name), d in sorted(by.items(), key=lambda kv: -sum(kv[1]['exec'])):
        out[f'{ex}:{name}'] = {
            'n': d['n'],
            'wait_med_ms': round(statistics.median(d['wait']), 1),
            'wait_max_ms': round(max(d['wait']), 1),
            'exec_med_ms': round(statistics.median(d['exec']), 1),
            'exec_max_ms': round(max(d['exec']), 1),
            'exec_sum_ms': round(sum(d['exec'])),
        }
    return out


def _validity_summary(run_dir):
    f = next(iter(run_dir.rglob('frame_validity_trace.csv')), None)
    if f is None:
        return {'frame_validity_trace.csv': 'absent'}
    events = collections.Counter()
    by_source = collections.Counter()
    with open(f) as fh:
        for row in csv.DictReader(fh):
            events[row['event']] += 1
            if row['event'] == 'invalidate':
                by_source[row['source']] += 1
    return {
        'events': dict(events),
        'invalidations_by_source': dict(by_source),
        'frames_drained': events['credit'],
    }


def body(session, runner, live):
    session.select_labware('6 well microplate')
    protocol = session.new_protocol(tiling='1x1')
    figure('S10.protocol_steps', protocol.num_steps())
    trace_dir = live / 'profile'
    profile_trace.enable(output_dir=str(trace_dir))
    outcome, started, ended, files_done, record = _run(
        session, runner, protocol, live / 'run', 'stack_run'
    )
    profile_trace.disable()
    check('S10: the run completed', outcome.status == 'completed', str(outcome.status))
    rows = _record_rows(record) if record else []
    figure('S10.captures', len(rows))
    figure('S10.run_wall_s', round(ended - started, 2))
    figure('S10.files_done_after_run_end_s', round(files_done - ended, 2))
    figure('S10.s_per_capture', round((ended - started) / max(1, len(rows)), 2))
    stamps = [datetime.datetime.fromisoformat(r['Timestamp']) for r in rows]
    gaps = sorted((b - a).total_seconds() for a, b in itertools.pairwise(stamps))
    if gaps:
        figure(
            'S10.capture_interval',
            {'median_s': round(statistics.median(gaps), 3), 'max_s': round(gaps[-1], 3)},
        )
    for k, v in _iotask_summary(trace_dir).items():
        figure(f'S10.iotask.{k}', v)
    validity = _validity_summary(trace_dir)
    figure('S10.frame_validity', validity)
    if rows and 'frames_drained' in validity:
        figure('S10.frames_drained_per_capture', round(validity['frames_drained'] / len(rows), 1))
    figure('S10.trace_dir', str(trace_dir))


def main():
    live = probe_dir('stack_p06')
    if HARDWARE:
        with hardware_session() as session:
            runner = session.create_protocol_runner()
            body(session, runner, live)
        sys.exit(report())
    with headless_session(live, acquiring=('BF', 'Blue')) as (session, runner):
        body(session, runner, live)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
