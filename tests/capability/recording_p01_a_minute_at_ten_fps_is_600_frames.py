"""A minute at 10 fps is 600 frames, manual and in a run.

A recording with a rate limit ends when it holds its frame budget,
``ceil(fps x duration)``, or when its frames' own times pass its last slot's
deadline, whichever comes first; a camera delivering at the rate fills the
budget. Before LVP 6648b240b both a manual recording and a protocol video
step at 10 fps for 60 s ended on the host's clock one frame short, 599 frames
``duration_elapsed``, in five of six simulated recordings.

Two recordings, each through the member the GUI's press reaches: a manual
recording (``session.manual_recording.start``) and one scan of a protocol
whose one step, added at the live position, records video
(``runner.start_single_scan``). Each is checked
by its manifest: 600 frames selected and written, budget 600, none failed,
ended ``frame_budget_filled``. The settings are written on the session in
memory (``update_settings``, ``set_channel_settings``), never saved: the
installation's current.json is left as it was, which ``hardware_session()``
checks.

    python tests/capability/recording_p01_a_minute_at_ten_fps_is_600_frames.py
    python tests/capability/recording_p01_a_minute_at_ten_fps_is_600_frames.py --hardware \\
        --layer BF
"""

import argparse
import json
import sys
import time

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report
from tests.stock_channels import LED0

_FPS = 10
_DURATION_S = 60
_BUDGET = _FPS * _DURATION_S


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--layer', default='BF', help='the layer both recordings are of')
    return parser.parse_args()


def _manifests(folder):
    return {
        path: path.stat().st_mtime_ns
        for pattern in ('*_manifest.json', 'recording_manifest.json')
        for path in folder.rglob(pattern)
    }


def _new_manifest(name, folder, before):
    new = [path for path in _manifests(folder) if path not in before]
    check(f'{name}: one manifest was written', len(new) == 1, str(new))
    return json.loads(new[0].read_text()) if len(new) == 1 else None


def _report_recording(name, manifest, waited_s):
    figure(f'{name}.waited_s', round(waited_s, 2))
    if manifest is None:
        return
    for key in (
        'frames_selected',
        'frames_written',
        'write_failures',
        'frame_budget',
        'end_reason',
        'measured_fps',
        'measured_duration_s',
        'timestamp_grade',
    ):
        figure(f'{name}.{key}', manifest.get(key))
    check(f'{name}: its budget is {_BUDGET}', manifest['frame_budget'] == _BUDGET)
    check(
        f'{name}: {_BUDGET} frames selected and written',
        manifest['frames_selected'] == _BUDGET == manifest['frames_written'],
        f'{manifest["frames_selected"]} selected, {manifest["frames_written"]} written',
    )
    check(f'{name}: no frame failed to write', manifest['write_failures'] == 0)
    check(
        f'{name}: it ended on its budget',
        manifest['end_reason'] == 'frame_budget_filled',
        manifest['end_reason'],
    )
    stamps = [entry['ts_s'] for entry in manifest['frame_index'] if entry['ts_s'] is not None]
    if len(stamps) > 1:
        span = stamps[-1] - stamps[0]
        figure(f'{name}.first_frame_ts_s', round(stamps[0], 3))
        figure(f'{name}.last_frame_ts_s', round(stamps[-1], 3))
        figure(f'{name}.rate_fps', round((len(stamps) - 1) / span, 3) if span > 0 else None)


def _manual(session, layer, live):
    session.update_settings('video.max_fps', float(_FPS))
    session.update_settings('video.max_duration_s', float(_DURATION_S))
    session.show_channel(layer)
    before = _manifests(live)
    started = time.monotonic()
    end_reason = session.manual_recording.start().result(timeout=_DURATION_S * 3).end_reason
    waited = time.monotonic() - started
    figure('manual.operation_end_reason', end_reason)
    _report_recording('manual', _new_manifest('manual', live, before), waited)


def _video_step(session, runner, layer, live):
    import _lumascope.modules.common_utils as common_utils
    from _lumascope.modules.acquire_mode import AcquireMode

    # The run's one step is this layer's video: every other layer the
    # installation acquires is set to nothing, on the session alone.
    for other in common_utils.get_channels():
        if other != layer:
            session.set_channel_settings(other, acquire=None)
    session.set_channel_settings(layer, acquire=AcquireMode.VIDEO)
    # A step records at the one global limit, as a manual recording does.
    session.update_settings('video.max_fps', float(_FPS))
    session.set_channel_settings(layer, video_duration_s=float(_DURATION_S))
    # One step at the live position, as the GUI's Add then Run One Scan.
    protocol = session.create_empty_protocol()
    session.add_step(protocol)
    figure('video_step.protocol_steps', protocol.num_steps())
    parent = live / 'video_step'
    parent.mkdir(parents=True, exist_ok=True)
    before = _manifests(parent)
    started = time.monotonic()
    pending = runner.start_single_scan(
        protocol, sequence_name='recording_p01', parent_dir=str(parent)
    )
    outcome = pending.result(timeout=_DURATION_S * 4)
    assert pending.files.result(timeout=_DURATION_S * 4) is not None, (
        "the run's files never finished"
    )
    waited = time.monotonic() - started
    check('video_step: the run completed', outcome.status == 'completed', str(outcome.status))
    _report_recording('video_step', _new_manifest('video_step', parent, before), waited)


def main():
    args = _arguments()
    live = probe_dir('recording_600')
    if HARDWARE:
        session_cm = hardware_session()
    else:
        session_cm = headless_session(live, acquiring=(args.layer, LED0))
    with session_cm as (session, runner):
        if HARDWARE:
            import pathlib

            live = pathlib.Path(session.get_settings_snapshot()['live_folder'])
            figure('live_folder', str(live))
        _manual(session, args.layer, live)
        _video_step(session, runner, args.layer, live)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
