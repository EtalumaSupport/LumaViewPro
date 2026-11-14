"""A recording's first kept frame counts for a change made just before it.

A recording waits, before its first frame, on the condition a capture drains
on (``ImagingAPI.changes_in_force``), so the first frame it keeps counts for
every change by its source's count of frames; a run's video step no longer sleeps after
it starts, so every camera's step starts one exposure sooner. Three
recordings, each through the member the GUI's press reaches:

1. a run's video step (``runner.start_single_scan``) right after a black
   level change;
2. a run's video step with Auto_Gain on, whose arm records a gain and an
   exposure change;
3. a manual recording (``session.manual_recording.start``) right after a
   gain change.

Each records as frames (``video_as_frames``) at every frame the camera
delivers, so the first file is the first frame after the start. Its
``frame_id`` (the file's record) names the frame the camera delivered,
caught by a frame listener, and that frame is judged through
``judge_frame`` as it arrives. The settings are written on the session
in memory, never saved.

    python tests/capability/recording_p02_the_first_frame_counts_for_a_change.py
    python tests/capability/recording_p02_the_first_frame_counts_for_a_change.py --hardware
"""

import argparse
import pathlib
import sys
import threading

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report

from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.frame_validity import FrameValiditySource
from _lumascope.modules.image_utils import read_file_record

_MANUAL_FRAMES = 5


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--layer', default='BF')
    return parser.parse_args()


class _Delivered:
    """Every frame the camera delivers, by its frame id, judged as it arrives.

    Judged at delivery: ``judge_frame`` judges a frame against each source's latest change, and a
    run's cleanup restores the camera after its step, so a judgment made
    after the run would be of the restore, not of the change before the
    recording.
    """

    def __init__(self, imaging):
        self._imaging = imaging
        self._lock = threading.Lock()
        self._awaited = None
        self.by_id = {}
        imaging.add_frame_listener(self._on_frame, name='recording_p02')

    def _on_frame(self, frame):
        judged = {j.change.source: j for j in self._imaging.judge_frame(frame)}
        with self._lock:
            self.by_id[int(frame.chunks['FrameID'])] = judged
            if self._awaited is not None and len(self.by_id) >= self._awaited[0]:
                self._awaited[1].set()

    def await_more(self, n, timeout=30.0):
        event = threading.Event()
        with self._lock:
            self._awaited = (len(self.by_id) + n, event)
        return event.wait(timeout)

    def close(self):
        self._imaging.remove_frame_listener(self._on_frame)


def _judge(name, delivered, first_file, sources):
    if first_file is None:
        check(f'{name}: the recording kept a frame', False, 'no first frame file')
        return
    frame_id = read_file_record(first_file)['frame_id']
    figure(f'{name}.first_file', first_file.name)
    figure(f'{name}.first_frame_id', frame_id)
    judged = delivered.by_id.get(int(frame_id))
    check(
        f'{name}: the first kept frame was delivered while listened',
        judged is not None,
        f'frame_id {frame_id}; delivered ids {min(delivered.by_id, default=None)}..'
        f'{max(delivered.by_id, default=None)}',
    )
    if judged is None:
        return
    for source in sources:
        judgment = judged.get(source)
        figure(
            f'{name}.{source.name}.frames_after',
            judgment.frames_after if judgment is not None else None,
        )
        check(
            f'{name}: the first kept frame counts for {source.name}',
            judgment is not None and judgment.counts,
            repr(judgment),
        )


def _first(folder, pattern, before):
    files = sorted(p for p in folder.rglob(pattern) if p not in before)
    return files[0] if files else None


def _video_step(session, runner, layer, parent, name, change, sources):
    scope = session.scope
    folder = parent / name
    folder.mkdir(parents=True, exist_ok=True)
    delivered = _Delivered(scope.imaging)
    try:
        change()
        protocol = session.create_empty_protocol()
        session.add_step(protocol)
        outcome = runner.start_single_scan(
            protocol, sequence_name=f'recording_p02_{name}', parent_dir=str(folder)
        ).result(timeout=120)
    finally:
        delivered.close()
    check(
        f'{name}: the run completed',
        outcome.status == 'completed',
        f'{outcome.status} {outcome.reason}',
    )
    _judge(name, delivered, _first(folder, '*_Frame_0000.tif*', set()), sources)


def _manual(session, live, change, source):
    scope = session.scope
    name = 'manual_gain'
    before = set(live.rglob('*.tif*'))
    delivered = _Delivered(scope.imaging)
    try:
        change()
        operation = session.manual_recording.start()
        check(
            f'{name}: {_MANUAL_FRAMES} frames came after start',
            delivered.await_more(_MANUAL_FRAMES),
        )
        operation.stop()
        finished = operation.result(timeout=60)
    finally:
        delivered.close()
    figure(f'{name}.end_reason', getattr(finished, 'end_reason', None))
    _judge(name, delivered, _first(live, '*_0000_*.tif*', before), (source,))


def main():
    import _lumascope.modules.common_utils as common_utils

    args = _arguments()
    live = probe_dir('recording_p02')
    session_cm = hardware_session() if HARDWARE else headless_session(live, acquiring=(args.layer,))
    with session_cm as (session, runner):
        scope = session.scope
        if HARDWARE:
            live = pathlib.Path(session.settings_snapshot()['live_folder'])
        else:
            scope._camera_driver.set_timing_mode('realistic')
        figure('live_folder', str(live))
        layer = args.layer
        # Every frame is kept as its own file, the step's one layer is this one.
        session.update_settings('video_as_frames', True)
        session.update_settings('video.max_fps', 0.0)
        session.set_channel_settings(layer, video_duration_s=1.0)
        for other in common_utils.get_channels():
            if other != layer:
                session.set_channel_settings(other, acquire=None)
        session.set_channel_settings(layer, acquire=AcquireMode.VIDEO)
        session.show_channel(layer)
        camera = scope.camera
        parent = live / 'recording_p02'

        camera.set_black_level(2.0)
        scope.imaging.capture_still()
        _video_step(
            session,
            runner,
            layer,
            parent,
            'step_black_level',
            lambda: camera.set_black_level(4.0),
            (FrameValiditySource.BLACK_LEVEL,),
        )

        session.set_channel_settings(layer, auto_gain=True)
        _video_step(
            session,
            runner,
            layer,
            parent,
            'step_auto_gain',
            lambda: None,
            (FrameValiditySource.GAIN, FrameValiditySource.EXPOSURE),
        )
        session.set_channel_settings(layer, auto_gain=False)

        gain = session.settings_snapshot()[layer]['gain_db']
        camera.set_gain_db(gain)
        scope.imaging.capture_still()
        _manual(session, live, lambda: camera.set_gain_db(gain + 6.0), FrameValiditySource.GAIN)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
