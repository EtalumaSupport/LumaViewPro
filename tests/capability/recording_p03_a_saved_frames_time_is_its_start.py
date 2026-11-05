"""A saved frame's time is its start on the camera's clock.

A still's file and every recorded frame (a frame file, or an MP4's manifest
entry) state the frame's start on the camera's own clock, its ``Timestamp``
chunk mapped onto the host's wall clock, with the source
``camera_frame_start``; a recording's manifest is graded the same and
counts the frames that came without their tick (``frames_unstamped``).
Three saves, each through the member the GUI's press reaches, at the
layer's exposure set to 25 ms:

1. a still, through a one-step image run (``runner.start_single_scan``);
2. a manual recording as frames (``video_as_frames`` on);
3. a manual recording as MP4.

Every frame the camera delivers is caught by a frame listener: its id, its
start mapped through a clock mapping latched beside the save
(``camera_frame_time``; the save's own latch is not public, so the two
differ by the mappings' uncertainty and drift), and its arrival, which
comes after the start by at least the exposure. A saved frame's stated time
is checked against both. The settings are written on the session in
memory, never saved.

    python tests/capability/recording_p03_a_saved_frames_time_is_its_start.py
    python tests/capability/recording_p03_a_saved_frames_time_is_its_start.py --hardware
"""

import argparse
import datetime
import json
import pathlib
import sys
import threading

import harness
from harness import HARDWARE, check, figure, hardware_session, headless_session, probe_dir, report

from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.image_utils import read_file_record
from _lumascope.modules.lumascope_api.frame_record import camera_frame_time

_EXPOSURE_MS = 25.0
_RECORDED_FRAMES = 40
# The stated time against this probe's own mapping of the same tick: both
# latches are off by their uncertainty (about 0.2 ms on the LS850T's Mac)
# and the clocks drift a few ppm over the seconds between them.
_MAPPING_AGREEMENT_MS = 2.0


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--layer', default='BF')
    return parser.parse_args()


class _Delivered:
    """Every delivered frame's mapped start and its arrival, by frame id."""

    def __init__(self, imaging):
        self._imaging = imaging
        self._lock = threading.Lock()
        self._awaited = None
        self.mapping = imaging.map_camera_clock()
        self.by_id = {}
        imaging.add_frame_listener(self._on_frame, name='recording_p03')

    def _on_frame(self, frame):
        start = camera_frame_time(frame, self.mapping) if self.mapping is not None else None
        arrival = datetime.datetime.fromtimestamp(frame.wall_ns / 1e9, datetime.UTC)
        with self._lock:
            self.by_id[int(frame.chunks['FrameID'])] = (start, arrival)
            if self._awaited is not None and len(self.by_id) >= self._awaited[0]:
                self._awaited[1].set()

    def await_more(self, n, timeout=30.0):
        event = threading.Event()
        with self._lock:
            self._awaited = (len(self.by_id) + n, event)
        return event.wait(timeout)

    def close(self):
        self._imaging.remove_frame_listener(self._on_frame)


def _against_delivered(delivered, stated_at, frame_id):
    """The stated time's distance from its delivered frame's mapped start, and its lead on arrival, ms."""
    entry = delivered.by_id.get(int(frame_id))
    if entry is None or entry[0] is None:
        return None, None
    start, arrival = entry
    off = (stated_at - start.at).total_seconds() * 1000
    lead = (arrival - stated_at).total_seconds() * 1000
    return off, lead


def _summary(name, offs, leads, count):
    offs = [o for o in offs if o is not None]
    leads = [lead for lead in leads if lead is not None]
    figure(f'{name}.compared', f'{len(offs)} of {count}')
    check(
        f'{name}: every saved frame was delivered while listened',
        count > 0 and len(offs) == count,
        f'{len(offs)} of {count}',
    )
    if not offs:
        return
    figure(f'{name}.off_ms', (round(min(offs), 3), round(max(offs), 3)))
    figure(f'{name}.arrival_after_stated_ms', (round(min(leads), 1), round(max(leads), 1)))
    check(
        f"{name}: every stated time is its frame start, within the mappings' agreement",
        max(abs(o) for o in offs) <= _MAPPING_AGREEMENT_MS,
        f'worst {max(offs, key=abs):+.3f} ms against {_MAPPING_AGREEMENT_MS} ms',
    )
    check(
        f'{name}: every frame arrived at least its exposure after its stated time',
        min(leads) >= _EXPOSURE_MS,
        f'earliest arrival {min(leads):.1f} ms after',
    )


def _still(session, runner, live):
    folder = live / 'recording_p03' / 'still'
    folder.mkdir(parents=True, exist_ok=True)
    delivered = _Delivered(session.scope.imaging)
    try:
        protocol = session.create_empty_protocol()
        session.add_step(protocol)
        pending = runner.start_single_scan(
            protocol, sequence_name='recording_p03_still', parent_dir=str(folder)
        )
        outcome = pending.result(timeout=120)
        # The run ends before its files are written; they are its own result.
        written = pending.files.result(timeout=120)
    finally:
        delivered.close()
    check(
        'still: the run completed',
        outcome.status == 'completed',
        f'{outcome.status} {outcome.reason}',
    )
    check("still: the run's files were written", written is not None, repr(written))
    files = sorted(p for p in folder.rglob('*') if p.suffix in ('.tif', '.tiff'))
    check('still: one image was saved', len(files) == 1, [f.name for f in files])
    if not files:
        return
    record = read_file_record(files[0])
    figure('still.timestamp_source', record.get('timestamp_source'))
    check(
        'still: its time is the camera frame start',
        record.get('timestamp_source') == 'camera_frame_start',
        record.get('timestamp_source'),
    )
    off, lead = _against_delivered(
        delivered, datetime.datetime.fromisoformat(record['timestamp_iso']), record['frame_id']
    )
    _summary('still', [off], [lead], 1)


def _frame_files(new, delivered):
    offs, leads = [], []
    files = sorted(p for p in new if p.suffix in ('.tif', '.tiff'))
    for path in files:
        record = read_file_record(path)
        if record.get('timestamp_source') != 'camera_frame_start':
            offs.append(None)
            continue
        off, lead = _against_delivered(
            delivered, datetime.datetime.fromisoformat(record['timestamp_iso']), record['frame_id']
        )
        offs.append(off)
        leads.append(lead)
    return offs, leads, len(files)


def _index_entries(manifest, delivered):
    offs, leads = [], []
    index = manifest.get('frame_index') or []
    for entry in index:
        frame_id = (entry.get('chunks') or {}).get('FrameID')
        if frame_id is None or entry.get('ts_s') is None:
            offs.append(None)
            continue
        stated = datetime.datetime.fromtimestamp(entry['ts_s'], datetime.UTC)
        off, lead = _against_delivered(delivered, stated, frame_id)
        offs.append(off)
        leads.append(lead)
    return offs, leads, len(index)


def _manual(session, live, as_frames):
    name = 'manual_frames' if as_frames else 'manual_mp4'
    session.update_settings('video_as_frames', as_frames)
    before = set(live.rglob('*'))
    delivered = _Delivered(session.scope.imaging)
    try:
        operation = session.manual_recording.start()
        check(
            f'{name}: {_RECORDED_FRAMES} frames came after start',
            delivered.await_more(_RECORDED_FRAMES),
        )
        operation.stop()
        finished = operation.result(timeout=120)
    finally:
        delivered.close()
    figure(f'{name}.end_reason', getattr(finished, 'end_reason', None))
    new = [p for p in live.rglob('*') if p not in before]
    manifests = [p for p in new if p.name.endswith('manifest.json')]
    check(f'{name}: one manifest was written', len(manifests) == 1, [str(p) for p in manifests])
    if len(manifests) != 1:
        return
    manifest = json.loads(manifests[0].read_text())
    for key in (
        'frames_selected',
        'frames_written',
        'write_failures',
        'frames_unstamped',
        'timestamp_grade',
    ):
        figure(f'{name}.{key}', manifest.get(key))
    check(
        f'{name}: the manifest is graded camera_frame_start',
        manifest.get('timestamp_grade') == 'camera_frame_start',
        manifest.get('timestamp_grade'),
    )
    check(
        f'{name}: no frame came without its tick',
        manifest.get('frames_unstamped') == 0,
        manifest.get('frames_unstamped'),
    )
    if as_frames:
        _summary(name, *_frame_files(new, delivered))
    else:
        _summary(name, *_index_entries(manifest, delivered))


def main():
    import _lumascope.modules.common_utils as common_utils

    args = _arguments()
    live = probe_dir('recording_p03')
    session_cm = hardware_session() if HARDWARE else headless_session(live, acquiring=(args.layer,))
    with session_cm as (session, runner):
        if HARDWARE:
            live = pathlib.Path(session.get_settings_snapshot()['live_folder'])
        else:
            session.scope._camera_driver.set_timing_mode('realistic')
        figure('live_folder', str(live))
        layer = args.layer
        session.update_settings(f'{layer}.exposure_ms', _EXPOSURE_MS)
        session.update_settings('video.max_fps', 0.0)
        for other in common_utils.get_layers():
            if other != layer:
                session.set_channel_acquire(other, None)
        session.set_channel_acquire(layer, AcquireMode.IMAGE)
        session.show_channel(layer)
        _still(session, runner, live)
        _manual(session, live, as_frames=True)
        _manual(session, live, as_frames=False)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
