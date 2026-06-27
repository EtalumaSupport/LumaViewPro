"""Tests for VideoWriter cv2-fallback channel handling.

The cv2 fallback path is the only consumer in the save path that
expects BGR; PyAV and tifffile both want RGB. add_frame converts
RGB->BGR at the cv2 boundary so callers can hand it RGB uniformly.
"""

from unittest import mock

import numpy as np
import pytest

import modules.video_writer as video_writer_module
from modules.video_writer import VideoWriter


class _FakeCv2VideoWriter:
    """Records frames passed to write() instead of writing to disk."""

    def __init__(self, *args, **kwargs):
        self.frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())
        return True

    def release(self):
        pass


@pytest.fixture
def cv2_writer(tmp_path):
    """VideoWriter forced onto the cv2 fallback path, capturing frames in memory."""
    output_path = tmp_path / 'test.avi'
    fake = _FakeCv2VideoWriter()
    with mock.patch('modules.video_writer.cv2.VideoWriter', return_value=fake):
        writer = VideoWriter(output_path=output_path, fps=30, include_timestamp_overlay=False)
        writer._use_pyav = False
        yield writer, fake


class TestVideoWriterCv2Fallback:
    """cv2.VideoWriter consumes BGR; callers pass RGB. Conversion happens in add_frame."""

    def test_rgb_red_becomes_bgr(self, cv2_writer):
        writer, fake = cv2_writer
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        writer.add_frame(image=rgb, timestamp=None)
        writer.close()
        bgr = fake.frames[0]
        assert bgr[:, :, 2].sum() > 0, 'Red lands at BGR index 2'
        assert bgr[:, :, 0].sum() == 0
        assert bgr[:, :, 1].sum() == 0

    def test_rgb_blue_becomes_bgr(self, cv2_writer):
        writer, fake = cv2_writer
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 2] = 200
        writer.add_frame(image=rgb, timestamp=None)
        writer.close()
        bgr = fake.frames[0]
        assert bgr[:, :, 0].sum() > 0, 'Blue lands at BGR index 0'
        assert bgr[:, :, 1].sum() == 0
        assert bgr[:, :, 2].sum() == 0

    def test_grayscale_frame_unchanged(self, cv2_writer):
        writer, fake = cv2_writer
        gray = np.full((100, 100), 128, dtype=np.uint8)
        writer.add_frame(image=gray, timestamp=None)
        writer.close()
        assert fake.frames[0].shape == (100, 100)
        assert fake.frames[0].sum() > 0


class TestEagerInitColorDeferral:
    """A writer built with explicit width/height but color=None must defer
    the is_color decision to the first frame: the eager path cannot know the
    frame ndim, so locking is_color to color-None gray would corrupt a caller
    that feeds pre-colored RGB. Mirrors the no-dimensions lazy path."""

    def test_color_none_eager_dims_encodes_rgb_as_color(self, tmp_path):
        captured = {}

        def _fake_ctor(*args, **kwargs):
            captured['isColor'] = kwargs.get('isColor')
            return _FakeCv2VideoWriter()

        out = tmp_path / 'eager.avi'
        with mock.patch('modules.video_writer.cv2.VideoWriter', side_effect=_fake_ctor):
            writer = VideoWriter(
                output_path=out,
                fps=30,
                width=32,
                height=24,
                color=None,
                include_timestamp_overlay=False,
            )
            writer._use_pyav = False
            # color=None -> encoder init deferred until the first frame's ndim.
            assert 'isColor' not in captured
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[:, :, 0] = 200
            writer.add_frame(image=rgb, timestamp=None)
            writer.close()
        assert captured.get('isColor') is True

    def test_color_set_eager_dims_inits_immediately(self, tmp_path):
        captured = {}

        def _fake_ctor(*args, **kwargs):
            captured['isColor'] = kwargs.get('isColor')
            return _FakeCv2VideoWriter()

        out = tmp_path / 'eager_color.avi'
        with mock.patch('modules.video_writer.cv2.VideoWriter', side_effect=_fake_ctor):
            with mock.patch('modules.video_writer._HAS_PYAV', False):
                writer = VideoWriter(
                    output_path=out,
                    fps=30,
                    width=32,
                    height=24,
                    color='Red',
                    include_timestamp_overlay=False,
                )
            writer.close()
        # color set -> output is always RGB; encoder opens eagerly as color.
        assert captured.get('isColor') is True


class _FakeStream:
    """Captures attributes the writer sets on the PyAV stream."""

    def __init__(self):
        self.thread_count = None
        self.width = None
        self.height = None
        self.pix_fmt = None
        self.options = None


class _FakeContainer:
    def __init__(self, stream):
        self._stream = stream

    def add_stream(self, codec, rate=None):
        return self._stream

    def close(self):
        pass


class TestPyavEncoderThreadCap:
    """libx264 runs multi-threaded but capped to cores-2, so the encode scales
    with the machine while always leaving headroom for the GUI/GL main thread.
    (Uncapped it grabs every core and froze the GUI mid-encode; the single-
    thread pin that dodged the libx264 teardown deadlock is lifted now that the
    av 17.0.1 libx264 fixes that teardown.)"""

    def test_thread_count_capped_to_cores_minus_two(self, tmp_path):
        fake_stream = _FakeStream()
        fake_av = mock.MagicMock()
        fake_av.open.return_value = _FakeContainer(fake_stream)

        out = tmp_path / 'capped.mp4'
        with (
            mock.patch.object(video_writer_module, '_HAS_PYAV', True),
            mock.patch.object(video_writer_module, 'av', fake_av, create=True),
            mock.patch.object(video_writer_module.os, 'cpu_count', return_value=8),
        ):
            VideoWriter(
                output_path=out,
                fps=30,
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

        assert fake_stream.thread_count == 6, 'cores-2 on an 8-core box leaves GUI headroom'

    def test_thread_count_floor_is_one(self, tmp_path):
        fake_stream = _FakeStream()
        fake_av = mock.MagicMock()
        fake_av.open.return_value = _FakeContainer(fake_stream)

        out = tmp_path / 'floor.mp4'
        with (
            mock.patch.object(video_writer_module, '_HAS_PYAV', True),
            mock.patch.object(video_writer_module, 'av', fake_av, create=True),
            mock.patch.object(video_writer_module.os, 'cpu_count', return_value=1),
        ):
            VideoWriter(
                output_path=out,
                fps=30,
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

        assert fake_stream.thread_count == 1, 'never below 1 thread on a 1-2 core box'


class _FailingCv2VideoWriter:
    """cv2 writer whose write() reports failure, to exercise drop accounting."""

    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self):
        return True

    def write(self, frame):
        return False

    def release(self):
        pass


class TestVideoWriterDropAccounting:
    """A frame the encoder fails to write is a dropped frame: it must be
    counted as dropped and must NOT inflate the written-frame count."""

    def test_cv2_write_failure_counts_as_drop(self, tmp_path):
        fake = _FailingCv2VideoWriter()
        with mock.patch('modules.video_writer.cv2.VideoWriter', return_value=fake):
            writer = VideoWriter(
                output_path=tmp_path / 'out.avi', fps=30, include_timestamp_overlay=False
            )
            writer._use_pyav = False
            writer.add_frame(image=np.zeros((24, 32, 3), dtype=np.uint8), timestamp=None)
        status = writer.get_progress()
        assert status['frame_count'] == 0, 'a failed write must not count as a written frame'
        assert status['dropped_frames'] == 1

    def test_pyav_encode_error_counts_as_drop(self, tmp_path):
        fake_stream = _FakeStream()

        def _raise(_frame):
            raise RuntimeError('encode boom')

        fake_stream.encode = _raise
        fake_av = mock.MagicMock()
        fake_av.open.return_value = _FakeContainer(fake_stream)
        fake_av.VideoFrame.from_ndarray.return_value = object()
        with (
            mock.patch.object(video_writer_module, '_HAS_PYAV', True),
            mock.patch.object(video_writer_module, 'av', fake_av, create=True),
        ):
            writer = VideoWriter(
                output_path=tmp_path / 'out.mp4',
                fps=30,
                width=32,
                height=24,
                include_timestamp_overlay=False,
            )
            writer.add_frame(image=np.zeros((24, 32, 3), dtype=np.uint8), timestamp=None)
        status = writer.get_progress()
        assert status['frame_count'] == 0
        assert status['dropped_frames'] == 1


class TestWriteVideoDropNotification:
    """write_video must surface a recording that lost frames -- otherwise a short
    video is discovered only by frame arithmetic, if ever. write_video runs only
    on the protocol path, so it surfaces drops via a LOG line, never a modal (an
    unattended protocol does not pop a dialog for a non-fatal drop)."""

    def _capture_warnings(self, monkeypatch):
        from modules.notification_center import notifications

        fired = []
        monkeypatch.setattr(notifications, 'warning', lambda *a, **k: fired.append((a, k)))
        return fired

    def _empty_result(self, dropped):
        import queue as _queue

        from modules.video_capture import VideoCaptureResult

        return VideoCaptureResult(
            captured_frames=5,
            calculated_fps=10,
            video_images=_queue.Queue(),
            duration_sec=0.5,
            dropped_frames=dropped,
        )

    def test_producer_drops_log_not_modal(self, tmp_path, monkeypatch):
        import modules.video_capture as vc
        from modules.video_capture import write_video

        fired = self._capture_warnings(monkeypatch)
        logged = []
        monkeypatch.setattr(vc.logger, 'warning', lambda msg, *a, **k: logged.append(str(msg)))
        write_video(
            result=self._empty_result(dropped=2),
            save_folder=tmp_path,
            name='clip',
            video_as_frames=True,
            step={'Color': 'Blue'},
            callbacks={},
        )
        assert fired == [], 'drops must not pop a modal during a protocol'
        assert any('dropped' in m and '2' in m for m in logged), (
            'a recording that dropped frames must log the count'
        )

    def test_no_drops_is_silent(self, tmp_path, monkeypatch):
        from modules.video_capture import write_video

        fired = self._capture_warnings(monkeypatch)
        write_video(
            result=self._empty_result(dropped=0),
            save_folder=tmp_path,
            name='clip',
            video_as_frames=True,
            step={'Color': 'Blue'},
            callbacks={},
        )
        assert fired == [], 'a clean recording must not warn'


class TestVideoBuilderDropAccounting:
    """The post-processing video builder skips unreadable source frames; those
    skips must be counted and returned, not silently dropped from the output."""

    def test_unreadable_sources_counted_as_dropped(self, tmp_path):
        from modules.video_builder import VideoBuilder

        src = tmp_path / 'frames'
        src.mkdir()
        (src / 'a.tiff').write_bytes(b'not a real tiff')
        (src / 'b.tiff').write_bytes(b'also garbage')
        builder = VideoBuilder(has_turret=False)
        result = builder.build_video(
            source_dir=src,
            output_file=tmp_path / 'out.mp4',
            fps=10,
        )
        assert result['frame_count'] == 0, 'no readable frame should encode'
        assert result['dropped_frames'] == 2, 'both unreadable sources must be counted'
