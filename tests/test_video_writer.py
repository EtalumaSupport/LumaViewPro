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


class TestVideoWriter16BitFallback:
    """A uint16 frame with no significant_bits routes through the one canonical
    converter at full 16-bit container depth, not a separate 16->8 entry point."""

    def test_uint16_no_sigbits_uses_canonical_converter(self, cv2_writer):
        writer, _fake = cv2_writer
        frame = np.zeros((100, 100), dtype=np.uint16)
        eight = np.zeros((100, 100), dtype=np.uint8)

        with (
            mock.patch.object(
                video_writer_module.image_utils, 'convert_16bit_to_8bit', return_value=eight
            ) as legacy,
            mock.patch.object(
                video_writer_module.image_utils, 'convert_to_8bit', return_value=eight
            ) as canonical,
        ):
            writer.add_frame(image=frame, timestamp=None, significant_bits=None)

        legacy.assert_not_called()
        canonical.assert_called_once()
        args, kwargs = canonical.call_args
        passed_sig = kwargs.get('significant_bits')
        if passed_sig is None and len(args) >= 2:
            passed_sig = args[1]
        assert passed_sig == 16


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
            save_encoding='8bit',
            capture_depth=8,
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
            save_encoding='8bit',
            capture_depth=8,
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


class TestBuildVideoSignificantBits:
    """build_video must read each source TIFF's significant-bit depth and pass
    it to the writer, exactly as the protocol-post pipeline (_create_video)
    does. A right-aligned 12-bit frame (full-scale payload 4095) scaled as if it
    were full 16-bit renders near-black -- white maps to ~15/255 -- so the depth
    has to reach add_frame on this path too, not just the pipeline path."""

    @staticmethod
    def _write_right_aligned_12bit(path, value=4095):
        import tifffile as tf

        # Store a uint16 frame whose payload is right-aligned 12-bit (0..4095)
        # and record SignificantBits=12 via the durable private tag the loader
        # reads back (image_utils._TIFF_TAG_SIGNIFICANT_BITS == 65123, type SHORT).
        frame = np.full((8, 8), value, dtype=np.uint16)
        tf.imwrite(str(path), frame, extratags=[(65123, 3, 1, 12, True)])

    def test_right_aligned_12bit_depth_reaches_writer(self, tmp_path):
        import modules.image_utils as image_utils
        import modules.video_builder as video_builder_module
        from modules.video_builder import VideoBuilder

        src = tmp_path / 'frames'
        src.mkdir()
        frame_path = src / 'frame_0000.tiff'
        self._write_right_aligned_12bit(frame_path)

        # Guard the fixture itself: the file must record 12-bit depth, not the
        # 16-bit container width, or the test could not distinguish the bug.
        assert image_utils.read_tiff_significant_bits(frame_path) == 12

        captured = []

        class _RecordingWriter:
            dropped_frames = 0

            def __init__(self, *args, **kwargs):
                pass

            def add_frame(self, image=None, timestamp=None, significant_bits=None):
                captured.append(significant_bits)

            def close(self):
                pass

        with mock.patch.object(video_builder_module, 'VideoWriter', _RecordingWriter):
            builder = VideoBuilder(has_turret=False)
            result = builder.build_video(
                source_dir=src,
                output_file=tmp_path / 'out.mp4',
                fps=10,
            )

        assert result['frame_count'] == 1, 'the readable frame must encode'
        # The bug passed significant_bits=None (16-bit fallback in add_frame),
        # which maps 4095 -> ~15/255 and renders the video near-black.
        assert captured == [12], (
            'build_video must hand the writer the source frame depth (12), '
            f'not the 16-bit container width; got {captured}'
        )


class _RateCapturingContainer:
    """Records the rate handed to add_stream so a test can assert the encoder
    was opened at the real frame rate, not a truncated one."""

    def __init__(self, stream):
        self._stream = stream
        self.rate = 'UNSET'

    def add_stream(self, codec, rate=None):
        self.rate = rate
        return self._stream

    def close(self):
        pass


class TestSubOneFpsRate:
    """A slow recording captures fewer frames than the seconds elapsed
    (long-exposure / timelapse), so its true rate is below 1 fps. The encoder
    must receive that real sub-1 rate: truncating it to an integer yields 0,
    which libx264/cv2 reject -- the output is an empty, unplayable file and the
    recording is lost. The rate is preserved as a Fraction so playback duration
    stays true."""

    def test_pyav_sub_one_fps_not_truncated_to_zero(self, tmp_path):
        fake_stream = _FakeStream()
        fake_container = _RateCapturingContainer(fake_stream)
        fake_av = mock.MagicMock()
        fake_av.open.return_value = fake_container

        out = tmp_path / 'slow.mp4'
        with (
            mock.patch.object(video_writer_module, '_HAS_PYAV', True),
            mock.patch.object(video_writer_module, 'av', fake_av, create=True),
        ):
            VideoWriter(
                output_path=out,
                fps=0.3,  # 3 frames over 10 seconds
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

        assert fake_container.rate != 0, 'sub-1 fps must not open the encoder at rate 0'
        assert float(fake_container.rate) == pytest.approx(0.3)


class _FpsGatedCv2VideoWriter:
    """Mimics OpenCV's built-in AVI/MJPEG encoder, the fallback used when no
    FFMPEG plugin is present: it asserts fps >= 1 and fails to open below it.
    Reports openness based on the fps it was constructed with."""

    def __init__(self, fps):
        self._fps = fps
        self.frames = []

    def isOpened(self):
        return self._fps is not None and self._fps >= 1

    def write(self, frame):
        self.frames.append(frame)
        return True

    def release(self):
        pass


class TestCv2SubOneFpsFallback:
    """The cv2/AVI fallback must not lose a true sub-1-fps recording. The
    FFMPEG-backed encoder honors a fractional rate, so the writer passes the
    real rate (0.3) -- never an int floored to 0. When the backend in use is the
    built-in AVI encoder (no FFMPEG plugin), which refuses sub-1 fps, the writer
    must still end up open (reopened at the 1 fps floor with a warning) rather
    than silently produce an empty, unplayable file."""

    def _make_writer(self, tmp_path, ctor):
        with (
            mock.patch('modules.video_writer.cv2.VideoWriter', side_effect=ctor),
            mock.patch('modules.video_writer._HAS_PYAV', False),
        ):
            return VideoWriter(
                output_path=tmp_path / 'slow.avi',
                fps=0.3,  # 3 frames over 10 seconds
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

    def test_ffmpeg_backend_preserves_true_sub_one_rate(self, tmp_path):
        # When the backend accepts sub-1 fps (FFMPEG), the real rate is kept
        # and the writer opens on the first try -- no clamp, no second open.
        fps_args = []

        def _ctor(*args, **kwargs):
            fps = kwargs.get('fps')
            fps_args.append(fps)
            return _FakeCv2VideoWriter()

        self._make_writer(tmp_path, _ctor)
        assert fps_args == [pytest.approx(0.3)], 'a supported sub-1 rate must pass through intact'

    def test_builtin_backend_rejecting_sub_one_does_not_lose_recording(self, tmp_path, monkeypatch):
        fps_args = []
        warnings = []
        monkeypatch.setattr(
            video_writer_module.logger, 'warning', lambda m, *a, **k: warnings.append(str(m))
        )

        def _ctor(*args, **kwargs):
            fps = kwargs.get('fps')
            fps_args.append(fps)
            return _FpsGatedCv2VideoWriter(fps)

        writer = self._make_writer(tmp_path, _ctor)

        # No init attempt may pass a rate of 0 -- that yields an empty file.
        assert 0 not in fps_args and 0.0 not in fps_args
        # The real sub-1 rate was tried first (preserve it where supported)...
        assert fps_args[0] == pytest.approx(0.3)
        # ...and because this backend rejected it, the writer reopened at the
        # 1 fps floor so the recording survives instead of being lost.
        assert fps_args[-1] >= 1
        assert writer._cv2_video is not None and writer._cv2_video.isOpened()
        assert any('sub-1' in w or 'fps >= 1' in w for w in warnings), (
            'the faster-than-real-time fallback must warn, naming the constraint'
        )


class _FpsGatedCv2VideoWriter:
    """Mimics OpenCV's built-in AVI/MJPEG encoder, the fallback used when no
    FFMPEG plugin is present: it asserts fps >= 1 and fails to open below it.
    Reports openness based on the fps it was constructed with."""

    def __init__(self, fps):
        self._fps = fps
        self.frames = []

    def isOpened(self):
        return self._fps is not None and self._fps >= 1

    def write(self, frame):
        self.frames.append(frame)
        return True

    def release(self):
        pass


class TestCv2SubOneFpsFallback:
    """The cv2/AVI fallback must not lose a true sub-1-fps recording. The
    FFMPEG-backed encoder honors a fractional rate, so the writer passes the
    real rate (0.3) -- never an int floored to 0. When the backend in use is the
    built-in AVI encoder (no FFMPEG plugin), which refuses sub-1 fps, the writer
    must still end up open (reopened at the 1 fps floor with a warning) rather
    than silently produce an empty, unplayable file."""

    def _make_writer(self, tmp_path, ctor):
        with (
            mock.patch('modules.video_writer.cv2.VideoWriter', side_effect=ctor),
            mock.patch('modules.video_writer._HAS_PYAV', False),
        ):
            return VideoWriter(
                output_path=tmp_path / 'slow.avi',
                fps=0.3,  # 3 frames over 10 seconds
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

    def test_ffmpeg_backend_preserves_true_sub_one_rate(self, tmp_path):
        # When the backend accepts sub-1 fps (FFMPEG), the real rate is kept
        # and the writer opens on the first try -- no clamp, no second open.
        fps_args = []

        def _ctor(*args, **kwargs):
            fps = kwargs.get('fps')
            fps_args.append(fps)
            return _FakeCv2VideoWriter()

        self._make_writer(tmp_path, _ctor)
        assert fps_args == [pytest.approx(0.3)], 'a supported sub-1 rate must pass through intact'

    def test_builtin_backend_rejecting_sub_one_does_not_lose_recording(self, tmp_path, monkeypatch):
        fps_args = []
        warnings = []
        monkeypatch.setattr(
            video_writer_module.logger, 'warning', lambda m, *a, **k: warnings.append(str(m))
        )

        def _ctor(*args, **kwargs):
            fps = kwargs.get('fps')
            fps_args.append(fps)
            return _FpsGatedCv2VideoWriter(fps)

        writer = self._make_writer(tmp_path, _ctor)

        # No init attempt may pass a rate of 0 -- that yields an empty file.
        assert 0 not in fps_args and 0.0 not in fps_args
        # The real sub-1 rate was tried first (preserve it where supported)...
        assert fps_args[0] == pytest.approx(0.3)
        # ...and because this backend rejected it, the writer reopened at the
        # 1 fps floor so the recording survives instead of being lost.
        assert fps_args[-1] >= 1
        assert writer._cv2_video is not None and writer._cv2_video.isOpened()
        assert any('sub-1' in w or 'fps >= 1' in w for w in warnings), (
            'the faster-than-real-time fallback must warn, naming the constraint'
        )


class TestFpsFromFrames:
    """fps_from_frames is the single owner of the frames-per-recorded-second
    rate both recording paths (protocol video and manual recording) use. A
    sub-1-fps recording must keep its true float rate, not floor to 0 (which
    loses the file) or clamp to 1 (which distorts duration)."""

    def test_sub_one_rate_is_preserved_not_floored(self):
        from modules.video_writer import fps_from_frames

        # 3 frames over 10 s -> 0.3 fps, not 0 and not 1.
        assert fps_from_frames(3, 10.0) == pytest.approx(0.3)

    def test_normal_rate(self):
        from modules.video_writer import fps_from_frames

        assert fps_from_frames(100, 10.0) == pytest.approx(10.0)

    def test_both_recording_paths_use_the_shared_helper(self):
        # The protocol and manual recording modules both import the one helper,
        # so the rate policy cannot drift between them again.
        import modules.manual_video_finalize as mvf
        import modules.video_capture as vc
        from modules.video_writer import fps_from_frames

        assert vc.fps_from_frames is fps_from_frames
        assert mvf.fps_from_frames is fps_from_frames
