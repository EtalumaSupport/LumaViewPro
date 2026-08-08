"""Tests for VideoWriter H.264 (PyAV) encoding behavior.

The writer has exactly one backend: PyAV/libx264. These tests pin the
encoder-boundary contracts -- bit-depth conversion, is_color deferral,
per-frame drop accounting, thread capping, and true sub-1-fps rates --
against a fake av module so no real encode runs.
"""

from unittest import mock

import numpy as np
import pytest

import modules.video_writer as video_writer_module
from modules.video_writer import VideoWriter


class _FakeStream:
    """Captures attributes the writer sets on the PyAV stream."""

    def __init__(self):
        self.thread_count = None
        self.width = None
        self.height = None
        self.pix_fmt = None
        self.options = None

    def encode(self, frame=None):
        return []


class _FakeContainer:
    def __init__(self, stream):
        self._stream = stream

    def add_stream(self, codec, rate=None):
        return self._stream

    def mux(self, packet):
        pass

    def close(self):
        pass


def _make_fake_av(captured_frames):
    """Fake av module recording every array handed to VideoFrame.from_ndarray.

    Returns (fake_av, fake_stream); appends (array_copy, format) tuples to
    captured_frames on each encode.
    """
    fake_stream = _FakeStream()
    fake_av = mock.MagicMock()
    fake_av.open.return_value = _FakeContainer(fake_stream)

    def _from_ndarray(arr, format=None):
        captured_frames.append((arr.copy(), format))
        return object()

    fake_av.VideoFrame.from_ndarray.side_effect = _from_ndarray
    return fake_av, fake_stream


class TestVideoWriter16Bit:
    """A uint16 frame with no significant_bits routes through the one canonical
    converter at full 16-bit container depth, not a separate 16->8 entry point."""

    def test_uint16_no_sigbits_uses_canonical_converter(self, tmp_path):
        captured = []
        fake_av, _ = _make_fake_av(captured)
        frame = np.zeros((100, 100), dtype=np.uint16)
        eight = np.zeros((100, 100), dtype=np.uint8)

        with (
            mock.patch.object(video_writer_module, 'av', fake_av),
            mock.patch.object(
                video_writer_module.image_utils, 'convert_16bit_to_8bit', return_value=eight
            ) as legacy,
            mock.patch.object(
                video_writer_module.image_utils, 'convert_to_8bit', return_value=eight
            ) as canonical,
        ):
            writer = VideoWriter(
                output_path=tmp_path / 'depth.mp4', fps=30, include_timestamp_overlay=False
            )
            writer.add_frame(image=frame, timestamp=None, significant_bits=None)

        legacy.assert_not_called()
        canonical.assert_called_once()
        args, kwargs = canonical.call_args
        passed_sig = kwargs.get('significant_bits')
        if passed_sig is None and len(args) >= 2:
            passed_sig = args[1]
        assert passed_sig == 16


class TestVideoBitDepth:
    """A uint16 frame must reach the encoder as uint8: the codec silently
    degrades deeper frames into corrupted video (the historic mp4v defect)."""

    def test_videowriter_converts_16bit_frame(self, tmp_path):
        captured = []
        fake_av, _ = _make_fake_av(captured)
        with mock.patch.object(video_writer_module, 'av', fake_av):
            writer = VideoWriter(
                output_path=tmp_path / 'c16.mp4', fps=10.0, include_timestamp_overlay=False
            )
            writer.add_frame(image=np.ones((100, 100), dtype=np.uint16) * 1000)
        assert len(captured) == 1
        written_image, _fmt = captured[0]
        assert written_image.dtype == np.uint8, (
            '16-bit frame must be converted to uint8 before encode'
        )

    def test_videowriter_passes_8bit_unchanged(self, tmp_path):
        captured = []
        fake_av, _ = _make_fake_av(captured)
        with mock.patch.object(video_writer_module, 'av', fake_av):
            writer = VideoWriter(
                output_path=tmp_path / 'c8.mp4', fps=10.0, include_timestamp_overlay=False
            )
            writer.add_frame(image=np.ones((100, 100), dtype=np.uint8) * 128)
        assert len(captured) == 1
        written_image, _fmt = captured[0]
        assert written_image.dtype == np.uint8
        assert written_image[0, 0] == 128, 'uint8 input must pass through unmodified'

    def test_videowriter_converts_color_16bit(self, tmp_path):
        captured = []
        fake_av, _ = _make_fake_av(captured)
        with mock.patch.object(video_writer_module, 'av', fake_av):
            writer = VideoWriter(
                output_path=tmp_path / 'c16c.mp4', fps=10.0, include_timestamp_overlay=False
            )
            writer.add_frame(image=np.ones((100, 100, 3), dtype=np.uint16) * 500)
        assert len(captured) == 1
        written_image, fmt = captured[0]
        assert written_image.dtype == np.uint8, '16-bit color frame must be converted to uint8'
        assert fmt == 'rgb24'


class TestEagerInitColorDeferral:
    """A writer built with explicit width/height but color=None must defer
    encoder init to the first frame: the eager path cannot know the frame
    ndim, so locking in a gray encoder would corrupt a caller that feeds
    pre-colored RGB. With color set, the output is always 3-channel, so the
    encoder opens eagerly."""

    def test_color_none_eager_dims_defers_to_first_frame(self, tmp_path):
        captured = []
        fake_av, _ = _make_fake_av(captured)
        with mock.patch.object(video_writer_module, 'av', fake_av):
            writer = VideoWriter(
                output_path=tmp_path / 'eager.mp4',
                fps=30,
                width=32,
                height=24,
                color=None,
                include_timestamp_overlay=False,
            )
            # color=None -> encoder init deferred until the first frame's ndim.
            fake_av.open.assert_not_called()
            rgb = np.zeros((24, 32, 3), dtype=np.uint8)
            rgb[:, :, 0] = 200
            writer.add_frame(image=rgb, timestamp=None)
            writer.close()
        fake_av.open.assert_called_once()
        assert writer._is_color is True
        assert captured[0][1] == 'rgb24', 'RGB input into a color=None writer must encode color'

    def test_color_set_eager_dims_inits_immediately(self, tmp_path):
        fake_av, _ = _make_fake_av([])
        with mock.patch.object(video_writer_module, 'av', fake_av):
            writer = VideoWriter(
                output_path=tmp_path / 'eager_color.mp4',
                fps=30,
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )
            writer.close()
        # color set -> output is always RGB; encoder opens eagerly as color.
        fake_av.open.assert_called_once()
        assert writer._is_color is True


class TestPyavEncoderThreadCap:
    """libx264 runs multi-threaded but capped to cores-2, so the encode scales
    with the machine while always leaving headroom for the GUI/GL main thread.
    (Uncapped it grabs every core and froze the GUI mid-encode; the single-
    thread pin that dodged the libx264 teardown deadlock is lifted now that the
    av 17.0.1 libx264 fixes that teardown.)"""

    def test_thread_count_capped_to_cores_minus_two(self, tmp_path):
        fake_av, fake_stream = _make_fake_av([])
        with (
            mock.patch.object(video_writer_module, 'av', fake_av),
            mock.patch.object(video_writer_module.os, 'cpu_count', return_value=8),
        ):
            VideoWriter(
                output_path=tmp_path / 'capped.mp4',
                fps=30,
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

        assert fake_stream.thread_count == 6, 'cores-2 on an 8-core box leaves GUI headroom'

    def test_thread_count_floor_is_one(self, tmp_path):
        fake_av, fake_stream = _make_fake_av([])
        with (
            mock.patch.object(video_writer_module, 'av', fake_av),
            mock.patch.object(video_writer_module.os, 'cpu_count', return_value=1),
        ):
            VideoWriter(
                output_path=tmp_path / 'floor.mp4',
                fps=30,
                width=32,
                height=24,
                color='Red',
                include_timestamp_overlay=False,
            )

        assert fake_stream.thread_count == 1, 'never below 1 thread on a 1-2 core box'


class TestVideoWriterDropAccounting:
    """A frame the encoder fails to write is a dropped frame: it must be
    counted as dropped and must NOT inflate the written-frame count."""

    def test_pyav_encode_error_counts_as_drop(self, tmp_path):
        fake_av, fake_stream = _make_fake_av([])

        def _raise(_frame):
            raise RuntimeError('encode boom')

        fake_stream.encode = _raise
        with mock.patch.object(video_writer_module, 'av', fake_av):
            writer = VideoWriter(
                output_path=tmp_path / 'out.mp4',
                fps=30,
                width=32,
                height=24,
                include_timestamp_overlay=False,
            )
            writer.add_frame(image=np.zeros((24, 32, 3), dtype=np.uint8), timestamp=None)
        assert writer._frame_count == 0, 'a failed encode must not count as a written frame'
        assert writer.dropped_frames == 1


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
                # Production reads back writer.output_path as the record
                # authority after close.
                self.output_path = kwargs.get('output_path') or (args[0] if args else None)

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
    which libx264 rejects -- the output is an empty, unplayable file and the
    recording is lost. The rate is preserved as a Fraction so playback duration
    stays true."""

    def test_pyav_sub_one_fps_not_truncated_to_zero(self, tmp_path):
        fake_stream = _FakeStream()
        fake_container = _RateCapturingContainer(fake_stream)
        fake_av = mock.MagicMock()
        fake_av.open.return_value = fake_container

        out = tmp_path / 'slow.mp4'
        with mock.patch.object(video_writer_module, 'av', fake_av):
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
