"""Regression tests for the significant-bits / native pixel-depth owner.

Pins the contract that the summed-capture reducer and the 8-bit display
downconvert key off a frame's significant bits, not its numpy container dtype.

Covered:
  - #570: a 12-bit frame summed (Sum > 1) overflows 4095 and, treated as a
    12-bit value, indexes the 4096-entry display LUT out of range -> IndexError.
    The summed result belongs in a 16-bit container and must downconvert as
    16-bit.
  - #424 depth half: a 10-bit frame must map full-white to full 8-bit white,
    not be treated as 12-bit (which crushes it ~4x dark).
  - The "sum into a 16-bit container" rule: accumulate, saturate at 65535.

These drive the REAL ImagingAPI.get_image path against the simulated camera in
a chosen pixel format + white test pattern, so every grabbed frame is uniform
and summed values are exact and deterministic.

xfail(strict) tests encode behavior that lands with the owner; each flips to
XPASS when its fix arrives, at which point the marker is removed in that commit.
"""

import numpy as np
import pytest

from modules import image_utils
from modules.exceptions import FrameDepthError
from modules.lumascope_api import Lumascope


def _configure_sim(scope, pixel_format, pattern='White'):
    """Put the simulated camera into a fixed depth + uniform bright frame."""
    cam = scope._camera_driver
    cam.set_timing_mode('fast')
    cam.set_pixel_format(pixel_format)
    cam.set_test_pattern(True, pattern)
    scope.imaging.start_streaming()
    return scope


@pytest.fixture
def make_scope():
    """Factory for simulated scopes at a chosen pixel format; auto-teardown."""
    scopes = []

    def _make(pixel_format, pattern='White'):
        scope = _configure_sim(Lumascope(simulate=True), pixel_format, pattern)
        scopes.append(scope)
        return scope

    yield _make
    for scope in scopes:
        try:
            scope.imaging.stop_streaming()
            scope.disconnect()
        except Exception:
            pass


# Uniform white-pattern pixel value the simulator emits per pixel format.
WHITE = {'Mono8': 255, 'Mono10': 1023, 'Mono12': 4095}


class TestSummedCaptureDepthCeiling:
    """The summed reducer + display downconvert must respect significant bits."""

    @pytest.mark.parametrize('sum_count', [2, 3, 30])
    def test_summed_12bit_display_no_crash(self, make_scope, sum_count):
        """A summed 12-bit capture forced to 8-bit returns a uint8 array.

        The summed result exceeds 4095 and rides in a 16-bit container, so the
        downconvert scales it as 16-bit instead of indexing the 12-bit table.
        """
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=sum_count)
        assert img is not None
        assert img.dtype == np.uint8

    def test_sum_into_16bit_container_accumulates(self, make_scope):
        """Summing accumulates into a 16-bit container (Sum=2 of 4095 -> 8190)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=False, sum_count=2)
        assert img is not None
        assert img.dtype == np.uint16
        assert int(img.max()) == 2 * WHITE['Mono12']  # 8190, no overflow, no over-clip

    def test_sum_saturates_at_container_ceiling(self, make_scope):
        """A high sum count saturates at the 16-bit container max (65535)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=False, sum_count=30)
        assert img is not None
        # 4095 * 30 = 122850 > 65535 -> saturates at the container ceiling.
        assert int(img.max()) == 65535

    def test_single_frame_identity(self, make_scope):
        """Sum=1 returns the native frame unchanged (no clip, no dtype change)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=False, sum_count=1)
        assert img is not None
        assert img.dtype == np.uint16
        assert int(img.max()) == WHITE['Mono12']  # 4095, untouched

    def test_8bit_camera_sum_unaffected(self, make_scope):
        """An 8-bit camera path keeps working: no crash, uint8, saturates at 255."""
        scope = make_scope('Mono8')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=2)
        assert img is not None
        assert img.dtype == np.uint8
        assert int(img.max()) == 255  # 255 * 2 -> saturates at 8-bit ceiling


class TestDisplayDownconvertGenericDepth:
    """The 8-bit display mapping must scale by the frame's real significant bits."""

    def test_10bit_white_maps_to_full_8bit_white(self, make_scope):
        """A full-white 10-bit frame must map to 8-bit 255, not ~63.

        Treating a 10-bit value (max 1023) as 12-bit divides by 4095 and crushes
        white to ~63. The display divisor must come from the significant bits.
        """
        scope = make_scope('Mono10')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=1)
        assert img is not None
        assert img.dtype == np.uint8
        assert int(img.max()) == 255

    def test_12bit_white_maps_to_full_8bit_white(self, make_scope):
        """A full-white 12-bit frame already maps to 255 (guards against regression)."""
        scope = make_scope('Mono12')
        img = scope.imaging.get_image(force_to_8bit=True, sum_count=1)
        assert img is not None
        assert img.dtype == np.uint8
        assert int(img.max()) == 255


def _save_meta(significant_bits=None):
    """Minimal metadata dict that write_tiff / generate_tiff_data accept."""
    meta = {
        'pixel_size_um': 0.5,
        'channel': 'Green',
        'objective': '10x',
        'exposure_time_ms': 50.0,
        'gain_db': 0.0,
        'illumination_ma': 100.0,
        'z_pos_um': 1000.0,
        'plate_pos_mm': {'x': 10.0, 'y': 20.0},
        'datetime': '2026:06:18 12:00:00',
        'camera_make': 'Test',
        'microscope': 'TestScope',
        'well_label': 'A1',
        'well_site': '1',
    }
    if significant_bits is not None:
        meta['significant_bits'] = significant_bits
    return meta


class TestSavePathNativeDepth:
    """Saved TIFFs store raw right-aligned values + an honest SignificantBits
    tag, and read-back scales to 8-bit by that tag -- not a hardcoded /256."""

    def _write(self, tmp_path, value, significant_bits):
        from modules import image_utils

        arr = np.full((8, 8), value, dtype=np.uint16)
        path = tmp_path / 'frame.tif'
        image_utils.write_tiff(
            data=arr,
            file_loc=path,
            metadata=_save_meta(significant_bits),
            ome=True,
            color='Green',
            significant_bits=significant_bits,
            save_encoding='right_aligned',
        )
        return path

    def test_tag_records_payload_depth_not_container(self, tmp_path):
        """A 12-bit frame is tagged SignificantBits=12 and stored right-aligned."""
        import tifffile as tf

        path = self._write(tmp_path, 4095, significant_bits=12)
        with tf.TiffFile(str(path)) as f:
            assert 'SignificantBits="12"' in (f.ome_metadata or '')
            assert int(f.pages[0].asarray().max()) == 4095  # raw, not left-justified to 65520

    def test_read_tiff_significant_bits_roundtrips(self, tmp_path):
        """The SignificantBits tag is recoverable from the written file."""
        from modules import image_utils

        path = self._write(tmp_path, 4095, significant_bits=12)
        assert image_utils.read_tiff_significant_bits(path) == 12

    def test_12bit_file_reads_back_to_full_white(self, tmp_path):
        """A right-aligned 12-bit file scales to 8-bit 255, not crushed to ~15."""
        from modules import image_utils

        path = self._write(tmp_path, 4095, significant_bits=12)
        sig = image_utils.read_tiff_significant_bits(path)
        eight = image_utils.convert_to_8bit(image_utils.read_tiff_with_legacy_collapse(path), sig)
        assert int(eight.max()) == 255

    def test_legacy_16bit_file_still_reads_back_correctly(self, tmp_path):
        """A legacy left-justified file (SignificantBits=16) is not crushed."""
        from modules import image_utils

        path = self._write(tmp_path, 4095 * 16, significant_bits=16)  # 65520, the old *16 form
        sig = image_utils.read_tiff_significant_bits(path)
        assert sig == 16
        eight = image_utils.convert_to_8bit(image_utils.read_tiff_with_legacy_collapse(path), sig)
        # 65520 / 65535 * 255 -> 254: near-white, not crushed to ~15.
        assert int(eight.max()) >= 254


class TestNonOmeNativeDepth:
    """Right-aligned native-depth must survive on the NON-OME path too -- plain
    TIFFs, ImageJ TIFFs, and the per-frame TIFFs the hyperstack workflow forces.
    These are the most common protocol outputs, and a reader that cannot recover
    the depth scales a 12-bit payload by 65535 instead of 4095 -- ~16x too dark."""

    def _write_plain(self, tmp_path, value, significant_bits):
        from modules import image_utils

        arr = np.full((8, 8), value, dtype=np.uint16)
        path = tmp_path / 'frame_plain.tif'
        image_utils.write_tiff(
            data=arr,
            file_loc=path,
            metadata=_save_meta(significant_bits),
            ome=False,
            color='Green',
            significant_bits=significant_bits,
            save_encoding='right_aligned',
        )
        return path

    def test_non_ome_significant_bits_roundtrips(self, tmp_path):
        """A plain (non-OME) 12-bit file recovers SignificantBits=12, not 16."""
        from modules import image_utils

        path = self._write_plain(tmp_path, 4095, significant_bits=12)
        assert image_utils.read_tiff_significant_bits(path) == 12

    def test_non_ome_12bit_reads_back_to_full_white(self, tmp_path):
        """A plain right-aligned 12-bit file scales to 8-bit 255, not crushed to ~15."""
        from modules import image_utils

        path = self._write_plain(tmp_path, 4095, significant_bits=12)
        sig = image_utils.read_tiff_significant_bits(path)
        eight = image_utils.convert_to_8bit(image_utils.read_tiff_with_legacy_collapse(path), sig)
        assert int(eight.max()) == 255


class TestLoadPixelsBoundary:
    """load_pixels is the one read that returns pixels AND their depth in a
    single call, so a caller cannot obtain a frame without the significant bits
    needed to scale it. These pin the back-compat matrix: every on-disk
    encoding the reader must round-trip to the correct depth."""

    def _write_tiff(
        self, tmp_path, value, significant_bits, ome, dtype=np.uint16, name='frame.tif'
    ):
        from modules import image_utils

        arr = np.full((8, 8), value, dtype=dtype)
        path = tmp_path / name
        image_utils.write_tiff(
            data=arr,
            file_loc=path,
            metadata=_save_meta(significant_bits),
            ome=ome,
            color='Green',
            significant_bits=significant_bits,
            save_encoding='8bit' if arr.dtype == np.uint8 else 'right_aligned',
        )
        return path

    def test_ome_right_aligned_12bit(self, tmp_path):
        """OME right-aligned 12-bit returns sig=12 with values stored raw."""
        from modules import image_utils

        path = self._write_tiff(tmp_path, 4095, significant_bits=12, ome=True)
        image, sig = image_utils.load_pixels(path)
        assert sig == 12
        assert int(image.max()) == 4095  # raw right-aligned, not left-justified

    def test_plain_private_tag_12bit(self, tmp_path):
        """Plain (non-OME) 12-bit recovers sig=12 from the private tag."""
        from modules import image_utils

        path = self._write_tiff(tmp_path, 4095, significant_bits=12, ome=False)
        image, sig = image_utils.load_pixels(path)
        assert sig == 12
        assert int(image.max()) == 4095

    def test_legacy_left_justified_reads_16(self, tmp_path):
        """A legacy left-justified file (SignificantBits=16) reads sig=16."""
        from modules import image_utils

        path = self._write_tiff(tmp_path, 4095 * 16, significant_bits=16, ome=True)
        image, sig = image_utils.load_pixels(path)
        assert sig == 16
        assert int(image.max()) == 65520

    def test_8bit_file_reads_8(self, tmp_path):
        """An 8-bit file reports sig=8 and preserves the uint8 container."""
        from modules import image_utils

        path = self._write_tiff(tmp_path, 255, significant_bits=8, ome=True, dtype=np.uint8)
        image, sig = image_utils.load_pixels(path)
        assert sig == 8
        assert image.dtype == np.uint8

    def test_non_tiff_png_uses_container_width(self, tmp_path):
        """A PNG carries no depth tag, so container width (8) is the depth."""
        import cv2

        from modules import image_utils

        arr = np.full((8, 8), 200, dtype=np.uint8)
        path = tmp_path / 'frame.png'
        cv2.imwrite(str(path), arr)
        image, sig = image_utils.load_pixels(path)
        assert sig == 8
        assert int(image.max()) == 200

    def test_returned_depth_scales_pixels_to_full_white(self, tmp_path):
        """The (pixels, depth) pair is self-consistent: feeding the returned
        depth back to convert_to_8bit maps full-payload to full 8-bit white."""
        from modules import image_utils

        path = self._write_tiff(tmp_path, 4095, significant_bits=12, ome=False)
        image, sig = image_utils.load_pixels(path)
        assert int(image_utils.convert_to_8bit(image, sig).max()) == 255

    def test_missing_file_raises(self, tmp_path):
        """A path that does not exist raises rather than returning a bare array."""
        from modules import image_utils

        with pytest.raises(FileNotFoundError):
            image_utils.load_pixels(tmp_path / 'does_not_exist.tif')

    def _write_single_channel_rgb(self, tmp_path):
        import tifffile as tf

        # A 3-channel frame with only one channel populated -- the shape a
        # false-color save produces and the shape the collapse rule targets.
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[..., 1] = 200  # green plane only
        path = tmp_path / 'single_channel_rgb.tiff'
        tf.imwrite(str(path), arr)
        return path

    def test_collapse_default_reduces_single_channel_rgb_to_mono(self, tmp_path):
        """The default collapses a single-populated-channel RGB to its mono
        plane so the mono-uniform consumers see one shape."""
        from modules import image_utils

        path = self._write_single_channel_rgb(tmp_path)
        image, _ = image_utils.load_pixels(path)
        assert image.ndim == 2

    def test_collapse_off_preserves_three_channel_shape(self, tmp_path):
        """Color-capable consumers (stitch / zproject / composite / stack) pass
        collapse_legacy_false_color=False to keep the raw RGB layout their own
        color handling expects -- collapsing it would turn a color stitch mono."""
        from modules import image_utils

        path = self._write_single_channel_rgb(tmp_path)
        image, _ = image_utils.load_pixels(path, collapse_legacy_false_color=False)
        assert image.shape == (8, 8, 3)

    @pytest.mark.parametrize('collapse', [True, False])
    def test_load_pixels_opens_tiff_once(self, tmp_path, monkeypatch, collapse):
        """load_pixels reads the pixels AND the depth from a single TiffFile
        open. Stitch / zproject / composite call it per tile, so a second open
        just for the depth tag would double the file opens of every run. The
        returned (pixels, sig) must stay byte-identical to the two-open form."""
        import tifffile as tf

        from modules import image_utils

        path = self._write_tiff(tmp_path, 4095, significant_bits=12, ome=False)

        # Baseline equal to the pre-refactor two-call form.
        expected_image = tf.imread(str(path))
        expected_sig = image_utils.read_tiff_significant_bits(path)

        real_tifffile = image_utils.tf.TiffFile
        opens = []

        def counting_tifffile(arg, *args, **kwargs):
            opens.append(arg)
            return real_tifffile(arg, *args, **kwargs)

        monkeypatch.setattr(image_utils.tf, 'TiffFile', counting_tifffile)
        image, sig = image_utils.load_pixels(path, collapse_legacy_false_color=collapse)

        assert len(opens) == 1
        assert sig == expected_sig
        assert np.array_equal(image, expected_image)


class TestWriteTiffDepthRequired:
    """A write that does not state its payload depth fails loudly instead of
    silently labeling a narrow payload as full container width."""

    def test_depth_less_write_raises(self, tmp_path):
        """write_tiff without significant_bits is a missing-argument error, not
        a default to 16. significant_bits is a required positional argument so
        a depth-less write cannot be expressed."""
        from modules import image_utils

        arr = np.full((8, 8), 4095, dtype=np.uint16)
        with pytest.raises(TypeError, match='significant_bits'):
            image_utils.write_tiff(
                data=arr,
                file_loc=tmp_path / 'frame.tif',
                metadata=_save_meta(),
                ome=True,
                color='Green',
            )

    def test_stated_depth_is_recorded(self, tmp_path):
        """A stated depth round-trips through the file, not the container width."""
        from modules import image_utils

        arr = np.full((8, 8), 4095, dtype=np.uint16)
        path = tmp_path / 'frame.tif'
        image_utils.write_tiff(
            data=arr,
            file_loc=path,
            metadata=_save_meta(),
            ome=True,
            color='Green',
            significant_bits=12,
            save_encoding='right_aligned',
        )
        assert image_utils.read_tiff_significant_bits(path) == 12


class TestConverterCollapse:
    """The depth-named converters delegate to the one significant-bits LUT, so
    there is a single canonical 8-bit mapping with no divergent per-depth tables."""

    def test_12bit_converter_matches_canonical(self):
        from modules import image_utils

        src = np.arange(4096, dtype=np.uint16).reshape(64, 64)
        assert np.array_equal(
            image_utils.convert_12bit_to_8bit(src),
            image_utils.convert_to_8bit(src, 12),
        )

    def test_16bit_converter_matches_canonical(self):
        from modules import image_utils

        src = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        assert np.array_equal(
            image_utils.convert_16bit_to_8bit(src),
            image_utils.convert_to_8bit(src, 16),
        )

    def test_no_standalone_depth_luts(self):
        """The divergent module-level tables are gone; the cache is the source."""
        from modules import image_utils

        assert not hasattr(image_utils, '_LUT_12_TO_8')
        assert not hasattr(image_utils, '_LUT_16_TO_8')

    def test_16to8_pins_deviation_from_legacy_truncation(self):
        """Pin the deliberate 16->8 map choice against the legacy >>8 (/256).

        The map is a linear rescale (value / 65535 * 255) then a truncating
        .astype(uint8), chosen over the old >>8. Both truncate and send full
        scale to 255, but they differ by at most 1 LSB at exactly 32640 of the
        65536 inputs because the divisor differs (65535 vs 65536), not because
        the rescale rounds. Locking the bound and the count documents this as a
        decision, so a future change to the map is caught rather than absorbed.
        """
        from modules import image_utils

        vals = np.arange(65536, dtype=np.uint16)
        new = image_utils.convert_to_8bit(vals.reshape(256, 256), 16).reshape(-1).astype(np.int32)
        legacy = (vals // 256).astype(np.uint8).astype(np.int32)  # the old >>8 form

        deviation = np.abs(new - legacy)
        assert deviation.max() <= 1  # never more than 1 LSB off the legacy map
        assert int((deviation != 0).sum()) == 32640  # the deliberate differing set
        assert new[0] == 0 and new[65535] == 255  # endpoints anchored


class TestVideoFrameCarriesDepth:
    """A video-frame TIFF carries its payload depth in the private tag. Video
    metadata is otherwise passed through untouched, so without this tag a
    right-aligned narrow frame reads back as full container width (~16x dark)."""

    def _write_video_frame(self, tmp_path, value, significant_bits):
        from modules import image_utils
        from modules.recording_frames import tiff_frame_metadata

        arr = np.full((8, 8), value, dtype=np.uint16)
        path = tmp_path / 'frame.tif'
        # The production builder both recording legs share, so this helper
        # cannot drift from the metadata shape the writer actually receives.
        metadata, _ = tiff_frame_metadata(
            timestamp_s=1781956800.0, frame_number=0, chunks=None, tick_freq_hz=None
        )
        image_utils.write_tiff(
            data=arr,
            file_loc=path,
            metadata=metadata,
            ome=False,
            color='BF',
            significant_bits=significant_bits,
            save_encoding='right_aligned',
            video_frame=True,
        )
        return path

    def test_12bit_video_frame_records_true_depth(self, tmp_path):
        from modules import image_utils

        path = self._write_video_frame(tmp_path, 4095, significant_bits=12)
        assert image_utils.read_tiff_significant_bits(path) == 12

    def test_video_frame_roundtrips_through_load_pixels(self, tmp_path):
        from modules import image_utils

        path = self._write_video_frame(tmp_path, 4095, significant_bits=12)
        _, sig = image_utils.load_pixels(path)
        assert sig == 12


class TestEncodeDisplayJpgDepth:
    """The JPEG export scales by the stated payload depth, so a summed 16-bit
    frame is not indexed against the 12-bit table."""

    def test_summed_16bit_frame_encodes_bright(self):
        import cv2

        from modules import image_utils

        frame = np.zeros((32, 32), dtype=np.uint16)
        frame[8:24, 8:24] = 60000  # near full 16-bit white
        jpg = image_utils.encode_display_jpg(frame, 'BF', significant_bits=16)
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert bgr[8:24, 8:24].mean() > 180  # bright, correctly scaled

    def test_wrong_depth_on_summed_frame_is_loud(self):
        from modules import image_utils

        # A summed value > 4095 declared as 12-bit violates the depth contract.
        # The explicit range check raises a typed FrameDepthError naming the
        # value and depth -- loud and independent of the downconvert arithmetic,
        # not the bare IndexError that LUT indexing happened to raise before, and
        # never a silent wrong image. Pinning it documents why the depth must be
        # stated correctly.
        frame = np.full((8, 8), 60000, dtype=np.uint16)
        with pytest.raises(FrameDepthError):
            image_utils.encode_display_jpg(frame, 'BF', significant_bits=12)


class TestCellCountConverterRouting:
    """Cell counting downconverts a 16-bit frame through the one canonical
    converter (significant_bits=16), not a separate 16->8 entry point."""

    def test_routes_16bit_through_convert_to_8bit(self):
        from unittest import mock

        from modules.cell_count import CellCount

        class _StopError(Exception):
            pass

        cc = CellCount()
        img = np.zeros((16, 16), dtype=np.uint16)

        with (
            mock.patch('modules.image_utils.convert_16bit_to_8bit') as legacy,
            mock.patch('modules.image_utils.convert_to_8bit', side_effect=_StopError) as canonical,
            pytest.raises(_StopError),
        ):
            cc.process_image(img, settings={}, significant_bits=16)

        canonical.assert_called_once()
        args, kwargs = canonical.call_args
        passed_sig = kwargs.get('significant_bits')
        if passed_sig is None and len(args) >= 2:
            passed_sig = args[1]
        assert passed_sig == 16
        legacy.assert_not_called()


class TestConvertTo8BitDepthGuard:
    """The downconvert fails loud + typed when a payload exceeds its declared depth.

    A value above ``(1 << significant_bits) - 1`` is a significant-bits contract
    violation -- the declared depth is wrong for the data. It must raise
    FrameDepthError carrying the value and the depth, not the IndexError that is
    an accident of LUT indexing and not a silent white-clip. Both the direct
    return and the reused-buffer (``out=``) preview path are covered, since the
    preview takes the ``out=`` path.
    """

    def test_over_range_value_raises_frame_depth_error(self):
        frame = np.full((4, 4), 5000, dtype=np.uint16)  # 5000 > 4095 = 12-bit max
        with pytest.raises(FrameDepthError) as exc:
            image_utils.convert_to_8bit(frame, 12)
        assert '5000' in str(exc.value)
        assert '12' in str(exc.value)
        assert exc.value.value == 5000
        assert exc.value.significant_bits == 12

    def test_over_range_value_raises_on_out_buffer_path(self):
        frame = np.full((4, 4), 1500, dtype=np.uint16)  # 1500 > 1023 = 10-bit max
        buf = np.empty((4, 4), dtype=np.uint8)
        with pytest.raises(FrameDepthError):
            image_utils.convert_to_8bit(frame, 10, out=buf)

    def test_boundary_value_converts_and_first_over_value_raises(self):
        # The exact edge the guard hinges on: the max in-range value (4095 for
        # 12-bit) is a valid LUT index and must convert to full scale, while the
        # first out-of-range value (4096) raises FrameDepthError.
        in_range = np.full((4, 4), 4095, dtype=np.uint16)
        result = image_utils.convert_to_8bit(in_range, 12)
        assert result.dtype == np.uint8
        assert int(result.max()) == 255
        over = np.full((4, 4), 4096, dtype=np.uint16)
        with pytest.raises(FrameDepthError):
            image_utils.convert_to_8bit(over, 12)

    def test_full_scale_value_maps_to_255_and_does_not_raise(self):
        frame = np.full((4, 4), 4095, dtype=np.uint16)  # exactly 12-bit max
        result = image_utils.convert_to_8bit(frame, 12)
        assert result.dtype == np.uint8
        assert np.all(result == 255)

    def test_sixteen_bit_container_never_raises(self):
        frame = np.full((4, 4), 65535, dtype=np.uint16)  # full uint16 under sig=16
        result = image_utils.convert_to_8bit(frame, 16)
        assert np.all(result == 255)
