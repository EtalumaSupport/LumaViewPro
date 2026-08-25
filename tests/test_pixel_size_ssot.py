# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: image scale resolves from real sources, with no hardcoded fallback.

Every Lumascope Classic (LS620) image carried a pixel size 9.1% too small
because scope_capabilities fell back to a hardcoded 2.0 um instead of the 2.2
the scope's optics declare. These pin the corrected resolution order
(motorconfig -> scopes.json -> camera profile -> None), the measured LS620
value, the LS850T no-op, and the honest-None degradation on the save path.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import modules.common_utils as common_utils
import modules.image_utils as image_utils
from drivers.motorconfig import MotorConfig
from modules.scope_capabilities import (
    _resolve_lens_focal_length_mm,
    _resolve_pixel_size_um,
)


def _camera_with_pixel_size(pixel_size_um):
    """Minimal camera exposing profile.pixel_size_um, matching what the resolver
    reads (getattr(camera, 'profile').pixel_size_um)."""
    return SimpleNamespace(profile=SimpleNamespace(pixel_size_um=pixel_size_um))


def _motorconfig_with_optics(pixel_size, lens_focal_length):
    mc = MotorConfig.__new__(MotorConfig)
    mc._config = {'Optics': {'PixelSize': pixel_size, 'LensFocalLength': lens_focal_length}}
    mc._defaults = {}
    return mc


class TestResolutionOrder:
    def test_classic_ls620_resolves_from_scopes_json(self):
        # No motorconfig (NullMotionBoard); scopes.json declares the LS620's
        # optics. This is the bug site: the old code fell back to 2.0 here
        # instead of consulting the 2.2 the scope actually has.
        cam = _camera_with_pixel_size(2.2)
        assert _resolve_pixel_size_um(None, 'LS620', cam) == 2.2
        assert _resolve_lens_focal_length_mm(None, 'LS620') == 47.8

    def test_ls850t_motorconfig_wins_and_is_a_no_op(self):
        # A scope WITH a motorconfig sources optics from it; the fix must not
        # move the LS850T's 2.0.
        mc = _motorconfig_with_optics(2.0, 47.8)
        assert _resolve_pixel_size_um(mc, 'LS850T', None) == 2.0
        assert _resolve_lens_focal_length_mm(mc, 'LS850T') == 47.8

    def test_motorconfig_optics_beats_scopes_json(self):
        # motorconfig is first in the order; a model that ALSO has a scopes.json
        # entry (LS620 -> 2.2) still takes the motorconfig value.
        mc = _motorconfig_with_optics(2.0, 47.8)
        assert _resolve_pixel_size_um(mc, 'LS620', None) == 2.0

    def test_camera_profile_fills_when_no_config_or_scopes_entry(self):
        # An unrecognized model with no motorconfig and no scopes.json entry
        # falls through to the camera's SDK-reported pitch.
        cam = _camera_with_pixel_size(2.19)
        assert _resolve_pixel_size_um(None, 'NoSuchScope', cam) == 2.19

    def test_unknown_scope_and_camera_yields_none(self):
        # Nothing can report a scale: stay None, never a guess. A generic camera
        # profile carries 0.0 until the SDK fills it, which must not count.
        cam = _camera_with_pixel_size(0.0)
        assert _resolve_pixel_size_um(None, 'NoSuchScope', cam) is None
        assert _resolve_lens_focal_length_mm(None, 'NoSuchScope') is None


class TestEffectivePixelSize:
    @staticmethod
    def _install_scale(monkeypatch, pixel_size_um, lens_focal_length_mm):
        import modules.app_context as app_context

        scope = SimpleNamespace(
            capabilities=SimpleNamespace(
                pixel_size_um=pixel_size_um, lens_focal_length_mm=lens_focal_length_mm
            )
        )
        monkeypatch.setattr(app_context, 'ctx', app_context.AppContext(scope=scope))

    def test_ls620_effective_pixel_size_at_20x(self, monkeypatch):
        # The bench-measured value: 2.2 / (47.8 / 9.0) = 0.41423 at 20x
        # (focal_length 9.0). The old 2.0 fallback gave 0.37657.
        self._install_scale(monkeypatch, 2.2, 47.8)
        assert common_utils.get_pixel_size(focal_length=9.0, binning_size=1) == pytest.approx(
            0.41423, abs=1e-4
        )

    def test_ls850t_effective_pixel_size_unchanged_at_20x(self, monkeypatch):
        self._install_scale(monkeypatch, 2.0, 47.8)
        assert common_utils.get_pixel_size(focal_length=9.0, binning_size=1) == pytest.approx(
            0.37657, abs=1e-4
        )

    def test_no_scale_returns_none_not_a_guess(self, monkeypatch):
        self._install_scale(monkeypatch, None, None)
        assert common_utils.get_pixel_size(focal_length=9.0, binning_size=1) is None
        assert (
            common_utils.get_field_of_view(
                focal_length=9.0, frame_size={'width': 100, 'height': 100}, binning_size=1
            )
            is None
        )


class TestVideoFrameScaleClaim:
    """A video frame carries no scale, so it must not claim one.

    The video-frame arm of generate_tiff_data returned before pixel_size_um was
    ever read and hardcoded ResolutionUnit CENTIMETER with no resolution value,
    so tifffile's default 1/1 made every frame of every recording assert
    1 px/cm = 10,000 um/px against a true ~2 um/px.
    """

    def test_video_frame_makes_no_absolute_unit_scale_claim(self, tmp_path):
        # Through the real path: the production metadata builder both recording
        # legs share, then the single canonical frame-save edge.
        import tifffile as tf

        from modules.image_save import write_video_frame
        from modules.recording_frames import tiff_frame_metadata

        metadata, _ts_filename = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=None,
        )
        path = tmp_path / 'ManualVideo_Frame_0000.tiff'

        write_video_frame(
            frame=np.zeros((32, 32), dtype=np.uint16),
            file_loc=path,
            metadata=metadata,
            channel='BF',
            false_color_on=False,
            save_encoding='right_aligned',
            capture_depth=12,
        )

        with tf.TiffFile(path) as t:
            resunit = t.pages[0].tags['ResolutionUnit'].value
        # NONE is the TIFF convention for "ratio only, no absolute unit", which
        # is what an unknown scale means. Do NOT assert XResolution is absent:
        # tifffile emits it as 1/1 regardless, and under NONE that reads as
        # square aspect with no absolute dimensions, not as a measurement.
        assert int(resunit) == int(tf.RESUNIT.NONE)

    def test_tiff_frame_metadata_carries_the_pixel_size_key(self):
        # The key is always present so a scale-less frame is expressible only as
        # an explicit None. The writer reads it as a hard key, so a builder that
        # forgot it fails loudly at the write instead of silently claiming 1/1.
        from modules.recording_frames import tiff_frame_metadata

        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=None,
        )

        assert 'pixel_size_um' in metadata
        assert metadata['pixel_size_um'] is None


class TestVideoFrameCarriesScale:
    """A video frame carries the scale measured at recording start.

    The two recording legs build metadata through one builder that had no scale
    input at all, so recordings were the only capture path writing files no one
    could measure. The value is snapshotted once per recording, not per frame:
    changing the objective mid-recording is not supported, and a per-frame
    resolve would let one stack hold frames that disagree.
    """

    def test_tiff_frame_metadata_requires_a_scale_argument(self):
        # Required, not defaulted: a default would let a future third builder
        # silently omit scale, which is the whole defect being closed here.
        from modules.recording_frames import tiff_frame_metadata

        with pytest.raises(TypeError):
            tiff_frame_metadata(
                timestamp_s=1755000000.0, frame_number=0, chunks=None, tick_freq_hz=None
            )

    def test_tiff_frame_metadata_carries_the_measured_scale(self):
        from modules.recording_frames import tiff_frame_metadata

        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=2.2,
        )

        assert metadata['pixel_size_um'] == 2.2

    def test_video_frame_written_with_a_scale_claims_it(self, tmp_path):
        import tifffile as tf

        from modules.image_save import write_video_frame
        from modules.recording_frames import tiff_frame_metadata

        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=2.2,
        )
        path = tmp_path / 'ManualVideo_Frame_0000.tiff'
        write_video_frame(
            frame=np.zeros((32, 32), dtype=np.uint16),
            file_loc=path,
            metadata=metadata,
            channel='BF',
            false_color_on=False,
            save_encoding='right_aligned',
            capture_depth=12,
        )

        with tf.TiffFile(path) as t:
            tags = t.pages[0].tags
            assert int(tags['ResolutionUnit'].value) == int(tf.RESUNIT.CENTIMETER)
            num, den = tags['XResolution'].value
        # The tag is pixels-per-centimetre, so inverting it recovers um/px.
        assert 1e4 / (num / den) == pytest.approx(2.2)

    def test_scale_less_video_frame_still_declares_none(self, tmp_path):
        # The honest-omission contract must survive the plumbing.
        import tifffile as tf

        from modules.image_save import write_video_frame
        from modules.recording_frames import tiff_frame_metadata

        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=None,
        )
        path = tmp_path / 'ManualVideo_Frame_0001.tiff'
        write_video_frame(
            frame=np.zeros((32, 32), dtype=np.uint16),
            file_loc=path,
            metadata=metadata,
            channel='BF',
            false_color_on=False,
            save_encoding='right_aligned',
            capture_depth=12,
        )

        with tf.TiffFile(path) as t:
            assert int(t.pages[0].tags['ResolutionUnit'].value) == int(tf.RESUNIT.NONE)


class TestReadPixelSizeUm:
    """One narrow reader for the one fact the hyperstack builder needs.

    StackBuilder used the full acquisition-context reader for a single field,
    and that reader returns None for any file with no OME Plane block -- which
    is every video frame. Reading the scale is its own question.
    """

    def _video_frame(self, tmp_path, pixel_size_um):
        from modules.image_save import write_video_frame
        from modules.recording_frames import tiff_frame_metadata

        metadata, _ = tiff_frame_metadata(
            timestamp_s=1755000000.0,
            frame_number=0,
            chunks=None,
            tick_freq_hz=None,
            pixel_size_um=pixel_size_um,
        )
        path = tmp_path / f'ManualVideo_Frame_{pixel_size_um}.tiff'
        write_video_frame(
            frame=np.zeros((32, 32), dtype=np.uint16),
            file_loc=path,
            metadata=metadata,
            channel='BF',
            false_color_on=False,
            save_encoding='right_aligned',
            capture_depth=12,
        )
        return path

    def test_recovers_scale_from_a_video_frame(self, tmp_path):
        path = self._video_frame(tmp_path, 2.2)

        assert image_utils.read_pixel_size_um(path) == pytest.approx(2.2)

    def test_returns_none_for_a_scale_less_video_frame(self, tmp_path):
        path = self._video_frame(tmp_path, None)

        assert image_utils.read_pixel_size_um(path) is None

    def test_recovers_scale_from_a_still(self, tmp_path):
        metadata = {
            'pixel_size_um': 0.41423,
            'channel': 'Green',
            'significant_bits': 8,
            'objective': '20x',
            'exposure_time_ms': 50.0,
            'gain_db': 0.0,
            'illumination_ma': 100.0,
            'z_pos_um': 1000.0,
            'plate_pos_mm': {'x': 10.0, 'y': 20.0},
            'datetime': '2026:03:16 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': 'A1',
            'well_site': '1',
        }
        path = tmp_path / 'still.tif'
        image_utils.write_tiff(
            data=np.zeros((32, 32), dtype=np.uint8),
            file_loc=path,
            metadata=metadata,
            ome=False,
            color='Green',
            significant_bits=8,
            save_encoding='8bit',
        )

        assert image_utils.read_pixel_size_um(path) == pytest.approx(0.41423)


class TestSaveWithoutScale:
    def test_save_with_none_pixel_size_omits_scale_and_does_not_raise(self, tmp_path):
        # Previously a save with no scale hit resolution_for_pixel_size(0.0) and
        # raised ZeroDivisionError. Now the writer omits the resolution tag: the
        # file makes no scale claim (ResolutionUnit NONE) rather than crashing.
        import tifffile as tf

        metadata = {
            'pixel_size_um': None,
            'channel': 'Green',
            'significant_bits': 8,
            'objective': '10x',
            'exposure_time_ms': 50.0,
            'gain_db': 0.0,
            'illumination_ma': 100.0,
            'z_pos_um': 1000.0,
            'plate_pos_mm': {'x': 10.0, 'y': 20.0},
            'datetime': '2026:03:16 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': 'A1',
            'well_site': '1',
        }
        path = tmp_path / 'no_scale.tif'

        image_utils.write_tiff(
            data=np.zeros((32, 32), dtype=np.uint8),
            file_loc=path,
            metadata=metadata,
            ome=False,
            color='Green',
            significant_bits=8,
            save_encoding='8bit',
        )

        assert path.exists()
        with tf.TiffFile(path) as t:
            resunit = t.pages[0].tags['ResolutionUnit'].value
        assert int(resunit) == int(tf.RESUNIT.NONE)
