"""Phase 1 mono-native regression matrix -- xfail-strict comprehensive
instrument (Rule 44).

Each test asserts the POST-mono-native shape / behavior of the save and
encode pipelines. Today's pipeline writes 3-channel false-color replicas
into fluorescence TIFFs; the assertions below describe what the pipeline
should produce after Phase 1d (atomic mono-native save migration). All
tests carry ``xfail(strict=True)`` -- they FAIL today and FLIP GREEN
when the 1d migration lands. ``strict=True`` catches a green test that
was never un-xfailed (regression-of-the-regression).

Distilled from the F0.5e historical-bug walk (8 commits, May 2025 - May
2026). The 4 MUST-HAVE tests catch >90% of the historical production
risk (the #657 fluorescence-channel-swap cluster); the 4 NICE-TO-HAVE
tests harden adjacent surfaces.

Pipeline coverage:
- TIFF save (must, R/G nice-to-have)
- Composite RGB intermediate read (must)
- MP4 encode through PyAV (must)
- cv2.VideoWriter BGR boundary (must)
- OME-TIFF axes (nice-to-have)
- Buffer-allocation O(1) reuse (nice-to-have)
- PNG composite output (nice-to-have)
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import tifffile as tf
from modules.layer_record import release_color
from modules.layer_record import release_colors
from tests.stock_channels import LED0, LED1, LED2


XFAIL_REASON = (
    'Phase 1 mono-native regression matrix -- flips green at 1d when the '
    'save pipeline writes 2D mono + layer metadata in place of 3-channel '
    'false-color replicas. xfail(strict=True) catches a stale marker.'
)


def _metadata(path, channel=LED0):
    """Minimal metadata dict matching ``write_tiff``'s generate_tiff_data."""
    return {
        'file_loc': str(path),
        'datetime': datetime.datetime(2026, 5, 27, 0, 0, 0),
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'z_pos_um': 0.0,
        'objective': {'id': 'test'},
        'exposure_time_ms': 1.0,
        'gain_db': 0.0,
        'illumination_ma': 0.0,
        'pixel_size_um': 1.0,
        'channel': channel,
    }


# ---------------------------------------------------------------------------
# MUST-HAVE 1: pure-blue 16-bit false-color TIFF round-trip
# ---------------------------------------------------------------------------


def test_pure_blue_16bit_falsecolor_tiff_roundtrip(tmp_path):
    """Synth uint16 LED 0 layer mono; save with the false-color opt-in ON;
    read back.

    Default OFF saves 2D mono + metadata (covered by test_tiff_format's
    default-off cases). With ``use_false_color_16bit=True`` the layer color
    is baked into 3-channel RGB so Windows Preview renders 16-bit
    fluorescence in color. The LED 0 layer (blue) must land in RGB index 2 with the
    value preserved exactly and the other planes zero (the R/B-swap and
    wrong-channel-on-video defects guard against an ordering regression).
    """
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'blue.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel=LED0),
        ome=False,
        color=release_color(LED0),
        significant_bits=16,
        save_encoding='rgb',
        white_bits=None,
    )

    result = tf.imread(str(out_path))

    assert result.shape == (8, 8, 3), (
        f'false_color_16bit=True must widen to 3-channel RGB, got {result.shape}.'
    )
    assert result.dtype == np.uint16
    assert result[0, 0, 2] == 42000, f'{LED0} lands at RGB index 2'
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == 0


# ---------------------------------------------------------------------------
# MUST-HAVE 2: pure-blue MP4 build round-trip
# ---------------------------------------------------------------------------


def test_pure_blue_mp4_roundtrip(tmp_path):
    """Synth a 1-frame mono TIFF input; run VideoBuilder; decode the
    first MP4 frame via PyAV; assert the Blue channel carries the data.

    VideoBuilder reads the mono input (no cv2 detour -- the failure
    surface that produced #657's wrong-channel decode), applies
    false-color via the boundary helper, then encodes RGB to MP4. The
    test is structural: the decoded frame's Blue channel must carry the
    source mono value at its correct linear scaling, and the other
    channels must stay dark, regardless of internal pipeline shape.
    """
    pytest.importorskip('av')
    import av

    from modules.video_builder import VideoBuilder

    # One mono input TIFF (post-1d input shape).
    input_dir = tmp_path / 'tiff_in'
    input_dir.mkdir()
    src = np.full((8, 8), 42000, dtype=np.uint16)
    tf.imwrite(str(input_dir / f'frame_0000_{LED0}.tiff'), src)

    out_path = tmp_path / 'out.mp4'

    # Direct invocation through VideoBuilder's frame-processing surface.
    # Post-1d API: VideoBuilder reads mono inputs + applies false-color
    # via boundary helper before encode.
    builder = VideoBuilder(has_turret=False)
    builder.build_video(
        source_dir=input_dir,
        output_file=out_path,
        false_color=True,
        color=LED0,
        channel_colors=release_colors(),
    )

    # Decode first frame via PyAV
    container = av.open(str(out_path))
    frame = next(container.decode(video=0))
    arr = frame.to_ndarray(format='rgb24')
    container.close()

    # The synthetic TIFF carries no significant-bits tag, so the loader
    # correctly treats it as full 16-bit: 42000/65535 scales to 163 in
    # uint8, and yuv420 encode rounding costs ~1 count either way. A
    # value outside this window means the pipeline mis-scaled (e.g.
    # treated the payload as 12-bit, the reads-near-black defect class).
    assert 155 <= arr[0, 0, 2] <= 170, (
        f'Decoded Blue channel must carry the source value at its 16-bit '
        f'linear scaling (42000/65535 -> ~163); got arr[0,0]={arr[0, 0]}.'
    )
    assert arr[0, 0, 0] < 20  # Red close to zero
    assert arr[0, 0, 1] < 20  # Green close to zero


# ---------------------------------------------------------------------------
# MUST-HAVE 3: composite RGB all-channels round-trip
# ---------------------------------------------------------------------------

_RUN_COLUMNS = [
    'Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination',
    'Gain', 'Auto_Gain', 'Exposure', 'Sum', 'Objective', 'Well', 'Tile', 'Z-Slice',
    'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config',
    'Stim_Config', 'Auto_Named', 'Label',
]  # fmt: skip
_CHANNEL_VALUES = {LED2: 200, LED1: 140, LED0: 80}


def _rgb_step(index: int, color: str) -> list[str]:
    return [
        f'custom000{index}_{color}', '10.0000', '10.0000', '5000.0', 'False', color,
        'True', '5.0', '1.0', 'False', '2.0', '1', '20x Oly', '', 'A1', '-1', 'True',
        str(index), '-1', 'image', '"{""duration"": 5, ""fps"": 30}"', '"{}"', 'True', '',
    ]  # fmt: skip


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    yield session
    session.shutdown()


@pytest.fixture
def rgb_run_folder(tmp_path):
    return _write_rgb_run_folder(tmp_path, ome=False)


def _write_rgb_run_folder(tmp_path, *, ome: bool):
    """One position imaged in LED 2, LED 1 and LED 0 (red, green, blue), each a flat 8-bit frame
    of its own value, as a run of this release saves it: OME-TIFF when ``ome``, else plain."""
    from modules import image_utils

    folder = tmp_path / 'live' / 'ProtocolData' / '20261013_120000'
    folder.mkdir(parents=True)
    steps = [_rgb_step(index, color) for index, color in enumerate(_CHANNEL_VALUES)]
    (folder / 'run.tsv').write_text(
        'LumaViewPro Protocol\nVersion\t8\nPeriod\t20.0\nDuration\t48.0\n'
        'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
        + '\t'.join(_RUN_COLUMNS)
        + '\n'
        + ''.join('\t'.join(step) + '\n' for step in steps)
    )
    record_rows = []
    for index, step in enumerate(steps):
        name, color = step[0], step[5]
        filename = f'{name}_0000{".ome.tiff" if ome else ".tiff"}'
        image_utils.write_tiff(
            data=np.full((16, 16), _CHANNEL_VALUES[color], dtype=np.uint8),
            file_loc=folder / filename,
            metadata={
                'pixel_size_um': 0.5,
                'channel': color,
                'objective': {'id': '20x Oly'},
                'exposure_time_ms': 2.0,
                'gain_db': 1.0,
                'illumination_ma': 5.0,
                'z_pos_um': 5000.0,
                'plate_pos_mm': {'x': 10.0, 'y': 10.0},
                'datetime': datetime.datetime(2026, 10, 13, 12, 0, 0),
                'well_label': '',
                'significant_bits': 8,
            },
            ome=ome,
            color=release_color(color),
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )
        record_rows.append(
            f'{filename}\t{name}\t{index}\t0\t2026-10-13 12:00:{index:02d}.000001\t1\t0.0\n'
        )
    (folder / 'protocol_record.tsv').write_text(
        'LumaViewPro Protocol Execution Record\nVersion\t3\nProtocol File\trun.tsv\n'
        'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
        + ''.join(record_rows)
    )
    return folder


def test_a_composite_from_a_folder_keeps_each_channel_on_its_own_plane(session, rgb_run_folder):
    """An LED 2, LED 1 and LED 0 position composited through the Session's
    post-processing reads back as one 8-bit RGB image with each channel on
    its own plane (R 0, G 1, B 2): the routing the #657 / #672 swaps broke.

    Historical bugs: ``cefdfcb`` (composite OME-TIFF axes mismatch),
    ``e2ef49e`` (#657 frames swap on save).
    """
    built = session.post_processing.composite(rgb_run_folder)

    assert built.new_count == 1
    result = tf.imread(str(built.artifact_paths[0]))
    assert result.dtype == np.uint8
    assert result.shape == (16, 16, 3)
    assert result[0, 0].tolist() == list(_CHANNEL_VALUES.values())


# ---------------------------------------------------------------------------
# NICE-TO-HAVE 5+6: pure-red + pure-green 16-bit false-color TIFF variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('color', [LED2, LED1])
def test_pure_color_16bit_falsecolor_tiff_roundtrip(tmp_path, color):
    """Per-color hardening for the false-color opt-in: confirm the LED 2 (red) and
    LED 1 (green) layer paths widen to 3-channel RGB with the value in the correct
    plane. Catches asymmetries in the layer-color lookup (a historical
    Red-specific R/B swap motivates the parametrize).
    """
    from modules.image_utils import write_tiff

    out_path = tmp_path / f'{color.lower()}.tiff'
    data = np.full((8, 8), 30000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel=color),
        ome=False,
        color=release_color(color),
        significant_bits=16,
        save_encoding='rgb',
        white_bits=None,
    )

    result = tf.imread(str(out_path))
    assert result.shape == (8, 8, 3), f'{color}: expected 3-channel RGB, got {result.shape}'
    expected_index = {LED2: 0, LED1: 1}[color]
    assert result[0, 0, expected_index] == 30000
    for other in (0, 1, 2):
        if other != expected_index:
            assert result[0, 0, other] == 0, f'{color}: plane {other} should be zero'


# ---------------------------------------------------------------------------
# NICE-TO-HAVE 7: composite OME-TIFF axes structural check
# ---------------------------------------------------------------------------


def test_a_composite_from_a_folder_in_ome_tiff_has_yxs_axes(session, tmp_path):
    """A composite of an OME-TIFF run's channels carries OME metadata and
    axes 'YXS' (Y, X, Sample) for its 3-channel image. ``cefdfcb`` was a
    crash on that axes mismatch.
    """
    built = session.post_processing.composite(_write_rgb_run_folder(tmp_path, ome=True))

    assert built.new_count == 1
    with tf.TiffFile(str(built.artifact_paths[0])) as t:
        assert t.ome_metadata is not None, 'composite OME-TIFF missing OME metadata block'
        assert t.series[0].axes == 'YXS'


def test_composite_sequenced_path_writes_the_format_its_name_states(tmp_path):
    """The sequenced composite path (_create_composite_image) writes the
    format its output's name states -- ``.ome.tiff`` writes OME, ``.tiff``
    plain. It once hardcoded ome=False.
    """
    import pandas as pd
    from modules.composite_generation import CompositeGeneration

    for ch in (LED2, LED1, LED0):
        tf.imwrite(str(tmp_path / f'{ch}.tiff'), np.full((8, 8), 20000, dtype=np.uint16))
    df = pd.DataFrame(
        [
            {'Color': LED2, 'Filepath': f'{LED2}.tiff'},
            {'Color': LED1, 'Filepath': f'{LED1}.tiff'},
            {'Color': LED0, 'Filepath': f'{LED0}.tiff'},
        ]
    )

    ome_out = tmp_path / 'comp.ome.tiff'
    CompositeGeneration._create_composite_image(
        path=tmp_path,
        df=df,
        brightness_thresholds_percent={},
        output_file_loc=ome_out,
        channel_colors=release_colors(),
    )
    with tf.TiffFile(str(ome_out)) as t:
        assert t.ome_metadata is not None, 'OME-TIFF format must produce OME metadata'

    plain_out = tmp_path / 'comp_plain.tiff'
    CompositeGeneration._create_composite_image(
        path=tmp_path,
        df=df,
        brightness_thresholds_percent={},
        output_file_loc=plain_out,
        channel_colors=release_colors(),
    )
    with tf.TiffFile(str(plain_out)) as t:
        assert t.ome_metadata is None, 'TIFF format must not produce OME metadata'


# ---------------------------------------------------------------------------
# NICE-TO-HAVE 8: PIW-6 buffer-allocation O(1) regression
# ---------------------------------------------------------------------------


def test_piw6_buffer_allocation_o1(tmp_path):
    """100 sequential ``write_tiff`` calls on the false-color widen path
    with a caller-supplied ``false_color_buf`` must allocate the
    3-channel false-color buffer ONCE, not 100 times -- a caller-supplied
    scratch buffer is reused across all saves; the O(1) reuse property
    must hold.
    """
    from modules.image_utils import write_tiff

    data = np.full((64, 64), 42000, dtype=np.uint16)
    # The widen path produces (H, W, 3); the caller-supplied scratch must
    # match that shape so add_false_color reuses it instead of allocating
    # a fresh 3-channel buffer on every call.
    scratch = np.zeros((64, 64, 3), dtype=data.dtype)

    # Track frame-sized allocations via np.zeros (add_false_color's fallback
    # when output shape mismatch) and np.empty (other internals). Filter to
    # frame-sized to avoid counting tifffile's many tiny scratch buffers.
    FRAME_SIZE = 64 * 64
    frame_alloc_count = {'n': 0}
    orig_zeros = np.zeros
    orig_empty = np.empty

    def _size_of(shape_arg):
        try:
            if hasattr(shape_arg, '__iter__'):
                n = 1
                for d in shape_arg:
                    n *= int(d)
                return n
            return int(shape_arg)
        except Exception:
            return 0

    def counting_zeros(shape, *args, **kwargs):
        if _size_of(shape) >= FRAME_SIZE:
            frame_alloc_count['n'] += 1
        return orig_zeros(shape, *args, **kwargs)

    def counting_empty(shape, *args, **kwargs):
        if _size_of(shape) >= FRAME_SIZE:
            frame_alloc_count['n'] += 1
        return orig_empty(shape, *args, **kwargs)

    with (
        patch('numpy.zeros', side_effect=counting_zeros),
        patch('numpy.empty', side_effect=counting_empty),
    ):
        for i in range(100):
            write_tiff(
                data=data,
                file_loc=tmp_path / f'frame_{i:04d}.tiff',
                metadata=_metadata(tmp_path / f'frame_{i:04d}.tiff', channel=LED0),
                ome=False,
                color=release_color(LED0),
                false_color_buf=scratch,
                significant_bits=16,
                save_encoding='rgb',
                white_bits=None,
            )

    # O(1) frame-sized allocations: the (H, W, 3) scratch fits the widen
    # output, so add_false_color reuses it across all 100 saves. A
    # mismatched scratch would fall back to np.zeros((H, W, 3), ...) on
    # every call -> O(n) = 100.
    assert frame_alloc_count['n'] < 10, (
        f'Expected O(1) frame-sized allocations across 100 saves; got '
        f'{frame_alloc_count["n"]}. A mismatched scratch buffer forces a '
        f'fresh 3-channel allocation per call.'
    )
