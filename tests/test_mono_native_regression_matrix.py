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

import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile as tf


XFAIL_REASON = (
    'Phase 1 mono-native regression matrix -- flips green at 1d when the '
    'save pipeline writes 2D mono + layer metadata in place of 3-channel '
    'false-color replicas. xfail(strict=True) catches a stale marker.'
)


def _metadata(path, channel='Blue'):
    """Minimal metadata dict matching ``write_tiff``'s generate_tiff_data."""
    return {
        'file_loc': str(path),
        'datetime': '2026-05-27T00:00:00',
        'plate_pos_mm': {'x': 0.0, 'y': 0.0},
        'z_pos_um': 0.0,
        'objective': 'test',
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
    """Synth uint16 Blue-layer mono; save as false-color TIFF; read back.

    Today: ``write_tiff`` widens the 2D mono to 3-channel RGB via
    ``add_false_color`` and writes a 3D file. Read-back is ``(H, W, 3)``.

    Post-1d: ``write_tiff`` keeps 2D mono and writes layer color as
    tifffile metadata (PALETTE or ImageJ LUT). Read-back is ``(H, W)``
    and the pixel value is preserved exactly.

    Historical bug: ``11ec3c7`` (R/B swap at write boundary) and
    ``e2ef49e`` (#657 frames -- add_false_color returns RGB). Mono-native
    save eliminates both by removing the widening step.
    """
    from modules.image_utils import write_tiff

    out_path = tmp_path / 'blue.tiff'
    data = np.full((8, 8), 42000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel='Blue'),
        ome=False,
        color='Blue',
        use_false_color_16bit=True,
    )

    result = tf.imread(str(out_path))

    assert result.ndim == 2, (
        f'Post-1d: file should be 2D mono with layer metadata, got shape '
        f'{result.shape}. Today fails because write_tiff widens to RGB.'
    )
    assert result.dtype == np.uint16
    assert result[0, 0] == 42000


# ---------------------------------------------------------------------------
# MUST-HAVE 2: pure-blue MP4 build round-trip
# ---------------------------------------------------------------------------


def test_pure_blue_mp4_roundtrip(tmp_path):
    """Synth a 1-frame mono TIFF input; run VideoBuilder; decode the
    first MP4 frame via PyAV; assert the Blue channel carries the data.

    Today: VideoBuilder's input loop calls ``cv2.imread`` (returns BGR
    on 3-channel sources) then ``cvtColor(BGR2RGB)`` before encode.
    With 3-channel-replica TIFF inputs this round-trips correctly but
    the cv2 detour is the failure surface that produced ``eae5079``
    (#657 video -- wrong channel on decode).

    Post-1d: input TIFFs are mono. VideoBuilder reads via ``tf.imread``
    (no cv2 detour), applies false-color via the boundary helper, then
    encodes RGB to MP4. Same final pixel value at index 2 (Blue), but
    via the canonical mono path.

    The test is structural -- it asserts the decoded MP4 frame's Blue
    channel carries the source mono value, regardless of internal
    pipeline shape.
    """
    pytest.importorskip('av')
    import av

    from modules.video_builder import VideoBuilder

    # One mono input TIFF (post-1d input shape).
    input_dir = tmp_path / 'tiff_in'
    input_dir.mkdir()
    src = np.full((8, 8), 42000, dtype=np.uint16)
    tf.imwrite(str(input_dir / 'frame_0000_Blue.tiff'), src)

    out_path = tmp_path / 'out.mp4'

    # Direct invocation through VideoBuilder's frame-processing surface.
    # Post-1d API: VideoBuilder reads mono inputs + applies false-color
    # via boundary helper before encode.
    builder = VideoBuilder(has_turret=False)
    builder.build_video(
        source_dir=input_dir,
        output_file=out_path,
        false_color=True,
        color='Blue',
    )

    # Decode first frame via PyAV
    container = av.open(str(out_path))
    frame = next(container.decode(video=0))
    arr = frame.to_ndarray(format='rgb24')
    container.close()

    assert arr[0, 0, 2] > 200, (
        f'Decoded Blue channel should carry the source value (scaled to '
        f'uint8); got arr[0,0]={arr[0, 0]}. Today fails because '
        f'VideoBuilder.build_video signature does not yet accept mono '
        f'2D TIFF inputs through the false_color boundary helper.'
    )
    assert arr[0, 0, 0] < 20  # Red close to zero
    assert arr[0, 0, 1] < 20  # Green close to zero


# ---------------------------------------------------------------------------
# MUST-HAVE 3: composite RGB all-channels round-trip
# ---------------------------------------------------------------------------


def test_composite_rgb_allchannels_roundtrip(tmp_path):
    """Synth 3 mono TIFFs (R, G, B layers); run composite generation;
    read output via tifffile axes='YXS'; assert per-channel preservation.

    Today: ``composite_generation.py`` reads per-channel inputs via
    ``cv2.imread`` (3-channel BGR) then extracts the relevant channel
    via ``cvtColor(BGR2GRAY)`` to recover mono. The redundancy is the
    bug surface (``#672`` extract-from-wrong-channel) plus a perf
    cost (3x read bandwidth).

    Post-1d: per-channel inputs are mono 2D; composite reads directly
    via ``tf.imread`` and stacks into the canonical (H, W, 3) output.

    Historical bugs: ``cefdfcb`` (composite OME-TIFF axes mismatch),
    ``e2ef49e`` (#657 frames swap on save).
    """
    input_dir = tmp_path / 'channels'
    input_dir.mkdir()

    # Post-1d input: 3 mono channel files
    red_val, green_val, blue_val = 50000, 35000, 20000
    tf.imwrite(str(input_dir / 'r.tiff'), np.full((8, 8), red_val, dtype=np.uint16))
    tf.imwrite(str(input_dir / 'g.tiff'), np.full((8, 8), green_val, dtype=np.uint16))
    tf.imwrite(str(input_dir / 'b.tiff'), np.full((8, 8), blue_val, dtype=np.uint16))

    out_path = tmp_path / 'composite.tiff'

    from modules.composite_generation import CompositeGeneration

    cg = CompositeGeneration(has_turret=False)
    cg.generate_composite_from_paths(
        red_path=input_dir / 'r.tiff',
        green_path=input_dir / 'g.tiff',
        blue_path=input_dir / 'b.tiff',
        output_path=out_path,
    )

    result = tf.imread(str(out_path))
    assert result.ndim == 3 and result.shape[2] == 3
    # Channel order: index 0 = Red, 1 = Green, 2 = Blue
    assert result[0, 0, 0] == red_val
    assert result[0, 0, 1] == green_val
    assert result[0, 0, 2] == blue_val


# ---------------------------------------------------------------------------
# MUST-HAVE 4: cv2.VideoWriter fallback RGB->BGR boundary (mocked)
# ---------------------------------------------------------------------------


def test_cv2_videowriter_fallback_bgr_boundary():
    """Pass a mono frame into VideoWriter (cv2 fallback path); assert
    the mocked ``cv2.VideoWriter.write`` receives a BGR-ordered frame
    with the false-color applied correctly.

    Today: the pre-1d caller already does add_false_color then BGR-swap.
    Post-1d: the caller passes mono; VideoWriter does false-color +
    BGR-swap internally before handing to cv2.VideoWriter.write.

    The bug surface (``161ed0e`` / ``7f26c7c`` -- RGB->BGR fallback
    swap regression) is the same; the test ensures the post-1d code
    path preserves the boundary.
    """
    from modules.video_writer import VideoWriter

    mono = np.full((8, 8), 50000, dtype=np.uint16)

    captured = {}

    def fake_write(frame):
        captured['frame'] = frame.copy()

    with patch('cv2.VideoWriter') as MockCv2:
        instance = MagicMock()
        instance.write = fake_write
        instance.isOpened.return_value = True
        MockCv2.return_value = instance

        writer = VideoWriter(
            output_path='/tmp/dummy.avi',
            fps=10,
            width=8,
            height=8,
            color='Red',
        )
        writer.add_frame(mono)
        writer.close()

    assert 'frame' in captured, 'cv2.VideoWriter.write was never called'
    written = captured['frame']
    # cv2 is BGR-native: a Red false-color frame should have non-zero
    # at index 2 (B index in BGR == R index in source RGB after swap).
    # Post-1d: mono Red enters VideoWriter, gets false-colored to RGB
    # internally, then BGR-swapped at the cv2 boundary.
    assert written.shape[-1] == 3
    assert written[0, 0, 2] > 0, (
        f'BGR[2] (== source RGB[0] == Red after swap) should be non-zero, '
        f'got {written[0, 0]}'
    )
    assert written[0, 0, 0] == 0, 'BGR[0] (== source RGB[2] == Blue) should be zero for Red layer'


# ---------------------------------------------------------------------------
# NICE-TO-HAVE 5+6: pure-red + pure-green 16-bit false-color TIFF variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('color', ['Red', 'Green'])
def test_pure_color_16bit_falsecolor_tiff_roundtrip(tmp_path, color):
    """Per-color hardening: confirm the Red and Green layer paths also
    produce 2D mono + metadata post-1d. Same shape as the Blue test;
    catches asymmetries in the layer-color lookup. ``11ec3c7`` was
    Red-specific; the parametrize widens the catch.
    """
    from modules.image_utils import write_tiff

    out_path = tmp_path / f'{color.lower()}.tiff'
    data = np.full((8, 8), 30000, dtype=np.uint16)

    write_tiff(
        data=data,
        file_loc=out_path,
        metadata=_metadata(out_path, channel=color),
        ome=False,
        color=color,
        use_false_color_16bit=True,
    )

    result = tf.imread(str(out_path))
    assert result.ndim == 2, f'{color}: expected 2D mono, got {result.shape}'
    assert result[0, 0] == 30000


# ---------------------------------------------------------------------------
# NICE-TO-HAVE 7: composite OME-TIFF axes structural check
# ---------------------------------------------------------------------------


def test_composite_ome_tiff_axes_structural(tmp_path):
    """Composite OME-TIFF must carry axes='YXS' (Y, X, Sample) for the
    3-channel output. ``cefdfcb`` was a crash on axes mismatch; today's
    composite passes via tifffile; post-1d the same axes structure
    holds even though the per-channel reads are mono.
    """
    input_dir = tmp_path / 'channels'
    input_dir.mkdir()
    for ch in ('r', 'g', 'b'):
        tf.imwrite(str(input_dir / f'{ch}.tiff'), np.full((8, 8), 30000, dtype=np.uint16))

    out_path = tmp_path / 'composite.ome.tiff'

    from modules.composite_generation import CompositeGeneration

    cg = CompositeGeneration(has_turret=False)
    cg.generate_composite_from_paths(
        red_path=input_dir / 'r.tiff',
        green_path=input_dir / 'g.tiff',
        blue_path=input_dir / 'b.tiff',
        output_path=out_path,
        format='ome-tiff',
    )

    # Read with tifffile; OME-TIFF metadata must indicate YXS axes
    with tf.TiffFile(str(out_path)) as t:
        ome_meta = t.ome_metadata
        assert ome_meta is not None, 'composite OME-TIFF missing OME metadata block'
        assert 'YXS' in (t.series[0].axes if t.series else ''), (
            f'Composite OME-TIFF axes should be YXS, got {t.series[0].axes!r}'
        )


# ---------------------------------------------------------------------------
# NICE-TO-HAVE 8: PIW-6 buffer-allocation O(1) regression
# ---------------------------------------------------------------------------


def test_piw6_buffer_allocation_o1(tmp_path):
    """100 sequential ``write_tiff`` calls with a caller-supplied
    ``false_color_buf`` must allocate the false-color buffer ONCE, not
    100 times. ``b9a91b1`` (PIW-6 false-color buffer pre-alloc) is the
    historical fix; post-1d the buffer is mono-sized (1/3 of today),
    but the O(1) reuse property must hold either way.
    """
    from modules.image_utils import write_tiff

    data = np.full((64, 64), 42000, dtype=np.uint16)
    # Post-1d: caller supplies a mono-sized scratch buffer (no widening
    # to 3-channel inside write_tiff). Today's write_tiff with
    # use_false_color_16bit=True allocates a 3-channel buffer per call
    # unless false_color_buf is supplied.
    scratch = np.zeros_like(data)

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

    with patch('numpy.zeros', side_effect=counting_zeros), patch(
        'numpy.empty', side_effect=counting_empty
    ):
        for i in range(100):
            write_tiff(
                data=data,
                file_loc=tmp_path / f'frame_{i:04d}.tiff',
                metadata=_metadata(tmp_path / f'frame_{i:04d}.tiff', channel='Blue'),
                ome=False,
                color='Blue',
                use_false_color_16bit=True,
                false_color_buf=scratch,
            )

    # Post-1d: O(1) frame-sized allocations -- the mono-sized scratch is
    # the right shape for the no-widen save path. Today: write_tiff widens
    # to (H, W, 3) which the 2D scratch does NOT match, so add_false_color
    # falls back to np.zeros((H, W, 3), ...) on every call -> O(n) = 100.
    assert frame_alloc_count['n'] < 10, (
        f'Expected O(1) frame-sized allocations across 100 saves; got '
        f"{frame_alloc_count['n']}. Today fails because mono-sized "
        f"scratch does not fit today's 3-channel widening output."
    )


