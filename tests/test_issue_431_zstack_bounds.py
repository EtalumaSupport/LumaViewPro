"""Regression for #431 (Z half): applying z-stacking must not create slices
outside the Z travel range.

XY tiling already skips out-of-bounds tiles; the matching Z-stack bounds check
was never implemented, so a z-stack range wider than the Z travel pushed the
protocol to the end of travel and crashed the run. apply_zstacking now skips
out-of-range slices and reports the count so the UI can warn.

Reuses the Protocol builders from test_protocol_roundtrip.
"""

from tests.test_protocol_roundtrip import _build_protocol, _make_step


# center reference at Z=5000 with range=100/step=20 -> 6 slices:
# 4950, 4970, 4990, 5010, 5030, 5050
_ZSTACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}


def _proto():
    # z_slice=-1 marks a not-yet-stacked step so apply_zstacking expands it.
    return _build_protocol([_make_step(name='A1_BF', z=5000.0, z_slice=-1)])


def test_out_of_range_zslices_are_skipped_and_counted():
    proto = _proto()
    axes_config = {'Z': {'limits': {'min': 4960.0, 'max': 5040.0}}}

    status = proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=axes_config)

    # 4950 and 5050 fall outside [4960, 5040].
    assert status['zslices_skipped'] == 2
    z_values = proto.steps()['Z'].tolist()
    assert len(z_values) == 4
    assert all(4960.0 <= z <= 5040.0 for z in z_values), z_values


def test_all_in_range_zslices_kept_no_skips():
    proto = _proto()
    axes_config = {'Z': {'limits': {'min': 0.0, 'max': 10000.0}}}

    status = proto.apply_zstacking(zstack_params=_ZSTACK, axes_config=axes_config)

    assert status['zslices_skipped'] == 0
    assert len(proto.steps()) == 6
