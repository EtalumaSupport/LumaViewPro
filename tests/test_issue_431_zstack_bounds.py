"""Regression for #431 (Z half): applying z-stacking must not create slices
outside the Z travel range.

XY tiling already checked its tiles against the travel; the matching Z-stack
bounds check was never implemented, so a z-stack range wider than the Z travel pushed the
protocol to the end of travel and crashed the run. Skipping the slices and
returning a count left a stack missing its ends that projected as if whole, so
apply_zstacking now refuses the stack and leaves the protocol as it was.

Reuses the Protocol builders from test_protocol_roundtrip.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from modules.exceptions import ProtocolRunRefusedError
from tests.test_protocol_roundtrip import _build_protocol, _make_step


# center reference at Z=5000 with range=100/step=20 -> 6 slices:
# 4950, 4970, 4990, 5010, 5030, 5050
_ZSTACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}


def _proto():
    # z_slice=-1 marks a not-yet-stacked step so apply_zstacking expands it.
    return _build_protocol([_make_step(name='A1_BF', z=5000.0, z_slice=-1)])


def test_out_of_range_zslices_refuse_the_stack():
    proto = _proto()
    before = proto.steps().copy()
    axis_limits = {'Z': {'min': 4960.0, 'max': 5040.0}}

    with (
        patch('modules.protocol.notifications.report_outcome'),
        pytest.raises(ProtocolRunRefusedError) as refusal,
    ):
        proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=axis_limits)

    # 4950 and 5050 fall outside [4960, 5040].
    assert refusal.value.reason == 'zslices_outside_travel'
    assert '2 of the 6 z-slices' in str(refusal.value)
    pd.testing.assert_frame_equal(proto.steps(), before)


def test_all_in_range_zslices_kept_no_skips():
    proto = _proto()
    axis_limits = {'Z': {'min': 0.0, 'max': 10000.0}}

    proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=axis_limits)

    assert len(proto.steps()) == 6
