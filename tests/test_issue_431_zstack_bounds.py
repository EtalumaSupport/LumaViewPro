"""A z-stack inside Z's travel builds whole: every slice kept, none skipped.

The refusal of a stack with a slice outside the travel is pinned in
test_a_build_outside_the_travel_is_refused.py; this is its admit side. A
stack that skipped its out-of-range ends and reported a count would have
projected as if whole (#431).

Reuses the Protocol builders from test_protocol_roundtrip.
"""

from tests.test_protocol_roundtrip import _build_protocol, _make_step


# center reference at Z=5000 with range=100/step=20 -> 6 slices:
# 4950, 4970, 4990, 5010, 5030, 5050
_ZSTACK = {'range': 100.0, 'step_size': 20.0, 'z_reference': 'center'}


def _proto():
    # z_slice=-1 marks a not-yet-stacked step so apply_zstacking expands it.
    return _build_protocol([_make_step(name='A1_BF', z=5000.0, z_slice=-1)])


def test_all_in_range_zslices_kept_no_skips():
    proto = _proto()
    axis_limits = {'Z': {'min': 0.0, 'max': 10000.0}}

    proto.apply_zstacking(zstack_params=_ZSTACK, axis_limits=axis_limits)

    assert len(proto.steps()) == 6
