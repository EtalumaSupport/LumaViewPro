# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A z-stack's slices are laid over the span they cover, placed at the focus.

``ZStackConfig.step_positions`` is the one place a stack's slice positions
are computed (``Protocol.apply_zstacking`` builds the slices from it, and the
travel check refuses from it). Its contract: the slices are ``step`` apart
and cover (n - 1) x step, and that span sits against the Z the stack is built
around by the reference -- a top stack ends at it, a bottom stack starts at
it, a centre stack is centred on it. So an odd stack, or any top or bottom
stack, has a slice at that Z, which is the focus its reference slice holds.

The span placed used to be the requested range. When the range is not a
whole number of steps the slices then missed the focus: 599 / 200 centred
sat at -299.5, -99.5 and +100.5, and a top stack's top slice 199 below it.
"""

from modules.zstack_config import ZStackConfig, ZStackReference

FOCUS = 5000.0

# (reference, range, step, the slices' distances from the focus, lowest first)
CASES = [
    # Whole numbers of steps: unchanged by the span being the covered one.
    (ZStackReference.CENTER, 400.0, 200.0, [-200.0, 0.0, 200.0]),
    (ZStackReference.TOP, 400.0, 200.0, [-400.0, -200.0, 0.0]),
    (ZStackReference.BOTTOM, 400.0, 200.0, [0.0, 200.0, 400.0]),
    # Not whole numbers of steps: the field's shape.
    (ZStackReference.CENTER, 599.0, 200.0, [-200.0, 0.0, 200.0]),
    (ZStackReference.TOP, 599.0, 200.0, [-400.0, -200.0, 0.0]),
    (ZStackReference.BOTTOM, 599.0, 200.0, [0.0, 200.0, 400.0]),
    # An even centre count straddles the focus symmetrically.
    (ZStackReference.CENTER, 350.0, 100.0, [-150.0, -50.0, 50.0, 150.0]),
    (ZStackReference.TOP, 350.0, 100.0, [-300.0, -200.0, -100.0, 0.0]),
]


def test_each_stack_is_laid_over_its_covered_span_at_the_focus():
    wrong = []
    for reference, range_um, step, expected in CASES:
        positions = ZStackConfig(
            range=range_um,
            step_size=step,
            current_z_reference=reference,
            current_z_value=FOCUS,
        ).step_positions()
        got = [positions[i] - FOCUS for i in sorted(positions)]
        if got != expected:
            wrong.append(f'{reference} {range_um}/{step}: {got}, expected {expected}')
    assert not wrong, '\n'.join(wrong)
