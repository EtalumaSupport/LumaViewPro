# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Regression: a zero-extent z-stack must not report itself as one plane.

``ZStackConfig.number_of_steps()`` is the authoritative count, and every gate
that decides whether a stack is configured asks it. It guarded ``step_size``
only, so ``range=0, step_size=10`` computed ``floor(0/10) + 1 == 1``: the
count answered "one plane" for a store with no extent, the standalone
Z-Stack button's refusal never fired, and the run acquired a single image and
reported success.

The count also returned ``np.float64`` against an ``-> int`` annotation, which
is why the Steps field rendered "1.0" -- and "11.0" for every healthy stack.
"""

import pytest

from modules.zstack_config import ZStackConfig


def _config(range_um: float, step_size: float) -> ZStackConfig:
    return ZStackConfig(
        range=range_um,
        step_size=step_size,
        current_z_reference='center',
        current_z_value=1000.0,
    )


class TestZeroExtentIsNotAStack:
    def test_a_zero_range_stack_reports_no_steps(self):
        # The defect: floor(0/10) + 1 == 1 reported a configured stack.
        assert _config(range_um=0, step_size=10).number_of_steps() == 0

    def test_a_zero_range_stack_yields_no_positions(self):
        # current_z_value and the reference are real, so an empty result is
        # the count's answer and not a ConfigError raised before we get here.
        assert _config(range_um=0, step_size=10).step_positions() == {}

    def test_a_zero_step_size_stack_still_reports_no_steps(self):
        # The axis that was already guarded, pinned so the rewrite of the
        # condition cannot silently drop it.
        assert _config(range_um=100, step_size=0).number_of_steps() == 0

    @pytest.mark.parametrize('range_um', [-1, 0, 10])
    @pytest.mark.parametrize('step_size', [-1, 0, 10])
    def test_a_stack_exists_only_when_both_axes_are_positive(self, range_um, step_size):
        # == not is: pre-fix the count is np.float64, so `> 0` yields
        # np.bool_ and an identity check would fail here for the return type
        # rather than for the extent this test is about. Int-ness is pinned
        # separately below.
        expected_steps = range_um > 0 and step_size > 0
        assert (_config(range_um, step_size).number_of_steps() > 0) == expected_steps


class TestTheCountIsAnInt:
    def test_a_healthy_stack_reports_an_int(self):
        # np.float64 satisfies neither isinstance(int) nor the annotation, and
        # renders into the Steps field as "11.0".
        steps = _config(range_um=100, step_size=10).number_of_steps()
        assert isinstance(steps, int)
        assert str(steps) == '11'

    def test_a_refused_stack_reports_an_int(self):
        assert isinstance(_config(range_um=0, step_size=10).number_of_steps(), int)


class TestApplyZStackingSaysWhyItDidNothing:
    """The Apply Z-Stacking button refused zero-extent params in silence.

    It logged 'Z-stacking parameters are zero. No changes applied.' and
    returned, so the press looked like it had worked and the protocol was
    unchanged with no reason given. Its sibling branch eight lines above --
    negative values -- had raised a popup since it was written.

    The Kivy UI classes cannot be instantiated headlessly (ids, _app_ctx,
    worker pool), so the contract is locked by AST the way the sibling
    refusal guards do it: no branch of this handler may return without
    delivering the reason to the user.
    """

    def test_no_branch_returns_without_telling_the_user(self):
        import ast

        from tests.ast_seams import find_def

        fn = find_def('ui/protocol_settings.py', 'apply_zstacking', 'ProtocolSettings')
        assert fn is not None, 'ProtocolSettings.apply_zstacking not found'

        silent = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            for branch in (node.body, node.orelse):
                if not any(isinstance(stmt, ast.Return) for stmt in branch):
                    continue
                delivers = any(
                    isinstance(inner, ast.Name) and inner.id == 'show_notification_popup'
                    for stmt in branch
                    for inner in ast.walk(stmt)
                )
                if not delivers:
                    silent.append(node.lineno)

        assert silent == [], (
            'apply_zstacking returns without showing the user why, at '
            f'ui/protocol_settings.py line(s) {silent}'
        )
