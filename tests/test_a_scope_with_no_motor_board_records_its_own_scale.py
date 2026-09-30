# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A scope with no motor board records its own image scale.

The pixel-size resolver asks the motor configuration first and the model
catalogue second. The null motor board used to carry a default motor
configuration -- a motorised scope's, 2.0 um pixels -- so an LS560 or
LS620, which has no motor board, answered with that scope's pixel and
every image it saved recorded the wrong scale. With no board there is no
motor configuration, and the model's own catalogue optics answer.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from drivers.null_motorboard import NullMotionBoard
from modules import common_utils
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# 20x Oly's focal length in objectives.json.
FOCAL_LENGTH_20X_MM = 9.0


def _um_per_pixel_at_20x(model: str) -> float | None:
    settings = complete_settings()
    settings['microscope'] = model
    session = ScopeSession.create(settings, simulate=True)
    try:
        return common_utils.get_pixel_size(
            focal_length=FOCAL_LENGTH_20X_MM,
            binning_size=1,
            capabilities=session.scope.capabilities,
        )
    finally:
        session.shutdown()


@pytest.mark.parametrize('model', ['LS620', 'LS560'])
def test_a_scope_with_no_motor_board_takes_its_catalogue_scale(model):
    # 2.2 um pixels behind a 47.8 mm tube lens at 20x.
    assert _um_per_pixel_at_20x(model) == pytest.approx(0.41423, abs=1e-4)


def test_a_scope_with_a_motor_board_keeps_its_own_scale():
    # 2.0 um pixels behind the same tube lens.
    assert _um_per_pixel_at_20x('LS850') == pytest.approx(0.37657, abs=1e-4)


class TestTheNullBoardDescribesNoMotors:
    def test_it_has_no_motor_configuration(self):
        assert NullMotionBoard().motorconfig is None

    def test_it_has_no_axes_and_no_limits(self):
        board = NullMotionBoard()

        assert dict(board.get_axes_config()) == {}
        for axis in ('X', 'Y', 'Z', 'T'):
            assert board.get_axis_limits(axis) is None

    @pytest.mark.parametrize(
        'conversion', ['z_um2ustep', 'z_ustep2um', 'xy_um2ustep', 'xy_ustep2um', 't_pos2ustep']
    )
    def test_it_converts_nothing(self, conversion):
        with pytest.raises(RuntimeError, match='no motor board'):
            getattr(NullMotionBoard(), conversion)(1)


def test_a_logs_only_report_reads_the_serial_without_a_motor_configuration(
    tmp_path, caplog, monkeypatch
):
    # The report's first serial source is the motor configuration; with no
    # motor board there is none, and asking it anyway logged an exception
    # on every report from such a scope before FULLINFO answered.
    from modules.tech_support_report import TechSupportReport

    report = TechSupportReport.__new__(TechSupportReport)
    report._meta = {}
    for step in ('_step_logs', '_step_data_folder', '_step_protocols', '_step_video_receipts'):
        monkeypatch.setattr(report, step, lambda tmp: None)
    monkeypatch.setattr(report, '_run_hardware_free_steps', lambda *args: None)
    report.diag = SimpleNamespace(motor_board=NullMotionBoard(), get_serial_number=lambda: '12062')

    with caplog.at_level(logging.ERROR):
        zip_path = report.generate_logs_only(output_dir=tmp_path)

    assert zip_path.name.startswith('SN12062-')
    assert 'SN lookup via motorconfig failed' not in caplog.text
