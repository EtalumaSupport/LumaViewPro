# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A new process learns from the hardware which axes are already homed.

Nothing about "we homed this stage" survives a process boundary: a fresh
MotionAPI has no memory, and the driver's own has_homed() answers only for
homes THIS process performed. Seeding every axis UNKNOWN therefore told the
pre-drive gate that a powered, already-homed scope had no valid reference
frame, and the gate refused to move it. The GUI never noticed because it homes
at startup. Every headless caller -- bench characterization scripts, REST, the
SDK -- hit it on every run, and their only escapes were to re-home (destroying
the focused position they attached to measure) or to bypass the gate.

The firmware is the one store that can answer: it clears its per-axis homed
flags when it boots and sets one only when that axis completes a home. These
tests pin both directions of that, because the safety half matters as much as
the fix -- a board that really has not been homed must still be refused.
"""

from __future__ import annotations

import pytest

from modules.lumascope_api._constants import AxisState
from drivers.motorboard import _parse_fullinfo


# A real EL-0940 response: one line, uneven padding, all four axes.
_FULLINFO_HOMED = (
    'EL-0940 Firmware:     2026-01-15 Model: LS850T   Serial: 12075'
    ' X homed: True   X present: True'
    ' Y homed: True   Y present: True'
    ' Z homed: True   Z present: True'
    ' T homed: True   T present: True'
)
_FULLINFO_FRESH_BOOT = _FULLINFO_HOMED.replace('homed: True', 'homed: False')


class TestFullinfoParser:
    """One parse of the response yields every field consumers used to rescan for."""

    def test_parses_identity_presence_and_homed_together(self):
        record = _parse_fullinfo(_FULLINFO_HOMED)
        assert record['model'] == 'LS850T'
        assert record['serial_number'] == '12075'
        assert record['has_turret'] is True
        assert record['present_axes'] == ['X', 'Y', 'Z', 'T']
        assert record['homed_axes'] == ['X', 'Y', 'Z', 'T']

    def test_a_freshly_booted_board_reports_present_but_not_homed(self):
        record = _parse_fullinfo(_FULLINFO_FRESH_BOOT)
        assert record['present_axes'] == ['X', 'Y', 'Z', 'T'], (
            'presence and homed-ness are independent facts: a board that has '
            'not homed still has its axes'
        )
        assert record['homed_axes'] == []

    def test_homed_is_read_per_axis_not_all_or_nothing(self):
        # A Z-only home (the partial-home path) must not claim X and Y.
        resp = _FULLINFO_FRESH_BOOT.replace('Z homed: False', 'Z homed: True')
        assert _parse_fullinfo(resp)['homed_axes'] == ['Z']

    def test_unparseable_response_reports_nothing_homed(self):
        # Legacy firmware answers UNKNOWN_CMD. Claiming a homed axis off a
        # response we could not read would drive against an invented frame,
        # so the fallback must stay at the refusing end of the gate.
        for resp in ('', 'UNKNOWN_CMD', 'garbage without the fields'):
            record = _parse_fullinfo(resp)
            assert record['model'] == 'unknown', resp
            assert record['homed_axes'] == [], resp
            assert record['present_axes'] == [], resp


class TestAxisStateSeeding:
    """_init_axes trusts the hardware's answer, in both directions."""

    @staticmethod
    def _seed(motion, present, homed):
        motion._init_axes(present, homed)
        return motion._axis_state

    def test_homed_hardware_seeds_idle_so_the_gate_permits_motion(self, monkeypatch):
        from modules.lumascope_api.motion import MotionAPI

        motion = MotionAPI.__new__(MotionAPI)
        state = self._seed(motion, ['X', 'Y', 'Z'], ['X', 'Y', 'Z'])
        assert state == dict.fromkeys(['X', 'Y', 'Z'], AxisState.IDLE), (
            'a fresh process attaching to an already-homed scope must not call its position unknown'
        )

    def test_unhomed_hardware_still_seeds_unknown(self):
        from modules.lumascope_api.motion import MotionAPI

        motion = MotionAPI.__new__(MotionAPI)
        state = self._seed(motion, ['X', 'Y', 'Z'], [])
        assert state == dict.fromkeys(['X', 'Y', 'Z'], AxisState.UNKNOWN), (
            'the gate must still refuse a scope that has never been homed'
        )

    def test_partially_homed_hardware_seeds_each_axis_on_its_own_answer(self):
        from modules.lumascope_api.motion import MotionAPI

        motion = MotionAPI.__new__(MotionAPI)
        state = self._seed(motion, ['X', 'Y', 'Z'], ['Z'])
        assert state == {
            'X': AxisState.UNKNOWN,
            'Y': AxisState.UNKNOWN,
            'Z': AxisState.IDLE,
        }

    def test_seeding_requires_the_hardware_answer(self):
        # Defaulting this argument would let a new construction site fall back
        # to blind UNKNOWN seeding, and nothing would fail until a headless
        # run was refused in the field.
        from modules.lumascope_api.motion import MotionAPI

        motion = MotionAPI.__new__(MotionAPI)
        with pytest.raises(TypeError):
            motion._init_axes(['Z'])


class TestDriversAnswerTheQuestion:
    """Every motion driver can be asked, so no construction path is blind."""

    def test_null_board_reports_nothing_homed(self):
        from drivers.null_motorboard import NullMotionBoard

        assert NullMotionBoard().detect_homed_axes() == []

    def test_simulated_board_reports_homed_only_after_a_home(self):
        from drivers.simulated_motorboard import SimulatedMotorBoard

        board = SimulatedMotorBoard()
        assert board.detect_homed_axes() == [], (
            'a simulated board that has not homed must look like a fresh boot'
        )
        board.home()
        assert set(board.detect_homed_axes()) == set(board.detect_present_axes()), (
            'after a home the simulator must report its axes homed, or no test '
            'can exercise the attach-to-homed-hardware path'
        )
