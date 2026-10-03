# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A board running the original firmware is named as such, with its date.

The original motor and LED firmware answer INFO with a date and no version
string. The API reported that as firmware_version None, the same answer as a
board that did not answer at all, so the support report and a bench verdict
could not say which firmware the scope ran. The API now names it 'original'
and carries the date INFO gave, for every firmware; None stays for a board
that did not answer.
"""

from __future__ import annotations

import pytest

from modules.lumascope_api.diagnostics import ORIGINAL_FIRMWARE
from tests.scope_fakes import build_scope


@pytest.fixture
def scope():
    return build_scope(simulate=True)


def _as_board_answered(driver, version, date, responding=True):
    driver.firmware_version = version
    driver.firmware_date = date
    driver.firmware_responding = responding


def test_the_original_motor_firmware_is_named_with_its_date(scope):
    _as_board_answered(scope._motion_driver, None, '2024-09-10')
    info = scope.diagnostics.get_motor_info()
    assert info['firmware_version'] == ORIGINAL_FIRMWARE == 'original'
    assert info['firmware_date'] == '2024-09-10'


def test_a_versioned_motor_firmware_keeps_its_version_and_date(scope):
    _as_board_answered(scope._motion_driver, '3.0.7', '2025-06-01')
    info = scope.diagnostics.get_motor_info()
    assert info['firmware_version'] == '3.0.7'
    assert info['firmware_date'] == '2025-06-01'


def test_a_board_that_did_not_answer_is_not_called_original(scope):
    _as_board_answered(scope._motion_driver, None, None, responding=False)
    assert scope.diagnostics.get_motor_info()['firmware_version'] is None


def test_the_original_led_firmware_is_named_with_its_date(scope):
    _as_board_answered(scope._led_driver, None, '2024-02-01')
    info = scope.diagnostics.get_led_info()
    assert info['firmware_version'] == ORIGINAL_FIRMWARE
    assert info['firmware_date'] == '2024-02-01'
