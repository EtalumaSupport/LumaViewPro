# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The shipped motor defaults, loaded the way a bring-up loads them.

A motor driver takes the defaults table as a required argument, so a test
that builds one hands it this.
"""

from _lumascope.modules.motor_defaults import read_motor_defaults
from _lumascope.modules.path_utils import resolve_data_file

SHIPPED_MOTOR_DEFAULTS = read_motor_defaults(resolve_data_file('motorconfig_defaults.json'))
