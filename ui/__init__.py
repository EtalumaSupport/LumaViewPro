# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""LVP UI package.

Runs before any `ui.*` module body, so we set the Kivy env vars here to
keep Kivy from writing logs to ~/.kivy/logs/. The LVP `kivy` logger is
still routed through lvp_logger's file handler, so diagnostics remain
in the main LVP logs -- just not under the user's home dir.
"""

import os

os.environ.setdefault('KIVY_NO_CONSOLELOG', '1')
os.environ.setdefault('KIVY_NO_FILELOG', '1')
