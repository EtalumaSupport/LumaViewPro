"""Regression: the root logger is captured in the shared LVP_Log bundle.

Standalone CLI scripts (the characterization tools) log on the root logger
via ``logging.info(...)`` / ``logging.basicConfig(...)``. Historically the
bundle's file handler was attached only to the named ``lvp_logger`` / ``LVP``
loggers, never to root, so a script's output never reached the shared bundle
-- the run was visible only by its serial-traffic motion signature.

``lvp_logger`` now owns the root config: root is the single owner of the bundle
handlers (so any root logging is captured by default, and the LVP loggers reach
the bundle by propagating to root instead of holding their own copies of the
handlers). The console handler is added only in debug and only when the
process has a stderr (a packaged windowed build has none, and a StreamHandler
over a missing stream raises on emit). It writes to ``sys.__stderr__``, never
to ``sys.stderr``: Kivy replaces ``sys.stderr`` with a stream that logs each
line, so a console made after Kivy loaded logged its own output until the
recursion limit (the sim-walk harness imports Kivy first; with debug on, every
one of its tests died in a RecursionError).

Source-level structural lock: the suite mocks ``lvp_logger`` at import
(conftest installs a MagicMock), so the real module is asserted on by source
text, the same approach as test_lvp_logger_marker_lookup. The runtime routing
(root INFO -> bundle; per-run DEBUG preserved; errors.log stays errors-only)
was verified by exercising the real module.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SRC = (ROOT / 'lvp_logger.py').read_text(encoding='utf-8')


class TestRootLoggerCapturedInBundle:
    def test_bundle_handler_attached_to_root(self):
        # The main-log handler is added to the root logger, not only the
        # named loggers.
        assert '_root_logger = logging.getLogger()' in _SRC
        assert '_root_logger.addHandler(file_handler)' in _SRC
        assert '_root_logger.addHandler(error_file_handler)' in _SRC

    def test_root_level_debug_so_per_run_debug_logs_survive(self):
        assert '_root_logger.setLevel(logging.DEBUG)' in _SRC

    def test_bundle_handler_filters_to_main_log_floor(self):
        # file_handler level set so a root/library DEBUG firehose does not
        # flood the bundle.
        assert 'file_handler.setLevel(_log_level)' in _SRC


class TestConsoleGuardedOnStderr:
    def test_console_only_in_debug_and_when_stderr_present(self):
        # The console echoes the bundle to the terminal only in debug: every
        # logger now propagates to root, so a non-debug console would surface
        # all LVP + framework output as terminal noise. It is also guarded on
        # the process's stderr -- a packaged windowed build has None, and a
        # StreamHandler over a missing stream raises on emit.
        assert 'if debug and sys.__stderr__ is not None:' in _SRC
        assert 'logging.StreamHandler(sys.__stderr__)' in _SRC


# A process that loads Kivy before lvp_logger, as the sim-walk harness does,
# with debug on: Kivy has replaced sys.stderr by the time the console is made.
_KIVY_FIRST = r"""
import os, sys
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_NO_CONSOLELOG'] = '1'
sys.path.insert(0, sys.argv[1])
import kivy.logger
import modules.settings_init as settings_init
settings_init.load_debug_setting = lambda directory: True
from lvp_logger import logger
logger.info('a line logged after Kivy loaded')
print('LOGGED')
"""


def test_a_debug_console_made_after_kivy_loaded_logs_without_looping(tmp_path):
    script = tmp_path / 'kivy_first.py'
    script.write_text(_KIVY_FIRST)

    ran = subprocess.run(
        [sys.executable, str(script), str(ROOT)], capture_output=True, text=True, timeout=120
    )

    assert 'LOGGED' in ran.stdout, ran.stderr[-2000:]
    assert 'a line logged after Kivy loaded' in ran.stderr
