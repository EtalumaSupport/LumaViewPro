"""Regression tests for the standalone-entry-point identity banner.

Standalone characterization / CLI scripts drive real hardware through the
production drivers but historically emitted no self-identifying record into
the shared LVP_Log bundle -- a run was visible only by its serial-traffic
motion signature. `log_standalone_banner` routes the launch fingerprint
(script name, the script's own repo SHA, argv) through the named lvp_logger
so it lands in the bundle; a root-logger line would miss that file handler.

Source-level structural lock: the suite mocks `lvp_logger` at import
(conftest installs a MagicMock in sys.modules), so behavioral testing of the
real banner is not possible here -- assert on the source text instead, the
same approach as test_lvp_logger_marker_lookup. The runtime behavior was
verified by exercising the real module directly.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_LVP_LOGGER_SRC = (ROOT / 'lvp_logger.py').read_text(encoding='utf-8')


class TestBannerCarriesInvocationIdentity:
    def test_banner_accepts_invocation_params(self):
        assert 'invocation_file' in _LVP_LOGGER_SRC
        assert 'invocation_argv' in _LVP_LOGGER_SRC

    def test_banner_emits_script_repo_and_args_lines(self):
        assert 'Script:' in _LVP_LOGGER_SRC
        assert 'ScriptRepo:' in _LVP_LOGGER_SRC
        assert 'Args:' in _LVP_LOGGER_SRC

    def test_shared_git_sha_helper_exists(self):
        # One helper serves both the LVP module SHA and the script's own SHA.
        assert 'def _git_short_sha(' in _LVP_LOGGER_SRC

    def test_script_repo_uses_invocation_file_dir(self):
        # The invoking script's SHA is looked up from its own directory, not
        # the LVP install path (they can be different repos).
        assert (
            '_git_short_sha(os.path.dirname(os.path.abspath(invocation_file)))' in _LVP_LOGGER_SRC
        )


class TestStandaloneWrapper:
    def test_wrapper_defined(self):
        assert 'def log_standalone_banner(' in _LVP_LOGGER_SRC

    def test_wrapper_forwards_invocation_to_banner(self):
        assert 'invocation_file=invocation_file' in _LVP_LOGGER_SRC
        assert 'invocation_argv=invocation_argv' in _LVP_LOGGER_SRC

    def test_wrapper_resolves_version_via_shared_reader(self):
        # Reuses the canonical version reader rather than re-parsing version.txt.
        assert 'read_version' in _LVP_LOGGER_SRC


class TestShippedEntryPointsCallBanner:
    def test_pylon_probe_sweep_calls_shared_banner(self):
        src = (ROOT / 'tools' / 'pylon_probe_sweep.py').read_text(encoding='utf-8')
        assert 'log_standalone_banner(__file__, sys.argv)' in src
        # The bespoke, partial run-start line is gone.
        assert 'run start:' not in src
