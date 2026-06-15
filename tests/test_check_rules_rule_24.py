"""Tests for tools/check_rules.py rule_24 (broad source-text scan).

Mirror of Firmware's tests/test_check_rules.py::TestRule24 after the
2026-05-28 broaden. The narrow logger/print/notification arg-string
implementation was replaced by a full-file source-text scan per
CLAUDE.md Rule 24 spec ('every string ... every comment ... every
docstring ... every identifier in .py / .c / .h / .kv / similar
files'). Companion to test_check_rules_rule_24_kv.py for the .kv path.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source


def _rule24_violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'rule_24']


class TestRule24Triggers:
    def test_degree_sign_in_logger_info(self):
        src = textwrap.dedent("""
            import logging
            logger = logging.getLogger(__name__)
            def f():
                logger.info(f"temp: 25 °C")
        """)
        vs = _rule24_violations(src, 'x.py')
        assert vs
        assert 'U+00B0' in vs[0].message

    def test_em_dash_in_comment(self):
        src = '# comment with — em-dash\nx = 1\n'
        vs = _rule24_violations(src, 'x.py')
        assert vs
        assert 'U+2014' in vs[0].message

    def test_non_ascii_in_docstring(self):
        src = textwrap.dedent('''
            def f():
                """Set temperature in °C; reject at 100 °C."""
                pass
        ''')
        vs = _rule24_violations(src, 'x.py')
        assert vs

    def test_non_ascii_in_bare_string(self):
        src = 'x = "non-ASCII °C bare string"\n'
        vs = _rule24_violations(src, 'x.py')
        assert vs

    def test_arrow_in_print(self):
        src = 'print(f"value → limit")\n'
        vs = _rule24_violations(src, 'x.py')
        assert vs

    def test_micro_sign_in_notifications(self):
        src = 'notifications.error("Title", "5 µm offset")\n'
        vs = _rule24_violations(src, 'x.py')
        assert vs


class TestRule24DoesNotTrigger:
    def test_clean_ascii_source_passes(self):
        src = textwrap.dedent('''
            # Clean ASCII comment with --, ->, degC.
            def f():
                """Return temperature in degC."""
                logger.info(f"temp: {25} degC -- ok")
                return "plain ascii"
        ''')
        vs = _rule24_violations(src, 'x.py')
        assert vs == []

    def test_escape_sequence_does_not_trigger(self):
        # '\\r\\n' in source is ASCII bytes; only the resolved string
        # value contains U+000D. Source-text scan correctly ignores it.
        src = "x = '\\r\\n'\n"
        vs = _rule24_violations(src, 'x.py')
        assert vs == []


class TestRule24FileLevelExempt:
    def test_test_check_rules_path_is_exempt(self):
        src = 'src = "logger.info(\\"got °C\\")"\n'
        vs = _rule24_violations(src, 'tests/test_check_rules.py')
        assert vs == []

    def test_test_check_rules_variant_path_is_exempt(self):
        src = 'fixture = "em-dash —"\n'
        vs = _rule24_violations(src, 'tests/test_check_rules_some_variant.py')
        assert vs == []

    def test_non_rule_check_test_path_not_exempt(self):
        # Other test files MUST hold ASCII discipline -- the 2026-05-28
        # LVP sweep + Firmware sweep confirmed it.
        src = '# em-dash —\n'
        vs = _rule24_violations(src, 'tests/test_imaging_api.py')
        assert vs


class TestRule24LineLevelExempt:
    def test_tech_support_report_block_glyphs_exempt(self):
        # _RULE_24_LINE_EXEMPT covers the CLI progress-bar block glyphs
        # at modules/tech_support_report.py:2475. Construct a source
        # with exactly 2475 lines + the glyph on the right line.
        prefix = '\n' * 2474
        src = prefix + 'print("[███░░]")\n'
        vs = _rule24_violations(src, 'modules/tech_support_report.py')
        assert vs == [], [v.format() for v in vs]

    def test_audit_fixes_ellipsis_exempt(self):
        # _RULE_24_LINE_EXEMPT covers the ellipsis assertion at
        # tests/test_audit_fixes.py:9716.
        prefix = '\n' * 9715
        src = prefix + 'assert "…" in text or "..." in text\n'
        vs = _rule24_violations(src, 'tests/test_audit_fixes.py')
        assert vs == [], [v.format() for v in vs]

    def test_other_line_in_exempt_file_still_blocks(self):
        # File-level exemption is line-specific in _RULE_24_LINE_EXEMPT.
        # Other lines in tech_support_report.py must still hold ASCII.
        src = 'x = "em-dash —"\nfoo = 1\n'
        vs = _rule24_violations(src, 'modules/tech_support_report.py')
        assert vs


class TestRule24UnparseableSourceStillScanned:
    def test_unparseable_source_still_scanned(self):
        # rule_24 is text-scan only; runs even on AST parse failure.
        src = "def f(:  # syntax error\n    x = '°C'\n"
        all_violations = check_source(src, 'x.py')
        assert any(v.rule == 'rule_24' for v in all_violations)
        assert any(v.rule == 'parse' for v in all_violations)
