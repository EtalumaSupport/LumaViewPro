"""Tests for tools/check_rules.py _check_rule_24_kv.

CLAUDE.md Rule 24 covers .kv files in addition to .py / .c / .h. The
.py path uses an AST gate (logger / print / notification arg strings
only); .kv has no AST and the rule covers the entire file. This test
file exercises the plain content scan.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import _check_rule_24_kv


class TestRule24KvBlocksNonAscii:
    def test_em_dash_in_comment_blocks(self):
        content = '# A button — used for foo\nLabel:\n    text: "x"\n'
        violations = _check_rule_24_kv(content, 'ui/foo.kv')
        assert len(violations) == 1
        assert violations[0].rule == 'rule_24'
        assert violations[0].line == 1

    def test_right_arrow_in_comment_blocks(self):
        content = 'Label:\n    text: "x"  # fallback → other\n'
        violations = _check_rule_24_kv(content, 'ui/foo.kv')
        assert len(violations) == 1
        assert violations[0].line == 2

    def test_micro_sign_in_label_text_blocks(self):
        content = "Label:\n    text: 'pixels/μm'\n"
        violations = _check_rule_24_kv(content, 'ui/foo.kv')
        assert len(violations) == 1
        assert violations[0].line == 2
        assert 'U+00B5' in violations[0].message or 'U+03BC' in violations[0].message

    def test_multiple_lines_each_reported(self):
        content = '# foo —\n# bar →\n# baz μ\n'
        violations = _check_rule_24_kv(content, 'ui/foo.kv')
        assert len(violations) == 3
        assert [v.line for v in violations] == [1, 2, 3]


class TestRule24KvPasses:
    def test_ascii_only_passes(self):
        content = (
            '# A button -- used for foo\n'
            'Label:\n'
            "    text: 'pixels/um'\n"
            "    tooltip_text: 'Camera pixels per micron of sample'\n"
        )
        violations = _check_rule_24_kv(content, 'ui/foo.kv')
        assert violations == []

    def test_empty_file_passes(self):
        violations = _check_rule_24_kv('', 'ui/empty.kv')
        assert violations == []
