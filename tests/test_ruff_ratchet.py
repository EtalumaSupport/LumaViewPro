"""Tests for tools/ruff_ratchet.py -- the pre-commit finding-count ratchet.

One lean pass over the pure logic (baseline parsing + verdict); the
real validation is the hook running on actual commits.
"""

import pytest

from tools.ruff_ratchet import decide, parse_baseline


class TestParseBaseline:
    def test_plain_integer(self):
        assert parse_baseline('495\n') == 495

    def test_comment_lines_and_blanks_skipped(self):
        assert parse_baseline('# ceiling\n\n# notes\n123\n') == 123

    def test_no_integer_raises(self):
        with pytest.raises(ValueError):
            parse_baseline('# only comments\n')

    def test_garbage_line_raises(self):
        with pytest.raises(ValueError):
            parse_baseline('four hundred\n')


class TestDecide:
    def test_increase_blocks(self):
        assert decide(496, 495) == 'block'

    def test_decrease_lowers(self):
        assert decide(370, 495) == 'lower'

    def test_equal_passes(self):
        assert decide(495, 495) == 'ok'

    def test_zero_count_lowers_to_zero(self):
        assert decide(0, 1) == 'lower'
