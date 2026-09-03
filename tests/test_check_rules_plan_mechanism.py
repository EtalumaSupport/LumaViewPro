"""Tests for tools/check_rules.py plan_undefined_mechanism.

The check blocks a plan document whose Stages section schedules a mechanism
the same document never specifies. It exists because a plan reached approval
naming three mechanisms whose entire specification had decayed, across
successive rewrites of one file, into a pointer at a revision that no longer
existed as a file.

The deferral detector must distinguish two things that look alike in text:

1. A body that really does defer -- "(carries)", "carry unchanged", "as rev 4"
   -- which leaves the mechanism unspecified once the pointed-at revision is
   gone.
2. Ordinary technical prose that happens to use the verb "carries". A bare
   word match rejected a mechanism whose specification was complete and
   self-contained, purely for the wording, which pushes an author toward
   rewriting correct prose instead of fixing the check.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_doc

_PLAN_PATH = 'docs/EXAMPLE_PLAN_2026-01-01.md'


def _plan(mechanism_body: str) -> str:
    return (
        '# Example -- PLAN\n\n'
        '## Status\n\n| Field | Value |\n|---|---|\n| Revision | 1 |\n\n'
        '## Mechanisms\n\n'
        '### M13 -- the starter\n\n'
        f'{mechanism_body}\n\n'
        '## Stages\n\n- **C3** -- M13.\n'
    )


def _violations(content: str) -> list:
    return [v for v in check_doc(content, _PLAN_PATH, None) if v.rule == 'plan_undefined_mechanism']


class TestDeferralIsStillCaught:
    def test_as_rev_n_blocks(self):
        assert _violations(_plan('Built as rev 4.'))

    def test_carry_unchanged_blocks(self):
        assert _violations(_plan('M13 -- carry unchanged.'))

    def test_carries_unchanged_blocks(self):
        assert _violations(_plan('This mechanism carries unchanged.'))

    def test_parenthesised_carries_blocks(self):
        assert _violations(_plan('M13 (carries)'))

    def test_unchanged_from_rev_blocks(self):
        assert _violations(_plan('Unchanged from rev 6.'))

    def test_a_stage_naming_an_undefined_mechanism_blocks(self):
        content = (
            '# Example -- PLAN\n\n## Status\n\n| Field | Value |\n|---|---|\n'
            '| Revision | 1 |\n\n## Stages\n\n- **C3** -- M13.\n'
        )
        assert _violations(content)


class TestOrdinaryProseIsNotADeferral:
    """The verb 'carries' in a specification is not a pointer at a revision."""

    def test_carries_no_callback_does_not_block(self):
        body = (
            "The starter's own task carries no callback, so the guard has no "
            'clearer once the worker is deleted; the starter supplies one.'
        )
        assert not _violations(_plan(body))

    def test_carries_no_channel_list_does_not_block(self):
        body = 'The outcome carries no channel list, so per-channel detail stays in the log.'
        assert not _violations(_plan(body))

    def test_carries_per_layer_autofocus_does_not_block(self):
        body = 'The run kind carries per-layer autofocus, so the wait is unbounded.'
        assert not _violations(_plan(body))
