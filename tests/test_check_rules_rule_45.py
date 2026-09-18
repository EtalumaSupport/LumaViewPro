"""Tests for tools/check_rules.py's Rule 45 doc predicate.

Rule 45 requires a current ``## Status`` section on plan / audit / design
docs. The checker finds those docs by filename: ``AUDIT_*`` or any basename
containing ``_PLAN``. A session handover's basename once carried a
free-text headline, and one headline read ``THE_PLAN_APPARATUS``, so the
substring match warned about a handover that has no Status section by
design. Handovers are exempt by name, whatever their headline says.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import _is_rule_45_doc


def test_plan_and_audit_docs_match():
    assert _is_rule_45_doc('docs/FOO_PLAN_2026-09-13.md')
    assert _is_rule_45_doc('docs/FIRMWARE_PLAN.md')
    assert _is_rule_45_doc('docs/AUDIT_SOMETHING_2026-09-13.md')


def test_handover_with_plan_in_its_headline_is_not_a_plan_doc():
    assert not _is_rule_45_doc(
        'docs/SESSION_HANDOVER_2026-09-12_triage_s23_THE_PLAN_APPARATUS_DIED.md'
    )
    assert not _is_rule_45_doc('docs/SESSION_HANDOVER_2026-09-13_skillspass.md')
    assert not _is_rule_45_doc('docs/SESSION_HANDOVER_2026-09-13_skillspass_2.md')


def test_archived_docs_are_exempt():
    assert not _is_rule_45_doc('docs/completed/FOO_PLAN_2026-01-01.md')
    assert not _is_rule_45_doc('docs/completed/AUDIT_OLD_2026-01-01.md')


def test_program_reference_is_a_rule_45_doc():
    # The program reference is the one document CLAUDE.md points at for
    # program facts; an edit that leaves its Status untouched is a stale
    # fact nobody dated, so it is gated like a plan.
    assert _is_rule_45_doc('docs/PROGRAM_OVERVIEW.md')
    assert not _is_rule_45_doc('docs/completed/PROGRAM_OVERVIEW.md')
    assert not _is_rule_45_doc('docs/PROGRAM_OVERVIEW_NOTES.md')
