# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A text box commits through CommitInput's one event, and nothing else.

A box that committed on ``on_focus`` committed on every focus loss, edited
or not: a click in and out of the Z box during a move moved Z to the
number it showed. A box that also bound ``on_text_validate`` ran its
handler twice per Enter, since Kivy dispatches the validate and then
clears focus. ``CommitInput`` (ui/ui_helpers.py) owns both answers -- it
fires ``on_commit`` once, for an edit or an Enter -- so a box that commits
binds ``on_commit`` and nothing else.

These tests read the kv as text (the suite mocks Kivy; it does not
instantiate widgets) and pin:

- no kv source binds ``on_focus`` or ``on_text_validate``, in any
  spelling, so a new box cannot commit around CommitInput;
- no CommitInput turns ``text_validate_unfocus`` off, which its Enter
  note rests on;
- the scan sees the boxes, so neither test above goes vacuous.

Whitespace tolerant: the kv mixes tabs and spaces, so indentation is
measured the way Kivy's parser measures it.
"""

from __future__ import annotations

import pathlib
import re

from tests.gui_logging_census import LOAD_STRING_MODULES


REPO = pathlib.Path(__file__).resolve().parent.parent
# Derived, not hand-copied: the census already registers every module holding a
# Builder.load_string block, and a module registered there but missing here
# would be scanned for logging coverage and not for a commit that escapes
# CommitInput. One list, so a new kv source cannot satisfy one guard while
# escaping the other. Python lines never carry a `<event>:` property at line
# start, so the same scan applies to both forms.
KV_SOURCES = (REPO / 'ui' / 'lumaviewpro.kv', *(REPO / m for m in LOAD_STRING_MODULES))

_FOCUS_OR_VALIDATE = re.compile(r'^[ \t]*(?P<event>on_focus|on_text_validate)\s*:')
_COMMIT = re.compile(r'^[ \t]*on_commit\s*:')
_UNFOCUS_OFF = re.compile(r'^[ \t]*text_validate_unfocus:\s*False\s*$')
# A widget starts a block: `TextInput:`, `MyWidget:` -- a bare capitalised
# name and a colon, no value.
_WIDGET_HEADER = re.compile(r'^[ \t]*(?P<name>[A-Z]\w*):\s*$')


def _indent(line: str) -> int:
    """Indentation width the way kivy/lang/parser.py measures it (tab = 4)."""
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _property_blocks(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """Each widget's name and own property lines, as (name, [(line_no, line)]).

    A block is the run of lines at one depth following a widget header at
    the shallower depth, ending at the next line shallower than that depth
    or the next widget header at the same depth (a sibling child widget).
    Comment lines are dropped so a commented-out binding cannot match.
    """
    blocks: list[tuple[str, list[tuple[int, str]]]] = []
    name: str | None = None
    current: list[tuple[int, str]] | None = None
    depth = None
    for no, raw in enumerate(text.split('\n'), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        d = _indent(raw)
        header = _WIDGET_HEADER.match(raw)
        if header:
            if current:
                blocks.append((name, current))
            name, current, depth = header.group('name'), [], None
            continue
        if current is None:
            continue
        if depth is None:
            depth = d
        if d < depth:
            blocks.append((name, current))
            name, current, depth = None, None, None
            continue
        if d == depth:
            current.append((no, raw))
    if current:
        blocks.append((name, current))
    return blocks


def _commit_input_blocks(text: str) -> list[list[tuple[int, str]]]:
    return [lines for name, lines in _property_blocks(text) if name == 'CommitInput']


def test_no_box_commits_on_focus_loss_or_enter_around_commit_input():
    for src in KV_SOURCES:
        offenders = [
            (no, m.group('event'))
            for no, ln in enumerate(src.read_text().split('\n'), start=1)
            if not ln.strip().startswith('#') and (m := _FOCUS_OR_VALIDATE.match(ln))
        ]
        assert not offenders, (
            f'{src.name}: {len(offenders)} binding(s) commit around CommitInput -- a focus '
            f'loss is not an edit, and an Enter already reaches on_commit: '
            + ', '.join(f'line {no} {event}' for no, event in offenders)
        )


def test_every_commit_input_keeps_kivy_unfocusing_on_enter():
    """CommitInput's Enter note is spent on the unfocus that follows the
    validate; a box that turns the unfocus off would carry the note to the
    next click away and commit an unedited box there."""
    for src in KV_SOURCES:
        for block in _commit_input_blocks(src.read_text()):
            off = [no for no, ln in block if _UNFOCUS_OFF.match(ln)]
            assert not off, (
                f'{src.name}: text_validate_unfocus: False at line {off[0]} in a CommitInput; '
                f'its Enter note would outlive the Enter'
            )


def test_the_scan_sees_the_commit_boxes():
    """Guards the two tests above against going vacuous if the kv shape changes:
    every committing box is found as a CommitInput block that binds on_commit."""

    def committing_boxes(src):
        return sum(
            1
            for block in _commit_input_blocks(src.read_text())
            if any(_COMMIT.match(ln) for _, ln in block)
        )

    kv_count = committing_boxes(KV_SOURCES[0])
    inline_count = committing_boxes(KV_SOURCES[1])
    assert kv_count >= 27, f'expected at least 27 committing boxes in the kv, saw {kv_count}'
    assert inline_count >= 2, (
        f'expected at least 2 committing boxes in the advanced settings kv, saw {inline_count}'
    )
