# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A committed text entry runs its handler once.

Kivy's single-line TextInput dispatches ``on_text_validate`` on Enter and
then, because ``text_validate_unfocus`` defaults True, clears its own focus,
which dispatches ``on_focus``. So a box that binds BOTH events to the same
handler runs it twice per Enter with identical text. At the stage-position
boxes that was a second move command queued to the driver; at the layer
boxes a second full settings apply; everywhere else a duplicate record in
``gui_interactions.log`` that a reader cannot tell from a real double entry.

The focus-loss binding alone covers Enter and click-away. These tests read
the kv as text (the suite mocks Kivy; it does not instantiate widgets) and
pin two things:

- no widget block binds ``on_text_validate`` and ``on_focus`` to the same
  call, in ``ui/lumaviewpro.kv`` or in the kv string inside
  ``ui/advanced_settings.py``;
- no focus-commit box turns ``text_validate_unfocus`` off, which is the
  Kivy default the single binding rests on.

Whitespace tolerant: the kv mixes tabs and spaces, so indentation is
measured the way Kivy's parser measures it.
"""

from __future__ import annotations

import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
KV_SOURCES = (
    REPO / 'ui' / 'lumaviewpro.kv',
    # Holds its kv in a Builder.load_string literal; python lines never carry
    # a `<event>:` property at line start, so the same scan applies.
    REPO / 'ui' / 'advanced_settings.py',
)

_VALIDATE = re.compile(r'^[ \t]*on_text_validate:\s*(?P<call>.+?)\s*$')
_FOCUS_COMMIT = re.compile(r'^[ \t]*on_focus:\s*if not self\.focus:\s*(?P<call>.+?)\s*$')
_UNFOCUS_OFF = re.compile(r'^[ \t]*text_validate_unfocus:\s*False\s*$')
# A widget starts a block: `TextInput:`, `MyWidget:` -- a bare capitalised
# name and a colon, no value.
_WIDGET_HEADER = re.compile(r'^[ \t]*[A-Z]\w*:\s*$')


def _indent(line: str) -> int:
    """Indentation width the way kivy/lang/parser.py measures it (tab = 4)."""
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _property_blocks(text: str) -> list[list[tuple[int, str]]]:
    """Each widget's own property lines, as (line_no, line) lists.

    A block is the run of lines at one depth following a widget header at
    the shallower depth, ending at the next line shallower than that depth
    or the next widget header at the same depth (a sibling child widget).
    Comment lines are dropped so a commented-out binding cannot match.
    """
    lines = text.split('\n')
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] | None = None
    depth = None
    for no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        d = _indent(raw)
        if _WIDGET_HEADER.match(raw):
            if current:
                blocks.append(current)
            current, depth = [], None
            continue
        if current is None:
            continue
        if depth is None:
            depth = d
        if d < depth:
            blocks.append(current)
            current, depth = None, None
            continue
        if d == depth:
            current.append((no, raw))
    if current:
        blocks.append(current)
    return blocks


def _double_bound(text: str) -> list[tuple[int, str]]:
    """(line, call) for every on_text_validate whose block also focus-commits the same call."""
    found = []
    for block in _property_blocks(text):
        focus_calls = {m.group('call') for _, ln in block if (m := _FOCUS_COMMIT.match(ln))}
        for no, ln in block:
            m = _VALIDATE.match(ln)
            if m and m.group('call') in focus_calls:
                found.append((no, m.group('call')))
    return found


def _focus_commit_blocks(text: str) -> list[list[tuple[int, str]]]:
    return [b for b in _property_blocks(text) if any(_FOCUS_COMMIT.match(ln) for _, ln in b)]


def test_no_text_box_binds_enter_and_focus_loss_to_the_same_handler():
    for src in KV_SOURCES:
        offenders = _double_bound(src.read_text())
        assert not offenders, (
            f'{src.name}: {len(offenders)} box(es) bind on_text_validate and on_focus to '
            f'the same call, so one Enter runs the handler twice: '
            + ', '.join(f'line {no} {call}' for no, call in offenders)
        )


def test_every_focus_commit_box_keeps_kivy_unfocusing_on_enter():
    """The single binding rests on Enter dropping focus; a box that turns that
    off would stop committing on Enter altogether."""
    for src in KV_SOURCES:
        text = src.read_text()
        for block in _focus_commit_blocks(text):
            off = [no for no, ln in block if _UNFOCUS_OFF.match(ln)]
            assert not off, (
                f'{src.name}: text_validate_unfocus: False at line {off[0]} in a box that '
                f'commits on focus loss; Enter would no longer commit'
            )


def test_the_scan_sees_the_focus_commit_boxes():
    """Guards the two tests above against going vacuous if the kv shape changes."""

    def commits(src):
        return sum(1 for ln in src.read_text().split('\n') if _FOCUS_COMMIT.match(ln))

    kv_count = commits(KV_SOURCES[0])
    inline_count = commits(KV_SOURCES[1])
    assert kv_count >= 29, f'expected at least 29 focus-commit boxes in the kv, saw {kv_count}'
    assert inline_count >= 3, (
        f'expected at least 3 focus-commit boxes in the advanced settings kv, saw {inline_count}'
    )
