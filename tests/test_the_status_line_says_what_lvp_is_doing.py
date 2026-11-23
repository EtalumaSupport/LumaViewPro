# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The status line says what LumaViewPro is doing, from the session's live work.

A recording building its stack refused captures for 393 s with nothing on
screen saying why (bundle SN12075-2026-11-10-144813; Eric: "it still
wouldn't let me do anything"). What a person waits on was written into the
window title by whichever GUI piece started it, and work no GUI piece
started -- a post-processing build -- had no writer at all. The status line
now reads ``ScopeSession.live_work`` each tick, so every piece of work the
session reports is on it, in the words the work carries.
"""

import ast

from _lumascope.modules import live_work
from tests.ast_seams import find_def
from ui.shader import work_stage, work_title
from ui.ui_helpers import FIGURE_SPACE, work_item_text


def test_each_kind_of_work_is_on_the_status_line_in_its_own_words():
    work = tuple(live_work.WorkItem(kind, f'the {kind} work') for kind in live_work.WorkKind)

    line = work_title(work, hold=None)

    missing = [item.name for item in work if work_item_text(item) not in line]
    assert missing == [], f'work the status line does not say: {missing}'


def test_with_nothing_being_done_the_status_line_is_its_readouts_alone():
    assert work_title((), hold=None) == ''


def test_a_figure_follows_the_work_it_counts():
    building = live_work.WorkItem(
        live_work.WorkKind.POST_PROCESSING,
        "Building a recording's hyperstack in V",
        progress_pct=40,
    )
    writing = live_work.WorkItem(live_work.WorkKind.RUN_FILES, "A run's images", left=12)

    assert work_title((building, writing), hold=None) == (
        " | Building a recording's hyperstack in V (40%) | A run's images (12 left)"
    )


def test_the_tick_reads_the_sessions_live_work():
    # pin-justified: the tick runs under Kivy's clock, with no headless
    # seam; what it hands work_title is the contract.
    update = find_def('ui/shader.py', '_update_status_bar', class_name='ShaderViewer')
    calls = [
        node
        for node in ast.walk(update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'work_title'
    ]
    assert len(calls) == 1
    (argument,) = calls[0].args
    reads = {
        ast.unparse(node.value)
        for node in ast.walk(update)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == ast.unparse(argument)
    }
    assert reads == {'ctx.session.live_work.work'}, reads


def test_a_held_still_is_said_after_the_work_with_its_seconds_counting_down():
    still = live_work.WorkItem(live_work.WorkKind.STILL, 'Capturing a summed still')

    assert work_title((still,), hold=('the captured still', 1.43)) == (
        f' | Capturing a summed still | Showing the captured still ({FIGURE_SPACE}1.4s)'
    )
    assert work_title((), hold=('the composite', 10.0)) == ' | Showing the composite (10.0s)'


def test_the_record_names_the_hold_without_its_seconds():
    # The tick's record is compared on this stage, so a hold's start and its
    # end are each one record and its ticking seconds none.
    assert work_stage((), hold=('the captured still', 1.43)) == 'Showing the captured still'
    assert work_stage((), hold=None) is None


def test_the_tick_reads_the_displays_named_hold():
    # pin-justified: the tick runs under Kivy's clock, with no headless
    # seam; what it hands work_title and work_stage is the contract.
    update = find_def('ui/shader.py', '_update_status_bar', class_name='ShaderViewer')
    handed = {
        ast.unparse(keyword.value)
        for node in ast.walk(update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {'work_title', 'work_stage'}
        for keyword in node.keywords
        if keyword.arg == 'hold'
    }
    assert handed == {'hold'}, handed
    reads = {
        ast.unparse(node.value)
        for node in ast.walk(update)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'hold'
    }
    assert reads == {'ctx.scope_display_thread.named_hold'}, reads
