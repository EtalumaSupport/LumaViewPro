# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A latest-wins post keeps one pending callback per key, running the newest function.

The live display posted each update to the UI thread as a fresh callback,
so a UI thread held up for four days held every one of them: 8.7 GB,
replayed in full when it ran again. ``schedule_ui_latest`` keeps at most one
callback pending per key, and a post while it is pending replaces what it
will run.
"""

from __future__ import annotations

import pytest

from modules.kivy_utils import UiDispatcher, schedule_ui_latest
from modules.scope_session import ScopeSession


class _Held:
    """A UI thread that is held up: callbacks queue until the test runs them."""

    def __init__(self):
        self.queue = []

    def schedule(self, func, timeout):
        self.queue.append(func)

    def run(self):
        queue, self.queue = self.queue, []
        for func in queue:
            func(0)

    def install(self):
        ScopeSession.set_ui_dispatcher(UiDispatcher(schedule=self.schedule, thread=None))


@pytest.fixture
def held():
    ui = _Held()
    ui.install()
    yield ui
    ScopeSession.set_ui_dispatcher(None)


def test_posts_while_held_leave_one_callback_running_the_newest(held):
    ran = []
    for i in range(100):
        schedule_ui_latest('k', lambda dt, i=i: ran.append(i))
    assert len(held.queue) == 1
    held.run()
    assert ran == [99]


def test_a_post_after_the_key_ran_is_delivered_again(held):
    ran = []
    schedule_ui_latest('k', lambda dt: ran.append('first'))
    held.run()
    schedule_ui_latest('k', lambda dt: ran.append('second'))
    held.run()
    assert ran == ['first', 'second']


def test_a_callback_left_from_before_a_swap_runs_nothing_and_order_holds(held):
    # The swap empties the pending entries; the callback queued for the old
    # one stays in the queue. Run in queue order, it must not take the
    # newer entry under its key ahead of the post made between them.
    ran = []
    schedule_ui_latest('k', lambda dt: ran.append('A'))
    stale = held.queue[:]
    held.queue.clear()
    held.install()
    held.queue.extend(stale)
    schedule_ui_latest('j', lambda dt: ran.append('C'))
    schedule_ui_latest('k', lambda dt: ran.append('B'))
    held.run()
    assert ran == ['C', 'B']


def test_an_inline_dispatcher_lets_a_function_post_under_its_own_key():
    # Each post is delivered at once, so the function posts again from
    # inside its own run; the store's lock is not held across it.
    ran = []

    def again(dt):
        ran.append(len(ran))
        if len(ran) < 3:
            schedule_ui_latest('k', again)

    ScopeSession.set_ui_dispatcher(UiDispatcher(schedule=lambda f, t: f(t), thread=None))
    try:
        schedule_ui_latest('k', again)
    finally:
        ScopeSession.set_ui_dispatcher(None)
    assert ran == [0, 1, 2]


def test_with_no_ui_thread_it_runs_inline():
    ran = []
    schedule_ui_latest('k', lambda dt: ran.append(dt))
    assert ran == [0]
