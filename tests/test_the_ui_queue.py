# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every post to the UI thread goes through one ordered, bounded queue.

The live display posted each update to the UI thread as a fresh callback,
so a UI thread held up for four days held every one of them: 8.7 GB,
replayed in full when it ran again. The dispatcher's queue
(``modules/kivy_utils.py``) holds two kinds of entry: an event
(``schedule_ui``) runs once, in post order; a state (``schedule_ui_latest``)
keeps one entry per key, moved to the back with its newest value, so a
held-up UI thread holds one entry per state however long it waits, and a
value never runs ahead of an event posted before it.
"""

from __future__ import annotations

import threading

import pytest

from modules.exceptions import BlockingCallRefusedError
from modules.kivy_utils import UiDispatcher, schedule_ui, schedule_ui_latest, ui_backlog
from modules.scope_session import ScopeSession


class _Held:
    """A UI thread that is held up: what the dispatcher is asked to run waits until the test runs it."""

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


def test_posts_of_one_state_leave_one_entry_running_the_newest(held):
    ran = []
    for i in range(1000):
        schedule_ui_latest('k', lambda dt, i=i: ran.append(i))
    assert ui_backlog() == 1
    held.run()
    assert ran == [999]


def test_a_state_posted_again_after_an_event_runs_after_it_once(held):
    ran = []
    schedule_ui_latest('k', lambda dt: ran.append('old'))
    schedule_ui(lambda dt: ran.append('event'))
    schedule_ui_latest('k', lambda dt: ran.append('new'))
    held.run()
    assert ran == ['event', 'new']


def test_a_run_held_up_holds_one_entry_per_state_whatever_its_events(held):
    # A run's shape: each step's event between its saved frame, live frame
    # and position. A state that kept its first slot whenever an event came
    # after it held one entry per step per state.
    ran = []
    for step in range(40):
        schedule_ui(lambda dt, step=step: ran.append(('step_started', step)))
        for state in ('saved_frame', 'live_frame', 'xy'):
            schedule_ui_latest(state, lambda dt, state=state, step=step: ran.append((state, step)))
    assert ui_backlog() == 40 + 3
    held.run()
    assert ran == [('step_started', step) for step in range(40)] + [
        ('saved_frame', 39),
        ('live_frame', 39),
        ('xy', 39),
    ]


def test_events_from_several_threads_each_run_once_in_their_post_order(held):
    ran = []
    start = threading.Barrier(4)

    def post(thread_id):
        start.wait()
        for i in range(250):
            schedule_ui(lambda dt, i=i: ran.append((thread_id, i)))

    threads = [threading.Thread(target=post, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    held.run()
    assert len(ran) == 1000
    for thread_id in range(4):
        assert [i for t, i in ran if t == thread_id] == list(range(250))


def test_a_post_made_while_the_queue_drains_runs_in_the_next_drain(held):
    ran = []

    def first(dt):
        ran.append('first')
        schedule_ui(lambda dt: ran.append('posted during'))

    schedule_ui(first)
    held.run()
    assert ran == ['first']
    assert ui_backlog() == 1
    held.run()
    assert ran == ['first', 'posted during']


def test_a_raising_entry_re_raises_and_loses_nothing_behind_it(held):
    # The entry raises after posting an event and a newer value of the
    # state queued behind it. The event runs, then the newer value once;
    # the stale value, requeued at the head, would run ahead of the event or
    # instead of the newer value.
    ran = []

    def raises(dt):
        schedule_ui(lambda dt: ran.append('event'))
        schedule_ui_latest('k', lambda dt: ran.append('new'))
        raise ValueError('the entry raised')

    schedule_ui(raises)
    schedule_ui_latest('k', lambda dt: ran.append('stale'))
    schedule_ui(lambda dt: ran.append('behind'))
    with pytest.raises(ValueError, match='the entry raised'):
        held.run()
    assert ran == []
    held.run()
    assert ran == ['behind', 'event', 'new']


def test_a_swap_hands_the_waiting_entries_to_the_new_dispatcher_in_order(held):
    ran = []
    schedule_ui(lambda dt: ran.append('event'))
    schedule_ui_latest('k', lambda dt: ran.append('state'))
    stale, held.queue = held.queue, []
    held.install()
    for func in stale:  # the old dispatcher's drain, still in its queue
        func(0)
    assert ran == []
    held.run()
    assert ran == ['event', 'state']


def test_a_swap_to_none_runs_the_waiting_entries_here_and_reports_a_raise(held, monkeypatch):
    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(notifications, 'report_outcome', lambda ex, **kw: reported.append(ex))
    ran = []

    def raises(dt):
        raise ValueError('the entry raised')

    schedule_ui(lambda dt: ran.append('first'))
    schedule_ui(raises)
    schedule_ui_latest('k', lambda dt: ran.append('state'))
    ScopeSession.set_ui_dispatcher(None)
    assert ran == ['first', 'state']
    assert [str(ex) for ex in reported] == ['the entry raised']
    assert ui_backlog() == 0
    held.run()
    assert ran == ['first', 'state']


def test_a_swap_from_inside_a_drain_is_refused(held):
    refused = []

    def swaps(dt):
        try:
            ScopeSession.set_ui_dispatcher(None)
        except BlockingCallRefusedError as ex:
            refused.append(ex.reason)

    schedule_ui(swaps)
    held.run()
    assert refused == ['swap_in_delivery']
    assert ui_backlog() == 0


def test_an_inline_dispatcher_lets_a_function_post_under_its_own_key():
    # Each drain is delivered at once, so the function posts again from
    # inside its own run; neither lock is held across the next drain.
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
    schedule_ui(lambda dt: ran.append(dt))
    assert ran == [0, 0]
