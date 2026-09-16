# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for how a failed executor task is reported in the log.

Two separate facts get logged when a background task raises: the RAISE record
(written where the exception is caught, carrying the traceback) and the
NOTIFICATION record (written by the notification center, recording what the
user was shown). Both land at ERROR, and they reach the log by different
routes -- the executor writes through lvp_logger, which the suite replaces
with one shared MagicMock, while the notification center holds a real
stdlib logger. Reading only one of the two sees half the picture.

Three defects these lock out:

  * The raise record used to read 'Uncaught Thread Exception in <name> Worker'
    while the exception was caught and returned to the caller -- nothing ever
    reached threading.excepthook. Only lvp_logger's excepthook may claim an
    exception escaped its thread, and a triage grep counting escapes has to be
    able to trust that.
  * The two records must stay distinguishable. If the raise record also says
    'task failed', one failure produces two near-identical ERROR lines and any
    count of them doubles.
  * concurrent.futures.CancelledError subclasses Exception, so a by-contract
    cancel fell into the general handler and was reported at ERROR as a
    failure before being logged at DEBUG as a cancel.
"""

import logging
from concurrent.futures import CancelledError

from lvp_logger import logger
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

ESCAPE_WORDING = ('uncaught', 'unhandled exception', 'escaped')

NOTIFICATION_LOGGER = 'LVP.notifications'


def _run_failing_task(action, caplog) -> list[str]:
    """Drive one task through raise-capture and the executor epilogue.

    Mirrors the worker's own sequence (dequeue, run, epilogue) without
    starting the worker thread, so the assertions cannot race a live lane.
    """
    logger.reset_mock()
    caplog.clear()
    executor = SequentialIOExecutor(name='TEST')
    task = IOTask(action)
    task.set_name(executor.executor_name)

    # task_done() in the epilogue must balance a real get(), or the queue
    # raises instead of the code under test.
    executor.queue.put(task)
    executor.queue.get()

    with caplog.at_level(logging.DEBUG, logger=NOTIFICATION_LOGGER):
        result, exception = task.run()
        executor._on_task_done(task, result, exception)

    records = [str(call.args[0]) for call in logger.error.call_args_list if call.args]
    records += [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.ERROR]
    return records


def test_raise_record_does_not_claim_the_exception_escaped(caplog):
    def boil_kettle():
        raise ValueError('kettle boiled dry')

    records = _run_failing_task(boil_kettle, caplog)
    assert records, 'a failed task must leave at least one ERROR record'
    for record in records:
        lowered = record.lower()
        for phrase in ESCAPE_WORDING:
            assert phrase not in lowered, (
                f'A caught exception is reported as {phrase!r}, which is what '
                f"lvp_logger's excepthook means: {record!r}"
            )


def test_raise_record_names_the_action_and_the_exception_type(caplog):
    def grind_beans():
        raise ValueError('burr jammed')

    records = _run_failing_task(grind_beans, caplog)
    named = [r for r in records if 'grind_beans' in r and 'ValueError' in r]
    assert len(named) == 1, (
        'Exactly one ERROR record should name the failing action and its '
        f'exception type; got {records!r}'
    )


def test_failure_produces_one_task_failed_record_not_two(caplog):
    def pull_shot():
        raise ValueError('portafilter empty')

    records = _run_failing_task(pull_shot, caplog)

    # The property is that the two records do not say the same thing -- the
    # raise record names the symbol and the exception type for a developer,
    # the notification record carries what the user was shown. Asserting a
    # count of one fixed phrase only worked while both happened to use it;
    # comparing the two directly survives either being reworded.
    raise_records = [r for r in records if 'raised' in r]
    notification_records = [r for r in records if r not in raise_records]
    assert len(raise_records) == 1, f'exactly one raise record expected: {records!r}'
    assert len(notification_records) == 1, f'exactly one user-facing record expected: {records!r}'
    assert 'pull_shot' in raise_records[0], (
        f'the raise record must still name the symbol for a developer: {records!r}'
    )
    # The notification logs as '[category] title: message'. The category is
    # the dedup identity and is deliberately the symbol -- it is never shown
    # to anyone. What follows it is what the user read, and that is the half
    # that must not be spelled from the callable.
    user_facing = notification_records[0].split('] ', 1)[1]
    assert 'pull_shot' not in user_facing, (
        f'the user-facing wording is a Python symbol: {user_facing!r}'
    )


def test_by_contract_cancel_is_not_reported_as_a_failure(caplog):
    def steam_milk():
        raise CancelledError()

    records = _run_failing_task(steam_milk, caplog)
    assert records == [], (
        f'A cancel is by-contract, not a failure, and must log no ERROR: {records!r}'
    )
