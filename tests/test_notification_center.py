# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the centralized notification system."""

import sys
import threading
import time
from unittest.mock import MagicMock

# Mock heavy deps before importing
sys.modules.setdefault('userpaths', MagicMock())
_mock_lvp_logger = MagicMock()
_mock_lvp_logger.logger = MagicMock()
_mock_lvp_logger.is_thread_paused = MagicMock(return_value=False)
sys.modules.setdefault('lvp_logger', _mock_lvp_logger)

from modules.notification_center import NotificationCenter, Severity, Notification


class TestNotificationCenter:
    """Core notification bus tests."""

    def test_listener_receives_notification(self):
        nc = NotificationCenter()
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.WARNING)
        nc.warning("Test", "Title", "Message")
        assert len(received) == 1
        assert received[0].title == "Title"
        assert received[0].category == "Test"
        assert received[0].severity == Severity.WARNING

    def test_severity_filter(self):
        nc = NotificationCenter()
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.ERROR)
        nc.info("Test", "Info", "msg")
        nc.warning("Test", "Warn", "msg")
        nc.error("Test", "Err", "msg")
        assert len(received) == 1
        assert received[0].severity == Severity.ERROR

    def test_debug_listener_gets_everything(self):
        nc = NotificationCenter()
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.DEBUG)
        nc.debug("A", "D", "m")
        nc.info("A", "I", "m")
        nc.warning("A", "W", "m")
        nc.error("A", "E", "m")
        nc.critical("A", "C", "m")
        assert len(received) == 5

    def test_dedup_suppresses_within_window(self):
        nc = NotificationCenter(dedup_window_s=1.0)
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.ERROR)
        nc.error("Motor", "Timeout", "msg1")
        nc.error("Motor", "Timeout", "msg2")  # same category+title
        nc.error("Motor", "Timeout", "msg3")
        assert len(received) == 1  # only first fires

    def test_dedup_allows_after_window(self):
        nc = NotificationCenter(dedup_window_s=0.05)
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.ERROR)
        nc.error("Motor", "Timeout", "first")
        time.sleep(0.1)  # wait past window
        nc.error("Motor", "Timeout", "second")
        assert len(received) == 2

    def test_dedup_different_titles_not_suppressed(self):
        nc = NotificationCenter(dedup_window_s=10.0)
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.ERROR)
        nc.error("Motor", "Timeout", "msg")
        nc.error("Motor", "Connection Lost", "msg")  # different title
        assert len(received) == 2

    def test_remove_listener(self):
        nc = NotificationCenter()
        received = []
        listener = lambda n: received.append(n)
        nc.add_listener(listener, min_severity=Severity.WARNING)
        nc.warning("A", "B", "C")
        assert len(received) == 1
        nc.remove_listener(listener)
        nc.warning("A", "B2", "C2")
        assert len(received) == 1  # still 1, listener removed

    def test_multiple_listeners(self):
        nc = NotificationCenter()
        r1, r2 = [], []
        nc.add_listener(lambda n: r1.append(n), min_severity=Severity.WARNING)
        nc.add_listener(lambda n: r2.append(n), min_severity=Severity.ERROR)
        nc.warning("A", "Warn", "m")
        nc.error("A", "Err", "m")
        assert len(r1) == 2  # gets both
        assert len(r2) == 1  # only error

    def test_listener_exception_doesnt_break_others(self):
        nc = NotificationCenter()
        received = []

        def bad_listener(n):
            raise RuntimeError("broken")

        nc.add_listener(bad_listener, min_severity=Severity.WARNING)
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.WARNING)
        nc.warning("A", "B", "C")
        assert len(received) == 1  # second listener still called

    def test_thread_safety(self):
        nc = NotificationCenter(dedup_window_s=0.0)  # no dedup for this test
        received = []
        lock = threading.Lock()

        def safe_append(n):
            with lock:
                received.append(n)

        nc.add_listener(safe_append, min_severity=Severity.INFO)

        def producer(category):
            for i in range(20):
                nc.info(category, f"T{i}", "msg")

        threads = [threading.Thread(target=producer, args=(f"P{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 100  # 5 threads × 20

    def test_notification_is_immutable(self):
        nc = NotificationCenter()
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.INFO)
        nc.info("A", "B", "C")
        n = received[0]
        assert isinstance(n, Notification)
        assert n.severity == Severity.INFO
        assert n.category == "A"
        assert n.title == "B"
        assert n.message == "C"

    def test_clear_resets_state(self):
        nc = NotificationCenter()
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.INFO)
        nc.info("A", "B", "C")
        assert len(received) == 1
        nc.clear()
        nc.info("A", "B", "C")
        assert len(received) == 1  # listener was cleared

    def test_convenience_methods(self):
        nc = NotificationCenter()
        received = []
        nc.add_listener(lambda n: received.append(n), min_severity=Severity.DEBUG)
        nc.debug("A", "D", "m")
        nc.info("A", "I", "m")
        nc.warning("A", "W", "m")
        nc.error("A", "E", "m")
        nc.critical("A", "C", "m")
        severities = [n.severity for n in received]
        assert severities == [Severity.DEBUG, Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL]
