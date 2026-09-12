# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Capture a production module's log records under the suite's mocked logger.

The suite replaces the LVP logger with a mock, so neither caplog nor a
handler on the real logger sees a module's log lines. This swaps the
module's ``logger`` for a private real logger with a capturing handler
for the test's duration and hands back the records.
"""

import logging


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def capture_module_log(monkeypatch, module) -> list:
    """Records logged through ``module.logger`` until the test ends."""
    private = logging.getLogger(f'test.capture.{module.__name__}')
    private.propagate = False
    private.setLevel(logging.DEBUG)
    handler = _CaptureHandler()
    private.addHandler(handler)
    monkeypatch.setattr(module, 'logger', private)
    return handler.records


def messages(records) -> list[str]:
    return [r.getMessage() for r in records]
