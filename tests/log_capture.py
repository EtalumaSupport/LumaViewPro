# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Read one production module's lines out of ``caplog``.

The suite's loggers are real and propagate to the root, so ``caplog`` holds
every line a test's code logged. A test about one module's lines reads them
by the file that wrote them, whichever of the module's loggers it used.
"""


def records_of(caplog, module) -> list:
    """The records ``caplog`` holds that *module*'s own code logged."""
    return [r for r in caplog.records if r.pathname == module.__file__]


def messages(records) -> list[str]:
    return [r.getMessage() for r in records]
