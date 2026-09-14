# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motor driver's refusal of an acceleration limit reaches the caller.

The API's setter used to wrap the driver call in a bare ``except Exception:
pass``, on the stated grounds that legacy boards lack the acceleration-limits
feature. That reason did not hold: the driver's own limit read already answers
with a default when the firmware has no AMAX / DMAX query, so a legacy board
COMPLETES the call rather than raising out of it. What the handler actually
absorbed was the driver's range rejection -- leaving a scope configured from a
value its firmware had refused, with nothing said to the caller, the log, or a
headless client.

Both directions are pinned here, because the fix has to keep the legacy case
working while letting the real refusal through: a driver that raises is
reported, and a driver that simply completes is still fine.
"""

# Heavy deps are mocked by tests/conftest.py at module-import time.

import pytest

from modules.lumascope_api import Lumascope


@pytest.fixture
def scope():
    return Lumascope(simulate=True)


def test_a_driver_refusal_is_raised_to_the_caller(scope, monkeypatch):
    """The rejection propagates instead of being swallowed."""

    def refuse(val_pct):
        raise ValueError(f'Acceleration limit of {val_pct}% is out of bounds.')

    monkeypatch.setattr(scope._motion_driver, 'set_acceleration_limits', refuse)

    with pytest.raises(ValueError):
        scope.motion.set_acceleration_limit(val_pct=500)


def test_firmware_without_the_command_still_completes(scope, monkeypatch):
    """The legacy case is absorbed in the driver, so the API call succeeds.

    Pinned so a future reading of this setter does not restore the handler to
    protect a case that never needed protecting here.
    """
    calls = []
    monkeypatch.setattr(
        scope._motion_driver, 'set_acceleration_limits', lambda val_pct: calls.append(val_pct)
    )

    scope.motion.set_acceleration_limit(val_pct=50)

    assert calls == [50]
