# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The interlocks a stage guards motion with, each with the heading the person reads.

Here, not in a driver or in ``modules/exceptions.py``, because both name
them: the driver whose board refuses a motion command raises its refusal
by reason, and the API answers it, and a driver may not import
``modules/``. Standard library only, so every layer may import it.
"""

from types import MappingProxyType

# Each interlock's reason, and its title. Read-only: the one list, which the
# driver's fault, the API's refusals and the run's admission all read.
INTERLOCK_TITLES = MappingProxyType(
    {
        'lid_open': 'Lid Open',
        'stage_unpowered': 'No Stage Power',
    }
)

# The reason a command the interlock stopped before anything moved is
# refused with, by the interlock: a different fact from the driver's fault,
# which may follow motion, so a string of its own.
REFUSED_BEFORE_MOTION = MappingProxyType(
    {reason: f'{reason}_before_motion' for reason in INTERLOCK_TITLES}
)
