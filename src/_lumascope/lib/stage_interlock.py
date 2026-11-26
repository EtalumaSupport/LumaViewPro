# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The interlocks a stage guards motion with, each with the heading the person reads.

Here, not in a driver or in ``src/_lumascope/modules/exceptions.py``, because both name
them: the driver whose board refuses a motion command raises its refusal
by reason, and the API answers it, and a driver may not import
``src/_lumascope/modules/``. Standard library only, so every layer may import it.
"""

from types import MappingProxyType

# Each interlock's reason, with its title and the person's words for a
# motion it refused before anything moved. Read-only: the one list, which the
# driver's fault, the API's refusals and the run's admission all read.
_INTERLOCKS = {
    'lid_open': (
        'Lid Open',
        "The microscope's lid is open. Close it to move or home the stage.",
    ),
    'stage_unpowered': (
        'No Stage Power',
        "The stage has no power. Check the stage's power supply, then home.",
    ),
}
INTERLOCK_TITLES = MappingProxyType({reason: title for reason, (title, _) in _INTERLOCKS.items()})
INTERLOCK_WORDS = MappingProxyType({reason: words for reason, (_, words) in _INTERLOCKS.items()})

# The reason a command the interlock stopped before anything moved is
# refused with, by the interlock: a different fact from the driver's fault,
# which may follow motion, so a string of its own.
REFUSED_BEFORE_MOTION = MappingProxyType(
    {reason: f'{reason}_before_motion' for reason in INTERLOCK_TITLES}
)
# Each of those refusals' interlock, so a refusal finds its words.
INTERLOCK_OF_REFUSAL = MappingProxyType(
    {refusal: reason for reason, refusal in REFUSED_BEFORE_MOTION.items()}
)

# The reason a move the interlock's stop halted on its way ends with, by the
# interlock: the stage stopped where it was, its position known, so neither
# the refusal nor the fault, and a string of its own. Only the lid stops a
# moving stage: the stage's power is read before a command, never during one.
STOPPED_MID_MOVE = MappingProxyType({'lid_open': 'lid_open_mid_move'})
