# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An objective's record, for tests that hand an objective to the code under test.

The catalogue answers frozen ``Objective`` records, so a stand-in objective
is one of those, taken from the shipped catalogue with the fields a test
states. One description here, so a field added to the record is added once.
"""

import dataclasses

from modules.objectives_loader import Objective, ObjectiveLoader
from tests.installation_fixtures import SHIPPED_DATA

SHIPPED_OBJECTIVES = ObjectiveLoader(source_path=SHIPPED_DATA.parent).objectives


def objective(objective_id: str = '10x Oly', **fields) -> Objective:
    """The shipped ``objective_id``, with any field overridden by name."""
    return dataclasses.replace(SHIPPED_OBJECTIVES[objective_id], **fields)
