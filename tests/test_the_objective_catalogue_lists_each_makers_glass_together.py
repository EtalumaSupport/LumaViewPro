# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective catalogue lists each maker's objectives together, by magnification.

The objective question offers the catalogue's list as its choices, and the
picker shows them in that order under a header per maker. The makers come
in the order each first appears in the catalogue file; within a maker the
objectives rise by magnification, and two of one magnification keep their
order in the file.
"""

import json

from _lumascope.modules.objectives_loader import ObjectiveLoader
from _lumascope.modules.path_utils import resolve_data_file


def test_each_makers_objectives_are_listed_together_by_magnification():
    loader = ObjectiveLoader()
    listed = loader.get_objectives_list()
    objectives = loader.objectives
    file_order = list(json.loads(resolve_data_file('objectives.json').read_text('utf-8')))

    makers_in_file = list(dict.fromkeys(objectives[o].manufacturer for o in file_order))
    makers_listed = list(dict.fromkeys(objectives[o].manufacturer for o in listed))
    assert makers_listed == makers_in_file

    for maker in makers_in_file:
        theirs = [o for o in listed if objectives[o].manufacturer == maker]
        first = listed.index(theirs[0])
        assert listed[first : first + len(theirs)] == theirs, f'{maker} is split: {listed}'
        assert theirs == sorted(
            theirs, key=lambda o: (objectives[o].magnification, file_order.index(o))
        ), f'{maker} out of order: {theirs}'


def test_the_shipped_catalogue_lists_olympus_then_meiji():
    objectives = ObjectiveLoader().objectives
    assert list(dict.fromkeys(o.manufacturer for o in objectives.values())) == [
        'Olympus',
        'Meiji',
    ]
