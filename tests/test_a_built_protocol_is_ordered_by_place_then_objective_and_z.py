# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol's steps are ordered for travel: by place, then by objective and Z at each place.

``Protocol.optimize_step_ordering`` is the order a tile or z-stack build leaves
a protocol in. The places keep the order each first appears in; at one place
the steps of one objective sit together, rising in Z; steps that tie keep the
order they were made in; steps with no X/Y are one place of their own.

The protocol is made as a user's would be: added on the simulated turret
scope (an objective change needs one), saved, and edited in the saved file,
then loaded through the session.
"""

from __future__ import annotations

import csv

from tests.test_a_command_for_absent_motion_hardware_is_refused import make_session  # noqa: F401

# Each step: X, Y, Z, objective, and a gain that names it. Places in the order
# they first appear: A (30, 40), the steps with no X/Y, then B (10, 20).
_STEPS = [
    ('30.0', '40.0', '5.0', '4x Oly', '1.0'),
    ('', '', '3.0', '4x Oly', '2.0'),
    ('10.0', '20.0', '6.0', '4x Oly', '3.0'),
    ('30.0', '40.0', '4.0', '10x Oly', '4.0'),
    ('30.0', '40.0', '4.0', '4x Oly', '5.0'),
    ('30.0', '40.0', '4.0', '4x Oly', '6.0'),
    ('', '', '1.0', '4x Oly', '7.0'),
    ('10.0', '20.0', '2.0', '4x Oly', '8.0'),
]


def _protocol_file(session, path):
    protocol = session.create_empty_protocol()
    session.set_channel_acquire('BF', 'image')
    for i in range(len(_STEPS)):
        session.scope.motion.move_xy(x_um=10000.0 + 5000.0 * i)
        session.add_step(protocol)
    protocol.to_file(path)

    with open(path, encoding='utf-8', newline='') as fp:
        rows = list(csv.reader(fp, delimiter='\t'))
    header = next(i for i, row in enumerate(rows) if row[:1] == ['Name'])
    at = [rows[header].index(column) for column in ('X', 'Y', 'Z', 'Objective', 'Gain')]
    for row, values in zip(rows[header + 1 :], _STEPS, strict=True):
        for column, value in zip(at, values, strict=True):
            row[column] = value
    with open(path, 'w', encoding='utf-8', newline='') as fp:
        csv.writer(fp, delimiter='\t', lineterminator='\n').writerows(rows)
    return path


def test_the_steps_are_ordered_by_place_then_objective_and_z(make_session, tmp_path):
    session = make_session('LS850T')
    protocol = session.load_protocol(_protocol_file(session, tmp_path / 'unordered.tsv'))

    protocol.optimize_step_ordering()

    order = [float(gain) for gain in protocol.steps()['Gain']]
    place_a, unplaced, place_b = order[:4], order[4:6], order[6:]
    assert sorted(place_a) == [1.0, 4.0, 5.0, 6.0], order
    # At place A the 10x step is its own block, at either end; the 4x steps
    # rise in Z, and the two at one Z keep the order they were made in.
    assert 4.0 in (place_a[0], place_a[-1]), order
    assert [gain for gain in place_a if gain != 4.0] == [5.0, 6.0, 1.0], order
    assert unplaced == [7.0, 2.0], order
    assert place_b == [8.0, 3.0], order
