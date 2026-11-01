# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Version-gate enforcement for the protocol file format.

The loader accepts CURRENT_VERSION and every version from 2 below it, so
a bump of Protocol.CURRENT_VERSION keeps every protocol file a user has
saved loading -- provided each column the bump adds is recovered for a
file without it; a file missing a current column refuses at construction.

These tests are the tripwire: bumping CURRENT_VERSION makes them fail
until the migration story is complete. The checklist the failure points
at:

1. Keep every previously released version loading in Protocol.from_file.
2. Add column recovery / defaults for the new columns, mirroring the
   existing Label recovery.
3. Add a load test here proving a previous-version file loads and
   re-saves at the new CURRENT_VERSION.
4. Update the pin below.
"""

import pytest

from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import _build_protocol, _make_step
from tests.protocol_fixtures import protocol_from_file


def _write_versioned_file(tmp_path, version, drop_columns=()):
    """Save a real protocol, then rewrite it as an older-version file."""
    protocol = _build_protocol([_make_step(color='BF', label='gate')])
    filepath = tmp_path / f'protocol_v{version}.tsv'
    protocol.to_file(filepath)

    lines = filepath.read_text().splitlines()
    out = []
    drop_indices = None
    for line in lines:
        cells = line.split('\t')
        if cells[0] == 'Version':
            out.append(f'Version\t{version}')
            continue
        if 'Name' in cells and 'Color' in cells:
            # The step-table header row: drop the requested columns here
            # and remember their positions for the data rows below.
            drop_indices = [cells.index(c) for c in drop_columns if c in cells]
            drop_indices.sort(reverse=True)
        if drop_indices and len(cells) > max(drop_indices):
            for idx in drop_indices:
                del cells[idx]
            out.append('\t'.join(cells))
        else:
            out.append(line)
    filepath.write_text('\n'.join(out) + '\n')
    return filepath


def test_current_version_pin():
    # This pin is the version-gate tripwire, not trivia: the loader's
    # migration branch and the fixtures below were written against this
    # exact value. Bumping CURRENT_VERSION must route you through the
    # module docstring's checklist; updating this line is its LAST step.
    assert Protocol.CURRENT_VERSION == 10


def test_previous_version_file_loads_and_upgrades(tmp_path):
    # A v9 file is a v10 file without the Z-Stack Offset column; the loader
    # gives a step in no z-stack 0 (the recovery of a stack's offsets is
    # tests/test_zstack_group_autofocus.py's), and hands back a protocol at
    # CURRENT_VERSION semantics.
    filepath = _write_versioned_file(tmp_path, version=9, drop_columns=('Z-Stack Offset',))
    protocol = protocol_from_file(
        filepath,
    )
    steps = protocol.steps()
    assert len(steps) == 1
    assert list(steps['Z-Stack Offset']) == [0.0]

    # Re-saving writes the CURRENT format, completing the upgrade.
    resaved = tmp_path / 'resaved.tsv'
    protocol.to_file(resaved)
    version_lines = [
        line for line in resaved.read_text().splitlines() if line.startswith('Version')
    ]
    assert version_lines == [f'Version\t{Protocol.CURRENT_VERSION}']


def test_every_enumerated_older_version_is_accepted(tmp_path):
    # v6-v9 share the v10 table shape closely enough to load via column
    # recovery; that acceptance must survive any bump. A v8 file without
    # Channel Group ID and Turret Slot gives its steps -1 in each, no Add
    # and no slot recorded.
    for version, dropped in (
        (6, ('Label', 'Channel Group ID', 'Turret Slot', 'Z-Stack Offset')),
        (7, ('Label', 'Channel Group ID', 'Turret Slot', 'Z-Stack Offset')),
        (8, ('Channel Group ID', 'Turret Slot', 'Z-Stack Offset')),
        (9, ('Z-Stack Offset',)),
    ):
        filepath = _write_versioned_file(tmp_path, version=version, drop_columns=dropped)
        protocol = protocol_from_file(
            filepath,
        )
        steps = protocol.steps()
        assert len(steps) == 1
        if version == 8:
            assert list(steps['Channel Group ID']) == [-1]
            assert list(steps['Turret Slot']) == [-1]


def test_future_version_refuses_loudly(tmp_path):
    filepath = _write_versioned_file(tmp_path, version=Protocol.CURRENT_VERSION + 1)
    with pytest.raises(ProtocolFormatError):
        protocol_from_file(
            filepath,
        )
