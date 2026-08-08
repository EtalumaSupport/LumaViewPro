# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Version-gate enforcement for the protocol file format.

The loader's version dispatch is literal: it accepts CURRENT_VERSION
exactly, plus the enumerated older versions ONLY while CURRENT_VERSION
is the value the migration branch was written against. A bump of
Protocol.CURRENT_VERSION without extending that dispatch silently
orphans every protocol file a user has ever saved -- the file that
loaded yesterday refuses today with "create a new protocol".

These tests are the tripwire: bumping CURRENT_VERSION makes them fail
until the migration story is complete. The checklist the failure points
at:

1. Extend the version dispatch in Protocol.from_file so every previously
   released version still loads (the elif pins the old CURRENT_VERSION).
2. Add column recovery / defaults for the new columns, mirroring the
   existing Label recovery.
3. Add a load test here proving a previous-version file loads and
   re-saves at the new CURRENT_VERSION.
4. Update the pin below.
"""

import pytest

from modules.protocol import Protocol, ProtocolFormatError
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step


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
    assert Protocol.CURRENT_VERSION == 8


def test_previous_version_file_loads_and_upgrades(tmp_path):
    # A v7 file is a v8 file without the Label column; the loader must
    # recover it and hand back a protocol at CURRENT_VERSION semantics.
    filepath = _write_versioned_file(tmp_path, version=7, drop_columns=('Label',))
    protocol = Protocol.from_file(filepath, tiling_configs_file_loc=TILING_CONFIGS)
    steps = protocol.steps()
    assert len(steps) == 1
    assert 'Label' in steps.columns

    # Re-saving writes the CURRENT format, completing the upgrade.
    resaved = tmp_path / 'resaved.tsv'
    protocol.to_file(resaved)
    version_lines = [
        line for line in resaved.read_text().splitlines() if line.startswith('Version')
    ]
    assert version_lines == [f'Version\t{Protocol.CURRENT_VERSION}']


def test_every_enumerated_older_version_is_accepted(tmp_path):
    # The dispatch enumerates 2-7; a bump that forgets the enumeration
    # (or leaves the CURRENT_VERSION == 8 conjunct stale) breaks these
    # loads. v6/v7 share the v8 table shape closely enough to load via
    # column recovery; that acceptance must survive any bump.
    for version in (6, 7):
        filepath = _write_versioned_file(tmp_path, version=version, drop_columns=('Label',))
        protocol = Protocol.from_file(filepath, tiling_configs_file_loc=TILING_CONFIGS)
        assert len(protocol.steps()) == 1


def test_future_version_refuses_loudly(tmp_path):
    filepath = _write_versioned_file(tmp_path, version=Protocol.CURRENT_VERSION + 1)
    with pytest.raises(ProtocolFormatError):
        Protocol.from_file(filepath, tiling_configs_file_loc=TILING_CONFIGS)
