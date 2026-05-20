# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#636 regression: protocol writes must not silently overwrite each other.

Defense-in-depth, two layers:

1. Load-time validation in ``Protocol.from_file`` — reject protocols
   where multiple rows share ``(Name, Well, Tile, Z-Slice, Tile Group
   ID)``. Tile Group ID is included so legitimate tiled protocols
   (same Name across different tile groups) are NOT rejected.

2. Write-time defense in ``Lumascope.generate_image_save_path`` — a
   new ``tail_id_mode="if_collision"`` mode that uses the plain
   filename when no file exists and only adds a numeric suffix on
   actual collision. ``protocol_image_writer.py`` passes this mode
   so a broken protocol that slips past validation cannot lose data.
"""
import pathlib
import textwrap

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TILING_CONFIGS = REPO_ROOT / "data" / "tiling.json"


# ---------------------------------------------------------------------------
# Static-source checks — pinpoint the regression sites.
# ---------------------------------------------------------------------------

def test_lumascope_api_supports_if_collision_mode():
    # Phase 6c (2026-05-19) relocated generate_image_save_path body
    # from Lumascope to modules.image_save; the if_collision branch
    # lives there now. Path retarget per Rule 48 (c); semantic intent
    # ("generate_image_save_path supports if_collision mode") preserved.
    src = (REPO_ROOT / "modules" / "image_save.py").read_text()
    assert 'tail_id_mode == "if_collision"' in src, (
        "image_save.generate_image_save_path must support the "
        '"if_collision" tail_id_mode for write-time defense against '
        "duplicate filenames. (#636)"
    )


def test_protocol_image_writer_uses_if_collision():
    src = (REPO_ROOT / "modules" / "protocol_image_writer.py").read_text()
    assert 'tail_id_mode="if_collision"' in src, (
        "protocol_image_writer.py must pass tail_id_mode=\"if_collision\" "
        "to scope.save_image — without it, duplicate step filenames "
        "silently overwrite. (#636)"
    )
    assert 'tail_id_mode=None' not in src, (
        "protocol_image_writer.py must not pass tail_id_mode=None on the "
        "save_image call (regressed to overwrite-prone behavior)."
    )


# ---------------------------------------------------------------------------
# Functional check — Protocol.from_file rejects duplicates.
# ---------------------------------------------------------------------------

def _build_tsv(step_rows: str) -> str:
    """Build a minimal v5 protocol TSV with the given step rows."""
    header = textwrap.dedent("""\
        LumaViewPro Protocol
        Version\t5
        Period\t60.0
        Duration\t1.0
        Labware\t96 well plate
        Capture Root\t

        Steps
        Name\tX\tY\tZ\tAuto_Focus\tColor\tFalse_Color\tIllumination\tGain\tAuto_Gain\tExposure\tSum\tObjective\tWell\tTile\tZ-Slice\tCustom Step\tTile Group ID\tZ-Stack Group ID\tAcquire\tVideo Config\tStim_Config
        """)
    return header + step_rows


_VIDEO_CFG = "{'duration': 30}"
_STIM_CFG = (
    "{'Blue': {'enabled': False, 'illumination': 250, 'frequency': 1, "
    "'pulse_width': 10, 'pulse_count': 100}, "
    "'Green': {'enabled': False, 'illumination': 200.0, 'frequency': 1, "
    "'pulse_width': 10, 'pulse_count': 100}, "
    "'Red': {'enabled': False, 'illumination': 350, 'frequency': 1, "
    "'pulse_width': 10, 'pulse_count': 100}}"
)


def _step_row(name, well, tile, z_slice, tile_group, x, y, z):
    cells = [
        name, str(x), str(y), str(z), 'False', 'BF', 'False', '50.0',
        '1.8', 'False', '4.0', '1', '20x Oly', well, tile, str(z_slice),
        'True', str(tile_group), '-1', 'image', _VIDEO_CFG, _STIM_CFG,
    ]
    return '\t'.join(cells) + '\n'


def test_load_rejects_duplicate_filename_keys(tmp_path):
    """Two rows with the same (Name, Well, Tile, Z-Slice, Tile Group ID)
    must be rejected. This is the user-reported #636 case where the
    second image silently overwrote the first.
    """
    from modules.protocol import Protocol

    rows = ''
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 60.1, 34.6, 5001.7)  # dup
    tsv = tmp_path / "dup.tsv"
    tsv.write_text(_build_tsv(rows))

    with pytest.raises(ValueError, match=r"duplicate"):
        Protocol.from_file(
            file_path=tsv,
            tiling_configs_file_loc=TILING_CONFIGS,
        )


def test_load_accepts_same_name_in_different_tile_groups(tmp_path):
    """Same Name + Well + Tile + Z-Slice across DIFFERENT Tile Group
    IDs is the legitimate tiled-acquisition pattern and must NOT be
    rejected. Tile Group ID is the disambiguator.
    """
    from modules.protocol import Protocol

    rows = ''
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('_PC_TA1', 'A1', '', -1, 1, 60.1, 34.6, 5001.7)
    rows += _step_row('_PC_TA1', 'A1', '', -1, 2, 73.7, 34.6, 5031.2)
    tsv = tmp_path / "tiled.tsv"
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(
        file_path=tsv,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    assert proto.num_steps() == 3


def test_load_accepts_unique_steps(tmp_path):
    """Sanity: unique steps load fine."""
    from modules.protocol import Protocol

    rows = ''
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('_PC_TA2', 'A2', '', -1, 0, 47.9, 34.6, 4972.9)
    tsv = tmp_path / "ok.tsv"
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(
        file_path=tsv,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    assert proto.num_steps() == 2
