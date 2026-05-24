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
import re
import textwrap

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TILING_CONFIGS = REPO_ROOT / 'data' / 'tiling.json'


# ---------------------------------------------------------------------------
# Static-source checks — pinpoint the regression sites.
# ---------------------------------------------------------------------------


def test_lumascope_api_supports_if_collision_mode():
    # Phase 6c (2026-05-19) relocated generate_image_save_path body
    # from Lumascope to modules.image_save; the if_collision branch
    # lives there now. Path retarget per Rule 48 (c); semantic intent
    # ("generate_image_save_path supports if_collision mode") preserved.
    #
    # Quote-agnostic: ruff format may emit either single- or double-
    # quoted string literals depending on the formatter pass that last
    # touched the line (commit 4a5a2422 converted production to single
    # quotes on 2026-05-22). Assert the comparison + the value rather
    # than coupling to the specific quote character.
    src = (REPO_ROOT / 'modules' / 'image_save.py').read_text()
    assert re.search(r"""tail_id_mode\s*==\s*['"]if_collision['"]""", src), (
        'image_save.generate_image_save_path must support the '
        '"if_collision" tail_id_mode for write-time defense against '
        'duplicate filenames. (#636)'
    )


def test_protocol_image_writer_uses_if_collision():
    # Quote-agnostic: same rationale as
    # test_lumascope_api_supports_if_collision_mode above.
    src = (REPO_ROOT / 'modules' / 'protocol_image_writer.py').read_text()
    assert re.search(r"""tail_id_mode\s*=\s*['"]if_collision['"]""", src), (
        'protocol_image_writer.py must pass tail_id_mode="if_collision" '
        'to scope.save_image -- without it, duplicate step filenames '
        'silently overwrite. (#636)'
    )
    assert not re.search(r'tail_id_mode\s*=\s*None\b', src), (
        'protocol_image_writer.py must not pass tail_id_mode=None on the '
        'save_image call (regressed to overwrite-prone behavior).'
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
        name,
        str(x),
        str(y),
        str(z),
        'False',
        'BF',
        'False',
        '50.0',
        '1.8',
        'False',
        '4.0',
        '1',
        '20x Oly',
        well,
        tile,
        str(z_slice),
        'True',
        str(tile_group),
        '-1',
        'image',
        _VIDEO_CFG,
        _STIM_CFG,
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
    tsv = tmp_path / 'dup.tsv'
    tsv.write_text(_build_tsv(rows))

    with pytest.raises(ValueError, match=r'duplicate'):
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
    tsv = tmp_path / 'tiled.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(
        file_path=tsv,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    assert proto.num_steps() == 3


def test_load_warns_on_cross_tgid_filename_collision(tmp_path, monkeypatch):
    """The customer's #636 case: 4 rows share (Name, Well, Tile, Z-Slice)
    across DIFFERENT Tile Group IDs. Strict dedup PASSES (TGID is part
    of the key, so all 4 tuples are unique). The softer check must
    detect the cross-TGID collision and fire a user-facing notification
    upfront so the user can fix their Name format BEFORE running the
    scan, not discover renamed files afterward.
    """
    from modules import protocol as protocol_mod
    from modules.protocol import Protocol

    captured_notifications: list = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            captured_notifications.append((category, title, message))

    monkeypatch.setattr(protocol_mod, 'notifications', _RecordingNotifier())

    rows = ''
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('_PC_TA1', 'A1', '', -1, 1, 60.1, 34.6, 5001.7)
    rows += _step_row('_PC_TA1', 'A1', '', -1, 2, 73.7, 34.6, 5031.2)
    rows += _step_row('_PC_TA1', 'A1', '', -1, 3, 46.0, 47.9, 4953.4)
    tsv = tmp_path / 'cross_tgid.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(
        file_path=tsv,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    assert proto.num_steps() == 4, (
        'Cross-TGID duplicates must NOT be rejected (the legitimate tiled-'
        'acquisition pattern still loads).'
    )
    assert len(captured_notifications) == 1, (
        f'Cross-TGID duplicate filenames must fire exactly one '
        f'notifications.warning at load time. Captured: {captured_notifications}'
    )
    category, title, message = captured_notifications[0]
    assert 'Tile Group ID' in message, (
        'Notification message must point the user at Tile Group ID as '
        'the actionable fix (per Rule 28 -- direct + action-focused).'
    )
    assert 'preserve' in message.lower() or 'no' in message.lower(), (
        'Notification must reassure the user that data is intact (the rename is data-preserving).'
    )


def test_load_no_warning_when_no_collisions(tmp_path, monkeypatch):
    """Happy path: unique (Name, Well, Tile, Z-Slice) tuples produce
    no notification."""
    from modules import protocol as protocol_mod
    from modules.protocol import Protocol

    captured_notifications: list = []

    class _RecordingNotifier:
        def warning(self, category, title, message, **kw):
            captured_notifications.append((category, title, message))

    monkeypatch.setattr(protocol_mod, 'notifications', _RecordingNotifier())

    rows = ''
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('_PC_TA2', 'A2', '', -1, 0, 47.9, 34.6, 4972.9)
    tsv = tmp_path / 'clean.tsv'
    tsv.write_text(_build_tsv(rows))

    Protocol.from_file(
        file_path=tsv,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    assert captured_notifications == [], (
        f'No notification should fire for a well-formed protocol. '
        f'Captured: {captured_notifications}'
    )


def test_load_accepts_unique_steps(tmp_path):
    """Sanity: unique steps load fine."""
    from modules.protocol import Protocol

    rows = ''
    rows += _step_row('_PC_TA1', 'A1', '', -1, 0, 46.5, 34.6, 4972.9)
    rows += _step_row('_PC_TA2', 'A2', '', -1, 0, 47.9, 34.6, 4972.9)
    tsv = tmp_path / 'ok.tsv'
    tsv.write_text(_build_tsv(rows))

    proto = Protocol.from_file(
        file_path=tsv,
        tiling_configs_file_loc=TILING_CONFIGS,
    )
    assert proto.num_steps() == 2


# ---------------------------------------------------------------------------
# Functional check -- if_collision warns the user when it triggers.
# ---------------------------------------------------------------------------


class _MinimalScope:
    """Bare-bones scope stand-in for generate_image_save_path tests.

    `generate_image_save_path` reads `scope.motion._last_turret_position`
    only when engineering mode is active; tests run with `_app_ctx.ctx`
    unset, so the engineering-mode branch is never entered and the
    `motion` attribute is not exercised here.
    """


def _patch_image_save_logger(monkeypatch):
    """Replace ``image_save.logger`` (a MagicMock under LVP's conftest mock)
    with a real captured-record list. Returns the list; tests assert
    against record messages directly.
    """
    from modules import image_save

    captured: list = []

    class _RecordingLogger:
        def warning(self, msg, *args, **kwargs):
            captured.append(('WARNING', msg % args if args else msg))

        def info(self, msg, *args, **kwargs):
            captured.append(('INFO', msg % args if args else msg))

    monkeypatch.setattr(image_save, 'logger', _RecordingLogger())
    return captured


def test_if_collision_emits_warning_on_rename(tmp_path, monkeypatch):
    """When a filename collides under if_collision mode, the user must
    be told via a WARNING log. The duplicate-key class (#636 customer
    case where TGIDs collide on filename) is silently data-preserving
    via the rename suffix; without the warning the user has no signal
    that their protocol Name format is producing collisions.
    """
    from modules.image_save import generate_image_save_path

    base = tmp_path / '_PC_TA1.tiff'
    base.write_bytes(b'')

    captured = _patch_image_save_logger(monkeypatch)

    path = generate_image_save_path(
        scope=_MinimalScope(),
        save_folder=tmp_path,
        file_root='_PC_TA1',
        append='',
        tail_id_mode='if_collision',
        output_format='TIFF',
    )

    assert path.name == '_PC_TA1_000001.tiff', (
        f'if_collision should append _000001 on first collision; got {path.name}'
    )
    assert any(
        level == 'WARNING' and 'filename collision' in msg and 'Tile Group ID' in msg
        for level, msg in captured
    ), (
        'if_collision must emit a WARNING naming the rename and pointing '
        'the user at Tile Group ID as the likely cause. (#636 follow-up). '
        f'Captured: {captured}'
    )


def test_if_collision_no_warning_when_no_collision(tmp_path, monkeypatch):
    """Happy path: no warning when the filename is fresh."""
    from modules.image_save import generate_image_save_path

    captured = _patch_image_save_logger(monkeypatch)

    path = generate_image_save_path(
        scope=_MinimalScope(),
        save_folder=tmp_path,
        file_root='_PC_TA1',
        append='',
        tail_id_mode='if_collision',
        output_format='TIFF',
    )

    assert path.name == '_PC_TA1.tiff'
    assert not any('filename collision' in msg for _, msg in captured), (
        f'if_collision must not warn when there is no actual collision. Captured: {captured}'
    )
