# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A derived file is written in the format its inputs were saved in.

A stitch, a z-projection, a composite and a quick-enhance of an OME-TIFF
run's frames are OME-TIFF; of a plain run's, plain TIFF. The answer is the
inputs' own, read from their names (``image_utils.derived_file_format``),
never from the settings when the build runs, so the same folder builds the
same files whenever, and from whichever host, it is built.

A plain build names and derives its files as every release before this
one did, so a folder already built builds nothing again; an OME build
states its format in its derivation, so it never shares a group or a name
with a plain build of the same images.

The builds run through ``session.post_processing`` on a run folder written
as a run of this release saves it (``write_run_folder``).
"""

import datetime

import cv2
import numpy as np
import pytest
import tifffile as tf

import modules.image_utils as image_utils
from modules.image_mode import OutputFormat
from modules.post_processing_api import Stitch, ZProject
from tests.stock_channels import LED0
from tests.test_a_derived_image_says_how_it_was_made import write_run_folder


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path / 'live')), simulate=True
    )
    yield session
    session.shutdown()


def _chain(session, folder):
    """Z-project, stitch and composite *folder* in one call; every file it made."""
    built = session.post_processing.build(
        folder, zproject=ZProject(), stitch=Stitch(), composite=True
    )
    return [path for step in built.built for path in step.result.artifact_paths]


def _record(folder) -> list[tuple[str, str]]:
    """Each derived file the folder's post-processing record names, with its derivation."""
    lines = (folder / 'protocol_post_record.tsv').read_text().splitlines()
    header = lines.index('Images') + 1
    columns = lines[header].split('\t')
    rows = [
        dict(zip(columns, line.split('\t'), strict=True)) for line in lines[header + 1 :] if line
    ]
    return [(row['Filepath'], row['Derivation']) for row in rows]


def _is_ome(path) -> bool:
    with tf.TiffFile(str(path)) as tif:
        return tif.ome_metadata is not None


@pytest.mark.parametrize('ome', [True, False], ids=['ome_run', 'plain_run'])
def test_every_build_of_a_run_is_written_in_the_runs_format(session, tmp_path, ome):
    made = _chain(session, write_run_folder(tmp_path, LED0, ome=ome))

    assert len(made) == 11, made
    for path in made:
        assert path.name.endswith('.ome.tiff') is ome, path
        assert _is_ome(path) is ome, path


# What the release before this one named and derived from a plain run of
# this folder's shape: a plain build is unchanged, so a folder built then
# builds nothing again.
_PLAIN_CHAIN = [
    ('ZProject/custom0000_BF_TA1_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0000_BF_TB1_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0000_BF_TA2_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0000_BF_TB2_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0001_Ex405_TA1_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0001_Ex405_TB1_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0001_Ex405_TA2_0000_zproj_max.tiff', 'ZProject(method=max)'),
    ('ZProject/custom0001_Ex405_TB2_0000_zproj_max.tiff', 'ZProject(method=max)'),
    (
        'Stitched-ZProject/custom0000_BF_0000_zproj_max_stitched.tiff',
        'ZProject(method=max)>Stitched(mode=quality)',
    ),
    (
        'Stitched-ZProject/custom0001_Ex405_0000_zproj_max_stitched.tiff',
        'ZProject(method=max)>Stitched(mode=quality)',
    ),
    (
        'Composite-Stitched-ZProject/custom0000_Composite_0000_zproj_max_stitched_a01a3c.tiff',
        'ZProject(method=max)>Stitched(mode=quality)'
        '>Composite(format=TIFF,Ex405.color=#0000ff,Ex405.threshold=50)',
    ),
]


def test_a_plain_runs_files_are_named_and_derived_as_they_always_were(session, tmp_path):
    folder = write_run_folder(tmp_path, LED0)
    _chain(session, folder)

    assert _record(folder) == _PLAIN_CHAIN


def test_an_ome_build_states_its_format_and_a_plain_one_does_not(session, tmp_path):
    folder = write_run_folder(tmp_path, LED0, ome=True)
    _chain(session, folder)

    derivations = dict(_record(folder))
    assert derivations['ZProject/custom0000_BF_TA1_0000_zproj_max.ome.tiff'] == (
        'ZProject(method=max,format=OME-TIFF)'
    )
    assert derivations['Stitched-ZProject/custom0000_BF_0000_zproj_max_stitched.ome.tiff'] == (
        'ZProject(method=max,format=OME-TIFF)>Stitched(mode=quality,format=OME-TIFF)'
    )


def test_the_format_is_the_one_every_input_shares(tmp_path):
    """OME-TIFF only when every input is; a mix, or a PNG or JPEG, is plain."""
    ome, plain = OutputFormat.OME_TIFF, OutputFormat.TIFF
    cases = {
        ('a.ome.tiff', 'b.ome.tiff'): ome,
        ('A.OME.TIF',): ome,
        ('exp1.5uM.ome.tiff',): ome,
        ('a.ome.tiff', 'b.tiff'): plain,
        ('exp1.5uM.tiff',): plain,
        ('a.png',): plain,
        ('a.jpg',): plain,
    }
    answered = {names: image_utils.derived_file_format(names) for names in cases}

    assert answered == cases


def test_a_composite_of_an_ome_run_is_ome_whatever_the_settings_say_when_built(session, tmp_path):
    """The files-and-readers walk's case: the setting moved to Hyperstack after an OME run."""
    folder = write_run_folder(tmp_path, LED0, ome=True)
    session.update_settings('image_output_format.sequenced', OutputFormat.HYPERSTACK)

    built = session.post_processing.composite(folder)

    assert built.new_count > 0
    assert all(_is_ome(path) and path.name.endswith('.ome.tiff') for path in built.artifact_paths)


def test_a_composite_runs_merge_is_written_in_the_format_its_frames_were(tmp_path):
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    settings = headless_settings(tmp_path, sequenced_format=OutputFormat.OME_TIFF)
    with open_composite_session(settings) as (_session, runner):
        artifact = runner.run_composite(sequence_name='ome', parent_dir=str(tmp_path)).artifact_path

    assert artifact.name.endswith('.ome.tiff'), artifact
    assert _is_ome(artifact)


def test_an_ome_folder_built_plain_before_builds_its_ome_chain_beside_it(
    session, tmp_path, monkeypatch
):
    """A folder built by the release before this one, which wrote every
    projection plain, gets an OME projection per group beside each old one
    and an OME stitch of them, and a further build makes nothing."""
    folder = write_run_folder(tmp_path, LED0, ome=True)
    # a stand-in by design: the release before this one, which wrote every
    # derived file plain whatever its inputs were.
    with monkeypatch.context() as before:
        before.setattr(image_utils, 'derived_file_format', lambda paths: OutputFormat.TIFF)
        session.post_processing.zproject(folder, method='Max')
    old = {name for name, _ in _record(folder)}

    built = session.post_processing.build(folder, zproject=ZProject(), stitch=Stitch())
    made = {
        path.relative_to(folder).as_posix()
        for step in built.built
        for path in step.result.artifact_paths
    }

    assert {name.replace('.tiff', '.ome.tiff') for name in old} <= made
    assert 'Stitched-ZProject/custom0000_BF_0000_zproj_max_stitched.ome.tiff' in made
    again = session.post_processing.build(folder, zproject=ZProject(), stitch=Stitch())
    assert [step.result.new_count for step in again.built] == [0, 0]


def test_a_deleted_projection_is_built_again_and_the_stitch_takes_it(session, tmp_path):
    folder = write_run_folder(tmp_path, LED0)
    session.post_processing.zproject(folder, method='Max')
    deleted = folder / 'ZProject' / 'custom0000_BF_TA1_0000_zproj_max.tiff'
    deleted.unlink()

    rebuilt = session.post_processing.zproject(folder, method='Max')
    stitched = session.post_processing.stitch(folder)

    assert [path.name for path in rebuilt.artifact_paths] == [deleted.name]
    assert not _is_ome(deleted)
    names = {path.name for path in stitched.artifact_paths}
    assert 'custom0000_BF_0000_zproj_max_stitched.tiff' in names


def _write_frame(path, *, ome: bool):
    image_utils.write_tiff(
        data=np.arange(64, dtype=np.uint8).reshape(8, 8),
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'BF',
            'objective': {'id': '20x Oly'},
            'exposure_time_ms': 2.0,
            'gain_db': 1.0,
            'illumination_ma': 5.0,
            'z_pos_um': 5000.0,
            'plate_pos_mm': {'x': 10.0, 'y': 10.0},
            'datetime': datetime.datetime(2026, 10, 29, 12, 0, 0),
            'well_label': '',
            'significant_bits': 8,
        },
        ome=ome,
        color=(1.0, 1.0, 1.0),
        significant_bits=8,
        save_encoding='right_aligned',
        white_bits=None,
    )


def _enhanced(session, source):
    (made,) = session.post_processing.enhance(source).created
    return made


def test_an_enhance_is_written_in_its_sources_format_beside_its_own_recipe(session, tmp_path):
    ome_source, plain_source = tmp_path / 'a.ome.tiff', tmp_path / 'a.tiff'
    _write_frame(ome_source, ome=True)
    _write_frame(plain_source, ome=False)

    first, second = _enhanced(session, ome_source), _enhanced(session, ome_source)
    plain = _enhanced(session, plain_source)

    assert (first.output_path.name, first.recipe_path.name) == (
        'a_enhanced.ome.tiff',
        'a_enhanced.recipe.json',
    )
    assert _is_ome(first.output_path)
    assert second.output_path.name == 'a_enhanced_2.ome.tiff'
    # a.tiff's base is a's too: it takes the next name free in both
    # formats, never a.ome.tiff's recipe.
    assert (plain.output_path.name, plain.recipe_path.name) == (
        'a_enhanced_3.tiff',
        'a_enhanced_3.recipe.json',
    )
    assert not _is_ome(plain.output_path)


def test_an_enhance_keeps_every_dot_of_its_sources_name_but_its_format(session, tmp_path):
    source = tmp_path / 'exp1.5uM_BF_0000.tiff'
    _write_frame(source, ome=False)

    assert _enhanced(session, source).output_path.name == 'exp1.5uM_BF_0000_enhanced.tiff'


def test_an_enhance_of_a_png_is_plain_tiff(session, tmp_path):
    source = tmp_path / 'a.png'
    cv2.imwrite(str(source), np.arange(64, dtype=np.uint8).reshape(8, 8))

    made = _enhanced(session, source)

    assert made.output_path.name == 'a_enhanced.tiff'
    assert not _is_ome(made.output_path)


def test_an_ome_output_states_the_facts_its_plain_twin_does(session, tmp_path):
    # Plated: a plain frame with no plate reads back no well label, which
    # an OME one does -- the saved-file record's own defect, not a build's.
    ome_made = _chain(session, write_run_folder(tmp_path / 'ome', LED0, ome=True, plated=True))
    plain_made = _chain(session, write_run_folder(tmp_path / 'plain', LED0, plated=True))

    differing = {}
    for ome_path, plain_path in zip(sorted(ome_made), sorted(plain_made), strict=True):
        ome_record = image_utils.read_file_record(ome_path)
        plain_record = image_utils.read_file_record(plain_path)
        # Not the time: a composite states when it was made, and the twins
        # are made one after the other.
        keys = (set(ome_record) | set(plain_record)) - {'datetime'}
        differing[plain_path.name] = {
            key: (ome_record.get(key, '<absent>'), plain_record.get(key, '<absent>'))
            for key in sorted(keys)
            if ome_record.get(key, '<absent>') != plain_record.get(key, '<absent>')
        }

    assert {name: diff for name, diff in differing.items() if diff} == {}
