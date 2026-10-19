# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A build is made from every image its records name, derived or raw, or not at all.

A derived image the post-processing record names and the folder lacks --
a projection deleted after it was made -- fails the group that needs it,
named with the build that makes it again, as a raw frame the run recorded
does; the other groups build. Every image of a group gone is a build's
output removed whole: the group is named and not built, and the answer is
complete. An output whose file is gone is not already made: it is built
again, and the record keeps one row for it.

Each test runs the real builds through ``session.post_processing`` on the
run folder of ``test_a_derived_image_says_how_it_was_made``: a 2x2 tiled
scan, a 3-slice Z-stack at each tile, BF and Blue.
"""

from __future__ import annotations

import pathlib

import pytest

from modules.exceptions import PostProcessingFailedError
from modules.protocol_post_record import ProtocolPostRecord
from tests.test_a_derived_image_says_how_it_was_made import (  # noqa: F401 -- fixtures
    run_folder,
    session,
)


def _record_rows(folder: pathlib.Path, name: str) -> int:
    """How many rows of the folder's post-processing record name *name*."""
    record = ProtocolPostRecord.from_file(folder / ProtocolPostRecord.DEFAULT_FILENAME)
    with record:
        return int((record.records()['Filepath'].map(lambda path: path.name) == name).sum())


def _projections(session, folder) -> list[pathlib.Path]:
    return sorted(session.post_processing.zproject(folder, method='Max').artifact_paths)


def test_a_stitch_missing_one_projection_fails_that_mosaic_and_builds_the_rest(session, run_folder):
    """One Blue projection deleted: the Blue mosaic of projections is not
    built -- never a mosaic with a hole -- and the failure names the
    projection and the build that restores it; BF's is built."""
    gone = next(path for path in _projections(session, run_folder) if '_Blue_' in path.name)
    gone.unlink()

    with pytest.raises(PostProcessingFailedError) as failed:
        session.post_processing.stitch(run_folder)

    (error,) = [error for error in failed.value.errors if 'Stitched-ZProject' in error]
    assert gone.name in error
    assert 'build Z-Projection again' in error
    mosaics = sorted(pathlib.Path(path).name for path in failed.value.produced_paths)
    assert 'custom0000_BF_0000_zproj_max_stitched.tiff' in mosaics
    assert not any('Blue_0000_zproj_max_stitched' in name for name in mosaics)
    assert not (
        run_folder / 'Stitched-ZProject' / 'custom0001_Blue_0000_zproj_max_stitched.tiff'
    ).exists()


def test_a_deleted_output_is_built_again_and_recorded_once(session, run_folder):
    """A projection deleted since it was made is not already made: z-project
    again makes it, and the record holds one row for it."""
    gone = _projections(session, run_folder)[0]
    gone.unlink()

    again = session.post_processing.zproject(run_folder, method='Max')

    assert again.new_count == 1
    assert [path.name for path in again.artifact_paths] == [gone.name]
    assert gone.is_file()
    assert _record_rows(run_folder, gone.name) == 1


def test_a_build_whose_inputs_were_all_removed_is_complete_and_names_them(session, run_folder):
    """Every projection deleted and the projections' folder selected: stitch
    builds nothing, answers complete, and names the two groups removed --
    neither refused for want of tiles nor failed."""
    for path in _projections(session, run_folder):
        path.unlink()

    stitched = session.post_processing.stitch(run_folder / 'ZProject')

    assert stitched.new_count == 0
    assert '2 group(s) not built: their Z-Projection images were removed from the folder.' in (
        stitched.message
    )


def test_a_selected_folder_missing_one_projection_fails_that_mosaic(session, run_folder):
    """The projections' folder selected, one projection deleted: the group
    that needs it fails, named, as with the whole run folder selected."""
    gone = next(path for path in _projections(session, run_folder) if '_Blue_' in path.name)
    gone.unlink()

    with pytest.raises(PostProcessingFailedError) as failed:
        session.post_processing.stitch(run_folder / 'ZProject')

    assert any(gone.name in error for error in failed.value.errors)


def test_a_deleted_composite_is_not_counted_among_the_inputs_left_out(session, run_folder):
    """Twelve composites made and one deleted: a stitch, which takes no
    composites, says it left out the eleven in the folder -- a file the
    record names and the folder lacks is not one the build left out."""
    composites = sorted(session.post_processing.composite(run_folder).artifact_paths)
    composites[0].unlink()

    stitched = session.post_processing.stitch(run_folder)

    assert [(item.kind, item.count) for item in stitched.excluded_inputs] == [('composite', 11)]
