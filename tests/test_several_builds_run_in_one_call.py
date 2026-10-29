# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Several builds in one call: ``session.post_processing.build``.

The member runs the chosen builds in the order that keeps the data honest
-- z-project, stitch, composite, video -- each taking only what the build
before it carried forward, so the call makes the chain's product and its
inputs. Every chosen build's settings are checked before any build starts;
a build with nothing to do is skipped and named; any other refusal or
failure stops the call as that build's own outcome.

Each test runs the real builds on the run folder of
``test_a_derived_image_says_how_it_was_made``: a 2x2 tiled scan, a 3-slice
Z-stack at each tile, BF and LED 0 (``LED0``) -- 24 images.
"""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import pytest

from modules import image_utils
from modules.exceptions import (
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from modules.layer_record import release_color
from modules.post_processing_api import Stitch, Video, ZProject
from tests.test_a_derived_image_says_how_it_was_made import (  # noqa: F401 -- fixtures
    run_folder,
    session,
)
from tests.stock_channels import LED0, LED1


def _made(result) -> dict[PostProcessingOperation, list[str]]:
    return {
        step.operation: sorted(
            str(pathlib.Path(path).relative_to(step.result.output_root))
            for path in step.result.artifact_paths
        )
        for step in result.built
    }


def test_zproject_stitch_composite_make_the_chains_product_and_its_inputs(session, run_folder):
    """Eight projections, the two stitched projections and the one stitched,
    projected composite -- not the 36 files the three builds make one by
    one -- and a video on a single-scan run is skipped and named."""
    result = session.post_processing.build(
        run_folder, zproject=ZProject(), stitch=Stitch(), composite=True, video=Video()
    )

    made = _made(result)
    assert list(made) == [
        PostProcessingOperation.Z_PROJECTION,
        PostProcessingOperation.STITCH,
        PostProcessingOperation.COMPOSITE,
    ]
    assert len(made[PostProcessingOperation.Z_PROJECTION]) == 8
    assert made[PostProcessingOperation.STITCH] == [
        'Stitched-ZProject/custom0000_BF_0000_zproj_max_stitched.tiff',
        f'Stitched-ZProject/custom0001_{LED0}_0000_zproj_max_stitched.tiff',
    ]
    (product,) = made[PostProcessingOperation.COMPOSITE]
    assert product.startswith('Composite-Stitched-ZProject/')
    assert [skip.operation for skip in result.skipped] == [PostProcessingOperation.VIDEO]
    assert 'Video requires multiple time points' in result.skipped[0].message


def test_the_same_call_again_makes_nothing_new(session, run_folder):
    """Made again, each build finds its outputs already made and the next
    still takes them: nothing new, and nothing from an earlier derivation."""
    chain = {'zproject': ZProject(), 'stitch': Stitch(), 'composite': True}
    session.post_processing.build(run_folder, **chain)

    again = session.post_processing.build(run_folder, **chain)

    assert [step.result.new_count for step in again.built] == [0, 0, 0]


def test_an_earlier_derivation_in_the_folder_is_no_input_of_the_chain(session, run_folder):
    """A Sum projection and per-slice mosaics already in the folder: the
    chain's stitch takes only its own Max projections."""
    session.post_processing.zproject(run_folder, method='Sum')
    session.post_processing.stitch(run_folder)

    result = session.post_processing.build(run_folder, zproject=ZProject(), stitch=Stitch())

    assert _made(result)[PostProcessingOperation.STITCH] == [
        'Stitched-ZProject/custom0000_BF_0000_zproj_max_stitched.tiff',
        f'Stitched-ZProject/custom0001_{LED0}_0000_zproj_max_stitched.tiff',
    ]


def _add_an_ordinary_step(folder: pathlib.Path) -> None:
    """One more Add beside the tiled stack: BF and LED 0 at one place, no
    tiles, no slices -- an image each that z-project and stitch pass over."""
    protocol = folder / 'run.tsv'
    rows = []
    for color, false_color in (('BF', 'False'), (LED0, 'True')):
        rows.append(
            [
                f'custom0002_{color}', '12.0000', '12.0000', '5000.0', 'False', color,
                false_color, '5.0', '1.0', 'False', '2.0', '1', '20x Oly', '', '', '',
                'True', '2', '', 'image', '"{""duration"": 5, ""fps"": 30}"', '"{}"', 'True',
                'custom0002',
            ]
        )  # fmt: skip
    protocol.write_text(protocol.read_text() + ''.join('\t'.join(row) + '\n' for row in rows))
    record = folder / 'protocol_record.tsv'
    lines = record.read_text()
    rng = np.random.default_rng(19)
    for offset, (color, _) in enumerate((('BF', 'False'), (LED0, 'True'))):
        name = f'custom0002_{color}'
        image_utils.write_tiff(
            data=(rng.random((64, 64)) * 80).astype(np.uint8),
            file_loc=folder / f'{name}_0000.tiff',
            metadata={
                'pixel_size_um': 0.5,
                'channel': color,
                'objective': {'id': '20x Oly'},
                'exposure_time_ms': 2.0,
                'gain_db': 1.0,
                'illumination_ma': 5.0,
                'z_pos_um': 5000.0,
                'plate_pos_mm': {'x': 12.0, 'y': 12.0},
                'datetime': datetime.datetime(2026, 10, 18, 12, 1, 0),
                'well_label': '',
                'significant_bits': 8,
            },
            ome=False,
            color=release_color(color),
            significant_bits=8,
            save_encoding='right_aligned',
            white_bits=None,
        )
        # Steps 24 and 25, after the stack's 24.
        lines += f'{name}_0000.tiff\t{name}\t{24 + offset}\t0\t2026-10-18 12:01:0{offset}.000001\t1\t0.0\n'
    record.write_text(lines)


def test_an_image_a_build_passes_over_reaches_the_next(session, run_folder):
    """An ordinary Add step's two images beside the tiled stack: z-project
    and stitch have nothing to combine in them, and the chain's composite
    still makes that step's composite, as composite alone does."""
    _add_an_ordinary_step(run_folder)

    result = session.post_processing.build(
        run_folder, zproject=ZProject(), stitch=Stitch(), composite=True
    )

    composites = _made(result)[PostProcessingOperation.COMPOSITE]
    assert any(name.startswith('Composite/custom0002_Composite') for name in composites), composites
    assert any(name.startswith('Composite-Stitched-ZProject/') for name in composites)


def test_a_bad_setting_on_the_last_build_refuses_before_the_first_writes(session, run_folder):
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.build(
            run_folder, zproject=ZProject(), video=Video(frame_rate_fps=0)
        )

    assert refused.value.reason == 'invalid_setting'
    assert refused.value.operation == PostProcessingOperation.VIDEO
    assert not (run_folder / 'ZProject').exists()


def test_a_call_that_chooses_no_build_is_refused(session, run_folder):
    with pytest.raises(PostProcessingRefusedError) as refused:
        session.post_processing.build(run_folder)

    assert refused.value.reason == 'invalid_setting'
    assert refused.value.operation == PostProcessingOperation.POST_PROCESSING


def test_a_build_that_fails_stops_the_call_as_its_own_outcome(session, run_folder):
    """A raw slice missing: z-project fails that tile's projection, and the
    call stops there, raised as z-project's failure -- the stitch after it
    would build a mosaic short of a tile."""
    next(run_folder.glob(f'custom0001_{LED0}_TA1_Z1_*.tiff')).unlink()

    with pytest.raises(PostProcessingFailedError) as failed:
        session.post_processing.build(run_folder, zproject=ZProject(), stitch=Stitch())

    assert failed.value.operation == PostProcessingOperation.Z_PROJECTION
    assert not (run_folder / 'Stitched-ZProject').exists()


def test_the_live_work_names_the_build_the_call_is_on(session, run_folder):
    seen = set()

    def progress(percent, detail=None):
        seen.update(item.name for item in session.post_processing.work())

    session.post_processing.build(
        run_folder, zproject=ZProject(), stitch=Stitch(), on_progress=progress
    )

    assert any(name.startswith('Building 1 of 2: Z-Projection') for name in seen), seen
    assert any(name.startswith('Building 2 of 2: Stitch') for name in seen), seen


def _stills_and_a_recording():
    """BF and LED 1 stills and an LED 0 recording, all at one place."""
    from tests.test_a_late_write_records_its_frame import _step

    steps = []
    for index, (color, acquire) in enumerate((('BF', 'image'), (LED1, 'image'), (LED0, 'video'))):
        step = _step('P1', index, x=20.0, gain=1.0)
        step['Color'] = color
        step['Acquire'] = acquire
        steps.append(step)
    return steps


def test_a_recordings_frames_pass_the_stills_builds_to_the_video(tmp_path):
    """A run's recording, saved as frames beside two stills, built composite
    then video in one call: the composite leaves none of the recording's
    frames out -- it carries them forward -- and the video is made from
    them."""
    from tests.test_a_late_write_records_its_frame import _protocol
    from tests.test_composite_run_e2e import headless_settings, open_composite_session

    settings = headless_settings(tmp_path)
    settings['video_as_frames'] = True
    with open_composite_session(settings) as (session, runner):
        pending = runner.start_single_scan(
            protocol=_protocol(_stills_and_a_recording()), parent_dir=str(tmp_path / 'runs')
        )
        assert pending.result(timeout=60).status == 'completed'
        assert pending.files.result(timeout=60) is not None, "the run's files never finished"
        (record,) = (tmp_path / 'runs').rglob('protocol_record.tsv')

        result = session.post_processing.build(record.parent, composite=True, video=Video())

    composite, _video = result.built
    assert composite.result.excluded_inputs == ()
    (video,) = _made(result)[PostProcessingOperation.VIDEO]
    assert video.startswith(f'Video/P1_{LED0}_10xOly_Z0_0000_video'), video
