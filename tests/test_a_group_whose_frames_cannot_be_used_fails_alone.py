# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A post-processing build fails only the group whose input frames cannot be used (#722).

A frame that will not decode (a write cut short), or frames of sizes or pixel
types that cannot share one output, raised out of the builder, and nothing in
the build's group loop caught it: the groups after it were never built, and the
raise was tifffile's own, naming no file. The image reader now says "this frame
cannot be read" as one type, naming the file, the canvas and pixel-type checks
say the same of their frames, and the loop answers that type as the one group's
failure: the build goes on, and ends "incomplete", naming the frame.

A hyperstack the builder refuses part-way through its write leaves nothing at
its output name, and a video none of whose frames could be read is a failed
group, not a recorded file that does not exist.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import image_utils
from modules.exceptions import InputFramesUnusableError, PostProcessingFailedError
from modules.stack_builder import StackBuilder
from modules.stitcher import Stitcher
from modules.video_builder import VideoBuilder
from modules.zprojector import ZProjector
from tests.test_capture_collision_policy import TILING_CONFIGS
from tests.test_post_processing_stays_inside_its_folder import _COLUMNS, _step


def _write_frame(path: pathlib.Path, *, x: float, well: str, seed: int, size: int = 256) -> None:
    # Random pixels, so an LZW strip barely compresses and a half truncation
    # cuts into the pixels while the header still reads.
    pixels = np.random.default_rng(seed).integers(0, 256, size=(size, size), dtype=np.uint8)
    image_utils.write_tiff(
        data=pixels,
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'BF',
            'objective': '20x',
            'exposure_time_ms': 2.0,
            'gain_db': 1.0,
            'illumination_ma': 5.0,
            'z_pos_um': 5000.0,
            'plate_pos_mm': {'x': x, 'y': 11.24},
            'datetime': '2026:10:07 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': well,
            'significant_bits': 8,
        },
        ome=False,
        color='BF',
        significant_bits=8,
        save_encoding='right_aligned',
    )


def _truncate(path: pathlib.Path) -> None:
    with open(path, 'r+b') as f:
        f.truncate(path.stat().st_size // 2)


def _run_folder(tmp_path, frames: list[dict]) -> pathlib.Path:
    """A run folder as a run of this release saves it, one step per frame.

    Each frame names its well, x, z-slice, scan and (for a tiled well) its
    tile; its file is ``<name>_<scan>.tiff``.
    """
    folder = tmp_path / 'live' / 'ProtocolData' / '20261007_120000'
    folder.mkdir(parents=True)
    steps, record = [], []
    names = {}
    for seed, frame in enumerate(frames):
        name = frame['name']
        if name not in names:
            names[name] = len(steps)
            step = _step(
                name,
                f'{5000.0 + 10 * frame.get("z_slice", 0)}',
                str(frame.get('z_slice', 0)),
                frame['well'],
            )
            step[1] = str(frame['x'])
            step[14] = frame.get('tile', '')
            step[17] = str(frame.get('tile_group', -1))
            steps.append(step)
        scan = frame.get('scan', 0)
        filename = f'{name}_{scan:04d}.tiff'
        record.append(
            f'{filename}\t{name}\t{names[name]}\t{scan}\t'
            f'2026-10-07 12:{scan:02d}:{seed:02d}.000001\t1\t0.0\n'
        )
        _write_frame(
            folder / filename,
            x=frame['x'],
            well=frame['well'],
            seed=seed,
            size=frame.get('size', 256),
        )
    (folder / 'run.tsv').write_text(
        'LumaViewPro Protocol\nVersion\t8\nPeriod\t1.0\nDuration\t0.5\n'
        'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
        + '\t'.join(_COLUMNS)
        + '\n'
        + ''.join('\t'.join(step) + '\n' for step in steps)
    )
    (folder / 'protocol_record.tsv').write_text(
        'LumaViewPro Protocol Execution Record\nVersion\t3\nProtocol File\trun.tsv\n'
        'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
        + ''.join(record)
    )
    return folder


def _z_stacks(tmp_path, **a1_slice1):
    return _run_folder(
        tmp_path,
        [
            {'name': f'{well}_BF_Z{z}', 'well': well, 'x': 14.38 + 9 * w, 'z_slice': z,
             **(a1_slice1 if (well, z) == ('A1', 1) else {})}
            for w, well in enumerate(('A1', 'A2'))
            for z in (0, 1)
        ],
    )  # fmt: skip


def _tiles(tmp_path):
    return _run_folder(
        tmp_path,
        [
            {'name': f'{well}_BF_T{t}', 'well': well, 'x': 14.38 + 9 * w + 0.1 * t,
             'tile': f'T{t}', 'tile_group': w}
            for w, well in enumerate(('A1', 'A2'))
            for t in (0, 1)
        ],
    )  # fmt: skip


def _outputs(folder: pathlib.Path) -> set[str]:
    made = {p for p in folder.rglob('*') if p.is_file() and p.parent != folder}
    return {p.name for p in made}


class TestTheReader:
    def test_a_frame_cut_short_is_refused_naming_it(self, tmp_path):
        frame = tmp_path / 'cut.tiff'
        _write_frame(frame, x=1.0, well='A1', seed=0)
        _truncate(frame)

        with pytest.raises(InputFramesUnusableError) as refused:
            image_utils.load_pixels(frame, collapse_legacy_false_color=False)

        assert refused.value.reason == 'unreadable'
        assert 'cut.tiff' in str(refused.value)

    def test_a_missing_frame_is_refused_naming_it(self, tmp_path):
        with pytest.raises(InputFramesUnusableError) as refused:
            image_utils.read_image_geometry(tmp_path / 'gone.tiff')

        assert refused.value.reason == 'unreadable'
        assert 'gone.tiff' in str(refused.value)


# parametrize-justified: each case is a different builder over its own folder shape
@pytest.mark.parametrize(
    ('folder_of', 'bad_frame', 'build'),
    [
        (_z_stacks, 'A1_BF_Z1_0000.tiff', lambda f: ZProjector(has_turret=False).load_folder(
            path=f, tiling_configs_file_loc=TILING_CONFIGS, method=ZProjector.methods()[0])),
        (_z_stacks, 'A1_BF_Z1_0000.tiff', lambda f: StackBuilder(has_turret=False).load_folder(
            path=f, tiling_configs_file_loc=TILING_CONFIGS, save_encoding='right_aligned')),
        (_tiles, 'A1_BF_T0_0000.tiff', lambda f: Stitcher(has_turret=False).load_folder(
            path=f, tiling_configs_file_loc=TILING_CONFIGS)),
    ],
    ids=['z-projection', 'hyperstack', 'stitch'],
)  # fmt: skip
def test_a_frame_cut_short_fails_its_group_and_the_others_are_built(
    tmp_path, folder_of, bad_frame, build
):
    folder = folder_of(tmp_path)
    _truncate(folder / bad_frame)

    with pytest.raises(PostProcessingFailedError) as incomplete:
        build(folder)

    assert bad_frame in str(incomplete.value)
    made = _outputs(folder)
    assert any(name.startswith('A2') for name in made), made
    assert not any(name.startswith('A1') for name in made), made


def test_a_slice_of_another_size_fails_its_group_and_the_others_are_built(tmp_path):
    folder = _z_stacks(tmp_path, size=128)

    with pytest.raises(PostProcessingFailedError) as incomplete:
        ZProjector(has_turret=False).load_folder(
            path=folder, tiling_configs_file_loc=TILING_CONFIGS, method=ZProjector.methods()[0]
        )

    assert '128x128' in str(incomplete.value)
    made = _outputs(folder)
    assert any(name.startswith('A2') for name in made), made
    assert not any(name.startswith('A1') for name in made), made


def test_a_video_none_of_whose_frames_can_be_read_is_a_failed_group(tmp_path):
    folder = _run_folder(
        tmp_path,
        [
            {'name': f'{well}_BF', 'well': well, 'x': 14.38 + 9 * w, 'scan': scan}
            for w, well in enumerate(('A1', 'A2'))
            for scan in range(3)
        ],
    )
    for scan in range(3):
        _truncate(folder / f'A1_BF_{scan:04d}.tiff')

    with pytest.raises(PostProcessingFailedError) as incomplete:
        VideoBuilder(has_turret=False).load_folder(
            path=folder,
            tiling_configs_file_loc=TILING_CONFIGS,
            frames_per_sec=5,
            enable_timestamp_overlay=False,
        )

    assert 'none of its 3 frames' in str(incomplete.value)
    assert all(pathlib.Path(p).exists() for p in incomplete.value.produced_paths)
    record = (folder / 'protocol_post_record.tsv').read_text()
    assert 'A1_BF' not in record and 'A2_BF' in record


class TestAHyperstackLeavesNothingWhenRefused:
    @staticmethod
    def _frames(tmp_path, second: np.ndarray) -> pd.DataFrame:
        _write_frame(tmp_path / 'a.tiff', x=1.0, well='A1', seed=0, size=8)
        tf.imwrite(str(tmp_path / 'b.tiff'), second)
        return pd.DataFrame(
            [
                {'Filepath': name, 'Color': 'BF', 'Scan Count': 0, 'Z-Slice': z,
                 'X': 1.0, 'Y': 2.0, 'Z': 3.0 + z, 'Well': 'A1'}
                for z, name in enumerate(('a.tiff', 'b.tiff'))
            ]
        )  # fmt: skip

    def test_a_two_page_frame_refused_part_way_through_the_write_leaves_no_file(self, tmp_path):
        # Its header reads as one 8x8 page; only the stream decodes it as two.
        # The pixel-type refusal's no-file pin is beside its own test, in
        # test_stack_builder_metadata_propagation.py.
        df = self._frames(tmp_path, np.zeros((2, 8, 8), dtype=np.uint8))

        with pytest.raises(InputFramesUnusableError) as refused:
            StackBuilder._create_stack(
                path=tmp_path,
                df=df,
                output_file_loc=pathlib.Path('out.ome.tiff'),
                save_encoding='right_aligned',
            )

        assert refused.value.reason == 'geometry'
        assert not (tmp_path / 'out.ome.tiff').exists()
        assert not (tmp_path / 'out.ome.tiff.tmp').exists()


def test_a_custom_steps_refusal_names_no_well(tmp_path):
    rows = []
    for z, color in ((0, 'BF'), (1, 'BF'), (0, 'Green')):
        name = f'{color}_{z}.tiff'
        _write_frame(tmp_path / name, x=1.0, well='', seed=z, size=8)
        rows.append({'Filepath': name, 'Color': color, 'Scan Count': 0, 'Z-Slice': z,
                     'X': 1.0, 'Y': 2.0, 'Z': 3.0, 'Well': ''})  # fmt: skip

    result = StackBuilder._create_stack(
        path=tmp_path,
        df=pd.DataFrame(rows),
        output_file_loc=pathlib.Path('out.ome.tiff'),
        save_encoding='right_aligned',
    )

    assert result['status'] is False
    assert result['error'].startswith('Cannot build a hyperstack: its channels')


def test_a_composite_of_a_simulator_runs_folder_fails_only_the_damaged_position(tmp_path):
    """The folder is a simulator run's: two positions, BF and Green at each."""
    from modules.composite_generation import CompositeGeneration
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import _settings
    from tests.test_a_runs_writes_run_on_its_lanes import _run_protocol
    from tests.test_run_refusal_contract import _make_single_step_protocol

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        base = _make_single_step_protocol().step(idx=0)
        steps = [
            {**base, 'Name': f'p{p}_{color}', 'Label': f'p{p}_{color}', 'X': x,
             'Color': color, 'Channel Group ID': p, 'Step Index': 2 * p + c}
            for p, x in enumerate((10.0, 30.0))
            for c, color in enumerate(('BF', 'Green'))
        ]  # fmt: skip
        _run_protocol(session, tmp_path, steps)
    finally:
        session.shutdown()
    run_dir = next(
        p for p in tmp_path.iterdir() if p.is_dir() and (p / 'protocol_record.tsv').exists()
    )
    bad_frame = next(run_dir.glob('p0_Green*.tiff'))
    _truncate(bad_frame)

    with pytest.raises(PostProcessingFailedError) as incomplete:
        CompositeGeneration(has_turret=False).load_folder(
            path=run_dir,
            tiling_configs_file_loc=TILING_CONFIGS,
            output_format='TIFF',
            brightness_thresholds_percent={'Green': 50},
        )

    assert bad_frame.name in str(incomplete.value)
    made = _outputs(run_dir)
    assert any(name.startswith('p1') for name in made), made
    assert not any(name.startswith('p0') for name in made), made
