# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import abc
import datetime
import pathlib
import time

import pandas as pd

import modules.image_utils as image_utils
import modules.post_derivation as post_derivation
import modules.path_utils as path_utils
import modules.recording_frames as recording_frames
from modules.api_surface import ProgressCallback
from modules.common_utils import PostFunction
from modules.exceptions import (
    ExcludedInput,
    InputFramesUnusableError,
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from modules.objectives_loader import objective_short_name
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
from modules.protocol_post_processing_result import PostProcResult
from modules.protocol_post_record import ProtocolPostRecord

from lvp_logger import logger


# What each post-processing function needs to find in a folder to produce
# output. Used in the empty-output user message so the popup explains which
# capture dimension was missing instead of saying "No images found" in a
# folder that visibly has images.
# What to do INSTEAD, when a function refuses a specific kind of derived input.
# Keyed by (function that refused, what was excluded) because the remedy is a
# property of neither alone: stitching refuses composites because the channels
# have to be stitched BEFORE they are merged, while that same pair the other way
# round is the supported route. A pair that is absent here adds no sentence --
# saying nothing beats guessing, and a remedy attached to a refusal it does not
# fit is the defect this table exists to avoid.
_EXCLUDED_REMEDY = {
    (PostFunction.STITCHED, PostFunction.COMPOSITE): (
        ' Composites are not stitched; Composite builds them from the stitched channels.'
    ),
}

# Each operation's name as a person reads it, for the titles and words of
# its outcomes; the enum values are file-record keys, not prose.
_OPERATION_LABELS = {
    PostFunction.COMPOSITE: PostProcessingOperation.COMPOSITE,
    PostFunction.STITCHED: PostProcessingOperation.STITCH,
    PostFunction.ZPROJECT: PostProcessingOperation.Z_PROJECTION,
    PostFunction.VIDEO: PostProcessingOperation.VIDEO,
    PostFunction.HYPERSTACK: PostProcessingOperation.HYPERSTACK,
}


def _remedy_for(post_function, excluded_kinds) -> str:
    """The 'do this instead' sentences for what a function turned away.

    Silent for any pair nobody wrote down, which is the point: the remedy has
    to fit the refusal, and one that does not is worse than none.
    """
    return ''.join(
        _EXCLUDED_REMEDY.get((post_function, kind), '')
        for kind in sorted(excluded_kinds, key=lambda k: k.value)
    )


_MULTI_FRAME_REQUIREMENT = {
    PostFunction.VIDEO: 'multiple time points per scan position',
    PostFunction.ZPROJECT: 'multiple Z-slices per scan position',
    PostFunction.COMPOSITE: 'multiple channels per scan position',
    PostFunction.STITCHED: 'multiple tile positions per scan',
}


def _missing(df: pd.DataFrame) -> pd.Series:
    """Which rows stand for images the records name and the folder lacks.

    The loader marks them (``Missing``): a raw frame the run recorded, or a
    derived image the post-processing record names; a table with no such
    column holds none.
    """
    marks = df.get('Missing')
    if marks is None:
        return pd.Series(False, index=df.index)
    return marks.fillna('').astype(str) != ''


def _recording_frames(df: pd.DataFrame) -> pd.Series:
    """Which rows are a recording's frames: told by the frame's file name,
    or, for a raw frame the folder lacks, by its step."""
    # astype(bool): on an EMPTY frame map() cannot infer a dtype, and
    # indexing with the resulting non-bool series degrades from a row
    # mask to column selection -- silently dropping every column.
    frames = df['Filepath'].map(recording_frames.is_video_frame).astype(bool)
    # Only a raw row has a step's Acquire: a derived image is never a
    # recording's frame, and a folder of derived images alone (a selected
    # 'ZProject/') has no Acquire column at all.
    missing = _missing(df)
    if missing.any():
        missing_raw = missing & ~_derived(df)
        if missing_raw.any():
            frames = frames | (missing_raw & (df['Acquire'] == 'video'))
    return frames


def _derived(df: pd.DataFrame) -> pd.Series:
    """Which rows are derived images: a raw capture's derivation is empty.

    Asked only of a table holding a missing row: what a gone image is
    decides how it is answered, and a table the loader marked carries
    every row's derivation.
    """
    return df['Derivation'].astype(str) != ''


def _made_by(derivation: str) -> PostProcessingOperation:
    """The build that made an image derived as *derivation*."""
    return _OPERATION_LABELS[post_derivation.steps(derivation)[-1].kind]


def _removed(group: pd.DataFrame) -> bool:
    """Whether every image of *group* is a derived image gone from the folder.

    A build's outputs deleted whole -- its folder emptied -- were removed,
    not damaged: the group is named and not built, where a group missing
    only some of its images fails. A group never mixes raw and derived
    images (every build groups by Derivation), and a raw frame gone is
    always a failure: nothing builds it again.
    """
    return bool(_missing(group).all() and _derived(group).all())


def _missing_frames_error(missing: pd.DataFrame) -> str:
    """Why a group was not built: each recorded image it lacks, and why."""
    if _derived(missing).all():
        named = [
            f'{row["Filepath"]} (not in the folder; build '
            f'{_made_by(row["Derivation"])} again to restore it)'
            for _, row in missing.iterrows()
        ]
        return f'{len(named)} image(s) its record names are missing -- ' + '; '.join(named)
    named = []
    for _, row in missing.iterrows():
        if row['Missing'] == 'not_on_disk':
            why = f'not in the folder: {row["Filepath"]}'
        else:
            why = f'its capture failed during the run: {row["Missing"]}'
        named.append(f'{row["Name"]} scan {row["Scan Count"]} ({why})')
    return f'{len(named)} frame(s) the run recorded are missing -- ' + '; '.join(named)


class ProtocolPostProcessor(abc.ABC):
    # The place an image fills in this build's output, as a person reads it.
    _PLACE = 'place'
    # What two acquisitions in one place should have been, for a build that
    # groups by Add; empty for one that does not.
    _ONE_PER_ADD = ''

    def __init__(
        self,
        post_function: PostFunction,
        *args,
        **kwargs,
    ):
        self._name = self.__class__.__name__
        self._post_function = post_function
        self._post_processing_helper = ProtocolPostProcessingHelper()
        self._has_turret = kwargs['has_turret']

    @staticmethod
    @abc.abstractmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Implement in child class')

    @staticmethod
    def _with_recording_scan(df: pd.DataFrame) -> pd.DataFrame:
        """Derive the 'Recording Scan' group key.

        The two source kinds carry different temporal identities: a
        still's time axis runs ACROSS scans (timelapse video, stills
        hyperstack -- every scan shares its group), while a recorded video
        frame's time axis runs WITHIN one recording, so the recording's
        scan is part of its group identity. Without the derived key, a
        multi-scan run's video frames land in ONE group whose per-scan
        frame numbers collide. Stills share the one sentinel so their
        cross-scan grouping is untouched.
        """
        recording_scan = df['Scan Count'].where(_recording_frames(df), -1)
        return df.assign(**{'Recording Scan': recording_scan})

    @staticmethod
    def _without_video_frames(df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw video-frame rows from a processor's input.

        A recording's frames are a TIME series: to Z-Projection they look
        like hundreds of slices at one Z, to Composite like whole
        recordings per channel, to Stitch like repeated tiles at one
        position -- each would emit a mislabeled artifact presented as
        real. Video frames belong to exactly two consumers, Create Video
        and the per-(well, scan) hyperstack, which do NOT call this.
        """
        return df[~_recording_frames(df)]

    @abc.abstractmethod
    def _generate_filename(self, df: pd.DataFrame, post: tuple[str, ...], **kwargs) -> str:
        """The output's file name, its post tokens ``post`` rendered from its derivation."""
        raise NotImplementedError('Implement in child class')

    @abc.abstractmethod
    def _derivation_step(self, group: pd.DataFrame, **kwargs) -> post_derivation.Step:
        """This build of ``group``: every setting that decides the output's pixels."""
        raise NotImplementedError('Implement in child class')

    @abc.abstractmethod
    def _places(self, group: pd.DataFrame) -> pd.Series | None:
        """The place each image of ``group`` fills in the output; None when each has its own."""
        raise NotImplementedError('Implement in child class')

    def _two_images_in_one_place(self, group: pd.DataFrame) -> str | None:
        """Why ``group`` is not built because two of its images fill one place, or None.

        An output takes one image per place: a mosaic one per tile position,
        a projection one per slice, a video one per scan, a composite and a
        hyperstack one per plane. Two in one place are two acquisitions (a
        run folder from before steps recorded their Add) or two derivations
        an earlier release made without recording how; building from both
        would keep one and drop the other, or mix them.
        """
        places = self._places(group)
        if places is None:
            return None
        doubled = places[places.duplicated(keep=False)]
        if doubled.empty:
            return None
        named = '; '.join(
            ', '.join(str(path) for path in group.loc[rows.index, 'Filepath'])
            for _, rows in doubled.groupby(doubled, sort=False)
        )
        said = f'Not built: more than one image for one {self._PLACE} ({named}).'
        if any(
            post_derivation.made_unrecorded(derivation)
            for derivation in group.loc[doubled.index, 'Derivation']
        ):
            return (
                f'{said} An earlier release made them without recording how; build them '
                'again with this release.'
            )
        return f'{said}{self._ONE_PER_ADD}'

    @abc.abstractmethod
    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Implement in child class')

    @abc.abstractmethod
    def _group_algorithm(self, path: pathlib.Path, df: pd.DataFrame):
        raise NotImplementedError('Implement in child class')

    @staticmethod
    @abc.abstractmethod
    def _add_record(
        protocol_post_record: ProtocolPostRecord,
        alg_metadata: dict,
        root_path: pathlib.Path,
    ):
        raise NotImplementedError('Implement in child class')

    def _planned_output(
        self, root_path: pathlib.Path, group: pd.DataFrame, **kwargs
    ) -> tuple[str, str, pathlib.Path]:
        """What ``group`` builds: its file name, its derivation, and where it lands.

        A group's images share one derivation (every build groups by it), so
        the output's is theirs extended by this build, and its name and
        folder are rendered from that.
        """
        derivation = post_derivation.extended(
            group.iloc[0]['Derivation'], self._derivation_step(group, **kwargs)
        )
        output_filename = self._generate_filename(
            df=group, post=post_derivation.name_tokens(derivation), **kwargs
        )
        output_subfolder = self._post_processing_helper.generate_output_dir_name(
            record=pd.Series(post_derivation.kinds(derivation))
        )
        return (
            output_filename,
            derivation,
            root_path / output_subfolder / output_filename,
        )

    def _get_objective_short_name_if_has_turret(self, objective_id: str) -> str | None:
        if not self._has_turret:
            return None

        # Derived from the id the run recorded, with no catalogue: a folder
        # post-processed on another installation, or after the catalogue was
        # edited, still names the objective the run used. A step with no
        # objective records '' (the protocol readers fill empty cells), and
        # its name carries no objective, as on a scope with no turret.
        if objective_id == '':
            return None
        return objective_short_name(objective_id)

    def _degraded_summary(self, count: int) -> str:
        """One clause naming what a degraded (fallback-produced) output means for
        this operation. The base states it generically; a subclass whose
        algorithm has a named fallback chain (e.g. stitching) overrides with its
        own wording so the shared loop does not hardcode a stitch-only vocabulary
        for zproject / composite / stack outputs.
        """
        return f'{count} {self._post_function.value.lower()} artifact(s) used a fallback method.'

    @staticmethod
    def _prepend_capture_root(name: str, kwargs: dict) -> str:
        """Prefix a post-processed output name with the protocol's capture_root.

        load_folder is the only caller of _generate_filename and always threads
        capture_root into kwargs, so it is read as a required key: a missing key
        is a caller bug and fails loud rather than silently dropping the root.
        An empty root is a valid state (no custom root set). The root is kept
        out of the name seed, so a root that happens to contain a token cannot
        perturb the derived name.
        """
        capture_root = kwargs['capture_root']
        return f'{capture_root}_{name}' if capture_root else name

    def load_folder(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        on_progress: ProgressCallback | None = None,
        **kwargs: dict,
    ) -> dict:
        """Run this operation over a captured folder; return the complete result.

        Returns only when every output asked for was produced (skipped
        single-image groups, already-recorded groups and a fallback
        algorithm's degraded output are complete results, reported in the
        dict). Anything else raises, and the caller reports it: this loader
        never posts, because only the caller knows whether someone is
        waiting on it.

        Raises:
            PostProcessingRefusedError: The folder cannot yield this output.
            PostProcessingFailedError: The build did not produce everything
                it was asked for; what it did produce rides along.
        """
        return self._load_folder_inner(
            path=path,
            tiling_configs_file_loc=tiling_configs_file_loc,
            on_progress=on_progress,
            **kwargs,
        )

    @property
    def operation_label(self) -> str:
        """This operation's name as a person reads it."""
        return _OPERATION_LABELS[self._post_function]

    @property
    def operation_key(self) -> str:
        """Ties a start notice to the outcome that answers it.

        A caller that announces a long build passes this with both ends, or
        the outcome opens a second modal instead of replacing the "please
        wait" one. Derived in one place so the two ends cannot drift apart.
        """
        return f'post-processing:{self._post_function.value}'

    def _incomplete(
        self,
        *,
        group_errors: list,
        attempted: int,
        refused_count: int,
        colliding_names: set,
        dropped_frames: int,
        artifact_paths: list,
        output_root: str,
        accounting_note: str,
        excluded_inputs: tuple[ExcludedInput, ...],
    ) -> PostProcessingFailedError:
        """The outcome of a build that did not produce everything asked of it.

        It carries what the build left out as a complete result does, so a
        group that failed does not also silence the inputs that were never
        tried.
        """
        label = self.operation_label.lower()
        missing = []
        if group_errors:
            shown = '; '.join(group_errors[:3])
            more = '; ...' if len(group_errors) > 3 else ''
            missing.append(
                f'{len(group_errors)} of {attempted} {label} group(s) failed ({shown}{more})'
            )
        if refused_count:
            names = ', '.join(sorted(colliding_names)[:3])
            more = ', ...' if len(colliding_names) > 3 else ''
            missing.append(
                f'{refused_count} group(s) were refused because more than one group '
                f'derives the same output filename ({names}{more})'
            )
        if dropped_frames:
            missing.append(
                f'{dropped_frames} frame(s) could not be added, so the video is '
                'shorter than its source'
            )
        sentence = '; '.join(missing)
        return PostProcessingFailedError(
            operation=self.operation_label,
            missing=f'{sentence[0].upper()}{sentence[1:]}.{accounting_note}',
            produced_paths=artifact_paths,
            output_root=output_root,
            errors=group_errors,
            excluded_inputs=excluded_inputs,
        )

    def _refuse(self, reason: str, message: str) -> PostProcessingRefusedError:
        return PostProcessingRefusedError(
            operation=self.operation_label, reason=reason, message=message
        )

    def _load_folder_inner(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        on_progress: ProgressCallback | None = None,
        **kwargs: dict,
    ) -> dict:
        start_ts = datetime.datetime.now()
        if not path:
            raise self._refuse('invalid_path', 'No folder was given to process.')

        selected_path = pathlib.Path(path)
        results = self._post_processing_helper.load_folder(
            path=selected_path,
            tiling_configs_file_loc=tiling_configs_file_loc,
        )

        if results['status'] is False:
            # The helper's own sentence says what it could not read; a
            # generic "failed to load" in its place sent people looking for
            # the wrong problem.
            raise self._refuse('protocol_data_unreadable', results['message'])

        df = results['images_df']
        if len(df) == 0:
            raise self._refuse(
                'no_images',
                'No image files were found in the selected folder. '
                'Check that the folder contains captured scan images.',
            )

        # Composite, stitch, and z-projection re-read the source frames via
        # tifffile, which cannot decode JPG. A scan saved as JPG has no
        # re-readable source for these outputs, so stop here with a clear
        # message instead of letting tifffile raise partway through a group.
        unsupported_source = next(
            (
                pathlib.Path(str(filepath))
                for filepath in df.loc[~_missing(df), 'Filepath']
                if pathlib.Path(str(filepath)).suffix.lower() not in image_utils.TIFF_SUFFIXES
            ),
            None,
        )
        if unsupported_source is not None:
            source_format = unsupported_source.suffix.lstrip('.').upper() or 'unknown'
            raise self._refuse(
                'unsupported_source_format',
                f'{self._post_function.value} requires TIFF or OME-TIFF source images. '
                f'First unsupported {source_format} file: {unsupported_source.name}. '
                'Reacquire the scan as TIFF or OME-TIFF before post-processing.',
            )

        root_path = results['root_path']
        protocol_post_record = results['protocol_post_record']

        # The per-image protocol writer (protocol_image_writer.py) prefixes
        # filenames with protocol.capture_prefix() so a scan run with Root
        # "experiment1" produces "experiment1_<step>_<color>_<...>.tiff".
        # Post-processed outputs (composite, stitch, z-proj, video, stack)
        # must use the same prefix; pipe it via kwargs to _generate_filename.
        protocol = results.get('protocol')
        kwargs.setdefault(
            'capture_root',
            protocol.capture_prefix() if protocol is not None else '',
        )

        # Account for every input the subclass filter drops: the completion
        # and empty-result messages are built from survivors, so uncounted
        # drops misattribute the outcome (a composite-only folder read as
        # "lacks multi-tile structure"). Dropped rows self-describe their
        # category through their own PostFunction flags. A dropped row that
        # carries this operation's own flag is its own earlier output, already
        # made rather than left out (a stitched composite is a stitch's own,
        # though it is a composite too): it is counted only to explain a
        # folder that holds nothing else.
        pre_filter_df = df
        df = self._filter_ignored_types(df=df)
        own_flag = self._post_function.value
        left_out_counts: dict[str, int] = {}
        excluded_kinds: set = set()
        already_made = 0
        dropped_rows = pre_filter_df.loc[pre_filter_df.index.difference(df.index)]
        # A derived image gone from the folder is not a file in it: neither
        # an output already made nor an input left out.
        gone = _missing(dropped_rows)
        if gone.any():
            dropped_rows = dropped_rows[~(gone & _derived(dropped_rows))]
        for _, dropped in dropped_rows.iterrows():
            if dropped[own_flag]:
                already_made += 1
                continue
            kind = next((flag for flag in PostFunction.list_values() if dropped[flag]), None)
            if kind is not None:
                excluded_kinds.add(PostFunction(kind))
            word = kind.lower() if kind is not None else 'other'
            left_out_counts[word] = left_out_counts.get(word, 0) + 1
        excluded_inputs = tuple(
            ExcludedInput(kind=word, count=count) for word, count in sorted(left_out_counts.items())
        )
        excluded_text = ', '.join(f'{item.count} {item.kind} file(s)' for item in excluded_inputs)
        derived_counts = dict(left_out_counts)
        if already_made:
            derived_counts[own_flag.lower()] = already_made
        derived_text = ', '.join(
            f'{count} {word} file(s)' for word, count in sorted(derived_counts.items())
        )
        if derived_text:
            logger.info(
                f'[{self._name} ] {len(dropped_rows)} input file(s) excluded from '
                f'{own_flag.lower()} generation: {derived_text}'
            )

        groups = self._get_groups(df)

        group_count = len(groups)

        # When two DIFFERENT groups would render one output filename,
        # generating them would produce only the first group's artifact (the
        # second is skipped as already-recorded) -- a silent data loss.
        # Refuse exactly those groups, loudly, and still process the
        # unambiguous ones: an already-captured folder with baked-in
        # colliding names (which renaming protocol steps cannot repair)
        # stays post-processable for everything else.
        #
        # An output is named from the protocol's step data, which a folder
        # carries as text: a well or label holding '..' or a separator would
        # put the output outside the folder it was asked to build in. Every
        # group is checked before any is built, so a refusal writes no output.
        planned_names = {}
        for _, group in groups:
            if len(group) <= 1 or _removed(group):
                continue
            name, _derivation, output_file_loc = self._planned_output(root_path, group, **kwargs)
            if not path_utils.resolves_inside(root_path, output_file_loc):
                raise self._refuse(
                    'output_outside_folder',
                    f'The output {name!r} would be written outside {root_path}: the '
                    'protocol in this folder names a step with a path. No output was written.',
                )
            planned_names[name] = planned_names.get(name, 0) + 1
        colliding_names = {name for name, count in planned_names.items() if count > 1}
        for name in sorted(colliding_names):
            logger.error(
                f'[{self._name} ] Refusing to generate {name}: more than one '
                f'image group derives this output filename, so the artifacts '
                f'would be indistinguishable.'
            )

        logger.info(f'{self._name}: Generating {self._post_function.value.lower()} images')

        new_count = 0
        existing_count = 0
        refused_count = 0
        current_group = 1
        last_error = None
        # Every failed group's error, not only the last: a build that lost
        # several groups for different reasons must say each.
        group_errors = []
        dropped_frames = 0
        degraded_outputs = []
        output_significant_bits = None
        completed_group_ms = []
        skipped_single_paths = []
        # The build that made each removed group's images (``_removed``).
        removed_groups: list[PostProcessingOperation] = []
        # Where each produced artifact landed. The return used to carry only
        # the output ROOT, which tells a caller a directory was touched but
        # not whether its own file exists -- an unattended caller could not
        # distinguish a successful run from one that produced nothing, and
        # had to re-derive the filename to find out.
        artifact_paths = []

        for _, group in groups:
            if len(group) == 0:
                continue

            # Before the single-image skip, so a removed group of one image
            # is named too.
            if _removed(group):
                removed_groups.append(_made_by(group.iloc[0]['Derivation']))
                logger.info(
                    f'[{self._name} ] Not building from {group.iloc[0]["Filepath"]} and the '
                    f'{len(group) - 1} other image(s) of its group: all removed from the folder.'
                )
                continue

            # A row for a missing recording stands for its frames, so the
            # group it is in counts as the group it would have been.
            missing = group[_missing(group)]
            frames = len(group) - len(missing) + int(missing['Frames'].sum() if len(missing) else 0)
            if frames == 1:
                if missing.empty:
                    skipped_single_paths.append(str(group.iloc[0]['Filepath']))
                    logger.debug(
                        f'[{self._name} ] Skipping generation for {group.iloc[0]["Filepath"]} since only {len(group)} image found.'
                    )
                continue

            output_filename, derivation, output_file_loc = self._planned_output(
                root_path, group, **kwargs
            )
            if output_filename in colliding_names:
                refused_count += 1
                continue
            row0 = group.iloc[0]
            output_file_loc_rel = output_file_loc.relative_to(root_path)
            # Each column names every value the group holds: a composite's or
            # a hyperstack's group spans its colours, so one row's colour names
            # only part of it. Its images are counted as images, since only a
            # stitch's are tiles.
            held = ' '.join(
                f'{key}={"+".join(str(value) for value in group[column].unique())}'
                for key, column in (
                    ('well', 'Well'),
                    ('color', 'Color'),
                    ('tile_group', 'Tile Group ID'),
                )
                if column in group
            )
            group_label = f'{held} images={len(group)} output={output_file_loc_rel}'

            # Already made only while its file is there: one deleted since is
            # built again, and the record's row for it replaced.
            if (
                protocol_post_record.file_exists_in_records(filepath=output_file_loc_rel)
                and (root_path / output_file_loc_rel).is_file()
            ):
                logger.info(
                    f'[{self._name} ] {output_file_loc_rel} already exists in record, skipping for generation.'
                )
                existing_count += (
                    1  # Count this so we don't error out if no other matches are found
                )
                continue

            doubled = self._two_images_in_one_place(group)
            if doubled is not None:
                last_error = doubled
                group_errors.append(f'{output_file_loc_rel}: {doubled}')
                logger.error(f'Not generating {output_file_loc_rel}: {doubled}')
                continue

            # Built whole or not at all: a group missing a frame the run
            # recorded -- deleted since, or never captured -- is this group's
            # failure, never an output short of it.
            if not missing.empty:
                last_error = _missing_frames_error(missing)
                group_errors.append(f'{output_file_loc_rel}: {last_error}')
                logger.error(f'Not generating {output_file_loc_rel}: {last_error}')
                continue

            kwargs['output_file_loc'] = output_file_loc_rel

            group_t0 = time.perf_counter()
            logger.info(f'[PostProcPerf] {self._name} group start: {group_label}')
            try:
                alg_results = self._group_algorithm(
                    path=root_path,
                    df=group,
                    on_progress=on_progress,
                    total_groups=group_count,
                    current_group=current_group,
                    **kwargs,
                )
            except InputFramesUnusableError as unusable:
                # Frames this group cannot use -- one that will not decode,
                # sizes or pixel types that cannot share one output -- cost
                # this group, not the groups after it: answered as its failure.
                alg_results = PostProcResult.failed(str(unusable))
            group_ms = (time.perf_counter() - group_t0) * 1000.0

            if not alg_results.status:
                last_error = alg_results.error
                group_errors.append(f'{output_file_loc_rel}: {alg_results.error}')
                logger.info(
                    f'[PostProcPerf] {self._name} group failed after {group_ms:.1f}ms: {group_label}'
                )
                logger.error(f'Failed to generate {output_file_loc_rel}: {alg_results.error}')
                continue

            completed_group_ms.append(group_ms)

            alg_metadata = alg_results.record_metadata
            # A video writes what frames it could read; the ones it could not
            # leave it shorter than its source, which the build must say.
            dropped_frames += alg_metadata.get('dropped_frames', 0)
            logger.info(
                f'[PostProcPerf] {self._name} group done in {group_ms:.1f}ms: '
                f'algorithm={alg_metadata.get("algorithm", "")} '
                f'degraded={bool(alg_metadata.get("fallback_reason"))} {group_label}'
            )
            fallback_reason = alg_metadata.get('fallback_reason')
            if fallback_reason:
                degraded_outputs.append(
                    {
                        'filepath': str(output_file_loc_rel),
                        'algorithm': alg_metadata.get('algorithm', ''),
                        'fallback_from': alg_metadata.get('fallback_from', ''),
                        'fallback_reason': fallback_reason,
                    }
                )

            # A subclass whose writer relocated the output (collision suffix,
            # container-format fallback) reports where the file really landed;
            # the record must point at that file, not the request.
            actual_output_file_loc = alg_results.actual_output_file_loc
            if actual_output_file_loc is not None:
                output_file_loc_rel = pathlib.Path(actual_output_file_loc).relative_to(root_path)

            # Each ProtocolPostProcessor subclass owns its own file write via
            # tifffile (RGB-native; auto-detects photometric). cv2.imwrite was
            # retired here -- cv2 is BGR-native and would silently swap channels
            # relative to the RGB-native readers (tifffile / FIJI / OS preview).
            # A subclass that fails to write must return a failed result.

            self._add_record(
                protocol_post_record=protocol_post_record,
                alg_metadata=alg_results.record_metadata,
                root_path=root_path,
                file_path=output_file_loc_rel,
                row0=row0,
                derivation=derivation,
            )

            # Recorded after the relocation resolve above, so the path names
            # the file that actually exists rather than the one requested.
            artifact_paths.append(str(root_path / output_file_loc_rel))

            new_count += 1
            current_group += 1
            # Carry the depth the artifact was written at so the completion line
            # states whether the input depth round-tripped through this operation
            # instead of requiring a tag read on the output file. The typed result
            # guarantees a successful group carries its output depth, so this read
            # is always present.
            output_significant_bits = alg_results.significant_bits

            if on_progress is not None:
                status = None
                if self._name == 'Stitcher':
                    mode_label = (
                        'Fast Preview'
                        if kwargs.get('stitching_mode') == 'fast_preview'
                        else 'Quality'
                    )
                    # current_group already names the group now starting, which
                    # has yet to run: it is one of the groups the estimate covers.
                    remaining = max(0, group_count - current_group + 1)
                    average_ms = sum(completed_group_ms) / len(completed_group_ms)
                    estimate_seconds = round((remaining * average_ms) / 1000.0)
                    status = (
                        f'Running {mode_label} Stitch -- group {current_group}/{group_count}.\n'
                        f'Estimated remaining time: about {estimate_seconds} seconds.\n'
                        'Source pixels and channel colors are preserved.'
                    )
                on_progress((new_count / group_count) * 100, status)

        protocol_post_record.complete()

        if on_progress is not None:
            on_progress(100, None)

        if skipped_single_paths:
            shown = ', '.join(skipped_single_paths[:3])
            more = ', ...' if len(skipped_single_paths) > 3 else ''
            logger.info(
                f'[{self._name} ] Skipped {len(skipped_single_paths)} single-image '
                f'group(s) (nothing to combine): {shown}{more}'
            )

        single_skip_note = (
            f' {len(skipped_single_paths)} single-image group(s) skipped.'
            if skipped_single_paths
            else ''
        )
        remedy = _remedy_for(self._post_function, excluded_kinds)
        excluded_note = (
            f' Excluded from this operation: {excluded_text}.{remedy}' if excluded_text else ''
        )
        removed_note = ''
        if removed_groups:
            makers = ', '.join(sorted(set(removed_groups)))
            removed_note = (
                f' {len(removed_groups)} group(s) not built: their {makers} images were '
                'removed from the folder.'
            )
        accounting_note = f'{single_skip_note}{excluded_note}{removed_note}'

        # A removed group is a complete answer, as an output already made is.
        if (new_count == 0) and (existing_count == 0) and not removed_groups:
            fname = self._post_function.value
            if refused_count > 0 and last_error is None:
                # Every eligible group collided; nothing could be generated.
                raise self._refuse(
                    'collision',
                    f'No {fname} was generated: every image group derives an '
                    f'output filename shared with another group, so their '
                    f'artifacts would be indistinguishable. See '
                    f'lumaviewpro.log for the colliding names.',
                )
            if derived_text and last_error is None:
                # The emptiness is explained by what the filter excluded, not
                # by the folder's structure -- a structural hint here sent
                # users hunting for missing tiles a composite folder has.
                # Own earlier outputs count here: a folder of nothing else
                # is explained by them.
                fname_lower = fname.lower()
                raise self._refuse(
                    'excluded_inputs',
                    f'No {fname_lower} was generated: this folder holds only '
                    f'{derived_text}, which are derived outputs.'
                    f'{remedy}{single_skip_note}',
                )
            needed = _MULTI_FRAME_REQUIREMENT.get(
                self._post_function, 'multiple frames per scan position'
            )
            if last_error is None:
                raise self._refuse(
                    'no_data',
                    f'No {fname} was generated. {fname} requires {needed}. '
                    f'The folder may have image files but not the structure '
                    f'this operation needs -- check the log if you expected '
                    f'the folder to be compatible.{accounting_note}',
                )
            # Usable groups WERE found and attempted, but every one failed in
            # the algorithm itself: that is a failure, not a folder lacking the
            # data -- reported as missing data, it sent people hunting for
            # Z-stacks when the operation had broken.

        if group_errors or refused_count or dropped_frames:
            raise self._incomplete(
                group_errors=group_errors,
                attempted=len(group_errors) + new_count,
                refused_count=refused_count,
                colliding_names=colliding_names,
                dropped_frames=dropped_frames,
                artifact_paths=artifact_paths,
                output_root=str(root_path),
                accounting_note=accounting_note,
                excluded_inputs=excluded_inputs,
            )

        end_ts = datetime.datetime.now()
        elapsed_time = end_ts - start_ts
        logger.info(
            f'{self._name}: Complete - Created {new_count} {self._post_function.value.lower()} '
            f'artifacts (significant_bits={output_significant_bits}) in {elapsed_time}.'
            f'{accounting_note}'
        )
        if degraded_outputs:
            summary = self._degraded_summary(len(degraded_outputs))
            logger.warning(f'{self._name}: Complete with degraded outputs: {summary}')
            return {
                'status': True,
                'message': f'Success with degraded output: {summary}{accounting_note}',
                'degraded': True,
                'degraded_outputs': degraded_outputs,
                'new_count': new_count,
                'output_root': str(root_path),
                'artifact_paths': artifact_paths,
                'accounting_note': accounting_note,
                'excluded_inputs': excluded_inputs,
            }
        return {
            'status': True,
            'message': f'Success.{accounting_note}',
            'new_count': new_count,
            'output_root': str(root_path),
            'artifact_paths': artifact_paths,
            'accounting_note': accounting_note,
            'excluded_inputs': excluded_inputs,
        }
