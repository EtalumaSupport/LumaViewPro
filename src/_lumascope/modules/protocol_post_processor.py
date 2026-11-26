# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import abc
import datetime
import pathlib
import time
import typing
from collections.abc import Callable, Sequence

import pandas as pd

from _lumascope.drivers import host_clock
import _lumascope.modules.image_utils as image_utils
import _lumascope.modules.post_derivation as post_derivation
import _lumascope.modules.path_utils as path_utils
import _lumascope.modules.recording_frames as recording_frames
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.api_surface import ProgressCallback
from _lumascope.modules.common_utils import PostFunction, causes_in_words, first_few
from _lumascope.modules.exceptions import (
    ExcludedInput,
    InputFramesUnusableError,
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
    VideoNotFinishedError,
    system_words,
)
from _lumascope.modules.objectives_loader import objective_short_name
from _lumascope.modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
from _lumascope.modules.protocol_post_processing_result import PostProcResult
from _lumascope.modules.protocol_post_record import ProtocolPostRecord
from _lumascope.modules.stitch_mode import StitchMode
from _lumascope.modules.tiling_config import TilingGrids

from _lumascope.lvp_logger import logger

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
    (PostFunction.ZPROJECT, PostFunction.STITCHED): (
        ' Stitched images are not projected; a projection is made from the slices, and'
        ' Stitch then stitches the projections.'
    ),
    (PostFunction.ZPROJECT, PostFunction.COMPOSITE): (
        ' Composites are not projected; a projection is made from the slices, and'
        ' Composite then composites the projections.'
    ),
}


class _Unwritten(typing.NamedTuple):
    """The output a build stopped at, and why it could not be written.

    *written* is True when the output is on disk and the record's append
    failed; *unrecorded* is the record's own rewrite failing at the end.
    """

    output: pathlib.Path
    failed: OSError | VideoNotFinishedError
    written: bool
    unrecorded: OSError | None = None

    def reason(self) -> str:
        if isinstance(self.failed, VideoNotFinishedError):
            # The writer's own words name its temporary, never the video.
            return f'it could not be finished ({system_words(self.failed.__cause__)})'
        if self.written:
            return (
                'it was saved, and the post-processing record could not list it '
                f'({system_words(self.failed)})'
            )
        return f'it could not be written ({system_words(self.failed)})'

    def sentence(self) -> str:
        return f'the build stopped at {self.output}: {self.reason()}'


# Each operation's name as a person reads it, for the titles and words of
# its outcomes; the enum values are file-record keys, not prose.
_OPERATION_LABELS = {
    PostFunction.COMPOSITE: PostProcessingOperation.COMPOSITE,
    PostFunction.STITCHED: PostProcessingOperation.STITCH,
    PostFunction.ZPROJECT: PostProcessingOperation.Z_PROJECTION,
    PostFunction.VIDEO: PostProcessingOperation.VIDEO,
    PostFunction.HYPERSTACK: PostProcessingOperation.HYPERSTACK,
}


def operation_key(post_function: PostFunction) -> str:
    """The key that ties a build's start notice to the outcome answering it.

    Derived in one place so the two ends cannot drift apart, for a caller
    that announces a build it does not construct.
    """
    return f'post-processing:{post_function.value}'


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
        missing_raw = missing & ~derived_rows(df)
        if missing_raw.any():
            frames = frames | (missing_raw & (df['Acquire'] == AcquireMode.VIDEO))
    return frames


def rows_of(*, stills: bool, recordings: bool) -> Callable[[pd.DataFrame], pd.Series]:
    """A build's ``select`` for the families it takes: a folder's stills, its recordings' frames, or both."""

    def select(df: pd.DataFrame) -> pd.Series:
        frames = _recording_frames(df)
        return (frames & recordings) | (~frames & stills)

    return select


def group_progress(
    on_progress: ProgressCallback | None, *, current_group: int, total_groups: int
) -> ProgressCallback | None:
    """*on_progress* for one group of a build of *total_groups*, or None without one.

    The group's own 0 to 100 is reported as its share of the build's, so the
    build's percent rises through its groups and never falls back to 0 at a
    group's start.
    """
    if on_progress is None:
        return None
    # One expression for every point, so a group's 100 and the next group's
    # 0 are the same number, not two roundings of it.
    return lambda percent, status: on_progress(
        (current_group - 1 + percent / 100) / total_groups * 100, status
    )


def derived_rows(df: pd.DataFrame) -> pd.Series:
    """Which rows are derived images, the one test: a raw capture's
    derivation is empty, and every row the loader gives carries one."""
    return df['Derivation'].astype(str) != ''


def raw_rows(df: pd.DataFrame) -> pd.Series:
    """Which rows are raw captures: those that are not derived."""
    return ~derived_rows(df)


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
    return bool(_missing(group).all() and derived_rows(group).all())


def _plans_of(
    get_groups: Callable[[pd.DataFrame], object], df: pd.DataFrame, planned: pd.DataFrame
) -> dict:
    """Each of *df*'s rows' plan, by its index: the steps the planned rows
    of its group hold.

    The planned rows are grouped with the captured ones, by the build's own
    groups, so a plan reaches exactly the group its steps' captures fill.
    """
    if df.empty or planned.empty:
        return {}
    both = pd.concat([df, planned], keys=['captured', 'planned'])
    group_of = get_groups(both).ngroup()
    steps: dict = {}
    for row, group_id in group_of.loc['planned'].items():
        steps.setdefault(group_id, set()).add(int(planned.at[row, 'Step Index']))
    return {
        row: frozenset(steps.get(group_id, ()))
        for row, group_id in group_of.loc['captured'].items()
    }


def _cut_short(group: pd.DataFrame, plan: frozenset, last_scan: int | None) -> pd.DataFrame:
    """The group's rows in the run's last scan when they hold less than
    *plan*, else none.

    A recorded capture holds its step whether its file is there or not: a
    scan that ran whole is never cut short, only refused for what it lost.
    A group with no row in that scan was not reached in it, and loses
    nothing.
    """
    if not plan:
        return group.iloc[0:0]
    in_last = group[group['Scan Count'] == last_scan]
    if in_last.empty:
        return in_last
    if {int(step) for step in in_last['Step Index']} >= plan:
        return in_last.iloc[0:0]
    return in_last


def _subject(group: pd.DataFrame) -> str:
    # A custom step has no well; its label names every step the group
    # spans, where a step's own name is one channel's or one slice's.
    row0 = group.iloc[0]
    return f'well {row0["Well"]}' if row0['Well'] else f'step {row0["Label"]}'


def _missing_frames_error(missing: pd.DataFrame) -> str:
    """Why a group was not built: each recorded image it lacks, and why."""
    if derived_rows(missing).all():
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
        named.append(f'{row["Name"]} scan {row["Scan Count"] + 1} ({why})')
    return f'{len(named)} frame(s) the run recorded are missing -- ' + '; '.join(named)


class ProtocolPostProcessor(abc.ABC):
    # The place an image fills in this build's output, as a person reads it.
    _PLACE = 'place'
    # What two acquisitions in one place should have been, for a build that
    # groups by Add; empty for one that does not.
    _ONE_PER_ADD = ''
    # Whether a group's share of the run's last scan is left out when it
    # holds less than the protocol planned there: the run ended partway
    # through it, and a build that needs every scan whole builds from the
    # scans before it.
    _WHOLE_SCANS_ONLY = False
    # Whether a folder whose every group holds one image is a complete
    # answer (nothing to combine) rather than a refusal. It is complete for
    # a build only the run that wrote the folder asks for, where it means
    # the run had nothing to combine; it is a refusal for a build a person
    # points at a folder, where it means the folder was the wrong one.
    _SINGLES_ARE_COMPLETE = False

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
        folder are rendered from that. Its format is the one its images
        share (``image_utils.derived_file_format``), whenever it is built;
        the name states it, and the build writes the format its name states.
        """
        output_format = image_utils.derived_file_format(group['Filepath'])
        derivation = post_derivation.extended(
            group.iloc[0]['Derivation'],
            self._derivation_step(group, output_format=output_format, **kwargs),
        )
        output_filename = self._generate_filename(
            df=group,
            post=post_derivation.name_tokens(derivation),
            output_format=output_format,
            **kwargs,
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

    def _degraded_summary(self, degraded_outputs: list[dict]) -> str:
        """What the degraded (fallback-produced) outputs mean for this
        operation, with why each fell back. The base states it generically; a
        subclass whose algorithm has a named fallback chain (e.g. stitching)
        overrides with its own wording so the shared loop does not hardcode a
        stitch-only vocabulary for zproject / composite / stack outputs.
        """
        return (
            f'{len(degraded_outputs)} {self.operation_label} artifact(s) used a fallback '
            f'method: {self._fallback_messages(degraded_outputs)}.'
        )

    @staticmethod
    def _fallback_messages(degraded_outputs: list[dict]) -> str:
        """Why each degraded output fell back, each reason said once."""
        return causes_in_words(
            (output['filepath'], output['fallback_message']) for output in degraded_outputs
        )

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
        tiling_grids: TilingGrids,
        on_progress: ProgressCallback | None = None,
        select: Callable[[pd.DataFrame], pd.Series] | None = None,
        **kwargs: dict,
    ) -> dict:
        """Run this operation over a captured folder; return the complete result.

        Returns only when every output asked for was produced (skipped
        single-image groups, already-recorded groups and a fallback
        algorithm's degraded output are complete results, reported in the
        dict). Anything else raises, and the caller reports it: this loader
        never posts, because only the caller knows whether someone is
        waiting on it.

        Args:
            select: Which of the folder's rows this build takes, for a build
                in a chain (``PostProcessingAPI.build``); None takes every
                row. A chained build's answer says what it carries forward
                (``carried_forward``): its input with the rows it consumed
                replaced by what it made from them. A row its own filter
                drops is carried, not left out.

        Raises:
            PostProcessingRefusedError: The folder cannot yield this output.
            PostProcessingFailedError: The build did not produce everything
                it was asked for; what it did produce rides along.
        """
        return self._load_folder_inner(
            path=path,
            tiling_grids=tiling_grids,
            on_progress=on_progress,
            select=select,
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
        wait" one.
        """
        return operation_key(self._post_function)

    def _incomplete(
        self,
        *,
        group_errors: 'list[tuple[str, str]]',
        attempted: int,
        refused_count: int,
        colliding_names: set,
        dropped_frames: 'Sequence[tuple[str, str]]',
        artifact_paths: list,
        output_root: str,
        accounting_note: str,
        excluded_inputs: tuple[ExcludedInput, ...],
        stopped: '_Unwritten | None' = None,
    ) -> PostProcessingFailedError:
        """The outcome of a build that did not produce everything asked of it.

        It carries what the build left out as a complete result does, so a
        group that failed does not also silence the inputs that were never
        tried. *stopped* is the output a build stopped at, which it says
        first. *group_errors* and *dropped_frames* are (item, cause) pairs,
        each cause said once.
        """
        label = self.operation_label.lower()
        missing = []
        errors = list(group_errors)
        if stopped is not None:
            missing.append(stopped.sentence())
            errors.append((str(stopped.output), stopped.reason()))
        if group_errors:
            missing.append(
                f'{len(group_errors)} of {attempted} {label} group(s) failed: '
                f'{causes_in_words(group_errors)}'
            )
        if refused_count:
            names = first_few(sorted(colliding_names), separator=', ')
            missing.append(
                f'{refused_count} group(s) were refused because more than one group '
                f'derives the same output filename ({names})'
            )
        if dropped_frames:
            errors.extend(dropped_frames)
            missing.append(
                f'{len(dropped_frames)} frame(s) could not be added, so the video is '
                f'shorter than its source: {causes_in_words(dropped_frames)}'
            )
        # A sentence each: a cause can hold a semicolon of its own.
        sentence = '. '.join(f'{part[0].upper()}{part[1:]}' for part in missing)
        record_note = ''
        if stopped is not None and stopped.unrecorded is not None:
            record_note = (
                ' The post-processing record could not be updated '
                f'({system_words(stopped.unrecorded)}).'
            )
        return PostProcessingFailedError(
            operation=self.operation_label,
            missing=f'{sentence}.{record_note}{accounting_note}',
            produced_paths=artifact_paths,
            output_root=output_root,
            errors=errors,
            excluded_inputs=excluded_inputs,
        )

    def _refuse(self, reason: str, message: str) -> PostProcessingRefusedError:
        return PostProcessingRefusedError(
            operation=self.operation_label, reason=reason, message=message
        )

    def _load_folder_inner(
        self,
        path: str | pathlib.Path,
        tiling_grids: TilingGrids,
        on_progress: ProgressCallback | None = None,
        select: Callable[[pd.DataFrame], pd.Series] | None = None,
        **kwargs: dict,
    ) -> dict:
        start_ns = host_clock.steady_ns()
        if not path:
            raise self._refuse('invalid_path', 'No folder was given to process.')

        selected_path = pathlib.Path(path)
        results = self._post_processing_helper.load_folder(
            path=selected_path,
            tiling_grids=tiling_grids,
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
        if select is not None:
            # A chained build takes only what the build before it carried
            # forward; the rest of the folder is not its input, so not an
            # input it left out either.
            df = df[select(df)]

        # A build takes TIFF or OME-TIFF. A JPG holds the display, 8-bit and
        # lossy with the display colors applied, and records no capture
        # facts: a projection or composite of it is a picture of the screen,
        # and a stitch of it is placed by geometry alone.
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
                f'{self.operation_label} requires TIFF or OME-TIFF source images. '
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
        left_out_counts: dict[PostFunction, int] = {}
        already_made = 0
        dropped_rows = pre_filter_df.loc[pre_filter_df.index.difference(df.index)]
        # A derived image gone from the folder is not a file in it: neither
        # an output already made nor an input left out.
        gone = _missing(dropped_rows)
        if gone.any():
            dropped_rows = dropped_rows[~(gone & derived_rows(dropped_rows))]
        # A recording's frames are no stills build's input, and nobody looks
        # for them in one: only Video takes them, so they are not left out.
        dropped_rows = dropped_rows[~_recording_frames(dropped_rows)]
        # What a chained build carries forward: every input row it does not
        # consume (its filter's drops, a single-image group's, ...), and in
        # place of each group it consumes, that group's output.
        carried = set(pre_filter_df['Filepath'])
        for _, dropped in dropped_rows.iterrows():
            if dropped[own_flag]:
                already_made += 1
                continue
            kind = next((flag for flag in PostFunction.list_values() if dropped[flag]), None)
            if kind is None:
                # Every filter drops a row by its output flag or as a
                # recording's frame, so a dropped row with neither is a filter
                # this accounting does not know.
                raise RuntimeError(
                    f'{self._name}: {dropped["Filepath"]} was dropped with no output kind'
                )
            left_out = PostFunction(kind)
            left_out_counts[left_out] = left_out_counts.get(left_out, 0) + 1
        excluded_kinds = set(left_out_counts)
        # The record names each kind by its operation; the words a person
        # reads name it by its files, as the build flags do.
        left_out_in_order = sorted(left_out_counts.items(), key=lambda item: item[0].value.lower())
        excluded_inputs = tuple(
            ExcludedInput(kind=_OPERATION_LABELS[kind], count=count)
            for kind, count in left_out_in_order
        )
        excluded_text = ', '.join(
            f'{count} {kind.value.lower()} file(s)' for kind, count in left_out_in_order
        )
        derived_counts = {kind.value.lower(): count for kind, count in left_out_counts.items()}
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

        # A folder whose only images are a recording holds nothing a stills
        # build takes. Asked of the frames present: a recording deleted since
        # its run is one Video cannot build either.
        if df.empty and (_recording_frames(pre_filter_df) & ~_missing(pre_filter_df)).any():
            raise self._refuse(
                'no_data',
                f'No {self.operation_label} was generated: this folder holds a recording, '
                'not stills; Video builds it.',
            )

        groups = self._get_groups(df)

        group_count = len(groups)

        # Each group's plan for the run's last scan, for a build that takes
        # only whole scans; none for any other.
        plans, last_scan = {}, None
        if self._WHOLE_SCANS_ONLY:
            plans = _plans_of(self._get_groups, df, results['planned_df'])
            last_scan = results['last_scan']
        # Each group whose share of the last scan was left out, as a person
        # reads it.
        left_out = []

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
            logger.info(
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
        # Each frame a video could not add, and why.
        dropped_frames = []
        degraded_outputs = []
        output_significant_bits = None
        completed_group_ms = []
        skipped_single_paths = []
        # Each one-image group whose image is not in the folder: named apart
        # from the skipped singles, which are saved and these are not.
        no_image_groups = []
        # The build that made each removed group's images (``_removed``).
        removed_groups: list[PostProcessingOperation] = []
        # Where each produced artifact landed. The return used to carry only
        # the output ROOT, which tells a caller a directory was touched but
        # not whether its own file exists -- an unattended caller could not
        # distinguish a successful run from one that produced nothing, and
        # had to re-derive the filename to find out.
        artifact_paths = []
        # The output the build could not write, when it stopped at one, and
        # why: the builds after it would fail the same way on a full disk.
        unwritten = None

        for _, group in groups:
            if len(group) == 0:
                continue

            # Before the single-image skip, so a removed group of one image
            # is named too.
            if _removed(group):
                removed_groups.append(_made_by(group.iloc[0]['Derivation']))
                carried -= set(group['Filepath'])
                logger.info(
                    f'[{self._name} ] Not building from {group.iloc[0]["Filepath"]} and the '
                    f'{len(group) - 1} other image(s) of its group: all removed from the folder.'
                )
                continue

            # Before the single-image skip and the missing-frame refusal: a
            # last scan the run ended partway through is neither a single
            # image nor a lost frame, and a group left with none of its own
            # scans is named, not skipped.
            whole = group
            cut = _cut_short(group, plans.get(group.index[0], frozenset()), last_scan)
            if not cut.empty:
                group = group.drop(index=cut.index)
                subject = _subject(whole)
                if group.empty:
                    _, _, cut_loc = self._planned_output(root_path, whole, **kwargs)
                    cut_loc_rel = cut_loc.relative_to(root_path)
                    last_error = (
                        f'{subject[0].upper()}{subject[1:]} has no complete scan: the run '
                        f'ended partway through its only scan. Its {len(cut)} image(s) '
                        'are saved.'
                    )
                    group_errors.append((str(cut_loc_rel), last_error))
                    logger.info(f'Not generating {cut_loc_rel}: {last_error}')
                    continue
                left_out.append(f'{subject} ({len(cut)} image(s))')

            # A row for a missing recording stands for its frames, so the
            # group it is in counts as the group it would have been.
            missing = group[_missing(group)]
            frames = len(group) - len(missing) + int(missing['Frames'].sum() if len(missing) else 0)
            if frames == 1:
                if len(group) > len(missing):
                    present = group[~_missing(group)]
                    skipped_single_paths.append(str(present.iloc[0]['Filepath']))
                    logger.debug(
                        f'[{self._name} ] Skipping generation for {present.iloc[0]["Filepath"]} since only {len(group)} image found.'
                    )
                else:
                    no_image_groups.append(_subject(group))
                    logger.info(
                        f'[{self._name} ] Nothing to combine at {_subject(group)}, and its '
                        f'one image is not in the folder: {_missing_frames_error(missing)}'
                    )
                continue

            # Named as the collision check named it, from every row it held.
            output_filename, derivation, output_file_loc = self._planned_output(
                root_path, whole, **kwargs
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
                    ('color', 'channel'),
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
                carried -= set(group['Filepath'])
                carried.add(output_file_loc_rel)
                continue

            doubled = self._two_images_in_one_place(group)
            if doubled is not None:
                last_error = doubled
                group_errors.append((str(output_file_loc_rel), doubled))
                logger.info(f'Not generating {output_file_loc_rel}: {doubled}')
                continue

            # Built whole or not at all: a group missing a frame the run
            # recorded -- deleted since, or never captured -- is this group's
            # failure, never an output short of it.
            if not missing.empty:
                last_error = _missing_frames_error(missing)
                group_errors.append((str(output_file_loc_rel), last_error))
                logger.info(f'Not generating {output_file_loc_rel}: {last_error}')
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
            except (OSError, VideoNotFinishedError) as failed:
                # Every read of a source raises the refusal above, so an
                # OSError here is the write of this group's output.
                unwritten = _Unwritten(output_file_loc_rel, failed, written=False)
                break
            group_ms = (time.perf_counter() - group_t0) * 1000.0

            if not alg_results.status:
                last_error = alg_results.error
                group_errors.append((str(output_file_loc_rel), alg_results.error))
                logger.info(
                    f'[PostProcPerf] {self._name} group failed after {group_ms:.1f}ms: {group_label}'
                )
                logger.info(f'Failed to generate {output_file_loc_rel}: {alg_results.error}')
                continue

            completed_group_ms.append(group_ms)

            alg_metadata = alg_results.record_metadata
            # A video writes what frames it could read; the ones it could not
            # leave it shorter than its source, which the build must say.
            dropped_frames.extend(alg_metadata.get('dropped_frames', ()))
            logger.info(
                f'[PostProcPerf] {self._name} group done in {group_ms:.1f}ms: '
                f'algorithm={alg_metadata.get("algorithm", "")} '
                f'degraded={bool(alg_metadata.get("fallback_message"))} {group_label}'
            )
            fallback_message = alg_metadata.get('fallback_message')
            if fallback_message:
                degraded_outputs.append(
                    {
                        'filepath': str(output_file_loc_rel),
                        'algorithm': alg_metadata.get('algorithm', ''),
                        'fallback_from': alg_metadata.get('fallback_from', ''),
                        'fallback_message': fallback_message,
                    }
                )

            # Each ProtocolPostProcessor subclass owns its own file write via
            # tifffile (RGB-native; auto-detects photometric). cv2.imwrite was
            # retired here -- cv2 is BGR-native and would silently swap channels
            # relative to the RGB-native readers (tifffile / FIJI / OS preview).
            # A write that fails raises, and the build stops there.
            try:
                self._add_record(
                    protocol_post_record=protocol_post_record,
                    alg_metadata=alg_results.record_metadata,
                    root_path=root_path,
                    file_path=output_file_loc_rel,
                    row0=row0,
                    derivation=derivation,
                )
            except OSError as failed:
                # The output is on disk, though the record does not list it.
                artifact_paths.append(str(root_path / output_file_loc_rel))
                unwritten = _Unwritten(output_file_loc_rel, failed, written=True)
                break

            artifact_paths.append(str(root_path / output_file_loc_rel))
            carried -= set(group['Filepath'])
            carried.add(output_file_loc_rel)

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
                        if kwargs.get('stitching_mode') == StitchMode.FAST_PREVIEW
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

        try:
            protocol_post_record.complete()
        except OSError as unrecorded:
            # Told beside the stop, never in its place.
            if unwritten is None:
                raise
            unwritten = unwritten._replace(unrecorded=unrecorded)

        if on_progress is not None:
            on_progress(100, None)

        if unwritten is not None:
            raise self._incomplete(
                group_errors=group_errors,
                attempted=len(group_errors) + new_count,
                refused_count=refused_count,
                colliding_names=colliding_names,
                dropped_frames=dropped_frames,
                artifact_paths=artifact_paths,
                output_root=str(root_path),
                accounting_note='',
                excluded_inputs=excluded_inputs,
                stopped=unwritten,
            ) from unwritten.failed

        if skipped_single_paths:
            shown = first_few(skipped_single_paths, separator=', ')
            logger.info(
                f'[{self._name} ] Skipped {len(skipped_single_paths)} single-image '
                f'group(s) (nothing to combine): {shown}'
            )

        single_skip_note = (
            f' {len(skipped_single_paths)} single-image group(s) skipped.'
            if skipped_single_paths
            else ''
        )
        no_image_note = (
            f' {len(no_image_groups)} single-image group(s) have no image in the folder: '
            f'{first_few(no_image_groups, separator=", ")}.'
            if no_image_groups
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
        left_out_note = (
            ' The run ended partway through its last scan, so that scan is not in the '
            f'{self.operation_label.lower()} of {", ".join(left_out)}; its images are saved.'
            if left_out
            else ''
        )
        accounting_note = (
            f'{single_skip_note}{no_image_note}{excluded_note}{removed_note}{left_out_note}'
        )

        # The answer for a build that declares a folder of single images
        # complete, so its reader need not infer it from the counts.
        nothing_to_combine = False
        # A removed group is a complete answer, as an output already made is.
        if (new_count == 0) and (existing_count == 0) and not removed_groups:
            fname = self.operation_label
            if refused_count > 0 and last_error is None:
                # Every eligible group collided; nothing could be generated.
                raise self._refuse(
                    'collision',
                    f'No {fname} was generated: every image group derives an '
                    f'output filename shared with another group, so their '
                    f'artifacts would be indistinguishable. The shared names: '
                    f'{first_few(sorted(colliding_names), separator=", ")}.',
                )
            if derived_text and last_error is None:
                # The emptiness is explained by what the filter excluded, not
                # by the folder's structure -- a structural hint here sent
                # users hunting for missing tiles a composite folder has.
                # Own earlier outputs count here: a folder of nothing else
                # is explained by them.
                raise self._refuse(
                    'excluded_inputs',
                    f'No {fname} was generated: this folder holds only '
                    f'{derived_text}, which are derived outputs.'
                    f'{remedy}{single_skip_note}',
                )
            needed = _MULTI_FRAME_REQUIREMENT.get(
                self._post_function, 'multiple frames per scan position'
            )
            # Every group held one image: for a build that declares that a
            # complete answer, it falls through to the success below.
            only_singles = bool(skipped_single_paths or no_image_groups)
            nothing_to_combine = last_error is None and self._SINGLES_ARE_COMPLETE and only_singles
            if last_error is None and not nothing_to_combine:
                found = (
                    'Every scan position in the folder holds one image.'
                    if only_singles
                    else 'The folder holds no images this build can take.'
                )
                raise self._refuse(
                    'no_data',
                    f'No {fname} was generated. {fname} requires {needed}. '
                    f'{found}{accounting_note}',
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

        elapsed_time = datetime.timedelta(microseconds=(host_clock.steady_ns() - start_ns) // 1000)
        logger.info(
            f'{self._name}: Complete - Created {new_count} {self._post_function.value.lower()} '
            f'artifacts (significant_bits={output_significant_bits}) in {elapsed_time}.'
            f'{accounting_note}'
        )
        if degraded_outputs:
            summary = self._degraded_summary(degraded_outputs)
            logger.warning(f'{self._name}: Complete with degraded outputs: {summary}')
            return {
                'status': True,
                'message': f'Success with degraded output: {summary}{accounting_note}',
                'degraded': True,
                'degraded_outputs': degraded_outputs,
                'output_root': str(root_path),
                'artifact_paths': artifact_paths,
                'carried_forward': frozenset(carried),
                'accounting_note': accounting_note,
                'excluded_inputs': excluded_inputs,
            }
        return {
            'status': True,
            'message': f'Success.{accounting_note}',
            'nothing_to_combine': nothing_to_combine,
            'output_root': str(root_path),
            'artifact_paths': artifact_paths,
            'carried_forward': frozenset(carried),
            'accounting_note': accounting_note,
            'excluded_inputs': excluded_inputs,
        }
