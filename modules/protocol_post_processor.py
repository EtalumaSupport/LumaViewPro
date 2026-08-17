# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import abc
import datetime
import pathlib
import time

import pandas as pd

import modules.image_utils as image_utils
import modules.recording_frames as recording_frames
from modules.common_utils import PostFunction
from modules.notification_center import notifications
from modules.objectives_loader import ObjectiveLoader
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
from modules.protocol_post_record import ProtocolPostRecord

from lvp_logger import logger


# What each post-processing function needs to find in a folder to produce
# output. Used in the empty-output user message so the popup explains which
# capture dimension was missing instead of saying "No images found" in a
# folder that visibly has images.
_MULTI_FRAME_REQUIREMENT = {
    PostFunction.VIDEO: 'multiple time points per scan position',
    PostFunction.ZPROJECT: 'multiple Z-slices per scan position',
    PostFunction.COMPOSITE: 'multiple channels per scan position',
    PostFunction.STITCHED: 'multiple tile positions per scan',
}


class ProtocolPostProcessor(abc.ABC):
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
        self._objectives_helper = ObjectiveLoader()

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
        recording_scan = df['Scan Count'].where(
            df['Filepath'].map(recording_frames.is_video_frame), -1
        )
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
        # astype(bool): on an EMPTY frame map() cannot infer a dtype, and
        # indexing with the resulting non-bool series degrades from a row
        # mask to column selection -- silently dropping every column.
        return df[~df['Filepath'].map(recording_frames.is_video_frame).astype(bool)]

    @abc.abstractmethod
    def _generate_filename(self, df: pd.DataFrame, **kwargs) -> str:
        raise NotImplementedError('Implement in child class')

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

    def _get_objective_short_name_if_has_turret(self, objective_id: str) -> str | None:
        if self._has_turret:
            short_name = self._objectives_helper.get_objective_info(objective_id=objective_id)[
                'short_name'
            ]
        else:
            short_name = None

        return short_name

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
        popup=None,
        **kwargs: dict,
    ) -> dict:
        """Run this operation over a captured folder; return the result dict.

        The popup is the attended lifecycle surface (the caller renders
        progress and completion). When no popup is supplied the run is
        UNATTENDED, so the notification bus becomes the surface: a start
        notice once the group count is known, and a completion or failure
        message on every exit path -- inherited by every subclass, so no
        unattended operation can finish (or fail) silently.
        """
        result = self._load_folder_inner(
            path=path,
            tiling_configs_file_loc=tiling_configs_file_loc,
            popup=popup,
            **kwargs,
        )
        if popup is None:
            self._notify_unattended_result(result)
        return result

    @property
    def _unattended_operation_key(self) -> str:
        """Ties the start notice to the outcome notice that answers it.

        Both ends must name the same operation or the outcome opens a second
        modal instead of replacing the "please wait" one. Derived in one place
        so the two ends cannot drift apart.
        """
        return f'post-processing:{self._post_function.value}'

    def _notify_unattended_result(self, result: dict) -> None:
        fname = self._post_function.value
        if result.get('status'):
            new_count = result.get('new_count')
            output_root = result.get('output_root')
            if result.get('degraded') or new_count is None or not output_root:
                body = result.get('message', 'Complete.')
            else:
                # The count-based body drops the message, so the drop/skip
                # accounting must ride along explicitly or unattended users
                # never see it (a silently skipped well looks like success).
                accounting = result.get('accounting_note', '')
                body = f'{new_count} {fname.lower()}(s) saved to {output_root}.{accounting}'
            notifications.notice(
                'Post-processing',
                f'{fname}s Saved',
                body,
                operation_key=self._unattended_operation_key,
            )
        else:
            notifications.error(
                'Post-processing',
                f'{fname} Save Failed',
                result.get('message', 'See lumaviewpro.log for details.'),
                operation_key=self._unattended_operation_key,
            )

    def _load_folder_inner(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        popup=None,
        **kwargs: dict,
    ) -> dict:
        start_ts = datetime.datetime.now()
        if not path:
            return {'status': False, 'message': 'Invalid path provided'}

        selected_path = pathlib.Path(path)
        results = self._post_processing_helper.load_folder(
            path=selected_path,
            tiling_configs_file_loc=tiling_configs_file_loc,
        )

        if results['status'] is False:
            return {
                'status': False,
                'message': f'Failed to load protocol data using path: {selected_path}',
            }

        df = results['images_df']
        if len(df) == 0:
            return {
                'status': False,
                'message': (
                    'No image files were found in the selected folder. '
                    'Check that the folder contains captured scan images.'
                ),
            }

        # Composite, stitch, and z-projection re-read the source frames via
        # tifffile, which cannot decode JPG. A scan saved as JPG has no
        # re-readable source for these outputs, so stop here with a clear
        # message instead of letting tifffile raise partway through a group.
        unsupported_source = next(
            (
                pathlib.Path(str(filepath))
                for filepath in df['Filepath']
                if pathlib.Path(str(filepath)).suffix.lower() not in image_utils.TIFF_SUFFIXES
            ),
            None,
        )
        if unsupported_source is not None:
            source_format = unsupported_source.suffix.lstrip('.').upper() or 'unknown'
            return {
                'status': False,
                'reason': 'unsupported_source_format',
                'message': (
                    f'{self._post_function.value} requires TIFF or OME-TIFF source images. '
                    f'First unsupported {source_format} file: {unsupported_source.name}. '
                    'Reacquire the scan as TIFF or OME-TIFF before post-processing.'
                ),
            }

        root_path = results['root_path']
        protocol_post_record = results['protocol_post_record']

        # The per-image protocol writer (protocol_image_writer.py) prefixes
        # filenames with protocol.capture_root() so a scan run with Root
        # "experiment1" produces "experiment1_<step>_<color>_<...>.tiff".
        # Post-processed outputs (composite, stitch, z-proj, video, stack)
        # must use the same prefix; pipe it via kwargs to _generate_filename.
        protocol = results.get('protocol')
        kwargs.setdefault(
            'capture_root',
            protocol.capture_root() if protocol is not None else '',
        )

        # Account for every input the subclass filter drops: the completion
        # and empty-result messages are built from survivors, so uncounted
        # drops misattribute the outcome (a composite-only folder read as
        # "lacks multi-tile structure"). Dropped rows self-describe their
        # category through their own PostFunction flags.
        pre_filter_df = df
        df = self._filter_ignored_types(df=df)
        excluded_counts: dict[str, int] = {}
        dropped_rows = pre_filter_df.loc[pre_filter_df.index.difference(df.index)]
        for _, dropped in dropped_rows.iterrows():
            for flag in PostFunction.list_values():
                if dropped[flag]:
                    category = f'{flag.lower()} file(s)'
                    break
            else:
                category = 'other file(s)'
            excluded_counts[category] = excluded_counts.get(category, 0) + 1
        if excluded_counts:
            excluded_text = ', '.join(
                f'{count} {category}' for category, count in sorted(excluded_counts.items())
            )
            logger.info(
                f'[{self._name} ] {len(dropped_rows)} input file(s) excluded from '
                f'{self._post_function.value.lower()} generation: {excluded_text}'
            )
        else:
            excluded_text = ''

        groups = self._get_groups(df)

        group_count = len(groups)

        if popup is None:
            # Unattended run: announce the start so a multi-minute build is
            # not a silent hang; the paired completion/failure message is
            # emitted by the load_folder wrapper.
            fname = self._post_function.value
            notifications.notice(
                'Post-processing',
                f'Saving {fname}s',
                f'Building {group_count} {fname.lower()}(s). This can take '
                f'several minutes; a message will confirm completion.',
                operation_key=self._unattended_operation_key,
            )

        # When two DIFFERENT groups would render one output filename,
        # generating them would produce only the first group's artifact (the
        # second is skipped as already-recorded) -- a silent data loss.
        # Refuse exactly those groups, loudly, and still process the
        # unambiguous ones: an already-captured folder with baked-in
        # colliding names (which renaming protocol steps cannot repair)
        # stays post-processable for everything else.
        planned_names = {}
        for _, group in groups:
            if len(group) <= 1:
                continue
            name = self._generate_filename(df=group, **kwargs)
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
        degraded_outputs = []
        output_significant_bits = None
        completed_group_ms = []
        skipped_single_paths = []

        for _, group in groups:
            if len(group) == 0:
                continue

            if len(group) == 1:
                skipped_single_paths.append(str(group.iloc[0]['Filepath']))
                logger.debug(
                    f'[{self._name} ] Skipping generation for {group.iloc[0]["Filepath"]} since only {len(group)} image found.'
                )
                continue

            output_filename = self._generate_filename(df=group, **kwargs)
            if output_filename in colliding_names:
                refused_count += 1
                continue
            row0 = group.iloc[0]
            record_data_post_functions = row0[PostFunction.list_values()]
            record_data_post_functions[self._post_function.value] = True
            output_subfolder = self._post_processing_helper.generate_output_dir_name(
                record=record_data_post_functions
            )
            output_path = root_path / output_subfolder
            output_file_loc = output_path / output_filename
            output_file_loc_rel = output_file_loc.relative_to(root_path)
            group_label = (
                f'well={row0.get("Well", "")} color={row0.get("Color", "")} '
                f'tile_group={row0.get("Tile Group ID", "")} '
                f'tiles={len(group)} output={output_file_loc_rel}'
            )

            if protocol_post_record.file_exists_in_records(filepath=output_file_loc_rel):
                logger.info(
                    f'[{self._name} ] {output_file_loc_rel} already exists in record, skipping for generation.'
                )
                existing_count += (
                    1  # Count this so we don't error out if no other matches are found
                )
                continue

            kwargs['output_file_loc'] = output_file_loc_rel

            group_t0 = time.perf_counter()
            logger.info(f'[PostProcPerf] {self._name} group start: {group_label}')
            alg_results = self._group_algorithm(
                path=root_path,
                df=group,
                popup=popup,
                total_groups=group_count,
                current_group=current_group,
                **kwargs,
            )
            group_ms = (time.perf_counter() - group_t0) * 1000.0

            if not alg_results.status:
                last_error = alg_results.error
                logger.info(
                    f'[PostProcPerf] {self._name} group failed after {group_ms:.1f}ms: {group_label}'
                )
                logger.error(f'Failed to generate {output_file_loc_rel}: {alg_results.error}')
                continue

            completed_group_ms.append(group_ms)

            alg_metadata = alg_results.record_metadata
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
                **record_data_post_functions.to_dict(),
            )

            new_count += 1
            current_group += 1
            # Carry the depth the artifact was written at so the completion line
            # states whether the input depth round-tripped through this operation
            # instead of requiring a tag read on the output file. The typed result
            # guarantees a successful group carries its output depth, so this read
            # is always present.
            output_significant_bits = alg_results.significant_bits

            if popup is not None:
                popup.progress = (new_count / group_count) * 100
                if self._name == 'Stitcher':
                    mode_label = (
                        'Fast Preview'
                        if kwargs.get('stitching_mode') == 'fast_preview'
                        else 'Quality'
                    )
                    remaining = max(0, group_count - current_group)
                    average_ms = sum(completed_group_ms) / len(completed_group_ms)
                    estimate_seconds = round((remaining * average_ms) / 1000.0)
                    popup.text = (
                        f'Running {mode_label} Stitch -- group {current_group}/{group_count}.\n'
                        f'Estimated remaining time: about {estimate_seconds} seconds.\n'
                        'Source pixels and channel colors are preserved.'
                    )

        protocol_post_record.complete()

        if popup is not None:
            popup.progress = 100

        collision_note = ''
        if refused_count > 0:
            collision_note = (
                f' {refused_count} group(s) were refused because more than '
                f'one group derives the same output filename '
                f'({", ".join(sorted(colliding_names)[:3])}'
                f'{", ..." if len(colliding_names) > 3 else ""}); their '
                f'artifacts were not generated.'
            )

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
        excluded_note = f' Excluded from this operation: {excluded_text}.' if excluded_text else ''
        accounting_note = f'{single_skip_note}{excluded_note}'

        if (new_count == 0) and (existing_count == 0):
            fname = self._post_function.value
            if refused_count > 0:
                # Every eligible group collided; nothing could be generated.
                msg = (
                    f'No {fname} was generated: every image group derives an '
                    f'output filename shared with another group, so their '
                    f'artifacts would be indistinguishable. See '
                    f'lumaviewpro.log for the colliding names.'
                )
                logger.info(f'[{self._name} ] {msg}')
                return {
                    'status': False,
                    'reason': 'collision',
                    'message': msg,
                }
            if excluded_text and last_error is None:
                # The emptiness is explained by what the filter excluded, not
                # by the folder's structure -- a structural hint here sent
                # users hunting for missing tiles a composite folder has.
                fname_lower = fname.lower()
                msg = (
                    f'No {fname_lower} was generated: this folder holds '
                    f'{excluded_text}, which are derived outputs excluded '
                    f'from {fname_lower} generation. Only source channel '
                    f'images are processed.{single_skip_note}'
                )
                logger.info(f'[{self._name} ] {msg}')
                return {
                    'status': False,
                    'reason': 'excluded_inputs',
                    'message': msg,
                }
            needed = _MULTI_FRAME_REQUIREMENT.get(
                self._post_function, 'multiple frames per scan position'
            )
            if last_error is not None:
                # Usable groups WERE found and attempted, but every one failed
                # in the algorithm itself. Surface the real failure instead of
                # implying the folder lacked the data -- the prior message sent
                # users hunting for missing Z-stacks when the operation broke.
                logger.info(f'[{self._name} ] No {fname} output -- all groups failed: {last_error}')
                return {
                    'status': False,
                    'reason': 'error',
                    'message': (
                        f'{fname} could not be generated: {last_error}. '
                        f'See lumaviewpro.log for details.'
                    ),
                }
            logger.info(
                f'[{self._name} ] No {fname} output generated -- '
                f'no usable image groups (need {needed})'
            )
            return {
                'status': False,
                'reason': 'no_data',
                'message': (
                    f'No {fname} was generated. {fname} requires {needed}. '
                    f'The folder may have image files but not the structure '
                    f'this operation needs -- check the log if you expected '
                    f'the folder to be compatible.{accounting_note}'
                ),
            }

        end_ts = datetime.datetime.now()
        elapsed_time = end_ts - start_ts
        logger.info(
            f'{self._name}: Complete - Created {new_count} {self._post_function.value.lower()} '
            f'artifacts (significant_bits={output_significant_bits}) in {elapsed_time}.'
            f'{collision_note}{accounting_note}'
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
                'accounting_note': accounting_note,
            }
        return {
            'status': True,
            'message': f'Success.{collision_note}{accounting_note}',
            'new_count': new_count,
            'output_root': str(root_path),
            'accounting_note': accounting_note,
        }
