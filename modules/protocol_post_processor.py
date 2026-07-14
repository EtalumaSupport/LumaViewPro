# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import abc
import datetime
import pathlib
import time

import pandas as pd

from modules.common_utils import PostFunction
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
        source_suffixes = {pathlib.Path(str(fp)).suffix.lower() for fp in df['Filepath']}
        if source_suffixes and source_suffixes.isdisjoint({'.tif', '.tiff'}):
            return {
                'status': False,
                'message': (
                    'Composite, stitch, and z-projection require TIFF or '
                    'OME-TIFF source images. This scan was saved as JPG, '
                    'which cannot be re-read for post-processing.'
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

        df = self._filter_ignored_types(df=df)
        groups = self._get_groups(df)

        group_count = len(groups)

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

        for _, group in groups:
            if len(group) == 0:
                continue

            if len(group) == 1:
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
            logger.info(f'[StitchPerf] {self._name} group start: {group_label}')
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
                    f'[StitchPerf] {self._name} group failed after {group_ms:.1f}ms: {group_label}'
                )
                logger.error(f'Failed to generate {output_file_loc_rel}: {alg_results.error}')
                continue

            alg_metadata = alg_results.record_metadata
            logger.info(
                f'[StitchPerf] {self._name} group done in {group_ms:.1f}ms: '
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
                    f'the folder to be compatible.'
                ),
            }

        end_ts = datetime.datetime.now()
        elapsed_time = end_ts - start_ts
        logger.info(
            f'{self._name}: Complete - Created {new_count} {self._post_function.value.lower()} '
            f'artifacts (significant_bits={output_significant_bits}) in {elapsed_time}.'
            f'{collision_note}'
        )
        if degraded_outputs:
            logger.warning(
                f'{self._name}: Complete with degraded outputs: {len(degraded_outputs)} '
                f'{self._post_function.value.lower()} artifact(s) used fallback stitching.'
            )
            return {
                'status': True,
                'message': (
                    f'Success with degraded output: {len(degraded_outputs)} '
                    f'{self._post_function.value.lower()} artifact(s) used fallback stitching.'
                ),
                'degraded': True,
                'degraded_outputs': degraded_outputs,
            }
        return {'status': True, 'message': f'Success.{collision_note}'}
