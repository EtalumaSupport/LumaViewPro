# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import abc
import datetime
import enum
import pathlib

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
        raise NotImplementedError(f'Implement in child class')

    @abc.abstractmethod
    def _generate_filename(self, df: pd.DataFrame, **kwargs) -> str:
        raise NotImplementedError(f'Implement in child class')

    @abc.abstractmethod
    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f'Implement in child class')

    @abc.abstractmethod
    def _group_algorithm(path: pathlib.Path, df: pd.DataFrame):
        raise NotImplementedError(f'Implement in child class')

    @staticmethod
    @abc.abstractmethod
    def _add_record(
        protocol_post_record: ProtocolPostRecord,
        alg_metadata: dict,
        root_path: pathlib.Path,
    ):
        raise NotImplementedError(f'Implement in child class')

    def _get_objective_short_name_if_has_turret(self, objective_id: str) -> str | None:
        if self._has_turret:
            short_name = self._objectives_helper.get_objective_info(objective_id=objective_id)[
                'short_name'
            ]
        else:
            short_name = None

        return short_name

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

        root_path = results['root_path']
        protocol_post_record = results['protocol_post_record']

        df = self._filter_ignored_types(df=df)
        groups = self._get_groups(df)

        group_count = len(groups)

        logger.info(f'{self._name}: Generating {self._post_function.value.lower()} images')

        new_count = 0
        existing_count = 0
        current_group = 1

        for _, group in groups:
            if len(group) == 0:
                continue

            if len(group) == 1:
                logger.debug(
                    f'[{self._name} ] Skipping generation for {group.iloc[0]["Filepath"]} since only {len(group)} image found.'
                )
                continue

            output_filename = self._generate_filename(df=group, **kwargs)
            row0 = group.iloc[0]
            record_data_post_functions = row0[PostFunction.list_values()]
            record_data_post_functions[self._post_function.value] = True
            output_subfolder = self._post_processing_helper.generate_output_dir_name(
                record=record_data_post_functions
            )
            output_path = root_path / output_subfolder
            output_file_loc = output_path / output_filename
            output_file_loc_rel = output_file_loc.relative_to(root_path)

            if protocol_post_record.file_exists_in_records(filepath=output_file_loc_rel):
                logger.info(
                    f'[{self._name} ] {output_file_loc_rel} already exists in record, skipping for generation.'
                )
                existing_count += (
                    1  # Count this so we don't error out if no other matches are found
                )
                continue

            kwargs['output_file_loc'] = output_file_loc_rel

            alg_results = self._group_algorithm(
                path=root_path,
                df=group,
                popup=popup,
                total_groups=group_count,
                current_group=current_group,
                **kwargs,
            )

            if not alg_results['status']:
                logger.error(f'Failed to generate {output_file_loc_rel}: {alg_results["error"]}')
                continue

            # Each ProtocolPostProcessor subclass owns its own file
            # write via tifffile (RGB-native; auto-detects photometric).
            # cv2.imwrite was retired here -- cv2 is BGR-native and
            # would silently swap channels relative to the RGB-native
            # readers (tifffile / FIJI / OS preview). Subclasses that
            # fail to write must signal status=False; an
            # alg_results['image'] payload is now informational, not
            # a save trigger.

            self._add_record(
                protocol_post_record=protocol_post_record,
                alg_metadata=alg_results['metadata'],
                root_path=root_path,
                file_path=output_file_loc_rel,
                row0=row0,
                **record_data_post_functions.to_dict(),
            )

            new_count += 1
            current_group += 1

            if popup is not None:
                popup.progress = (new_count / group_count) * 100

        protocol_post_record.complete()

        if popup is not None:
            popup.progress = 100

        if (new_count == 0) and (existing_count == 0):
            fname = self._post_function.value
            needed = _MULTI_FRAME_REQUIREMENT.get(
                self._post_function, 'multiple frames per scan position'
            )
            logger.info(
                f'[{self._name} ] No {fname} output generated -- '
                f'no usable image groups (need {needed})'
            )
            return {
                'status': False,
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
            f'{self._name}: Complete - Created {new_count} {self._post_function.value.lower()} artifacts in {elapsed_time}.'
        )
        return {'status': True, 'message': 'Success'}
