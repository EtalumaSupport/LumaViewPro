
import abc
import datetime
import pathlib

import cv2
import pandas as pd

from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
from modules.protocol_post_record import ProtocolPostRecord
import image_utils

from lvp_logger import logger


class ProtocolPostProcessingExecutor(abc.ABC):

    def __init__(
        self,
        post_function: PostFunction
    ):
        self._name = self.__class__.__name__
        self._post_function = post_function
        self._post_processing_helper = ProtocolPostProcessingHelper()
        

    @staticmethod
    @abc.abstractmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Implement in child class")


    @staticmethod
    @abc.abstractmethod
    def _generate_filename(df: pd.DataFrame, **kwargs) -> str:
        raise NotImplementedError(f"Implement in child class")


    @abc.abstractmethod
    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Implement in child class")
    

    @abc.abstractmethod
    def _group_algorithm(
        path: pathlib.Path,
        df: pd.DataFrame
    ):
        raise NotImplementedError(f"Implement in child class")
    

    @staticmethod
    @abc.abstractmethod
    def _add_record(
        protocol_post_record: ProtocolPostRecord,
        alg_metadata: dict,
        root_path: pathlib.Path,
    ):
        raise NotImplementedError(f"Implement in child class")


    def load_folder(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        **kwargs: dict,
    ) -> dict:
        start_ts = datetime.datetime.now()
        selected_path = pathlib.Path(path)
        results = self._post_processing_helper.load_folder(
            path=selected_path,
            tiling_configs_file_loc=tiling_configs_file_loc,
            axis_limits_mm=kwargs['axis_limits_mm'],
        )

        if results['status'] is False:
            return {
                'status': False,
                'message': f'Failed to load protocol data using path: {selected_path}'
            }
        
        df = results['images_df']
        if len(df) == 0:
            return {
                'status': False,
                'message': 'No images found in selected folder'
            }
        
        root_path = results['root_path']
        protocol_post_record = results['protocol_post_record']

        df = self._filter_ignored_types(df=df)
        groups = self._get_groups(df)

        logger.info(f"{self._name}: Generating {self._post_function.value.lower()} images")

        new_count = 0
        existing_count = 0

        for _, group in groups:
            if len(group) == 0:
                continue

            if len(group) == 1:
                logger.debug(f"[{self._name} ] Skipping generation for {group.iloc[0]['Filepath']} since only {len(group)} image found.")
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

            if protocol_post_record.file_exists_in_records(
                filepath=output_file_loc_rel
            ):
                logger.info(f"[{self._name} ] {output_file_loc_rel} already exists in record, skipping for generation.")
                existing_count += 1 # Count this so we don't error out if no other matches are found
                continue

            kwargs['output_file_loc'] = output_file_loc_rel

            alg_results = self._group_algorithm(
                path=root_path,
                df=group,
                **kwargs,
            )

            if not alg_results['status'] == True:
                logger.error(f"Failed to generate {output_file_loc_rel}: {alg_results['error']}")
                continue

            if 'image' in alg_results:
                if not output_path.exists():
                    output_path.mkdir(exist_ok=True, parents=True)
            
                logger.debug(f"[{self._name} ] Writing {output_file_loc_rel}")

                if not cv2.imwrite(
                    filename=str(output_file_loc),
                    img=alg_results['image']
                ):
                    logger.error(f"[{self._name} ] Unable to write image {output_file_loc}")
                    continue

            
            self._add_record(
                protocol_post_record=protocol_post_record,
                alg_metadata=alg_results['metadata'],
                root_path=root_path,
                file_path=output_file_loc_rel,
                row0=row0,
                **record_data_post_functions.to_dict(),
            )
      
            new_count += 1

        protocol_post_record.complete()

        if (new_count == 0) and (existing_count == 0):
            logger.info(f"[{self._name} ] No sets of images found")
            return {
                'status': False,
                'message': 'No images found'
            }

        end_ts = datetime.datetime.now()
        elapsed_time = end_ts - start_ts
        logger.info(f"{self._name}: Complete - Created {new_count} {self._post_function.value.lower()} artifacts in {elapsed_time}.")
        return {
            'status': True,
            'message': 'Success'
        }
    