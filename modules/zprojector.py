
import itertools
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

import modules.artifact_locations as artifact_locations
import modules.common_utils as common_utils
import image_utils
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper

import modules.imagej_helper as imagej_helper

from lvp_logger import logger


class ZProjector:

    def __init__(self, ij_helper: imagej_helper.ImageJHelper = None):
        self._name = self.__class__.__name__
        self._protocol_post_processing_helper = ProtocolPostProcessingHelper()
        
        if ij_helper is None:
            self._ij_helper = imagej_helper.ImageJHelper()
        else:
            self._ij_helper = ij_helper
        

    @staticmethod
    def methods() -> list[str]:
        return imagej_helper.ZProjectMethod.list()


    @staticmethod
    def _generate_zproject_filename(
        df: pd.DataFrame,
        method
    ) -> str:
        row0 = df.iloc[0]

        name = common_utils.generate_default_step_name(
            custom_name_prefix=row0['Name'],
            well_label=row0['Well'],
            color=row0['Color'],
            z_height_idx=None,
            scan_count=row0['Scan Count'],
            tile_label=row0['Tile'],
        )
        
        outfile = f"{name}_zproj_{method.value}.tiff"
        return outfile
    
    
    def _zproject(self, path: pathlib.Path, df: pd.DataFrame, method):

        images = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filename']
            images.append(tf.imread(str(image_filepath)))
        
        project_result = self._ij_helper.zproject(
            images_data=images,
            method=method
        )

        if project_result is None:
            return False
        
        filename = self._generate_zproject_filename(df=df, method=method)
        file_loc = path / filename

        write_result = tf.imwrite(
            file_loc,
            data=project_result,
        )

        if not write_result:
            return False
        
        return True
    

    def load_folder(self, path: str | pathlib.Path, tiling_configs_file_loc: pathlib.Path, method_name: str) -> dict:
        path = pathlib.Path(path)
        results = self._protocol_post_processing_helper.load_folder(
            path=path,
            tiling_configs_file_loc=tiling_configs_file_loc,
            include_stitched_images=False,
            include_composite_images=False,
            include_composite_and_stitched_images=False,
        )

        if results['status'] is False:
            return {
                'status': False,
                'message': f'Failed to load protocol data from {path}'
            }
        
        method = imagej_helper.ZProjectMethod[method_name]

        df = results['image_tile_groups']
        loop_list = df.groupby(by=['Z-Project Group Index'])
        logger.info(f"{self._name}: Generating Z-Projected images")

        count = 0
        for _, group in loop_list:

            if len(group) == 0:
                continue

            if len(group) == 1:
                logger.debug(f"{self._name}: Skipping Z-Project generation for {group.iloc[0]['Filename']} since only {len(group)} image found.")
                continue

            result = self._zproject(
                path=path,
                df=df,
                method=method,
            )
            
            count += 1

        if count == 0:
            logger.info(f"{self._name}: No sets of images found to run Z-Projection on")
            return {
                'status': False,
                'message': 'No images found'
            }
        
        logger.info(f"{self._name}: Complete")
        return {
            'status': True,
            'message': 'Success'
        }


