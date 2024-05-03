
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
    

    def _zproject_for_multi_channel(
        self,
        images_data: list[np.ndarray],
        method: str
    ) -> np.ndarray | None:
        sample_image = images_data[0]
        used_color_planes = image_utils.get_used_color_planes(image=sample_image)
        out_image = np.zeros_like(sample_image, dtype=sample_image.dtype)

        for used_color_plane in used_color_planes:
            images_for_color_plane = []

            for image_data in images_data:
                images_for_color_plane.append(image_data[:,:,used_color_plane])

            project_result = self._ij_helper.zproject(
                images_data=images_for_color_plane,
                method=method
            )

            if project_result is None:
                logger.error(f"Failed to create Z-Projection for color plane {used_color_plane}")
                return None
            
            out_image[:,:,used_color_plane] = project_result
        
        return out_image
    

    def _zproject_for_single_channel(
        self,
        images_data: list[np.ndarray],
        method: str
    ) -> np.ndarray | None:
        project_result = self._ij_helper.zproject(
            images_data=images_data,
            method=method
        )

        if project_result is None:
            logger.error(f"Failed to create Z-Projection")
            return None
        
        return project_result
    
    
    def _zproject(self, path: pathlib.Path, df: pd.DataFrame, method):

        orig_images = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filename']
            orig_images.append(tf.imread(str(image_filepath)))

        # If working with color images, split the list of color images into separate lists for 
        # each color plane
        if image_utils.is_color_image(image=orig_images[0]):
            result = self._zproject_for_multi_channel(
                images_data=orig_images,
                method=method,
            )
        
        else: # Grayscale images
            result = self._zproject_for_single_channel(
                images_data=orig_images,
                method=method
            )

        if result is None:
            logger.error(f"Failed to create Z-Projection")
            return False
        
        filename = self._generate_zproject_filename(df=df, method=method)
        file_loc = path / filename

        write_result = tf.imwrite(
            file_loc,
            data=result,
            compression='lzw',
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


