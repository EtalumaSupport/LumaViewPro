
import itertools
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper
# import image_utils

from lvp_logger import logger


class CompositeGeneration:

    def __init__(self):
        self._name = self.__class__.__name__
        self._protocol_post_processing_helper = ProtocolPostProcessingHelper()

    
    def load_folder(self, path: str | pathlib.Path) -> dict:
        results = self._protocol_post_processing_helper.load_folder(
            path=path,
            include_stitched_images=True,
            include_composite_images=False
        )

        df = results['image_tile_groups']
        df['Composite Group Index'] = df.groupby(by=['Scan Count','Z-Slice','Well','Objective','X','Y'], dropna=False).ngroup()
        
        # Handle composite generation for stitched images also
        stitched_images_df = results['stitched_images']

        if stitched_images_df is not None:
            stitched_images_df['Composite Group Index'] = stitched_images_df.groupby(by=['Scan Count', 'Z-Slice', 'Well'], dropna=False).ngroup()
            loop_list = itertools.chain(
                df.groupby(by=['Composite Group Index']),
                stitched_images_df.groupby(by=['Composite Group Index'])
            )
        else:
            loop_list = df.groupby(by=['Composite Group Index'])

        logger.info(f"{self._name}: Generating composite images")
        for _, composite_group in loop_list:
            
            if len(composite_group) == 0:
                continue

            if len(composite_group) == 1:
                logger.debug(f"{self._name}: Skipping composite generation for {composite_group.iloc[0]['Filename']} since {len(composite_group)} layer(s) found.")
                continue

            composite_filename = common_utils.replace_layer_in_step_name(
                step_name=composite_group.iloc[0]['Filename'],
                new_layer_name='Composite'
            )

            # Create parent folder if needed
            split_name = os.path.split(composite_filename)
            if len(split_name) == 2:
                composite_path = path / split_name[0]
                pathlib.Path(composite_path).mkdir(parents=True, exist_ok=True)

            # Filter out non-fluorescence layers
            allowed_layers = common_utils.get_fluorescence_layers()
            composite_group = composite_group[composite_group['Color'].isin(allowed_layers)]


            composite_image = self._create_composite_image(
                path=path,
                df=composite_group[['Filename','Color']]
            )

            logger.debug(f"{self._name}: - {composite_filename}")

            _ = cv2.imwrite(
                filename=str(path / composite_filename),
                img=composite_image
            )
        
        logger.info(f"{self._name}: Complete")
        return {
            'status': True,
            'message': 'Success'
        }


    @staticmethod
    def _create_composite_image(path: pathlib.Path, df: pd.DataFrame):

        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filename']
            images[row['Filename']] = cv2.imread(str(image_filepath))

        source_image_sample = df['Filename'].values[0]
        img = np.zeros_like(images[source_image_sample])

        for _, row in df.iterrows():
            layer = row['Color']
            source_image = images[row['Filename']]

            if layer == 'Blue':
                img[:,:,0] = source_image[:,:,0]
            elif layer == 'Green':
                img[:,:,1] = source_image[:,:,1]
            elif layer == 'Red':
                img[:,:,2] = source_image[:,:,2]

        return img
       

if __name__ == "__main__":
    composite_gen = CompositeGeneration()
    composite_gen.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
