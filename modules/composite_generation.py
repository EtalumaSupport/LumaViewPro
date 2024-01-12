
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
        results = self._protocol_post_processing_helper.load_folder(path=path)
        df = results['image_tile_groups']
        df['composite_group_index'] = df.groupby(by=['scan_count','z_slice','well','objective','x','y']).ngroup()
        
        logger.info(f"{self._name}: Generating composite images")
        for _, composite_group in df.groupby(by=['composite_group_index']):

            composite_filename = common_utils.replace_layer_in_step_name(
                step_name=composite_group.iloc[0]['filename'],
                new_layer_name='Composite'
            )

            # Filter out non-fluorescence layers
            allowed_layers = common_utils.get_fluorescence_layers()
            composite_group = composite_group[composite_group['color'].isin(allowed_layers)]

            if len(composite_group) <= 1:
                logger.debug(f"{self._name}: Skipping composite generation for {composite_filename} since {len(composite_group)} layer(s) found.")
                continue

            composite_image = self._create_composite_image(
                path=path,
                df=composite_group[['filename','color']]
            )

            logger.debug(f"{self._name}: - {composite_filename}")

            img_write_result = cv2.imwrite(
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
            image_filepath = path / row['filename']
            images[row['filename']] = cv2.imread(str(image_filepath))

        source_image_sample = df['filename'].values[0]
        img = np.zeros_like(images[source_image_sample])

        for _, row in df.iterrows():
            layer = row['color']
            source_image = images[row['filename']]

            if layer == 'Blue':
                img[:,:,0] = source_image[:,:,0]
            elif layer == 'Green':
                img[:,:,1] = source_image[:,:,1]
            elif layer == 'Red':
                img[:,:,2] = source_image[:,:,2]

        return img
        # for layer in common_utils.get_layers():
        #     if settings[layer]['acquire'] == True:

        #         # Go to focus and wait for arrival
        #         lumaview.ids['imagesettings_id'].ids[layer].goto_focus()

        #         while not lumaview.scope.get_target_status('Z'):
        #             time.sleep(.001)

        #         # set the gain and exposure
        #         gain = settings[layer]['gain']
        #         lumaview.scope.set_gain(gain)
        #         exposure = settings[layer]['exp']
        #         lumaview.scope.set_exposure_time(exposure)

        #         # update illumination to currently selected settings
        #         illumination = settings[layer]['ill']

        #         # Dark field capture
        #         scope_leds_off()

        #         # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        #         time.sleep(2*exposure/1000+0.2)
        #         scope_display.update_scopedisplay() # Why?
        #         darkfield = lumaview.scope.get_image()

        #         # Florescent capture
        #         if lumaview.scope.led:
        #             lumaview.scope.led_on(lumaview.scope.color2ch(layer), illumination)
        #             logger.info('[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch(layer), illumination)')
        #         else:
        #             logger.warning('LED controller not available.')

        #         # TODO: replace sleep + get_image with scope.capture - will require waiting on capture complete
        #         time.sleep(2*exposure/1000+0.2)
        #         exposed = lumaview.scope.get_image()

        #         scope_display.update_scopedisplay() # Why?
        #         corrected = exposed - np.minimum(exposed,darkfield)
        #         # buffer the images
        #         if layer == 'Blue':
        #             img[:,:,0] = corrected
        #         elif layer == 'Green':
        #             img[:,:,1] = corrected
        #         elif layer == 'Red':
        #             img[:,:,2] = corrected


if __name__ == "__main__":
    composite_gen = CompositeGeneration()
    composite_gen.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
