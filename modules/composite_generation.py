
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import image_utils
from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord
from settings_init import settings
from lvp_logger import logger


class CompositeGeneration(ProtocolPostProcessingExecutor):

    def __init__(self):
        super().__init__(
            post_function=PostFunction.COMPOSITE
        )
        self._name = self.__class__.__name__


    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Scan Count',
                'Well',
                'Objective',
                'X',
                'Y',
                'Z-Slice',
                'Tile',
                'Custom Step',
                'Raw',
                *PostFunction.list_values()
            ],
            dropna=False
        )
    

    @staticmethod
    def _generate_filename(df: pd.DataFrame, **kwargs) -> str:
        row0 = df.iloc[0]

        name = common_utils.generate_default_step_name(
            custom_name_prefix=row0['Name'],
            well_label=row0['Well'],
            color='Composite',
            z_height_idx=row0['Z-Slice'],
            scan_count=row0['Scan Count'],
            tile_label=row0['Tile'],
            stitched=row0['Stitched'],
        )

        outfile = f"{name}.tiff"
        return outfile


    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already composited outputs
        df = df[df[self._post_function.value] == False]

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]

        # Skip stacks
        df = df[df[PostFunction.HYPERSTACK.value] == False]

        return df
    

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return CompositeGeneration._create_composite_image(
            path=path,
            df=df[['Filepath','Color']]
        )
    

    @staticmethod
    def _add_record(
        protocol_post_record: ProtocolPostRecord,
        alg_metadata: dict,
        root_path: pathlib.Path,
        file_path: pathlib.Path,
        row0: pd.Series,
        **kwargs: dict,
    ):
        protocol_post_record.add_record(
            root_path=root_path,
            file_path=file_path,
            timestamp=row0['Timestamp'],
            name=row0['Name'],
            scan_count=row0['Scan Count'],
            x=row0['X'],
            y=row0['Y'],
            z=row0['Z'],
            z_slice=row0['Z-Slice'],
            well=row0['Well'],
            color=alg_metadata['color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )


    @staticmethod
    def _create_composite_image(path: pathlib.Path, df: pd.DataFrame):

        # Ratio for the amount that the transmitted channel is added to the composite ([0,1])
        BF_present = False
        BF_channel = ""

        allowed_BF_layers = common_utils.get_transmitted_layers()
        allowed_layers = [*common_utils.get_fluorescence_layers(), *common_utils.get_luminescence_layers()]
        img = None

        for layer in allowed_BF_layers:
            if (df['Color'] == layer).any():
                # A BF layer is present. Use first found one as base for the composite.
                BF_present = True
                BF_channel = layer
                allowed_layers.append(layer)

                break

        
        df = df[df['Color'].isin(allowed_layers)]

        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        color_index_map = {
            'Blue': 0,
            'Green': 1,
            'Red': 2,
            'Lumi': 0,
        }
        
        error = None
        status = True
        
        # Transmitted channel present
        if BF_present:
            try:
                logger.info("CompositeGeneration] Generating transmitted channel composite")
                BF_row = df[df['Color'] == BF_channel]
                try:
                    BF_image_filename = BF_row['Filepath'].iloc[0]
                except Exception as e:
                    BF_present = False
                    logger.error(f"CompositeGeneration] Error details: {e}")
                    raise e
                
                BF_image = images[BF_image_filename]

                img = np.array(BF_image, dtype=BF_image.dtype)

                # Init mask to keep track of changed pixels
                # Set all values in the mask for changed to False
                mask_transmitted_changed = img == None
                
                # Prep transmitted channel to have 3 channels for RGB value manipulation
                img = np.repeat(BF_image[:, :, None], 3, axis=2)

                img_dtype = BF_image.dtype

                for _, row in df.iterrows():
                    layer = row['Color']

                    # Exempt BF layer
                    if layer == BF_channel:
                        continue

                    f_image = images[row['Filepath']]
                    f_is_color = image_utils.is_color_image(image=f_image)

                    if img_dtype == "uint8":
                        brightness_threshold = settings[layer]["composite_brightness_threshold"] / 100 * 255
                    elif img_dtype == "uint16":
                        brightness_threshold = settings[layer]["composite_brightness_threshold"] / 100 * 65536

                    channel_index = color_index_map[layer]

                    # Convert to np array
                    f_image = np.array(f_image)
                    
                    # Convert to 1 channnel image instead of RGB
                    if f_is_color:
                        img_gray = f_image[:, :, channel_index]
                    else:
                        img_gray = f_image

                    # Create mask of every pixel > brightness threshold in channel image
                    channel_above_threshold_mask = img_gray > brightness_threshold

                    # Create masks for pixels that correspond to changed/unchanged pixels in the transmitted image
                    not_changed_mask = channel_above_threshold_mask & (~mask_transmitted_changed)
                    changed_mask = channel_above_threshold_mask & mask_transmitted_changed

                    # For not-yet changed pixels, set every other channel to 0, then the desired color channel value
                    # Allows desired channel to show up fully
                    img[not_changed_mask, 0] = 0
                    img[not_changed_mask, 1] = 0
                    img[not_changed_mask, 2] = 0

                    img[not_changed_mask, channel_index] = img_gray[not_changed_mask]

                    # Update changed pixels
                    mask_transmitted_changed[not_changed_mask] = True

                    # For already changed pixels, only update the current channel value (allows stacking of RGB values)
                    img[changed_mask, channel_index] = img_gray[changed_mask]

            except Exception as e:
                # Issue with BF composite, so treat as normal flourescent composite generation.
                logger.error(f"CompositeGeneration] Error generating transmitted channel composite: {e}")
                error = "Error generating transmitted channel composite"
                status = False

        # No transmitted channel present
        if not BF_present:
            try:
                logger.info("CompositeGeneration] Generating flourescent channel composite")
                row0 = df.iloc[0]
                source_image_sample_filename = row0['Filepath']
                source_image_sample = images[source_image_sample_filename]
                source_image_sample_shape = source_image_sample.shape

                img = np.zeros(
                    shape=(source_image_sample_shape[0], source_image_sample_shape[1], 3),
                    dtype=source_image_sample.dtype
                )

                img_dtype = source_image_sample.dtype

                for _, row in df.iterrows():
                    try:
                        layer_index = color_index_map[layer]
                    except:
                        # If color not flourescent, skip this image
                        continue

                    layer = row['Color']
                    source_image = images[row['Filepath']]
                    source_is_color = image_utils.is_color_image(image=source_image)
                    
                    if source_is_color:
                        img[:,:,layer_index] = source_image[:,:,layer_index]
                    else:
                        img[:,:,layer_index] = source_image
                status = True
                error = None

            except Exception as e:
                logger.error(f"CompositeGeneration] Error generating flourescent channel composite: {e}")
                error = "Error generating flourescent channel composite"
                status = False

        if img is None:
            status = False
            error = "Composite Generation Error: No final image"
            
        return {
            'status': status,
            'error': error,
            'image': img,
            'metadata': {
                'color': 'Composite',
            }
        }
       

if __name__ == "__main__":
    composite_gen = CompositeGeneration()
    composite_gen.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
