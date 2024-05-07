
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
    def _generate_filename(df: pd.DataFrame) -> str:
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

        return df
    

    @staticmethod
    def _group_algorithm(
        path: pathlib.Path,
        df: pd.DataFrame,
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

        allowed_layers = common_utils.get_fluorescence_layers()
        df = df[df['Color'].isin(allowed_layers)]

        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        row0 = df.iloc[0]
        source_image_sample_filename = row0['Filepath']
        source_image_sample = images[source_image_sample_filename]
        source_image_sample_shape = source_image_sample.shape
        
        img = np.zeros(
            shape=(source_image_sample_shape[0], source_image_sample_shape[1], 3),
            dtype=source_image_sample.dtype
        )

        for _, row in df.iterrows():
            layer = row['Color']
            source_image = images[row['Filepath']]
            source_is_color = image_utils.is_color_image(image=source_image)

            color_index_map = {
                'Blue': 0,
                'Green': 1,
                'Red': 2
            }

            layer_index = color_index_map[layer]
            if source_is_color:
                img[:,:,layer_index] = source_image[:,:,layer_index]
            else:
                img[:,:,layer_index] = source_image

        return {
            'status': True,
            'error': None,
            'image': img,
            'metadata': {
                'color': 'Composite',
            }
        }
       

if __name__ == "__main__":
    composite_gen = CompositeGeneration()
    composite_gen.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
