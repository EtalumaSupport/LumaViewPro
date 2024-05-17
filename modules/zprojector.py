
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf

import modules.common_utils as common_utils
import image_utils

from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord

import modules.imagej_helper as imagej_helper

from lvp_logger import logger


class ZProjector(ProtocolPostProcessingExecutor):

    def __init__(
        self,
        ij_helper: imagej_helper.ImageJHelper = None
    ):
        super().__init__(
            post_function=PostFunction.ZPROJECT
        )
        self._name = self.__class__.__name__
        
        if ij_helper is None:
            self._ij_helper = imagej_helper.ImageJHelper()
        else:
            self._ij_helper = ij_helper
        

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Scan Count',
                'Well',
                'Color',
                'Objective',
                'X',
                'Y',
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
            color=row0['Color'],
            z_height_idx=None,
            scan_count=row0['Scan Count'],
            tile_label=row0['Tile'],
            stitched=row0['Stitched'],
            zprojection=kwargs['method'].lower(),
        )

        outfile = f"{name}.tiff"
        return outfile
    

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already composited outputs
        df = df[df[self._post_function.value] == False]

        # Skip videos
        df = df[df[PostFunction.VIDEO.value] == False]

        # Skip stacks
        df = df[df[PostFunction.STACK.value] == False]

        return df
    

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return self._zproject(
            path=path,
            df=df[['Filepath','Color']],
            method=kwargs['method'],
            output_file_loc=kwargs['output_file_loc'],
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
            z="",
            z_slice="",
            well=row0['Well'],
            color=row0['Color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )


    @staticmethod
    def methods() -> list[str]:
        return imagej_helper.ZProjectMethod.list()


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
                error = f"Failed to create Z-Projection for color plane {used_color_plane}"
                logger.error(error)
                return {
                    'status': False,
                    'error': error,
                }
            
            out_image[:,:,used_color_plane] = project_result
        
        return {
            'status': True,
            'error': None,
            'image': out_image,
            'metadata': {},
        }
    

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
            error = f"Failed to create Z-Projection"
            logger.error(error)
            return {
                'status': False,
                'error': error,
            }
        
        return {
            'status': True,
            'error': None,
            'image': project_result,
            'metadata': {},
        }
    
    
    def _zproject(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        method: str,
        output_file_loc: pathlib.Path,
    ):
        method = imagej_helper.ZProjectMethod[method]

        orig_images = []
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
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

        if result['status'] == False:
            return result
        
        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        tf.imwrite(
            output_file_loc_abs,
            data=result['image'],
            compression='lzw',
        )

        del result['image']

        return result
