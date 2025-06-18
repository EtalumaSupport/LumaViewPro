
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf

import modules.common_utils as common_utils
import image_utils

from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord
from modules.objectives_loader import ObjectiveLoader
import psfmodels as psfm
from skimage import restoration

import modules.imagej_helper as imagej_helper

from lvp_logger import logger


class Deconvolutor(ProtocolPostProcessingExecutor):

    def __init__(
        self,
        ij_helper: imagej_helper.ImageJHelper = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            post_function=PostFunction.DECONVOLUTION,
            *args,
            **kwargs,
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
    

    def _generate_filename(self, df: pd.DataFrame, **kwargs) -> str:
        row0 = df.iloc[0]
        objective_short_name = self._get_objective_short_name_if_has_turret(objective_id=row0['Objective'])

        name = common_utils.generate_default_step_name(
            custom_name_prefix=row0['Name'],
            well_label=row0['Well'],
            color=row0['Color'],
            z_height_idx=row0['Z-Slice'],
            scan_count=row0['Scan Count'],
            tile_label=row0['Tile'],
            objective_short_name=objective_short_name,
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
        df = df[df[PostFunction.HYPERSTACK.value] == False]

        return df
    

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return self._deconvolve(
            path=path,
            df=df[['Filepath','Color']],
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
            z=row0['Z'],
            z_slice=row0['Z-Slice'],
            well=row0['Well'],
            color=row0['Color'],
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )

    def _deconvolve_for_multi_channel(
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
    

    def _deconvolve_for_single_channel(
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
    
    
    def _deconvolve(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        objective_aperture: float,
        objective_focal: float,
        output_file_loc: pathlib.Path,
    ):
        
        color_to_wavelength = {
            'Red': 0,
            'Green': 0,
            'Blue': 0,
        }
        #method = imagej_helper.ZProjectMethod[method]

        # Available Data
        #      {
        #     'Filepath': image_name,
        #     'Name': step['Name'],
        #     'Scan Count': file_data['Scan Count'],
        #     'Step Index': step_idx,
        #     'X': step['X'],
        #     'Y': step['Y'],
        #     'Z': step['Z'],
        #     'Z-Slice': step['Z-Slice'],
        #     'Well': step['Well'],
        #     'Color': step['Color'],
        #     'Objective': step['Objective'],
        #     'Tile': step['Tile'],
        #     'Tile Group ID': step['Tile Group ID'],
        #     'Z-Stack Group ID': step['Z-Stack Group ID'],
        #     'Custom Step': step['Custom Step'],
        #     'Timestamp': file_data['Timestamp'],
        # }

        orig_images = []
        df_sorted = df.sort_values('Z')

        for _, row in df_sorted.iterrows():
            image_dict = {}
            image_filepath = path / row['Filepath']
            image_dict['image_np'] = tf.imread(str(image_filepath))
            image_dict['Color'] = row['Color']
            image_dict['Z'] = row['Z']

            image_dict['size'] = image_dict['image_np'].shape

            orig_images.append(image_dict)


        z_step = abs(orig_images[1]['Z'] - orig_images[0]['Z']) # Should be in um
        wavelength = color_to_wavelength[orig_images[0]['Color']] # um
        shape = (len(orig_images), orig_images[0]['size'][0], orig_images[0]['size'][1])
        focal_depth = 0
        refractive_index_immersion = 1.518
        refractive_index_sample = 1.33
        coverslip_thickness = 0.17
        
        pixel_size = common_utils.get_pixel_size(objective_focal, 1)
        aperture = objective_aperture

        num_z_steps = len(orig_images)

        # NOTE
        # We need to check to make sure that we are only doing this for 1 flourescent channel at a time
        # Otherwise, we will get issues

        psf3d = psfm.scalar_psf(
            nz=num_z_steps,
            nx=shape[0],
            ny=shape[1],
            dxy=pixel_size,
            dz=z_step,
            pz=focal_depth,
            ni=refractive_index_immersion,
            ns=refractive_index_sample,
            tg=coverslip_thickness,
            wvl=wavelength,
            NA=aperture,
        )

        # Normalize psf
        psf3d /= psf3d.sum()

        raw_stack = np.stack([orig['image_np'] for orig in orig_images], axis=0)

        deconv_stack = restoration.richardson_lucy(
            image = raw_stack,
            psf=psf3d,
            iterations=20,
            clip=True
        )

        # Now recover each z-stack slice

        # 5) Recover individual slices
        #    For slice index i (0 ≤ i < Z):




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
