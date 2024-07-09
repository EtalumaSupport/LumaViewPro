import os
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf
# from ome_types.model import OME, Image, Pixels, UnitsTime, UnitsLength, Channel

import image_utils

import modules.common_utils as common_utils
from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord


class StackBuilder(ProtocolPostProcessingExecutor):

    def __init__(self):
        super().__init__(
            post_function=PostFunction.HYPERSTACK
        )
        self._name = self.__class__.__name__
        

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Well',
                'Objective',
                'X',
                'Y',
                'Tile',
                'Tile Group ID',
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
            color=None,
            z_height_idx=None,
            scan_count=None,
            tile_label=None,
            hyperstack=True,
        )
        
        outfile = f"{name}.ome.tiff"
        return outfile
    

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Only process raw
        df = df[df['Raw'] == True]

        return df
    

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return StackBuilder._create_stack(
            path=path,
            df=df,
            output_file_loc=kwargs['output_file_loc'],
            focal_length=kwargs['focal_length'],
            binning_size=kwargs['binning_size'],
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
            scan_count=-1,
            x=row0['X'],
            y=row0['Y'],
            z=-1,
            z_slice=-1,
            well=row0['Well'],
            color="Stack",
            objective=row0['Objective'],
            tile_group_id=row0['Tile Group ID'],
            tile=row0['Tile'],
            custom_step=row0['Custom Step'],
            **kwargs,
        )


    @staticmethod
    def _generate_image_metadata(
        df: pd.DataFrame,
        path: pathlib.Path,
        output_file_loc: pathlib.Path,
        plane_metadata: dict,
        binning_size: int,
        focal_length: float,
    ):

        axes = "TZCYX"
        photometric = 'minisblack'

        channel_names = df['Color'].unique().tolist()
        z_vals = df['Z'].unique().tolist()
        row0 = df.iloc[0]
        # num_planes = len(z_vals)
        # num_channels = len(channel_names)

        num_t = df['Scan Count'].nunique()
        num_z = df['Z-Slice'].nunique()
        num_c = df['Color'].nunique()

        pixel_size_um = round(
            common_utils.get_pixel_size(
                focal_length=focal_length,
                binning_size=binning_size,
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )

        row0 = df.iloc[0]
        sample_image_file_loc = path / row0['Filepath']
        sample_image = tf.imread(sample_image_file_loc)

        metadata={
            'axes': axes,
            'SignificantBits': sample_image.itemsize*8,
            'Pixels': {
                'PhysicalSizeX': pixel_size_um,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': pixel_size_um,
                'PhysicalSizeYUnit': 'µm',
            },
            'Channel': {'Name': channel_names},

            # 'Plane': plane_metadata, #,{
            #     'PositionX': num_planes*[row0['X']],
            #     'PositionY': num_planes*[row0['Y']],
            #     'PositionZ': z_vals, #['z_pos_um'],
            #     'PositionXUnit': num_planes*['mm'],
            #     'PositionYUnit': num_planes*['mm'],
            #     'PositionZUnit': num_planes*['um'],
            #     # 'ExposureTime': num_planes*[0], #metadata['exposure_time_ms'],
            #     # 'ExposureTimeUnit': num_planes*['ms'],
            #     # 'Gain': num_planes*[0], #metadata['gain_db'],
            #     # 'GainUnit': num_planes*['dB'],
            #     # 'Illumination': num_planes*[0], #metadata['illumination_ma'],
            #     # 'IlluminationUnit': num_planes*['mA'],
            # }
        }

        options=dict(
            photometric=photometric,
            tile=(128, 128),
            compression='lzw',
            resolutionunit='CENTIMETER',
            maxworkers=2
        )
        
        resolution = (1e4 / pixel_size_um, 1e4 / pixel_size_um)

        return {
            'metadata': metadata,
            'options': options,
            'resolution': resolution,
        }
    

    # @staticmethod
    # def _generate_image_metadata_ome_types(
    #     df: pd.DataFrame,
    #     path: pathlib.Path,
    #     output_file_loc: pathlib.Path,
    #     plane_metadata: dict,
    #     binning_size: int,
    #     focal_length: float,
    # ):
    #     num_t = df['Scan Count'].nunique()
    #     num_z = df['Z-Slice'].nunique()
    #     num_c = df['Color'].nunique()

    #     channel_names = df['Color'].unique().tolist()

    #     row0 = df.iloc[0]
    #     sample_image_file_loc = path / row0['Filepath']
    #     sample_image = tf.imread(sample_image_file_loc)
    #     sample_image_shape = sample_image.shape
    #     h, w = sample_image_shape[0], sample_image_shape[1]

    #     pixel_size_um = round(
    #         common_utils.get_pixel_size(
    #             focal_length=focal_length,
    #             binning_size=binning_size,
    #         ),
    #         common_utils.max_decimal_precision('pixel_size'),
    #     )
    #     ome = OME()


    #     channels = []
    #     for channel_name in channel_names:
    #         channels.append(
    #             Channel(
    #                 name=channel_name,
    #             )
    #         )

    #     pixels=Pixels(
    #         dimension_order='XYCZT',
    #         size_c=num_c,
    #         size_t=num_t,
    #         size_z=num_z,
    #         size_x=w,
    #         size_y=h,
    #         type=str(sample_image.dtype),
    #         channels=channels,
    #         physical_size_x=pixel_size_um,
    #         physical_size_x_unit=UnitsLength.MICROMETER,
    #         physical_size_y=pixel_size_um,
    #         physical_size_y_unit=UnitsLength.MICROMETER,
    #         time_increment=1,
    #         time_increment_unit=UnitsTime.SECOND,
    #     )
    #     image_metadata = Image(
    #         # id=output_file_loc.name,
    #         name=output_file_loc.name,
    #         pixels=pixels,
    #     )
    #     ome.images.append(image_metadata)

    #     return ome

    #     axes = "TZCYX"
    #     photometric = 'minisblack'

        
    #     z_vals = df['Z'].unique().tolist()
    #     row0 = df.iloc[0]
    #     # num_planes = len(z_vals)
    #     # num_channels = len(channel_names)


        # row0 = df.iloc[0]
        # sample_image_file_loc = path / row0['Filepath']
        # sample_image = tf.imread(sample_image_file_loc)

        # metadata={
        #     'axes': axes,
        #     'SignificantBits': sample_image.itemsize*8,
        #     'Pixels': {
        #         'PhysicalSizeX': pixel_size_um,
        #         'PhysicalSizeXUnit': 'µm',
        #         'PhysicalSizeY': pixel_size_um,
        #         'PhysicalSizeYUnit': 'µm',
        #     },
        #     'Channel': {'Name': channel_names},

        #     # 'Plane': plane_metadata, #,{
        #     #     'PositionX': num_planes*[row0['X']],
        #     #     'PositionY': num_planes*[row0['Y']],
        #     #     'PositionZ': z_vals, #['z_pos_um'],
        #     #     'PositionXUnit': num_planes*['mm'],
        #     #     'PositionYUnit': num_planes*['mm'],
        #     #     'PositionZUnit': num_planes*['um'],
        #     #     # 'ExposureTime': num_planes*[0], #metadata['exposure_time_ms'],
        #     #     # 'ExposureTimeUnit': num_planes*['ms'],
        #     #     # 'Gain': num_planes*[0], #metadata['gain_db'],
        #     #     # 'GainUnit': num_planes*['dB'],
        #     #     # 'Illumination': num_planes*[0], #metadata['illumination_ma'],
        #     #     # 'IlluminationUnit': num_planes*['mA'],
        #     # }
        # }

        # options=dict(
        #     photometric=photometric,
        #     tile=(128, 128),
        #     compression='lzw',
        #     resolutionunit='CENTIMETER',
        #     maxworkers=2
        # )
        
        # resolution = (1e4 / pixel_size_um, 1e4 / pixel_size_um)

        # return {
        #     'metadata': metadata,
        #     'options': options,
        #     'resolution': resolution,
        # }



    @staticmethod
    def _create_stack(
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path,
        focal_length: float,
        binning_size: int,
    ):
        num_t = df['Scan Count'].nunique()
        num_z = df['Z-Slice'].nunique()
        num_c = df['Color'].nunique()

        _, color_idx_map = np.unique(df['Color'], return_inverse=True)
        df['Color Index'] = color_idx_map
        
        row0 = df.iloc[0]
        sample_image_file_loc = path / row0['Filepath']
        sample_image = tf.imread(sample_image_file_loc)
        sample_image_shape = sample_image.shape
        h, w = sample_image_shape[0], sample_image_shape[1]

        stacked_image = np.zeros(
            shape=(num_t, num_z, num_c, h, w), # Hyperstack order TZCYX
            dtype=sample_image.dtype,
        )

        # ome = OME()
        # pixels = Pixels(
        #     size_c=num_c,
        #     size_t=num_t,
        #     size_z=num_z,
        #     size_x=w,
        #     size_y=h,
        #     type=str(sample_image.dtype),
        #     dimension_order="XYCZT", #"TZCYX",
        #     metadata_only=True,
        #     # physical_size_x=0,
        #     # physical_size_x_unit='µm',
        #     # physical_size_y=0,
        #     # physical_size_y_unit='µm',
        #     significant_bits=sample_image.itemsize*8,

        # )
        # img_meta = Image(
        #     pixels=pixels,
        # )
        # ome.images.append(img_meta)
        # print(ome.to_xml())
        df = df.sort_values(by=['Scan Count', 'Z-Slice', 'Color Index'], ascending=True)

        plane_metadata = {
            'PositionX': [],
            'PositionY': [],
            'PositionZ': [],
        }
                # 'PositionX': num_planes*[row0['X']],
                # 'PositionY': num_planes*[row0['Y']],
                # 'PositionZ': z_vals, #['z_pos_um'],
                # 'PositionXUnit': num_planes*['mm'],
                # 'PositionYUnit': num_planes*['mm'],
                # 'PositionZUnit': num_planes*['um'],

        for _, row in df.iterrows():
            t = row['Scan Count']
            z = row['Z-Slice']
            c = row['Color Index']
            image = tf.imread(path / row['Filepath'])

            if image_utils.is_color_image(image):
                image = image_utils.rgb_image_to_gray(image=image)

            stacked_image[t,z,c,:,:] = image
            plane_metadata['PositionX'].append(row['X'])
            plane_metadata['PositionY'].append(row['Y'])
            plane_metadata['PositionZ'].append(row['Z'])


        num_planes = len(plane_metadata['PositionX'])
        plane_metadata['PositionXUnit'] = num_planes*['mm']
        plane_metadata['PositionYUnit'] = num_planes*['mm']
        plane_metadata['PositionZUnit'] = num_planes*['um']

        # ome_info = StackBuilder._generate_image_metadata_ome_types(
        ome_info = StackBuilder._generate_image_metadata(
            df=df,
            path=path,
            output_file_loc=output_file_loc,
            plane_metadata=plane_metadata,
            focal_length=focal_length,
            binning_size=binning_size,
        )

        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        tf.imwrite(
            output_file_loc_abs,
            data=stacked_image,
            bigtiff=False, #True,
            ome=True,
            imagej=True,
            metadata=ome_info['metadata'],
            # photometric=ome_info['options']['photometric'],
            # description=ome.to_xml(),
            resolution=ome_info['resolution'],
            **ome_info['options'],
        )

        # tf.imwrite(
        #     output_file_loc_abs,
        #     data=stacked_image,
        #     bigtiff=False, #True,
        #     ome=True,
        #     imagej=True,
        #     # description=ome_info.to_xml(),
        #     metadata=ome_info.model_dump()
        #     # metadata=None,
        #     # metadata=ome_info['metadata'],
        #     # # photometric=ome_info['options']['photometric'],
        #     # # description=ome.to_xml(),
        #     # resolution=ome_info['resolution'],
        #     # **ome_info['options'],
        # )


        return {
            'status': True,
            'error': None,
            'metadata': {}
        }


if __name__ == "__main__":
    stack_builder = StackBuilder()
    tiling_configs_file_loc=pathlib.Path(os.getenv('SOURCE_ROOT')) / "data" / "tiling.json"
    stack_builder.load_folder(
        path=os.getenv('SAMPLE_IMAGE_FOLDER'),
        tiling_configs_file_loc=tiling_configs_file_loc,
        focal_length=45.0,
        binning_size=1,
    )
