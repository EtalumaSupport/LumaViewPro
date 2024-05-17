import os
import pathlib

import numpy as np
import pandas as pd
import tifffile as tf

import modules.common_utils as common_utils
import image_utils

from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord


class StackBuilder(ProtocolPostProcessingExecutor):

    def __init__(self):
        super().__init__(
            post_function=PostFunction.STACK
        )
        self._name = self.__class__.__name__
        

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                # 'Scan Count',
                # 'Z-Slice',
                'Well',
                # 'Color',
                'Objective',
                'X',
                'Y',
                # 'Z',
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
        df: pd.DataFrame
    ):
        
        """
        px, py = self._coordinate_transformer.stage_to_plate(
            labware=self._labware,
            stage_offset=self._stage_offset,
            sx=self.get_current_position(axis='X'),
            sy=self.get_current_position(axis='Y')
        )
        z = self.get_current_position(axis='Z')

        px = round(px, common_utils.max_decimal_precision('x'))
        py = round(py, common_utils.max_decimal_precision('y'))
        z  = round(z,  common_utils.max_decimal_precision('z'))

        pixel_size_um = round(
            common_utils.get_pixel_size(
                focal_length=self._objective['focal_length'],
                binning=self._binning_size,
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )
        
        metadata = {
            'channel': color,
            'focal_length': self._objective['focal_length'],
            'plate_pos_mm': {'x': px, 'y': py},
            'z_pos_um': z,
            'exposure_time_ms': round(self.get_exposure_time(), common_utils.max_decimal_precision('exposure')),
            'gain_db': round(self.get_gain(), common_utils.max_decimal_precision('gain')),
            'illumination_ma': round(self.get_led_ma(color=color), common_utils.max_decimal_precision('illumination')),
            'binning_size': self._binning_size,
            'pixel_size_um': pixel_size_um,
        }
        """

        """
        use_color = image_utils.is_color_image(data)

        if use_color:
            photometric = 'rgb'
            axes = 'YXS'
        else:
            photometric = 'minisblack'
            axes = 'YX'

        ome_metadata={
            'axes': axes,
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': metadata['pixel_size_um'],
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': metadata['pixel_size_um'],
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': [metadata['channel']]},
            'Plane': {
                'PositionX': metadata['plate_pos_mm']['x'],
                'PositionY': metadata['plate_pos_mm']['y'],
                'PositionZ': metadata['z_pos_um'],
                'PositionXUnit': 'mm',
                'PositionYUnit': 'mm',
                'PositionZUnit': 'um',
                'ExposureTime': metadata['exposure_time_ms'],
                'ExposureTimeUnit': 'ms',
                'Gain': metadata['gain_db'],
                'GainUnit': 'dB',
                'Illumination': metadata['illumination_ma'],
                'IlluminationUnit': 'mA'
            }
        }

        options=dict(
            photometric=photometric,
            tile=(128, 128),
            compression='lzw',
            resolutionunit='CENTIMETER',
            maxworkers=2
        )

        resolution = (1e4 / metadata['pixel_size_um'], 1e4 / metadata['pixel_size_um'])

        return {
            'metadata': ome_metadata,
            'options': options,
            'resolution': resolution,
        }
        """

        axes = "TZCYX"
        photometric = 'minisblack'

        channel_names = df['Color'].unique().tolist()
        z_vals = df['Z'].unique().tolist()
        row0 = df.iloc[0]

        ome_metadata={
            'axes': axes,
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': metadata['pixel_size_um'],
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': metadata['pixel_size_um'],
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': channel_names},
            'Plane': {
                'PositionX': row0['X'],
                'PositionY': row0['Y'],
                'PositionZ': metadata['z_pos_um'],
                'PositionXUnit': 'mm',
                'PositionYUnit': 'mm',
                'PositionZUnit': 'um',
                'ExposureTime': 0, #metadata['exposure_time_ms'],
                'ExposureTimeUnit': 'ms',
                'Gain': 0, #metadata['gain_db'],
                'GainUnit': 'dB',
                'Illumination': 0, #metadata['illumination_ma'],
                'IlluminationUnit': 'mA'
            }
        }


        

    @staticmethod
    def _create_stack(
        path: pathlib.Path,
        df: pd.DataFrame,
        output_file_loc: pathlib.Path,
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

        for _, row in df.iterrows():
            t = row['Scan Count']
            z = row['Z-Slice']
            c = row['Color Index']
            image = tf.imread(path / row['Filepath'])

            if image_utils.is_color_image(image):
                image = image_utils.rgb_image_to_gray(image=image)

            stacked_image[t,z,c,:,:] = image
        
        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        tf.imwrite(
            output_file_loc_abs,
            data=stacked_image,
            bigtiff=False,
            ome=True,
            imagej=True,
            # resolution=
        )
        return {
            'status': True,
            'error': None,
            'metadata': {}
        }


if __name__ == "__main__":
    stack_builder = StackBuilder()
    stack_builder.load_folder(
        path=os.getenv('SAMPLE_IMAGE_FOLDER'),
        tiling_configs_file_loc=pathlib.Path(os.getenv('SOURCE_ROOT')) / "data" / "tiling.json",
    )
