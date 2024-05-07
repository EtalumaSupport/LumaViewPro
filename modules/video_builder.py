
import itertools
import pathlib

import cv2
import pandas as pd

import image_utils
import modules.common_utils as common_utils
from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_executor import ProtocolPostProcessingExecutor
from modules.protocol_post_record import ProtocolPostRecord

from lvp_logger import logger


class VideoBuilder(ProtocolPostProcessingExecutor):

    CODECS = [
        0,
        'mp4v',
        'mjpg'
    ]

    def __init__(self):
        super().__init__(
            post_function=PostFunction.VIDEO
        )
        self._name = self.__class__.__name__

    
    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(
            by=[
                'Well',
                'Color',
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
            color=row0['Color'],
            z_height_idx=row0['Z-Slice'],
            scan_count=None,
            tile_label=row0['Tile'],
            stitched=row0['Stitched'],
            video=True,
        )

        outfile = f"{name}.avi"
        return outfile
    

    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:

        # Skip already composited outputs
        df = df[df[self._post_function.value] == False]

        return df
    

    def _group_algorithm(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        **kwargs,
    ):
        return self._create_video(
            path=path,
            df=df[['Filepath', 'Scan Count', 'Timestamp']],
            frames_per_sec=kwargs['frames_per_sec'],
            enable_timestamp_overlay=kwargs['enable_timestamp_overlay'],
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


    @staticmethod
    def _get_fourcc_code(codec: str | int):
        if type(codec) == str:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        else:
            fourcc = codec

        return fourcc
    
    
    def _create_video(
        self,
        path: pathlib.Path,
        df: pd.DataFrame,
        frames_per_sec: int,
        enable_timestamp_overlay: bool,
        output_file_loc: pathlib.Path
    ) -> bool:
        df = df.sort_values(by=['Scan Count'], ascending=True)

        # codec = self.CODECS[0] # Set to AVI
        codec = self.CODECS[1] # Set to mp4v
        fourcc = self._get_fourcc_code(codec=codec)

        def _get_image_info() -> tuple:
            source_image_sample_filename = df['Filepath'].values[0]
            source_image_sample_filepath = path / source_image_sample_filename
            source_image_sample = cv2.imread(str(source_image_sample_filepath), cv2.IMREAD_UNCHANGED)
            is_color = True if source_image_sample.ndim == 3 else False
            
            if is_color:
                frame_height, frame_width, _ = source_image_sample.shape
            else:
                frame_height, frame_width = source_image_sample.shape
            
            return (frame_height, frame_width), is_color
        
        def _get_timestamp_str(val):
            frame_ts = val.to_pydatetime()
            frame_ts_str = frame_ts.strftime("%Y-%m-%d %H:%M:%S")
            return frame_ts_str

        (frame_height, frame_width), is_color = _get_image_info()
        output_file_loc_abs = path / output_file_loc
        output_file_loc_abs.parent.mkdir(exist_ok=True, parents=True)
        video = cv2.VideoWriter(
            filename=str(output_file_loc_abs),
            fourcc=fourcc,
            fps=frames_per_sec,
            frameSize=(frame_width, frame_height),
            isColor=is_color
        )

        logger.info(f"[{self._name}] Writing video to {output_file_loc}")
        
        for _, row in df.iterrows():
            image_path = path / row['Filepath']
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if enable_timestamp_overlay:
                frame_ts = _get_timestamp_str(row['Timestamp'])
                image = image_utils.add_timestamp(image=image, timestamp_str=frame_ts)
                
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

        logger.debug(f"[{self._name}] - Complete")

        return {
            'status': True,
            'error': None,
            'metadata': {},
        }
