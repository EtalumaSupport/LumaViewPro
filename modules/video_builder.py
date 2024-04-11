
import itertools
import pathlib

import cv2
import pandas as pd

import modules.artifact_locations as artifact_locations
import modules.common_utils as common_utils
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper

from lvp_logger import logger


class VideoBuilder:

    CODECS = [
        0,
        'mp4v',
        'mjpg'
    ]

    def __init__(self):
        self._name = self.__class__.__name__
        self._protocol_post_processing_helper = ProtocolPostProcessingHelper()

    
    def load_folder(
        self, path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
        frames_per_sec: int
    ) -> dict:
        results = self._protocol_post_processing_helper.load_folder(
            path=path,
            tiling_configs_file_loc=tiling_configs_file_loc,
            include_stitched_images=True,
            include_composite_images=True,
            include_composite_and_stitched_images=True,
        )

        if results['status'] is False:
            return {
                'status': False,
                'message': f'Failed to load protocol data from {path}'
            }

        df = results['image_tile_groups']
        
        if len(df) == 0:
            return {
                'status': False,
                'message': 'No images found in selected folder'
            }
        
        grouping_key = 'Video Group Index'

        # Raw images first
        loop_list = df.groupby(by=[grouping_key])

        stitched_images_df = results['stitched_images']
        if stitched_images_df is not None:
            loop_list = itertools.chain(
                loop_list,
                stitched_images_df.groupby(by=[grouping_key])
            )

        composite_images_df = results['composite_images']
        if composite_images_df is not None:
            loop_list = itertools.chain(
                loop_list,
                composite_images_df.groupby(by=[grouping_key])
            )

        composite_and_stitched_images_df = results['composite_and_stitched_images']
        if composite_and_stitched_images_df is not None:
            loop_list = itertools.chain(
                loop_list,
                composite_and_stitched_images_df.groupby(by=[grouping_key])
            )

        logger.info(f"{self._name}: Generating video(s)")
        metadata = []
        
        for _, video_group in loop_list:
            
            if len(video_group) == 0:
                continue

            if len(video_group) == 1:
                logger.debug(f"{self._name}: Skipping video generation for {video_group.iloc[0]['Filename']} since only {len(video_group)} image found.")
                continue

            first_row = video_group.iloc[0]
            video_filename_base = common_utils.generate_default_step_name(
                well_label=first_row['Well'],
                color=first_row['Color'],
                z_height_idx=first_row['Z-Slice'],
                tile_label=first_row['Tile'],
                custom_name_prefix=first_row['Name'],
                stitched=first_row['Stitched']
            )
            video_filename = f"{video_filename_base}.avi"

            output_path = path / artifact_locations.video_output_dir()
            if not output_path.exists():
                output_path.mkdir(exist_ok=True, parents=True)

            output_file_loc = output_path / video_filename

            status = self._create_video(
                path=path,
                df=video_group[['Filename', 'Scan Count']],
                frames_per_sec=frames_per_sec,
                output_file_loc=str(output_file_loc)
            )

            if status == False:
                logger.error(f"{self._name}: Unable to create video {output_file_loc}")
                continue
            
            # logger.debug(f"{self._name}: - {output_file_loc}")
            # if not cv2.imwrite(
            #     filename=str(output_file_loc),
            #     img=composite_image
            # ):
            #     logger.error(f"{self._name}: Unable to write image {output_file_loc}")
            #     continue

            metadata.append({
                'Filename': video_filename,
                'Name': first_row['Name'],
                'Protocol Group Index': first_row['Protocol Group Index'],
                'X': first_row['X'],
                'Y': first_row['Y'],
                'Z-Slice': first_row['Z-Slice'],
                'Well': first_row['Well'],
                'Color': first_row['Color'],
                'Objective': first_row['Objective'],
                'Tile Group ID': first_row['Tile Group ID'],
                'Custom Step': first_row['Custom Step'],
                'Stitch Group Index': first_row['Stitch Group Index'],
                'Composite Group Index': first_row['Composite Group Index'],
                'Video Group Index': first_row['Video Group Index'],
                'Stitched': first_row['Stitched'],
                'Composite': first_row['Composite'],
            })

        metadata_df = pd.DataFrame(metadata)
        
        if len(metadata_df) == 0:
            return {
                'status': False,
                'message': 'No images found'
            }

        if not output_path.exists():
            path.mkdir(parents=True, exist_ok=True)

        metadata_filename = artifact_locations.video_output_metadata_filename()
        metadata_df.to_csv(
            path_or_buf=output_path / metadata_filename,
            header=True,
            index=False,
            sep='\t',
            lineterminator='\n',
        )
        
        logger.info(f"{self._name}: Complete")
        return {
            'status': True,
            'message': 'Success'
        }


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
        output_file_loc: pathlib.Path
    ) -> bool:
        df = df.sort_values(by=['Scan Count'], ascending=True)

        # codec = self.CODECS[0] # Set to AVI
        codec = self.CODECS[1] # Set to mp4v
        fourcc = self._get_fourcc_code(codec=codec)

        def _get_image_info() -> tuple:
            source_image_sample_filename = df['Filename'].values[0]
            source_image_sample_filepath = path / source_image_sample_filename
            source_image_sample = cv2.imread(str(source_image_sample_filepath), cv2.IMREAD_UNCHANGED)
            is_color = True if source_image_sample.ndim == 3 else False
            
            if is_color:
                frame_height, frame_width, _ = source_image_sample.shape
            else:
                frame_height, frame_width = source_image_sample.shape
            
            return (frame_height, frame_width), is_color

        (frame_height, frame_width), is_color = _get_image_info()
        video = cv2.VideoWriter(
            filename=str(output_file_loc),
            fourcc=fourcc,
            fps=frames_per_sec,
            frameSize=(frame_width, frame_height),
            isColor=is_color
        )

        logger.info(f"[{self._name}] Writing video to {output_file_loc}")
        
        for _, row in df.iterrows():
            image_path = path / row['Filename']
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            video.write(image)

        cv2.destroyAllWindows()
        video.release()

        logger.debug(f"[{self._name}] - Complete")

        return True
