
import abc
import copy
import itertools
import os
import pathlib

import cv2
import numpy as np
import pandas as pd

import modules.common_utils as common_utils
import image_utils
from modules.protocol_post_processing_functions import PostFunction
from modules.protocol_post_processing_helper import ProtocolPostProcessingHelper

from lvp_logger import logger


class ProtocolPostProcessingExecutor(abc.ABC):

    def __init__(
        self,
        post_function: PostFunction
    ):
        self._name = self.__class__.__name__
        self._post_function = post_function
        self._post_processing_helper = ProtocolPostProcessingHelper()
        

    @abc.abstractmethod
    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Implement in child class")
    

    @abc.abstractmethod
    def _filter_ignored_types(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(f"Implement in child class")
    

    @abc.abstractmethod
    @staticmethod
    def _group_algorithm(
        path: pathlib.Path,
        df: pd.DataFrame
    ):
        raise NotImplementedError(f"Implement in child class")


    def load_folder(
        self, path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path
    ) -> dict:
        selected_path = pathlib.Path(path)
        results = self._post_processing_helper.load_folder(
            path=selected_path,
            tiling_configs_file_loc=tiling_configs_file_loc,
        )

        if results['status'] is False:
            return {
                'status': False,
                'message': f'Failed to load protocol data using path: {selected_path}'
            }
        
        df = results['images_df']
        if len(df) == 0:
            return {
                'status': False,
                'message': 'No images found in selected folder'
            }
        
        root_path = results['root_path']
        protocol_post_record = results['protocol_post_record']

        df = self._filter_ignored_types(df=df)
        groups = self._get_groups(df)

        logger.info(f"{self._name}: Generating {self._post_function.value.lower()} images")

        new_count = 0
        existing_count = 0

        for _, group in groups:
            if len(group) == 0:
                continue

            if len(group) == 1:
                logger.debug(f"{self._name}: Skipping generation for {group.iloc[0]['Filepath']} since only {len(group)} image found.")
                continue

            output_filename = self._generate_stitched_filename(df=group)
            first_row = group.iloc[0]
            record_data_post_functions = first_row[PostFunction.list_values()]
            record_data_post_functions[self._post_function.value] = True
            output_subfolder = self._post_processing_helper.generate_output_dir_name(record=record_data_post_functions)
            output_path = root_path / output_subfolder
            output_file_loc = output_path / output_filename
            output_file_loc_rel = output_file_loc.relative_to(root_path)

            if protocol_post_record.file_exists_in_records(
                filepath=output_file_loc_rel
            ):
                logger.info(f"{output_file_loc_rel} already exists in record, skipping for generation.")
                existing_count += 1 # Count this so we don't error out if no other matches are found
                continue

            image, center = self.simple_position_stitcher(
                path=path,
                df=group[['Filepath', 'X', 'Y']]
            )

            if not output_path.exists():
                output_path.mkdir(exist_ok=True, parents=True)

            # stitched_image = self.position_stitcher(
            #     path=path,
            #     df=stitch_group[['filename', 'x', 'y']],
            #     pos2pix=int(pos2pix * tiling_config.TilingConfig.DEFAULT_FILL_FACTORS['position'])
            # )
            
            logger.debug(f"{self._name}: - {output_file_loc}")

            if not cv2.imwrite(
                filename=str(output_file_loc),
                img=image
            ):
                logger.error(f"{self._name}: Unable to write image {output_file_loc}")
                continue

            
            protocol_post_record.add_record(
                root_path=root_path,
                file_path=output_file_loc_rel,
                timestamp=first_row['Timestamp'],
                name=first_row['Name'],
                # protocol_group_index=first_row['Protocol Group Index'],
                scan_count=first_row['Scan Count'],
                x=center['x'],
                y=center['y'],
                z=first_row['Z'],
                z_slice=first_row['Z-Slice'],
                well=first_row['Well'],
                color=first_row['Color'],
                objective=first_row['Objective'],
                tile_group_id=first_row['Tile Group ID'],
                custom_step=first_row['Custom Step'],
                **record_data_post_functions.to_dict(),
                # composite=first_row[PostFunction.COMPOSITE.value],
                # stitched=record_data[self._post_function.value],
                # z_project=first_row[PostFunction.ZPROJECT.value],
                # video=first_row[PostFunction.VIDEO.value],
                
                
                
                
                # stitch_group_index=first_row['Stitch Group Index'],
            )
      
            new_count += 1

        protocol_post_record.complete()

        if (new_count == 0) and (existing_count == 0):
            logger.info(f"{self._name}: No sets of images found to stitch")
            return {
                'status': False,
                'message': 'No images found'
            }

        # for metadata, path, metadata_filename in (
        #     (metadata, output_path, artifact_locations.stitcher_output_metadata_filename()),
        #     # (stitched_metadata_composite, output_path_stitched_and_composite, artifact_locations.composite_and_stitched_output_metadata_filename())
        # ):
        #     metadata_df = pd.DataFrame(metadata)
        #     if len(metadata_df) == 0:
        #         continue

        #     if not path.exists():
        #         path.mkdir(parents=True, exist_ok=True)
        
        #     metadata_df = pd.DataFrame(metadata)
        #     metadata_df.to_csv(
        #         path_or_buf=path / metadata_filename,
        #         header=True,
        #         index=False,
        #         sep='\t',
        #         lineterminator='\n',
        #     )
        
        logger.info(f"{self._name}: Complete - Created {new_count} stitched images.")
        return {
            'status': True,
            'message': 'Success'
        }


    @staticmethod
    def _calc_pos2pix_from_objective(objective: str) -> int:
        # TODO
        if objective == '4x':
            return 625 #550
        else:
            raise NotImplementedError(f"Image stitching for objectives other than 4x is not implemented")
    

    @staticmethod
    def _generate_stitched_filename(df: pd.DataFrame) -> str:
        row0 = df.iloc[0]
        # custom_step = common_utils.to_bool(row0['Custom Step'])
        # custom_step = row0['Custom Step']

        name = common_utils.generate_default_step_name(
            custom_name_prefix=row0['Name'],
            well_label=row0['Well'],
            color=row0['Color'],
            z_height_idx=row0['Z-Slice'],
            scan_count=row0['Scan Count']
        )
        
        outfile = f"{name}_stitched.tiff"
        return outfile


    @staticmethod
    def simple_position_stitcher(path: pathlib.Path, df: pd.DataFrame):
        '''
        Performs a simple concatenation of images, given a set of X/Y positions the images were captured from.
        Assumes no overlap between images.
        '''
        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        df = df.copy()

        num_x_tiles = df['X'].nunique()
        num_y_tiles = df['Y'].nunique()

        # Used to find the center of the image in X/Y coordinates
        x_center = df['X'].unique().mean()
        y_center = df['Y'].unique().mean()
        center = {
            'x': round(x_center, common_utils.max_decimal_precision(parameter='x')),
            'y': round(y_center, common_utils.max_decimal_precision(parameter='y')),
        }

        source_image_sample_row = df.iloc[0]
        source_image_sample_filename = source_image_sample_row['Filepath']
        source_image_sample = images[source_image_sample_filename]
        source_image_h, source_image_w = source_image_sample.shape
        
        df = df.sort_values(['X','Y'], ascending=False)
        df['x_index'] = df.groupby(by=['X']).ngroup()
        df['y_index'] = df.groupby(by=['Y']).ngroup()
        df['x_pix_range'] = df['x_index'] * source_image_w
        df['y_pix_range'] = df['y_index'] * source_image_h
            
        stitched_im_x = source_image_w * num_x_tiles
        stitched_im_y = source_image_h * num_y_tiles

        reverse_x = True
        reverse_y = False
        if reverse_x:
            df['x_pix_range'] = stitched_im_x - df['x_pix_range']

        if reverse_y:
            df['y_pix_range'] = stitched_im_y - df['y_pix_range']

        
        is_color_image = image_utils.is_color_image(image=source_image_sample)
        if is_color_image:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype=source_image_sample.dtype)
        else:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x), dtype=source_image_sample.dtype)

        for _, row in df.iterrows():
            filename = row['Filepath']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x] = image
            else:

                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image

        return stitched_img, center


    @staticmethod
    def position_stitcher(path: pathlib.Path, df: pd.DataFrame, pos2pix):
        
        # Load source images
        images = {}
        for _, row in df.iterrows():
            image_filepath = path / row['Filepath']
            images[row['Filepath']] = cv2.imread(str(image_filepath), cv2.IMREAD_UNCHANGED)

        df = df.copy()

        df['x_pos_range'] = df['X'] - df['X'].min()
        df['y_pos_range'] = df['Y'] - df['Y'].min()
        df['x_pix_range'] = df['x_pos_range']*pos2pix
        df['y_pix_range'] = df['y_pos_range']*pos2pix
        df = df.sort_values(['x_pix_range','y_pix_range'], ascending=False)
        df[['x_pix_range','y_pix_range']] = df[['x_pix_range','y_pix_range']].apply(np.floor).astype(int)

        source_image_sample_filename = df['Filepath'].values[0]
        source_image_sample = images[source_image_sample_filename]
        stitched_im_x = source_image_sample.shape[1] + df['x_pix_range'].max()
        stitched_im_y = source_image_sample.shape[0] + df['y_pix_range'].max()

        reverse_x = True
        reverse_y = False
        if reverse_x:
            df['x_pix_range'] = stitched_im_x - df['x_pix_range']

        if reverse_y:
            df['y_pix_range'] = stitched_im_y - df['y_pix_range']

        
        is_color_image = image_utils.is_color_image(image=source_image_sample)

        if is_color_image:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x, 3), dtype=source_image_sample.dtype)
        else:
            stitched_img = np.zeros((stitched_im_y, stitched_im_x), dtype=source_image_sample.dtype)

        for _, row in df.iterrows():
            filename = row['Filepath']
            image = images[filename]
            im_x = image.shape[1]
            im_y = image.shape[0]

            x_val = row['x_pix_range']
            y_val = row['y_pix_range']

            if reverse_y:
                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val-im_y:y_val, x_val:x_val+im_x] = image
            else:

                if reverse_x:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val-im_x:x_val] = image
                else:
                    if is_color_image:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image
                    else:
                        stitched_img[y_val:y_val+im_y, x_val:x_val+im_x,:] = image


        return stitched_img
            

if __name__ == "__main__":
    stitcher = Stitcher()
    stitcher.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))
