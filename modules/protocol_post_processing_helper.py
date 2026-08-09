# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
import modules.recording_frames as recording_frames
from modules.protocol import Protocol
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules.common_utils import PostFunction
from modules.protocol_post_record import ProtocolPostRecord

from lvp_logger import logger


class ProtocolPostProcessingHelper:
    def __init__(self):
        self._name = self.__class__.__name__

    @staticmethod
    def _get_image_filenames_from_folder(
        path: pathlib.Path,
        exclude_subpaths: list | None = None,
        include_subpaths: list | None = None,
    ) -> dict[str, list[pathlib.Path]]:

        if exclude_subpaths is None:
            exclude_subpaths = []
        if include_subpaths is None:
            include_subpaths = []

        raw_image_names = []
        post_image_names = []
        raw_image_dirs = ['.', *common_utils.get_layers()]

        if (len(include_subpaths) != 0) and (len(exclude_subpaths) != 0):
            raise Exception('Specify only include_subpaths OR exclude_subpaths. Not both.')

        images = image_utils.find_tiff_files(path, recursive=True)

        # A video recording folder holds frames one level below the
        # protocol folder that owns the execution record; step up so
        # relative names compute against the protocol folder. Tested on
        # the final path component only -- a protocol ROOT that happens
        # to carry the token in its name must not step out of itself.
        if recording_frames.is_video_recording_dir_name(path.name):
            path = path.parent

        for image in images:
            image_name = pathlib.Path(os.path.relpath(image, path))

            if recording_frames.is_protocol_video_frame(image_name.name):
                parent_dir = str(image_name.parent.parent)

            else:
                parent_dir = str(image_name.parent)

            if (len(exclude_subpaths) > 0 and (parent_dir in exclude_subpaths)) or (
                len(include_subpaths) > 0 and (parent_dir not in include_subpaths)
            ):
                continue

            if parent_dir not in raw_image_dirs:
                post_image_names.append(image_name)
            else:
                raw_image_names.append(image_name)

        return {
            'raw': raw_image_names,
            'post': post_image_names,
        }

    @staticmethod
    def generate_output_dir_name(record: pd.Series) -> pathlib.Path:
        # Filter to only the true values
        record = record[record == True]  # noqa: E712 -- pandas mask

        # Get the post-processing function names, in alphabetical order
        used_functions = sorted(record.keys().to_list())

        return pathlib.Path('-'.join(used_functions))

    def _find_protocol_tsvs(self, path: pathlib.Path) -> dict[str, pathlib.Path] | None:

        # If provided a file, change to the parent folder
        try:
            if not path.is_dir():
                path = path.parent
        except Exception:
            return None

        loc_data = {}

        # Search for the protocol execution record TSV in the current directory and the parent directory
        protocol_execution_record_filename = ProtocolExecutionRecord.DEFAULT_FILENAME
        protocol_execution_record_file_loc = path / protocol_execution_record_filename
        if protocol_execution_record_file_loc.is_file():
            protocol_root_dir = path
        else:
            try:
                protocol_execution_record_file_loc = (
                    path.parent / protocol_execution_record_filename
                )
                if protocol_execution_record_file_loc.is_file():
                    protocol_root_dir = path.parent
                else:
                    return None
            except Exception:
                return None

        loc_data['protocol_root_dir'] = protocol_root_dir
        loc_data['protocol_execution_record'] = protocol_execution_record_file_loc
        protocol_execution_record = ProtocolExecutionRecord.from_file(
            file_path=protocol_execution_record_file_loc
        )

        # Search for the post-processing record TSV
        post_record_filename = ProtocolPostRecord.DEFAULT_FILENAME
        post_record_file_loc = protocol_root_dir / post_record_filename
        if post_record_file_loc.is_file():
            loc_data['protocol_post_record'] = post_record_file_loc
        else:
            loc_data['protocol_post_record'] = None

        # Search for the protocol TSV
        protocol_file_relative_loc = protocol_execution_record.protocol_file_loc()
        protocol_file_loc = protocol_root_dir / protocol_file_relative_loc
        if not protocol_file_loc.is_file():
            return None

        loc_data['protocol'] = protocol_file_loc

        return loc_data

    def _get_raw_images_df(
        self,
        image_names: list,
        protocol: Protocol,
        protocol_execution_record: ProtocolExecutionRecord,
    ) -> pd.DataFrame | None:

        image_data = []

        for image_name in image_names:
            # A video frame's execution-record row is keyed by its
            # recording folder, not the frame file.
            if recording_frames.is_protocol_video_frame(image_name.name):
                file_data = protocol_execution_record.get_data_from_filename(
                    file_path=image_name.parent
                )
            else:
                file_data = protocol_execution_record.get_data_from_filename(file_path=image_name)

            if file_data is None:
                logger.warning(f'No info found in protocol execution record for {image_name}')
                continue

            step_idx = file_data['Step Index']
            step = protocol.step(idx=step_idx)

            image_data.append(
                {
                    'Filepath': image_name,
                    'Name': step['Name'],
                    'Label': step['Label'],
                    'Scan Count': file_data['Scan Count'],
                    'Step Index': step_idx,
                    'X': step['X'],
                    'Y': step['Y'],
                    'Z': step['Z'],
                    'Z-Slice': step['Z-Slice'],
                    'Well': step['Well'],
                    'Color': step['Color'],
                    'Objective': step['Objective'],
                    'Tile': step['Tile'],
                    'Tile Group ID': step['Tile Group ID'],
                    'Z-Stack Group ID': step['Z-Stack Group ID'],
                    'Custom Step': step['Custom Step'],
                    'Timestamp': file_data['Timestamp'],
                }
            )

        df = pd.DataFrame(image_data)
        df = df.fillna('')

        return df

    def _get_post_images_df(
        self,
        image_names: list[pathlib.Path],
        protocol_post_record: ProtocolPostRecord,
    ) -> pd.DataFrame | None:

        df = protocol_post_record.records()
        if len(df) == 0:
            return df

        # Filter out any images that are missing from the filesystem
        # This is not strictly needed since the following filter using 'image_names'
        # will also inherently remove non-existent files
        df = df[df['File Exists'] == True]  # noqa: E712 -- pandas mask

        # Filter out any images that are not path of the selected images provided
        df = df[df['Filepath'].isin(image_names)]

        return df

    @staticmethod
    def _add_zproject_group_index(df: pd.DataFrame) -> pd.DataFrame:
        df['Z-Project Group Index'] = df.groupby(
            by=['Scan Count', 'Well', 'Color', 'Objective', 'X', 'Y', 'Tile', 'Custom Step'],
            dropna=False,
        ).ngroup()
        return df

    def load_folder(
        self,
        path: str | pathlib.Path,
        tiling_configs_file_loc: pathlib.Path,
    ) -> dict:
        selected_path = pathlib.Path(path)
        logger.info(f'{self._name}: Loading folder {selected_path}')

        protocol_tsvs = self._find_protocol_tsvs(path=selected_path)

        if protocol_tsvs is None:
            logger.error(f'{self._name}: Protocol and/or protocol record not found in folder')
            return {
                'status': False,
                'message': 'Protocol and/or Protocol Record not found in folder',
            }

        root_path = protocol_tsvs['protocol_root_dir']

        # Special handling for logging this since it may be None or a pathlib file
        protocol_post_record_str = (
            'None'
            if protocol_tsvs['protocol_post_record'] is None
            else protocol_tsvs['protocol_post_record'].name
        )

        logger.info(f"""{self._name}: Found ->
            Selected dir:                      {selected_path}
            Protocol root dir:                 {root_path}
            Protocol:                          {protocol_tsvs['protocol'].name}
            Protocol execution record:         {protocol_tsvs['protocol_execution_record'].name}
            Protocol post-processing metadata: {protocol_post_record_str}
        """)

        try:
            protocol = Protocol.from_file(
                file_path=protocol_tsvs['protocol'], tiling_configs_file_loc=tiling_configs_file_loc
            )
        except Exception as e:
            msg = f'Unable to load protocol file: {e}'
            return {
                'status': False,
                'message': msg,
            }

        if protocol is None:
            msg = 'Protocol not loaded'
            logger.error(f'{self._name}: {msg}')
            return {
                'status': False,
                'message': msg,
            }

        protocol_execution_record = ProtocolExecutionRecord.from_file(
            file_path=protocol_tsvs['protocol_execution_record'],
        )

        if protocol_execution_record is None:
            msg = 'Protocol Execution Record not loaded'
            logger.error(f'{self._name}: {msg}')
            return {
                'status': False,
                'message': msg,
            }

        if protocol_execution_record.num_records() == 0:
            msg = 'Protocol Execution Record has no records'
            logger.error(f'{self._name}: {msg}')
            return {
                'status': False,
                'message': msg,
            }

        protocol_post_record = None
        if protocol_tsvs['protocol_post_record'] is not None:
            try:
                protocol_post_record = ProtocolPostRecord.from_file(
                    file_path=protocol_tsvs['protocol_post_record'],
                )
                logger.info(
                    f'Loaded existing protocol post record {protocol_tsvs["protocol_post_record"]}'
                )
            except Exception as e:
                # An unreadable record must be moved aside, not appended to:
                # constructing a fresh ProtocolPostRecord on it reopens it in
                # append mode, and current-format rows under its old header
                # would misalign every column on the next load.
                record_loc = protocol_tsvs['protocol_post_record']
                preserved_loc = record_loc.with_name(record_loc.name + '.unreadable')
                try:
                    os.replace(record_loc, preserved_loc)
                except OSError as move_error:
                    msg = (
                        f'The post-processing record {record_loc.name} could not be '
                        f'read ({e}) or moved aside ({move_error}). Close any program '
                        f'holding it open, or remove it, then retry.'
                    )
                    logger.error(f'{self._name}: {msg}')
                    return {
                        'status': False,
                        'message': msg,
                    }
                logger.error(
                    f'{self._name}: Unable to load the protocol post record file '
                    f'{record_loc} ({e}). Preserved it as {preserved_loc.name}; '
                    f'starting a new record.'
                )

        if protocol_post_record is None:
            protocol_post_record = ProtocolPostRecord(
                file_loc=root_path / ProtocolPostRecord.DEFAULT_FILENAME
            )

        if selected_path == root_path:
            include_subpaths = []
        else:
            include_subpaths = [selected_path.name]

        # A selected video recording folder scans itself for images
        # instead of the root folder. Tested on the final path component
        # only -- a root that carries the token elsewhere in its path
        # must keep the normal root scan.
        if recording_frames.is_video_recording_dir_name(path.name):
            image_names = self._get_image_filenames_from_folder(
                path=path, exclude_subpaths=[], include_subpaths=[]
            )
        else:
            image_names = self._get_image_filenames_from_folder(
                path=root_path, exclude_subpaths=[], include_subpaths=include_subpaths
            )

        raw_images_df = self._get_raw_images_df(
            image_names=image_names['raw'],
            protocol=protocol,
            protocol_execution_record=protocol_execution_record,
        )

        raw_images_df['Raw'] = True

        post_processing_names = PostFunction.list_values()
        raw_images_df[post_processing_names] = False

        post_images_df = self._get_post_images_df(
            image_names=image_names['post'],
            protocol_post_record=protocol_post_record,
        )
        post_images_df['Raw'] = False

        if (len(raw_images_df) == 0) and (len(post_images_df) == 0):
            log_msg = 'No image files found in folder to process'
            user_msg = (
                'No image files were found in this folder to process. '
                'Check that the folder contains captured scan images.'
            )
            logger.error(f'{self._name}: {log_msg}')
            return {
                'status': False,
                'message': user_msg,
            }

        df_list = [raw_images_df, post_images_df]
        images_df = pd.concat([df for df in df_list if not df.empty])

        return {
            'status': True,
            'root_path': root_path,
            'selected_path': selected_path,
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'protocol_post_record': protocol_post_record,
            'images_df': images_df,
        }
