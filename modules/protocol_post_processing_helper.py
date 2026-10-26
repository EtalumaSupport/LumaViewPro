# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import os
import pathlib

import pandas as pd

import modules.common_utils as common_utils
import modules.image_utils as image_utils
import modules.path_utils as path_utils
import modules.recording_frames as recording_frames
from modules.protocol import Protocol, StoredNames
from modules.protocol_execution_record import ProtocolExecutionRecord
from modules import post_derivation
from modules.protocol_post_record import ProtocolPostRecord

from lvp_logger import logger


def _raw_row(
    image_name: pathlib.Path, recorded: dict, protocol: Protocol, *, missing: str, frames: int
) -> dict:
    """One raw frame's row: its file, its step's columns, and its record's.

    ``Missing`` is empty for a frame in the folder, else why it is not;
    ``Frames`` is how many frames the row stands for.
    """
    step_index = recorded['Step Index']
    step = protocol.step_row(step_index=step_index)
    return {
        'Filepath': image_name,
        'Name': step['Name'],
        'Label': step['Label'],
        'Scan Count': recorded['Scan Count'],
        'Step Index': step_index,
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
        'Channel Group ID': step['Channel Group ID'],
        'Custom Step': step['Custom Step'],
        'Acquire': step['Acquire'],
        'Timestamp': recorded['Timestamp'],
        'Missing': missing,
        'Frames': frames,
    }


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
        # A run saved a folder per channel under the names it stored, an old
        # run's under its old names.
        raw_image_dirs = ['.', *common_utils.get_stored_channel_names()]

        if (len(include_subpaths) != 0) and (len(exclude_subpaths) != 0):
            raise Exception('Specify only include_subpaths OR exclude_subpaths. Not both.')

        images = image_utils.find_tiff_files(path, recursive=True)

        # A video recording folder holds frames one level below the
        # protocol folder that owns the execution record; step up so
        # relative names compute against the protocol folder. Told from the
        # frames it holds -- a protocol ROOT whose name carries the token
        # must not step out of itself.
        if recording_frames.is_video_recording_dir(path):
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

    def _find_protocol_tsvs(self, path: pathlib.Path) -> dict | None:
        """The run folder's files, or None when it holds no run record and protocol.

        Raises:
            ValueError: the run record cannot be read, or names a protocol
                outside the run folder -- the message says which, for a
                person.
        """

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
        try:
            protocol_execution_record = ProtocolExecutionRecord.from_file(
                file_path=protocol_execution_record_file_loc
            )
        except (OSError, ValueError) as e:
            raise ValueError(
                f'The run record {protocol_execution_record_file_loc.name} could not be read ({e}).'
            ) from e
        loc_data['execution_record'] = protocol_execution_record

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
        # The record is data in the folder, not the program's own: a path in
        # it that climbs out of the run folder would read a file the run
        # never wrote.
        if not path_utils.resolves_inside(protocol_root_dir, protocol_file_loc):
            raise ValueError(
                f'The run record names its protocol as {str(protocol_file_relative_loc)!r}, '
                f'which is outside the run folder {protocol_root_dir}.'
            )
        if not protocol_file_loc.is_file():
            return None

        loc_data['protocol'] = protocol_file_loc

        return loc_data

    def _get_raw_images_df(
        self,
        image_names: list,
        protocol: Protocol,
        protocol_execution_record: ProtocolExecutionRecord,
    ) -> pd.DataFrame:

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

            image_data.append(_raw_row(image_name, file_data, protocol, missing='', frames=1))

        return pd.DataFrame(image_data)

    @staticmethod
    def _get_missing_images(
        protocol: Protocol,
        protocol_execution_record: ProtocolExecutionRecord,
        root_path: pathlib.Path,
        include_subpaths: list[str],
    ) -> list[dict]:
        """A marked row for each frame the run recorded that the folder lacks.

        Two kinds: a recorded file that is not in the folder (``Missing`` is
        ``'not_on_disk'``), and a capture the run recorded as leaving no
        file (``Missing`` is the reason word the run wrote). Only what the
        disk walk would have taken counts: a TIFF or a recording's frames
        folder, in the selected subfolder when one is selected; a capture
        that left no file belongs to a subfolder named for its channel.
        ``Frames`` is how many frames the row stands for, so a missing
        recording is a group's worth, not one.
        """
        rows = []
        for entry in protocol_execution_record.saved_entries():
            reason = entry['No File']
            if reason is None:
                recorded = pathlib.Path(entry['Filename'])
                takes = (
                    recorded.suffix.lower() in image_utils.TIFF_SUFFIXES
                    or recording_frames.is_video_recording_dir_name(recorded.name)
                )
                if not takes or (root_path / recorded).exists():
                    continue
                if include_subpaths and str(recorded.parent) not in include_subpaths:
                    continue
                missing = 'not_on_disk'
            else:
                recorded = pathlib.Path(reason)
                color = protocol.step_row(step_index=entry['Step Index'])['Color']
                if include_subpaths and color not in include_subpaths:
                    continue
                missing = reason
            rows.append(
                _raw_row(
                    recorded,
                    entry,
                    protocol,
                    missing=missing,
                    frames=max(1, entry['Frame Count']),
                )
            )
        return rows

    def _get_post_images_df(
        self,
        image_names: list[pathlib.Path],
        protocol_post_record: ProtocolPostRecord,
        *,
        include_subpaths: list[str],
    ) -> pd.DataFrame | None:
        """The record's derived images the build may take, as rows.

        An image the disk walk found is taken. One the record names and the
        folder lacks is taken too, marked ``Missing`` ``'not_on_disk'`` as a
        raw frame the run recorded is, so the group it belongs to fails
        instead of building short -- under the walk's own filters: only an
        image the walk would have taken (a TIFF), and only in the selected
        subfolder when one is selected.
        """
        df = protocol_post_record.records()
        if len(df) == 0:
            return df

        def walk_would_take(path) -> bool:
            path = pathlib.Path(path)
            return path.suffix.lower() in image_utils.TIFF_SUFFIXES and (
                not include_subpaths or str(path.parent) in include_subpaths
            )

        found = df['Filepath'].isin(image_names)
        gone = ~df['File Exists'].astype(bool) & df['Filepath'].map(walk_would_take).astype(bool)
        df = df[found | gone].copy()
        df['Missing'] = gone[found | gone].map({True: 'not_on_disk', False: ''})
        df['Frames'] = 1
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

        # Each 'status': False below is a folder this cannot process, which
        # the caller raises as a refusal; the reporter logs it once, so none
        # is logged here.
        try:
            protocol_tsvs = self._find_protocol_tsvs(path=selected_path)
        except ValueError as e:
            return {
                'status': False,
                'message': str(e),
            }
        if protocol_tsvs is None:
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
            # The run's record, read for its steps in the names its files
            # carry and never run or edited: a period saved before the floor,
            # or a step cell a run would refuse, does not refuse it here.
            protocol = Protocol.from_file(
                file_path=protocol_tsvs['protocol'],
                tiling_configs_file_loc=tiling_configs_file_loc,
                unit=StoredNames.KEPT,
            )
        except Exception as e:
            msg = f'Unable to load protocol file: {e}'
            return {
                'status': False,
                'message': msg,
            }

        protocol_execution_record = protocol_tsvs['execution_record']

        if protocol_execution_record.num_records() == 0:
            msg = 'Protocol Execution Record has no records'
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
                try:
                    preserved_loc = path_utils.move_aside(record_loc, '.unreadable')
                except OSError as move_error:
                    msg = (
                        f'The post-processing record {record_loc.name} could not be '
                        f'read ({e}) or moved aside ({move_error}). Close any program '
                        f'holding it open, or remove it, then retry.'
                    )
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
        # instead of the root folder. Told from the frames it holds -- a
        # root whose name carries the token must keep the normal root scan.
        if recording_frames.is_video_recording_dir(path):
            image_names = self._get_image_filenames_from_folder(
                path=path, exclude_subpaths=[], include_subpaths=[]
            )
        else:
            image_names = self._get_image_filenames_from_folder(
                path=root_path, exclude_subpaths=[], include_subpaths=include_subpaths
            )

        raw_images = self._get_raw_images_df(
            image_names=image_names['raw'],
            protocol=protocol,
            protocol_execution_record=protocol_execution_record,
        ).to_dict('records')
        # A selected recording folder is its frames only; any other
        # selection also takes what the run recorded and the folder lacks.
        if not recording_frames.is_video_recording_dir(path):
            raw_images += self._get_missing_images(
                protocol=protocol,
                protocol_execution_record=protocol_execution_record,
                root_path=root_path,
                include_subpaths=include_subpaths,
            )
        raw_images_df = pd.DataFrame(raw_images).fillna('')

        raw_images_df['Raw'] = True
        raw_images_df = raw_images_df.assign(**post_derivation.raw_columns())

        post_images_df = self._get_post_images_df(
            image_names=image_names['post'],
            protocol_post_record=protocol_post_record,
            include_subpaths=include_subpaths,
        )
        post_images_df['Raw'] = False

        if (len(raw_images_df) == 0) and (len(post_images_df) == 0):
            user_msg = (
                'No image files were found in this folder to process. '
                'Check that the folder contains captured scan images.'
            )
            return {
                'status': False,
                'message': user_msg,
            }

        # Renumbered: each frame arrives with its own 0-based index, and a
        # label shared by a raw row and a post-processed row lets a caller
        # that finds rows by label mistake one for the other.
        df_list = [raw_images_df, post_images_df]
        images_df = pd.concat([df for df in df_list if not df.empty], ignore_index=True)

        return {
            'status': True,
            'root_path': root_path,
            'selected_path': selected_path,
            'protocol': protocol,
            'protocol_execution_record': protocol_execution_record,
            'protocol_post_record': protocol_post_record,
            'images_df': images_df,
        }
