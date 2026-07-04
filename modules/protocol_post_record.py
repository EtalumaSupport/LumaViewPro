# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import csv
import datetime
import io
import os
import pathlib

import numpy as np
import pandas as pd

from modules.common_utils import PostFunction, recover_step_label, to_int

from lvp_logger import logger


class ProtocolPostRecord:
    FILE_HEADER = 'LumaViewPro Protocol Post-Processing Record'
    # v2 adds Label: the source step's base text (user text or the
    # 'custom<NNNN>' prefix) travels with every record so output filenames
    # derive from the persisted field instead of re-parsing the rendered
    # Name, which truncated user labels containing token-shaped segments.
    CURRENT_VERSION = 2
    DEFAULT_FILENAME = 'protocol_post_record.tsv'

    def __del__(self):
        try:
            self._close_outfile()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_outfile()
        return False

    COLUMNS = (
        'Filepath',
        'Timestamp',
        'Name',
        'Label',
        'Scan Count',
        'X',
        'Y',
        'Z',
        'Z-Slice',
        'Well',
        'Color',
        'Objective',
        'Tile Group ID',
        'Tile',
        'Custom Step',
        *PostFunction.list_values(),
    )

    def __init__(self, file_loc: pathlib.Path, records: pd.DataFrame | None = None):
        self._name = self.__class__.__name__

        self._file_loc = file_loc

        if not self._file_loc.exists():
            self._initialize_outfile(outfile=self._file_loc)
        else:
            self._reopen_outfile(outfile=self._file_loc)

        if records is None:
            self._records = self._create_empty_df()
        else:
            self._records = records
        self._pending_records = []  # Accumulated dicts, merged lazily

    def _initialize_outfile(self, outfile: pathlib.Path):
        self._outfile_fp = open(outfile, 'w')  # noqa: SIM115 -- instance-lifetime handle closed in _close_outfile()
        self._outfile_csv = csv.writer(self._outfile_fp, delimiter='\t', lineterminator='\n')
        self._outfile_csv.writerow([self.FILE_HEADER])
        self._outfile_csv.writerow(['Version', self.CURRENT_VERSION])
        self._outfile_csv.writerow([])
        self._outfile_csv.writerow(['Images'])
        self._outfile_csv.writerow(self.COLUMNS)

    def _reopen_outfile(self, outfile: pathlib.Path):
        self._outfile_fp = open(outfile, 'a')  # noqa: SIM115 -- instance-lifetime handle closed in _close_outfile()
        self._outfile_csv = csv.writer(self._outfile_fp, delimiter='\t', lineterminator='\n')

    def complete(self):
        self._close_outfile()

    def _close_outfile(self):
        if self._outfile_fp is None:
            return

        self._outfile_fp.close()
        self._outfile_fp = None
        self._outfile_csv = None

    @staticmethod
    def _create_empty_df() -> pd.DataFrame:
        post_functions = PostFunction.list_values()
        post_function_tuples = [(post_function, bool) for post_function in post_functions]
        dtypes = np.dtype(
            [
                ('Filepath', str),
                ('Timestamp', str),
                ('Name', str),
                ('Label', str),
                ('Scan Count', int),
                ('X', float),
                ('Y', float),
                ('Z', float),
                ('Z-Slice', int),
                ('Well', str),
                ('Color', str),
                ('Objective', str),
                ('Tile Group ID', int),
                ('Tile', str),
                ('Custom Step', bool),
                *post_function_tuples,
            ]
        )
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
        return df

    def _flush_pending(self):
        """Merge accumulated record dicts into the DataFrame."""
        if not self._pending_records:
            return
        new_df = pd.DataFrame(self._pending_records)
        df_list = [self._records, new_df]
        self._records = pd.concat(
            [df for df in df_list if not df.empty], ignore_index=True
        ).reset_index(drop=True)
        self._pending_records.clear()

    def records(self) -> pd.DataFrame:
        self._flush_pending()
        return self._records

    def file_exists_in_records(self, filepath: pathlib.Path) -> bool:
        # Check pending records first (avoid flushing for every lookup)
        for rec in self._pending_records:
            if rec.get('Filepath') == filepath:
                return True

        df = self._records
        df = df[df['Filepath'] == filepath]
        num_matches = len(df)
        if num_matches == 0:
            return False

        if num_matches == 1:
            return True

        if num_matches > 1:
            raise Exception(
                f'Expected 0 or 1 matched in post record for {filepath}, but found {num_matches}.'
            )

    @staticmethod
    def _create_record_dict(
        root_path: pathlib.Path,
        file_path: pathlib.Path,
        timestamp: datetime.datetime,
        name: str,
        label: str,
        scan_count: int,
        x: float,
        y: float,
        z: float,
        z_slice: int,
        well: str,
        color: str,
        objective: str,
        tile_group_id: int | str,
        tile: str,
        custom_step: bool,
        **kwargs: dict,
    ) -> dict:
        abs_path = root_path / file_path

        return {
            'Filepath': file_path,
            'Timestamp': timestamp,
            'Name': name,
            'Label': label,
            'Scan Count': scan_count,
            'X': x,
            'Y': y,
            'Z': z,
            'Z-Slice': z_slice,
            'Well': well,
            'Color': color,
            'Objective': objective,
            'Tile Group ID': tile_group_id,
            'Tile': tile,
            'Custom Step': custom_step,
            'Raw': False,
            'File Exists': abs_path.exists(),
            **kwargs,
        }

    def add_record(
        self,
        root_path: pathlib.Path,
        file_path: pathlib.Path,
        timestamp: datetime.datetime,
        name: str,
        label: str,
        scan_count: int,
        x: float,
        y: float,
        z: float,
        z_slice: int,
        well: str,
        color: str,
        objective: str,
        tile_group_id: int | str,
        tile: str,
        custom_step: bool,
        **kwargs: dict,
    ):

        if self.file_exists_in_records(filepath=file_path):
            logger.info(f'[{self._name} ] File {file_path} already exists in records. Skipping.')

        record_dict = self._create_record_dict(
            root_path=root_path,
            file_path=file_path,
            timestamp=timestamp,
            name=name,
            label=label,
            scan_count=scan_count,
            x=x,
            y=y,
            z=z,
            z_slice=z_slice,
            well=well,
            color=color,
            objective=objective,
            tile_group_id=tile_group_id,
            tile=tile,
            custom_step=custom_step,
            **kwargs,
        )

        self._pending_records.append(record_dict)

        self._add_record_to_file(
            file_path=file_path,
            timestamp=timestamp,
            name=name,
            label=label,
            scan_count=scan_count,
            x=x,
            y=y,
            z=z,
            z_slice=z_slice,
            well=well,
            color=color,
            objective=objective,
            tile_group_id=tile_group_id,
            tile=tile,
            custom_step=custom_step,
            **kwargs,
        )

    def _add_record_to_file(
        self,
        file_path: pathlib.Path,
        timestamp: datetime.datetime,
        name: str,
        label: str,
        scan_count: int,
        x: float,
        y: float,
        z: float,
        z_slice: int,
        well: str,
        color: str,
        objective: str,
        tile_group_id: int | str,
        tile: str,
        custom_step: bool,
        **kwargs: dict,
    ):

        self._outfile_csv.writerow(
            [
                file_path,
                timestamp,
                name,
                label,
                scan_count,
                x,
                y,
                z,
                z_slice,
                well,
                color,
                objective,
                tile_group_id,
                tile,
                custom_step,
                *kwargs.values(),
            ]
        )
        self._outfile_fp.flush()

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        with open(file_path) as fp:
            csvreader = csv.reader(fp, delimiter='\t')
            header = next(csvreader)
            if header[0] != cls.FILE_HEADER:
                raise Exception('Invalid protocol post-processing record')

            version = next(csvreader)
            if version[0] != 'Version':
                raise Exception('Version key not found')

            file_version = int(version[1])
            if file_version not in (1, 2):
                raise Exception('Unsupported protocol post-processing record version')

            # Search for "Images" to indicate start of images data
            while True:
                tmp = next(csvreader)
                if len(tmp) == 0:
                    continue

                if tmp[0] == 'Images':
                    break

            table_lines = []
            for line in fp:
                table_lines.append(line)

            table_str = ''.join(table_lines)
            # Pin the text-identity columns to str at read time: pandas type
            # inference otherwise turns a numeric-looking name or label
            # ('0600') into a float ('600.0') that corrupts derived output
            # filenames.
            df = pd.read_csv(
                io.StringIO(table_str),
                sep='\t',
                lineterminator='\n',
                dtype={'Name': str, 'Label': str, 'Well': str, 'Tile': str},
            ).fillna('')

            if len(df) == 0:
                # A record with no data rows is a legitimate state (a prior
                # session generated no outputs); keep the typed empty frame
                # so the append path stays usable instead of discarding the
                # whole record file.
                df = cls._create_empty_df()
            else:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

                # Convert filename to pathlib type
                df['Filepath'] = df.apply(lambda row: pathlib.Path(row['Filepath']), axis=1)

                root_path = file_path.parent
                df['File Exists'] = df.apply(
                    lambda row: (root_path / row['Filepath']).is_file(), axis=1
                )

                # Z-Slice arrives as float64 whenever any cell is blank
                # (pandas NaN inference); normalize to the int / -1-sentinel
                # form so a z index can never render as 'Z3.0' in a name.
                df['Z-Slice'] = df['Z-Slice'].apply(to_int)

            if 'Label' not in df.columns:
                # Pre-v2 records never persisted the label; recover it per
                # row from the Name's shape (machine-rendered or
                # machine-anchored names yield their machine base, anything
                # else is user text kept verbatim). Post rows' columns are
                # output-adjusted (a stitch blanks the tile), so the
                # anchor-shape test, not a plain render compare, is what
                # keeps their machine names from being misread as labels.
                df['Label'] = [recover_step_label(row)[0] for _, row in df.iterrows()]

            record = ProtocolPostRecord(
                file_loc=file_path,
                records=df,
            )

            if file_version < cls.CURRENT_VERSION:
                # The instance appends rows in the current column order, so
                # an older file must be upgraded in place; appending
                # current-format rows under an old header would silently
                # misalign every column after Name.
                record._rewrite_outfile()

            return record

    def _rewrite_outfile(self):
        """Rewrite the on-disk file in the current format, keeping all rows.

        Writes to a sibling temp file and atomically replaces the original,
        so a crash or error mid-rewrite cannot truncate the record -- the
        old file survives intact until the new one is complete.
        """
        self._close_outfile()
        tmp_loc = self._file_loc.with_name(self._file_loc.name + '.tmp')
        post_function_columns = PostFunction.list_values()
        try:
            self._initialize_outfile(outfile=tmp_loc)
            for _, row in self.records().iterrows():
                self._add_record_to_file(
                    file_path=row['Filepath'],
                    timestamp=row['Timestamp'],
                    name=row['Name'],
                    label=row['Label'],
                    scan_count=row['Scan Count'],
                    x=row['X'],
                    y=row['Y'],
                    z=row['Z'],
                    z_slice=row['Z-Slice'],
                    well=row['Well'],
                    color=row['Color'],
                    objective=row['Objective'],
                    tile_group_id=row['Tile Group ID'],
                    tile=row['Tile'],
                    custom_step=row['Custom Step'],
                    **{column: row[column] for column in post_function_columns},
                )
            self._close_outfile()
            os.replace(tmp_loc, self._file_loc)
        except Exception:
            self._close_outfile()
            tmp_loc.unlink(missing_ok=True)
            raise
        self._reopen_outfile(outfile=self._file_loc)
