
import csv
import datetime
import io
import pathlib

import numpy as np
import pandas as pd

from modules.protocol_post_processing_functions import PostFunction

from lvp_logger import logger


class ProtocolPostRecord:

    FILE_HEADER = "LumaViewPro Protocol Post-Processing Record"
    CURRENT_VERSION = 1
    DEFAULT_FILENAME = 'protocol_post_record.tsv'
    COLUMNS = (
        'Filepath',
        'Timestamp',
        'Name',
        'Scan Count',
        'X',
        'Y',
        'Z',
        'Z-Slice',
        'Well',
        'Color',
        'Objective',
        'Tile Group ID',
        'Custom Step',
        *PostFunction.list_values(),
    )

    def __init__(
        self,
        file_loc: pathlib.Path,
        records: pd.DataFrame | None = None
    ):
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


    def _initialize_outfile(self, outfile: pathlib.Path):
        self._outfile_fp = open(outfile, 'w')
        self._outfile_csv = csv.writer(self._outfile_fp, delimiter='\t', lineterminator='\n')
        self._outfile_csv.writerow([self.FILE_HEADER])
        self._outfile_csv.writerow(['Version', self.CURRENT_VERSION])
        self._outfile_csv.writerow([])
        self._outfile_csv.writerow(['Images'])
        self._outfile_csv.writerow(self.COLUMNS)
 

    def _reopen_outfile(self, outfile: pathlib.Path):
        self._outfile_fp = open(outfile, 'a')
        self._outfile_csv = csv.writer(self._outfile_fp, delimiter='\t', lineterminator='\n')


    def complete(self):
        self._close_outfile()


    def _close_outfile(self):
        if self._outfile_fp is None:
            return
        
        self._outfile_fp.close()


    @staticmethod
    def _create_empty_df() -> pd.DataFrame:
        post_functions = PostFunction.list_values()
        post_function_tuples = [(post_function, bool) for post_function in post_functions]
        dtypes = np.dtype(
            [
                ("Filepath", str),
                ("Timestamp", str),
                ("Name", str),
                ("Scan Count", int),
                ("X", float),
                ("Y", float),
                ("Z", float),
                ("Z-Slice", int),
                ("Well", str),
                ("Color", str),
                ("Objective", str),
                ("Tile Group ID", int),
                ("Custom Step", bool),
                *post_function_tuples,
            ]
        )
        df = pd.DataFrame(np.empty(0, dtype=dtypes))
        return df


    def records(self) -> pd.DataFrame:
        return self._records
    

    def file_exists_in_records(self, filepath: pathlib.Path) -> bool:
        df = self._records
        df = df[df['Filepath'] == filepath]
        num_matches = len(df)
        if num_matches == 0:
            return False
        
        if num_matches == 1:
            return True
        
        if num_matches > 1:
            raise Exception(f"Expected 0 or 1 matched in post record for {filepath}, but found {num_matches}.")


    @staticmethod
    def _create_record_dict(
        root_path: pathlib.Path,
        file_path: pathlib.Path,
        timestamp: datetime.datetime,
        name: str,
        scan_count: int,
        x: float,
        y: float,
        z: float,
        z_slice: int,
        well: str,
        color: str,
        objective: str,
        tile_group_id: int | str,
        custom_step: bool,
        **kwargs: dict,
    ) -> dict:
        abs_path = root_path / file_path

        return {
            "Filepath": file_path,
            "Timestamp": timestamp,
            "Name": name,
            "Scan Count": scan_count,
            "X": x,
            "Y": y,
            "Z": z,
            "Z-Slice": z_slice,
            "Well": well,
            "Color": color,
            "Objective": objective,
            "Tile Group ID": tile_group_id,
            "Custom Step": custom_step,
            "Raw": False,
            "File Exists": abs_path.exists(),
            **kwargs,
        }

    def add_record(
        self,
        root_path: pathlib.Path,
        file_path: pathlib.Path,
        timestamp: datetime.datetime,
        name: str,
        scan_count: int,
        x: float,
        y: float,
        z: float,
        z_slice: int,
        well: str,
        color: str,
        objective: str,
        tile_group_id: int | str,
        custom_step: bool,
        **kwargs: dict,
    ):
        
        if self.file_exists_in_records(filepath=file_path):
            logger.info(f"[{self._name} ] File {file_path} already exists in records. Skipping.")

        record_dict = self._create_record_dict(
            root_path=root_path,
            file_path=file_path,
            timestamp=timestamp,
            name=name,
            scan_count=scan_count,
            x=x,
            y=y,
            z=z,
            z_slice=z_slice,
            well=well,
            color=color,
            objective=objective,
            tile_group_id=tile_group_id,
            custom_step=custom_step,
            **kwargs
        )

        new_record_df = pd.DataFrame([record_dict])

        df_list = [self._records, new_record_df]
        self._records = pd.concat([df for df in df_list if not df.empty], ignore_index=True).reset_index(drop=True)

        self._add_record_to_file(
            file_path=file_path,
            timestamp=timestamp,
            name=name,
            scan_count=scan_count,
            x=x,
            y=y,
            z=z,
            z_slice=z_slice,
            well=well,
            color=color,
            objective=objective,
            tile_group_id=tile_group_id,
            custom_step=custom_step,
            **kwargs
        )


    def _add_record_to_file(
        self,
        file_path: pathlib.Path,
        timestamp: datetime.datetime,
        name: str,
        scan_count: int,
        x: float,
        y: float,
        z: float,
        z_slice: int,
        well: str,
        color: str,
        objective: str,
        tile_group_id: int | str,
        custom_step: bool,
        **kwargs: dict,
    ):

        self._outfile_csv.writerow(
            [
                file_path,
                timestamp,
                name,
                scan_count,
                x,
                y,
                z,
                z_slice,
                well,
                color,
                objective,
                tile_group_id,
                custom_step,
                *kwargs.values(),
            ]
        )
        self._outfile_fp.flush()


    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        with open(file_path, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t') 
            header = next(csvreader)
            if header[0] != cls.FILE_HEADER:
                raise Exception(f"Invalid protocol post-processing record")
            
            version = next(csvreader)
            if version[0] != 'Version':
                raise Exception(f"Version key not found")
            
            if int(version[1]) not in (1,):
                raise Exception(f"Unsupported protocol execution record version")
            
            # Search for "Images" to indicate start of images data
            while True:
                tmp = next(csvreader)
                if len(tmp) == 0:
                    continue

                if tmp[0] == "Images":
                    break

            table_lines = []
            for line in fp:
                table_lines.append(line)

            table_str = ''.join(table_lines)
            df = pd.read_csv(io.StringIO(table_str), sep='\t', lineterminator='\n').fillna('')

            # Convert filename to pathlib type
            df['Filepath'] = df.apply(lambda row: pathlib.Path(row['Filepath']), axis=1)

            root_path = file_path.parent
            df['File Exists'] = df.apply(lambda row: True if (root_path / row['Filepath']).is_file() else False, axis=1)

            if len(df) == 0:
                cls._create_empty_df()

            return ProtocolPostRecord(
                file_loc=file_path,
                records=df,
            )
