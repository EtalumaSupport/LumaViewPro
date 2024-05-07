
import csv
import datetime
import pathlib

import pandas as pd


class ProtocolExecutionRecord:

    FILE_HEADER = "LumaViewPro Protocol Execution Record"
    CURRENT_VERSION = 2
    DEFAULT_FILENAME = 'protocol_record.tsv'
    COLUMNS = ('Filename', 'Step Name', 'Step Index', 'Scan Count', 'Timestamp')

    def __init__(
        self,
        protocol_file_loc: pathlib.Path, # | None = None,
        outfile: pathlib.Path | None = None,
        records: pd.DataFrame | None = None
    ):
        if (outfile is not None) and (records is not None):
            raise Exception(f"Specify only outfile OR records")
        
        if (outfile is None) and (records is None):
            raise Exception(f"Must specify outfile or records")
        
        self._protocol_file_loc = pathlib.Path(protocol_file_loc)

        self._outfile_fp = None
        if outfile is not None:
            self._mode = "to_file"
            self._outfile = outfile
            self._initialize_outfile(outfile=outfile)
        else:
            self._mode = "from_file"
            self._records = records


    def _initialize_outfile(self, outfile: pathlib.Path):
        self._outfile_fp = open(outfile, 'w')
        self._outfile_csv = csv.writer(self._outfile_fp, delimiter='\t', lineterminator='\n')
        self._outfile_csv.writerow([self.FILE_HEADER])
        self._outfile_csv.writerow(['Version', self.CURRENT_VERSION])
        self._outfile_csv.writerow(['Protocol File', str(self._protocol_file_loc)])
        self._outfile_csv.writerow(self.COLUMNS)

    
    def protocol_file_loc(self) -> pathlib.Path:
        return self._protocol_file_loc
    

    def complete(self):
        self._close_outfile()


    def _close_outfile(self):
        if self._outfile_fp is None:
            return
        
        self._outfile_fp.close()


    def add_step(
        self,
        image_file_name: pathlib.Path,
        step_name: str,
        step_index: int,
        scan_count: int,
        timestamp: datetime.datetime
    ):
        if self._mode != "to_file":
            raise Exception(f"add_step() can only be called when the instance is initialized with an 'outfile'.")
        
        self._outfile_csv.writerow([image_file_name, step_name, step_index, scan_count, timestamp])
        self._outfile_fp.flush()
        
    
    def get_data_from_filename(self, file_path: str | pathlib.Path) -> dict | None:
        record = self._records.loc[self._records['Filename'] == str(file_path)]
        if len(record) != 1:
            return None
        
        first_row = record.iloc[0]
        return {
            'Step Index': first_row['Step Index'],
            'Scan Count': first_row['Scan Count'],
            'Timestamp': first_row['Timestamp']
        }


    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        with open(file_path, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t') 
            header = next(csvreader)
            if header[0] != cls.FILE_HEADER:
                raise Exception(f"Invalid protocol execution record")
            
            version = next(csvreader)
            if version[0] != 'Version':
                raise Exception(f"Version key not found")
            
            if int(version[1]) not in (2,):
                raise Exception(f"Unsupported protocol execution record version")
            
            protocol_file_loc_row = next(csvreader)
            if protocol_file_loc_row[0] != "Protocol File":
                raise Exception(f"Protocol file location not found in file")
            
            protocol_file_loc = protocol_file_loc_row[1]
            
            _ = next(csvreader) # Column names

            records = []
            for row in csvreader:
                records.append(
                    {
                        'Filename': row[0],
                        'Step Name': row[1],
                        'Step Index': int(row[2]),
                        'Scan Count': int(row[3]),
                        'Timestamp': datetime.datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S.%f")
                    }
                )

            df = pd.DataFrame(records)

            return ProtocolExecutionRecord(
                records=df,
                protocol_file_loc=protocol_file_loc
            )

            

            

        

    
    