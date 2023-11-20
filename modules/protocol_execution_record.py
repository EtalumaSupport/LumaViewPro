
import csv
import datetime
import pathlib

import pandas as pd


class ProtocolExecutionRecord:

    FILE_HEADER = "LumaViewPro Protocol Execution Record"
    CURRENT_VERSION = 1
    COLUMNS = ('Filename', 'Step Name', 'Step Index', 'Timestamp')

    def __init__(
        self,
        outfile: pathlib.Path | None = None,
        records: pd.DataFrame | None = None
    ):
        if (outfile is not None) and (records is not None):
            raise Exception(f"Specify only outfile OR records")
        
        if (outfile is None) and (records is None):
            raise Exception(f"Must specify outfile or records")
        
        if outfile is not None:
            self._mode = "to_file"
            self._outfile = outfile
            self._initialize_outfile(outfile=outfile)
        else:
            self._mode = "from_file"
            self._records = records
            
    
    def _initialize_outfile(self, outfile: pathlib.Path):
        self._outfile_fp = open(outfile, 'w')
        self._outfile_fp.write(f"{self.FILE_HEADER}\n")
        self._outfile_fp.write(f"Version\t{self.CURRENT_VERSION}\n")
        column_line = '\t'.join(self.COLUMNS)
        self._outfile_fp.write(f"{column_line}\n")

    
    def complete(self):
        self._close_outfile()


    def _close_outfile(self):
        self._outfile_fp.close()


    def add_step(
        self,
        image_file_name: pathlib.Path,
        step_name: str,
        step_index: int,
        timestamp: datetime.datetime
    ):
        if self._mode != "to_file":
            raise Exception(f"add_step() can only be called when the instance is initialized with an 'outfile'.")
        
        self._outfile_fp.write(f"{image_file_name}\t{step_name}\t{step_index}\t{timestamp}\n")
        self._outfile_fp.flush()
        
    
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
            
            if int(version[1]) not in (1,):
                raise Exception(f"Unsupported protocol execution record version")
            
            _ = next(csvreader) # Column names

            records = []
            for row in csvreader:
                records.append(
                    {
                        'filename': row[0],
                        'step_name': row[1],
                        'step_index': int(row[2]),
                        'timestamp': datetime.datetime(row[3])
                    }
                )

            df = pd.DataFrame(records)

            return ProtocolExecutionRecord(records=df)

            

            

        

    
    