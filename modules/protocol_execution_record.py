
import csv
import datetime
import pathlib

import pandas as pd


class ProtocolExecutionRecord:

    FILE_HEADER = "LumaViewPro Protocol Execution Record"
    CURRENT_VERSION = 1
    COLUMNS = ('Filename', 'Step Index', 'Timestamp')

    def __init__(self, outfile: pathlib.Path | None = None):
        self._outfile = outfile

    
    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        with open(file_path, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t') 
            header = next(csvreader)
            if header[0] != cls.FILE_HEADER:
                raise Exception(f"Invalid protocol execution record")
            
            version = next(csvreader)
            if int(version[0]) not in (1,):
                raise Exception(f"Unsupported protocol execution record version")
            
            _ = next(csvreader) # Column names

            records = []
            for row in csvreader:
                records.append(
                    {
                        'filename': row[0],
                        'step_index': int(row[1]),
                        'timestamp': datetime.datetime(row[2])
                    }
                )

            df = pd.DataFrame(records)

            

            

        

    
    