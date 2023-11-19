

import csv
import datetime
import numpy as np
import os
import pathlib

import modules.tiling_config as tiling_config
import modules.protocol_step as protocol_step


class Protocol:

    PROTOCOL_FILE_HEADER = "LumaViewPro Protocol"

    def __init__(self, config=None):

        if config is None:
            self._config = {}
        else:
            self._config = config

    
    @staticmethod
    def _build_z_height_map(values) -> dict:
            z_height_map = {}

            z_heights = sorted(set(values.astype('float').tolist()))
            if len(z_heights) >= 2:
                z_height_map = {z_height: idx for idx, z_height in enumerate(z_heights)}

            return z_height_map
    
    
    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        
        config = {}
        print(os.getcwd())
        with open(file_path, 'r') as fp:
            csvreader = csv.reader(fp, delimiter='\t')
            verify = next(csvreader)
            if not (verify[0] == cls.PROTOCOL_FILE_HEADER):
                raise Exception(f"Not a valid LumaViewPro Protocol")
            
            period = next(csvreader)        
            config['period'] = datetime.timedelta(minutes=float(period[1]))
            
            duration = next(csvreader)
            config['duration'] = datetime.timedelta(hours=float(duration[1]))

            labware = next(csvreader)
            config['labware'] = labware[1]

            # orig_labware = labware
            # labware_valid, labware = self._validate_labware(labware=orig_labware)
            # if not labware_valid:
                # raise Exception(f"Invalid labware: {labware}")
                # logger.error(f'[LVP Main  ] ProtocolSettings.load_protocol() -> Invalid labware in protocol: {orig_labware}, setting to {labware}')

            _ = next(csvreader) # Skip the column header line

            step_names = list()
            step_values = np.empty((0,10), float)

            for row in csvreader:
                steps.append(
                    protocol_step.ProtocolStep(
                        name=row[0],
                        x=row[1],
                        y=row[2],
                        z=row[3],
                        auto_focus=row[4],
                        channel=row[5],
                        false_color=row[6],
                        illumination=row[7],
                        gain=row[8]
                        auto_gain=row[9]
                        exposure=row[10]
                    )
                )
                # step_names.append(row[0])
                # step_values = np.append(step_values, np.array([row[1:]]), axis=0)

        config['step_names'] = step_names
        config['step_values'] = step_values

        # Index and build a map of Z-heights. Indicies will be used in step/file naming
        # Only build the height map if we have at least 2 heights in the protocol.
        # Otherwise, we don't want "_Z<slice>" added to the name
        config['z_height_map'] = cls._build_z_height_map(values=step_values[:,2])

        # Extract tiling config from step names 
        tc = tiling_config.TilingConfig()
        config['tiling_config_label'] = tc.determine_tiling_label_from_names(
             names=step_names
        )

        return Protocol(
            config=config
        )


if __name__ == "__main__":
    protocol = Protocol.from_file(file_path=pathlib.Path("modules/protocol_test6.tsv"))
    print("Done")
