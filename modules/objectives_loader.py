# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import json
import logging
import pathlib

import pandas as pd

from modules.path_utils import resolve_data_file

logger = logging.getLogger('LVP.modules.objectives_loader')


class ObjectiveLoader:

    def __init__(self, *arg, source_path: str | pathlib.Path | None = None):

        with open(resolve_data_file("objectives.json", source_path=source_path), "r") as read_file:
            self._objectives = json.load(read_file)

        self._generate_short_names()
        self._objectives_df = pd.DataFrame.from_dict(self._objectives, orient='index')
        

    def _create_short_name_from_objective_id(self, objective_id: str) -> str:


        tmp = objective_id.replace('w/o', 'No')
        tmp = tmp.replace('W/o', 'No')
        tmp = tmp.replace('W/O', 'No')

        tmp = tmp.replace('w/', '')
        tmp = tmp.replace('W/', '')

        # Remove illegal path characters
        tmp = tmp.replace('/', '')
        tmp = tmp.replace('\\', '')
        tmp = tmp.replace('-', '')
        tmp = tmp.replace('_', '')
        
        # Split on whitespace
        tmp = tmp.split(' ')

        # Capitalize the first letter of each word
        tmp = [v.capitalize() for v in tmp]

        # Rejoin into single key
        tmp = ''.join(tmp)

        return tmp
    

    def _generate_short_names(self):
        # Generate short name to be used for protocol step names
        for objective_key, objective_info in self._objectives.items():
            if 'short_name' not in objective_info:
                short_name = self._create_short_name_from_objective_id(objective_id=objective_key)
                self._objectives[objective_key]['short_name'] = short_name

        # Confirm there are no collisions
        short_names = [v['short_name'] for v in self._objectives.values()]
        short_names_set = set(short_names)
        if len(short_names_set) < len(short_names):
            raise Exception(f"Duplicate short names for objectives were generated")


    def find_objective_id_from_short_name(self, short_name: str) -> str | None:
        for k, v in self._objectives.items():
            if v['short_name'] == short_name:
                return k
            
        return None
    

    def get_objective_info(
        self,
        objective_id: str | None = None,
        short_name: str | None = None,
    ) -> dict:
        
        if ((objective_id is None) and (short_name is None)) or \
           ((objective_id is not None) and (short_name is not None)):
            raise Exception(f"Must supply objective ID or short name, but not both")
        
        if short_name is not None:
            objective_id = self.find_objective_id_from_short_name(short_name=short_name)             
            
            if objective_id not in self._objectives:
                raise Exception(f"No objective found with short name {short_name}")
            
        try:
            objective_info = None
            if objective_id in self._objectives:
                objective_info = self._objectives[objective_id]
            else:
                logger.warning(f"Exact match for objective ID {objective_id} not found, attmempting to use closest match")
                for key in self._objectives:
                    if key.startswith(objective_id):
                        objective_info = self._objectives[key]
                        break

                if objective_info is None:
                    logger.error(f"No close match found for objective ID {objective_id}")
                    return None

        except Exception:
            raise Exception(f"Unable to retrieve information for objective {objective_id}")
        
        return objective_info
    

    def get_objectives_list(self) -> list:
        return list(self._objectives.keys())
    

    def get_objectives_dataframe(self) -> pd.DataFrame:
        return self._objectives_df
    
