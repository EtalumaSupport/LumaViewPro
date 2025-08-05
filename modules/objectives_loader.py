
'''
MIT License

Copyright (c) 2024 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.
'''

import json

import pandas as pd


class ObjectiveLoader:

    def __init__(self, *arg):

        with open('./data/objectives.json', "r") as read_file:
            self._objectives = json.load(read_file)

        self._generate_short_names()
        self._objectives_df = pd.DataFrame.from_dict(self._objectives, orient='index')
        

    def _create_short_name_from_objective_id(self, objective_id: str) -> str:

        tmp = objective_id.replace('w/', '')
        tmp = objective_id.replace('W/', '')
        tmp = objective_id.replace('W/', '')
        tmp = objective_id.replace('w/o', 'No')
        tmp = objective_id.replace('W/o', 'No')
        tmp = objective_id.replace('W/O', 'No')

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
            objective_info = self._objectives[objective_id]
        except:
            raise Exception(f"Unable to retrieve information for objective {objective_id}")
        
        return objective_info
    

    def get_objectives_list(self) -> list:
        return list(self._objectives.keys())
    

    def get_objectives_dataframe(self) -> pd.DataFrame:
        return self._objectives_df
    