
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

AUTHORS:
Kevin Peter Hickerson, The Earthineering Company
Anna Iwaniec Hickerson, Keck Graduate Institute
'''

import json
import numpy as np

class LabWare(object):
    """A class that stores and computes actions for objective labware"""

    def __init__(self, *arg):
        self.x = 75
        self.y = 25
        self.z = 1

class Slide(LabWare):
    """A class that stores and computes actions for slides labware"""

    def __init__(self, *arg):
        super(Slide, self).__init__()
        self.covered = True


class WellPlate(LabWare):
    """A class that stores and computes actions for wellplate labware"""

    def __init__(self, config: dict, *arg):
        super(WellPlate, self).__init__()

        self.config = config
        self.ind_list = [] # ordered list of all well indices 
        self.pos_list = [] # ordered list of all well positions


    # set indices based on plate and motion
    def set_indices(self, stitch=1):

        self.ind_list = []
        self.stitch_list = []
        # 'i' represents column index (x-direction)
        # 'j' represents row index (y-direction)
        # (0, 0) on the bottom left looking from above

        # start at the top-left when looking from above
        for j in range(self.config['rows']):            
            for i in range(self.config['columns']):
                if j % 2 == 1:
                    i = self.config['columns'] - i - 1
                self.ind_list.append([i,j])
                
        if stitch > 1:
            for i in range(stitch):
                for j in range(stitch):
                    if j % 2 == 1:
                        i = self.config['columns'] - i - 1
                    self.stitch_list.append([i,j])

    # set positions based on indices
    def set_positions(self):

        self.set_indices(stitch=1)
        self.pos_list = []

        for well in self.ind_list:
            x, y = self.get_well_position(well[0], well[1])
            self.pos_list.append([x, y])

    
    def get_positions_with_labels(self) -> tuple[float,float,str]:
        self.set_positions()
        tmp = []
        for pos in self.pos_list:
            x, y = pos
            label = self.get_well_label(x=x, y=y)
            tmp.append((x, y, label))
        
        return tmp
    

    # Get center position of well on plate in mm given its index (i, j)
    def get_well_position(self, i, j):

        dx = self.config['spacing']['x'] # distance b/w wells x-dir
        ox = self.config['offset']['x']  # offset to first well x-dir
        x = ox+i*dx                     # x position in mm of well
        
        dy = self.config['spacing']['y']    # distance b/w wells y-dir
        oy = self.config['offset']['y']     # offset to top well y-dir
        y = oy+j*dy
            
        return x, y
    
    # Get index of closest well based on plate position (x, y) in mm
    def get_well_index(self, x, y):

        ox = self.config['offset']['x']     # offset to first well x-dir
        dx = self.config['spacing']['x']    # distance b/w wells x-dir
        i = (x-ox)/dx

        dy = self.config['spacing']['y']    # distance b/w wells y-dir
        oy = self.config['offset']['y']     # offset to top well y-dir
        j = (y-oy)/dy
        
        i = round(i)
        j = round(j)
        i = np.clip(i, 0, self.config['columns']-1)
        j = np.clip(j, 0, self.config['rows']-1)
        return i, j


    def get_well_label(self, x, y):
        well_x, well_y = self.get_well_index(x=x, y=y)

        # Handling for labware with more than 26 rows
        letter = ''
        if well_y >= 26:
            letter += 'A'
            well_y -= 26

        letter += chr(ord('A') + well_y)
        return f'{letter}{well_x + 1}'
    

    def get_dimensions(self):
        return self.config['dimensions']
    
    
class PitriDish(LabWare):
    """A class that stores and computes actions for petri dish labware"""

    def __init__(self, *arg):
        super(PitriDish, self).__init__()
        self.diameter = 100
        self.z = 20
