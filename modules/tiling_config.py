
import itertools
import json


class TilingConfig:

    def __init__(self):

        with open('./data/tiling.json', "r") as fp:
            self._available_configs = json.load(fp)


    def available_configs(self) -> list[str]:
        return list(self._available_configs['data'].keys())
    

    def get_mxn_size(self, config_label: str) -> dict:
        return self._available_configs['data'][config_label]
    
    
    def get_label_from_mxn_size(self, m: int, n: int) -> str | None:
        for config_label, config_data in self._available_configs['data'].items():
            if (config_data['m'] == m) and (config_data['n'] == n):
                return config_label
            
        return None
    

    def default_config(self) -> str:
        return self._available_configs["metadata"]["default"]


    def _calc_range(
        self, 
        config_label: str,
        focal_length: float,
        frame_size: dict[int],
        fill_factor: int
    ) -> dict[dict]:
        tiling_mxn = self.get_mxn_size(config_label)

        magnification = 47.8 / focal_length # Etaluma tube focal length [mm]
                                            # in theory could be different in different scopes
                                            # could be looked up by model number
                                            # although all are currently the same
        pixel_width = 2.0 # [um/pixel] Basler pixel size (could be looked up from Camera class)
        um_per_pixel = pixel_width / magnification

        fov_size_x = um_per_pixel * frame_size['width']
        fov_size_y = um_per_pixel * frame_size['height']       
        x_fov = fill_factor * fov_size_x
        y_fov = fill_factor * fov_size_y
        #x_current = lumaview.scope.get_current_position('X')
        #x_current = np.clip(x_current, 0, 120000) # prevents crosshairs from leaving the stage area
        x_center = 60000 # TODO make center of a well
        #y_current = lumaview.scope.get_current_position('Y')
        #y_current = np.clip(y_current, 0, 80000) # prevents crosshairs from leaving the stage area
        y_center = 40000 # TODO make center of a well
        tiling_min = {
            "x": x_center - tiling_mxn["n"]*x_fov/2,
            "y": y_center - tiling_mxn["m"]*y_fov/2
        }

        tiling_max = {
            "x": x_center + tiling_mxn["n"]*x_fov/2,
            "y": y_center + tiling_mxn["m"]*y_fov/2
        }

        return {
            'mxn': tiling_mxn,
            'min': tiling_min,
            'max': tiling_max,
        }


    def get_tile_centers(
            self,
            config_label: str,
            focal_length: float,
            frame_size: dict[int],
            fill_factor: int
    ) -> dict:
        ranges = self._calc_range(
            config_label=config_label,
            focal_length=focal_length,
            frame_size=frame_size,
            fill_factor=fill_factor
        )

        mxn = ranges['mxn']
        min = ranges['min']
        max = ranges['max']

        tiles = {}
        ax = (max["x"] + min["x"])/2
        ay = (max["y"] + min["y"])/2
        dx = (max["x"] - min["x"])/mxn["n"]
        dy = (max["y"] - min["y"])/mxn["m"]

        PRECISION = 2 # Digits

        for i, j in itertools.product(range(mxn["m"]), range(mxn["n"])):
            
            if (mxn["m"] == 1) and (mxn["n"] == 1):
                # Handle special case where tiling is 1x1 (i.e. no tiling)
                tile_label = ""
            else:
                row_letter = chr(i+ord('A'))
                col_number = j+1
                tile_label = f"{row_letter}{col_number}"

            tiles[tile_label] = {
                "x": round(min["x"] + (j+0.5)*dx - ax, PRECISION),
                "y": round(max["y"] + (i+0.5)*dy - ay, PRECISION)
            }

        return tiles
