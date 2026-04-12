# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import itertools
import json
import pathlib

from lvp_logger import logger

import modules.common_utils as common_utils

class TilingConfig:

    DEFAULT_FILL_FACTORS = {
        'position': 1.0 # No overlap needed for position-based tiling
    }

    def __init__(self, tiling_configs_file_loc: pathlib.Path):
        try:
            with open(tiling_configs_file_loc, "r") as fp:
                self._available_configs = json.load(fp)
        except FileNotFoundError:
            logger.error(f'[Tiling    ] tiling.json not found at {tiling_configs_file_loc}')
            raise RuntimeError(
                f"Required file tiling.json not found at {tiling_configs_file_loc}. "
                "Please reinstall or restore from backup."
            )
        except json.JSONDecodeError as e:
            logger.error(f'[Tiling    ] tiling.json is corrupt: {e}')
            raise RuntimeError(
                f"tiling.json is corrupt ({e}). "
                "Please restore from backup or reinstall."
            )

        self._validate_tiling(tiling_configs_file_loc)

    def _validate_tiling(self, filepath):
        """Check tiling.json has required structure."""
        cfg = self._available_configs
        if not isinstance(cfg, dict):
            raise ValueError(f"tiling.json at {filepath}: expected dict, got {type(cfg).__name__}")
        if 'metadata' not in cfg:
            logger.warning(f"[Tiling    ] missing 'metadata' key in {filepath}")
        if 'data' not in cfg:
            raise ValueError(f"tiling.json at {filepath}: missing required 'data' key")
        if not isinstance(cfg['data'], dict):
            raise ValueError(f"tiling.json at {filepath}: 'data' must be a dict")
        for label, entry in cfg['data'].items():
            if not isinstance(entry, dict):
                logger.warning(f"[Tiling    ] '{label}' should be dict in {filepath}")
                continue
            for field in ('m', 'n'):
                if field not in entry:
                    logger.warning(f"[Tiling    ] '{label}' missing '{field}' in {filepath}")
                elif not isinstance(entry[field], int):
                    logger.warning(
                        f"[Tiling    ] '{label}'.'{field}' should be int, "
                        f"got {type(entry[field]).__name__} in {filepath}"
                    )


    def available_configs(self) -> list[str]:
        return list(self._available_configs['data'].keys())
    

    def get_mxn_size(self, config_label: str) -> dict:
        return self._available_configs['data'][config_label]
    
    
    def get_label_from_mxn_size(self, m: int, n: int) -> str | None:
        for config_label, config_data in self._available_configs['data'].items():
            if (config_data['m'] == m) and (config_data['n'] == n):
                return config_label
            
        return None
    

    def determine_tiling_label_from_names(self, names: list):
        label_letters = set()
        label_numbers = set()
        for name in names:
            label = common_utils.get_tile_label_from_name(name=name)
            if label is None:
                continue

            label_letter = label[0]
            label_number = int(label[1:])

            label_letters.add(label_letter)
            label_numbers.add(label_number)

        m = len(label_letters)
        n = len(label_numbers)
        if m != n:
            logger.warning(f"TilingConfig] Tiling configuration found as non-symmetric ({m}x{n}). Protocol tiling label will be innacurate.")
            return None
            # raise Exception(f"Tiling configuration requires equal dimensions, but found {m}x{n}")
        
        return self.get_label_from_mxn_size(m=m, n=n)


    def default_config(self) -> str:
        return self._available_configs["metadata"]["default"]
    

    def no_tiling_label(self) -> str:
        return "1x1"
    

    def _calc_range(
        self, 
        config_label: str,
        focal_length: float,
        frame_size: dict[int],
        fill_factor: int,
        binning_size: int,
    ) -> dict[dict]:
        
        tiling_mxn = self.get_mxn_size(config_label)

        fov_size = common_utils.get_field_of_view(
            focal_length=focal_length,
            frame_size=frame_size,
            binning_size=binning_size,
        )  
        
        x_fov = fill_factor * fov_size['width']
        y_fov = fill_factor * fov_size['height']

        # Stage center derived from motorconfig travel limits
        import modules.app_context as _app_ctx
        ctx = _app_ctx.ctx
        if ctx is not None and ctx.scope is not None:
            x_center = ctx.scope.travel_limit_um('X') / 2
            y_center = ctx.scope.travel_limit_um('Y') / 2
        else:
            x_center = 60000
            y_center = 40000
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
            fill_factor: int,
            binning_size: int,
    ) -> dict:
        ranges = self._calc_range(
            config_label=config_label,
            focal_length=focal_length,
            frame_size=frame_size,
            fill_factor=fill_factor,
            binning_size=binning_size,
        )

        tiling_mxn = ranges['mxn']
        tiling_min = ranges['min']
        tiling_max = ranges['max']

        tiles = {}
        ax = (tiling_max["x"] + tiling_min["x"])/2
        ay = (tiling_max["y"] + tiling_min["y"])/2
        dx = (tiling_max["x"] - tiling_min["x"])/tiling_mxn["n"]
        dy = (tiling_max["y"] - tiling_min["y"])/tiling_mxn["m"]

        PRECISION = 2 # Digits

        for i, j in itertools.product(range(tiling_mxn["m"]), range(tiling_mxn["n"])):
            
            if (tiling_mxn["m"] == 1) and (tiling_mxn["n"] == 1):
                # Handle special case where tiling is 1x1 (i.e. no tiling)
                tile_label = ""
            else:
                row_letter = chr(i+ord('A'))
                col_number = j+1
                tile_label = f"{row_letter}{col_number}"

            tiles[tile_label] = {
                "x": round(tiling_min["x"] + (j+0.5)*dx - ax, PRECISION),
                "y": round(tiling_min["y"] + (i+0.5)*dy - ay, PRECISION)
            }

        return tiles
