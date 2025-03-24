
import json
import pathlib

import mergedeep

from lvp_logger import logger

class MotorConfig:

    def __init__(
        self,
        default_config_file_loc: pathlib.Path,
    ):
        self._name = self.__class__.__name__
        self._config = {}
        self._default_config = self._get_config_from_file(default_config_file_loc)
        self.update(config=self._default_config)


    def _get_config_from_file(self, file_loc: pathlib.Path) -> dict:
        if file_loc is None:
            logger.error(f"[{self._name} ] No default configuration file specified for motorboard")
            return {}
        
        if not file_loc.is_file():
            logger.error(f"[{self._name} ] Motorboard default config file {self.file_loc} not found.")
            return {}
        
        try:
            with open(file=file_loc, mode='r') as fp:
                config = json.load(fp)

        except json.decoder.JSONDecodeError as ex:
            logger.exception(f'[{self._name} ] Unable to parse JSON from {self.file_loc}: {ex}')
            return {}

        return config
    

    def update(self, config: dict):
        merged_config = mergedeep.merge(
            {},
            self._config,
            config,
            strategy=mergedeep.Strategy.REPLACE
        )

        self._config = merged_config


    def _axis_lookup(self, key: str, axis: str, cast_to: type):
        subconfig = self._config.get(key, None)
        if subconfig is None:
            logger.exception(f'[{self._name} ] {key} not found in motor config')
            return cast_to(0)
        
        axis = axis.upper()
        val = subconfig.get(axis, None)
        if val is None:
            logger.exception(f'[{self._name} ] Axis {axis} not in motor config {key}')
            return cast_to(0)
        
        return cast_to(val)
        
        
    def axis_usteps_per_mm_per_obj(self, axis: str) -> int:
        return self._axis_lookup(
            key="Axis Microsteps per mm / Objective",
            axis=axis,
            cast_to=int
        )
    

    def axis_travel_limit_mm(self, axis: str) -> int:
        return self._axis_lookup(
            key="Axis Travel Limit",
            axis=axis,
            cast_to=int
        )
    

    def axis_present(self, axis: str) -> bool:
        return self._axis_lookup(
            key="Axis Present",
            axis=axis,
            cast_to=bool
        )
    

    def axis_antibacklash(self, axis: str) -> int:
        return self._axis_lookup(
            key="Axis antibacklash",
            axis=axis,
            cast_to=int
        )


    def model(self) -> str:
        key = "Microscope"
        return self._config.get(key, "LS850")
    

    def serial_number(self) -> str:
        key = "Serial Number"
        return self._config.get(key, "Unknown")
    

    def led_channel_name(self, channel_index: int):
        key = "LEDChannel"
        result = self._config.get(key, None)

        if result is None:
            logger.exception(f'[{self._name} ] {key} not in motor config')
            return "Unknown"

        # Move the "Ch" from the key "Ch<index>"
        tmp_dict = {int(k[2:]): v for k, v in result.items()}

        val = tmp_dict.get(channel_index, None)

        if val is None:
            logger.exception(f'[{self._name} ] Channel {channel_index} not in motor config {key}')
            return "Unknown"
        
        return val
        
    
    def turret_position_usteps(self, position: int) -> int:
        key = "TurretPosition"
        result = self._config.get(key, None)

        if result is None:
            logger.exception(f'[{self._name} ] {key} not in motor config')
            return 0
        
        # Convert the keys from str to int, and move to 0-based index
        tmp_dict = {int(k)-1: v for k, v in result.items()}

        val = tmp_dict.get(position, None)
        if val is None:
            logger.exception(f'[{self._name} ] Position {position} not in motor config {key}')
            return 0
        
        return val
    

    def lens_focal_length(self) -> float:
        try:
            val = self._config['Optics']['LensFocalLength']
        except:
            logger.exception(f'[{self._name} ] LensFocalLength not in motor config Optics')
            val = 47.8

        return val
    

    def pixel_size(self) -> float:
        try:
            val = self._config['Optics']['PixelSize']
        except:
            logger.exception(f'[{self._name} ] PixelSize not in motor config Optics')
            val = 2.0

        return val


    def image_center(self) -> dict:
        key = "ImageCenter"
        try:
            val = self._config[key]
        except:
            logger.exception(f'[{self._name} ] {key} not in motor config')
            val = {
                'X': 0,
                'Y': 0,
            }

        return val

