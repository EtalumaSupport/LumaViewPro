
import importlib.util
import inspect
import logging
import pathlib
import sys

import numpy as np

import image_utils
from modules.image_capture.image_capture_format_base import ImageCaptureFormatBase
from modules.image_capture.image_capture_enums import ImageCaptureConfig, ImageFileFormat, ImageChannelCount, ImagePixelDepth


class ImageCaptureRouter:

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._capture_configs = self._load_supported_image_capture_formats()
        self._capture_config_lookup = self._build_capture_config_lookup()

    
    def _load_supported_image_capture_formats(self) -> dict[str, ImageCaptureConfig]:
        instances = {}
        current_dir = pathlib.Path(__file__).resolve().parent

        for file_path in current_dir.glob("*.py"):
            module_name = file_path.stem

            # Don't import the __init__.py file
            if module_name == "__init__":
                continue

            # Don't import base classes
            if module_name.lower().endswith("base"):
                continue

            try:

                # Dynamically import the modules
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Extract classes from import module
                for name, obj in inspect.getmembers(module, inspect.isclass):

                    # Ensure the class is defined in this module, not an imported dependency
                    if obj.__module__ == module_name:

                        # Only import classes which are subclasses of the ImageCaptureFormatBase
                        if False == issubclass(obj, ImageCaptureFormatBase):
                            continue

                        instances[name] = obj()
                        self._logger.debug(f"  Found class: {name}")

            except Exception as e:
                self._logger.error(f"Error importing {file_path}: {e}")

        return instances
            
    
    def _build_capture_config_lookup(self) -> dict[tuple[ImageCaptureConfig], ImageCaptureFormatBase]:
        config_lookup = {}
        for k, v in self._capture_configs.items():
            for config in v.supported_configs():
                if config in config_lookup:
                    self._logger.warning(f"{config} already in capture config lookup. Skipping from {k}")
                    continue

                config_lookup[config] = v
        
        return config_lookup


    def save(
        self,
        image_data: np.ndarray,
        file_format: ImageFileFormat,
        **kwargs
    ):
        
        if image_data.dtype == np.uint8:
            pixel_depth = ImagePixelDepth.BIT8
        elif image_data.dtype == np.uint16:
            pixel_depth = ImagePixelDepth.BIT12
        else:
            raise NotImplementedError

        if image_utils.is_color_image(image_data):
            channel_count = ImageChannelCount.THREE_CHANNEL
        else:
            channel_count = ImageChannelCount.SINGLE_CHANNEL

        image_capture_config = ImageCaptureConfig(
            pixel_depth=pixel_depth,
            channel_count=channel_count,
            file_format=file_format,
        )

        try:
            image_plugin = self._capture_config_lookup[image_capture_config]
        except:
            self._logger.error(f"Unable to save image with config {image_capture_config}. No plugin found.")

        try:
            image_plugin.save(image_data=image_data, **kwargs)
        except Exception as ex:
            self._logger.error(f"Unable to save image (config {image_capture_config}) with plugin ({image_plugin}): {ex}")

