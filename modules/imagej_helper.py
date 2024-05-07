
import enum
import logging

import numpy as np

from lvp_logger import logger


try:
    import imagej
    import scyjava
    imagej_imported = True
    logging.getLogger('scyjava').setLevel(level=logging.INFO)
    logging.getLogger('imagej').setLevel(level=logging.INFO)
except ImportError:
    imagej_imported = False


class ZProjectMethod(enum.Enum):
    Min                 = "min"
    Max                 = "max"
    Average             = "avg"
    Median              = "median"
    Sum                 = "sum"
    StdDev              = "sd"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.name, cls))


class ImageJHelper:

    def __init__(self):
        if imagej_imported == False:
            logger.error(f"[ImageJ Helper] ImageJ module failed to import, unable to use {self.__class__.__name__}")
            self._ij = None
            return
        
        try:
            self._ij = imagej.init("sc.fiji:fiji:2.14.0", add_legacy=False, mode="headless") # mode="interactive"
            logger.info(f"[ImageJ Helper] ImageJ version: {self._ij.getVersion()}")
        except Exception as ex:
            self._ij = None
            logger.error(f"[ImageJ Helper] Unable to initialize ImageJ: {ex}")


    def _log_uninitialized(self):
        logger.error(f"[ImageJ Helper] ImageJ not initialized")


    def zproject(self, images_data: list[np.ndarray], method: ZProjectMethod) -> np.ndarray:
        if not self._ij:
            self._log_uninitialized()
            return None
        
        if len(images_data) == 0:
            logger.error(f"[ImageJ Helper] zproject -> No images provided")
            return None
        
        orig_dtype = images_data[0].dtype
        images_to_stack = scyjava.jimport("ij.plugin.ImagesToStack")()
        z_projector = scyjava.jimport("ij.plugin.ZProjector")()

        jimages = []
        for image_data in images_data:
            jimage = self._ij.py.to_java(image_data)
            jimp = self._ij.py.to_imageplus(jimage)
            jimages.append(jimp)

        
        jstack = images_to_stack.run(jimages)

        j_z_project_result = z_projector.run(jstack, method.value)
        z_project_result = self._ij.py.from_java(j_z_project_result)

        # Convert back to integer if needed
        z_project_result = z_project_result.round().astype(orig_dtype)

        return z_project_result
