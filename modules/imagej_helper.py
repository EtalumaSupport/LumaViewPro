
import numpy as np

from lvp_logger import logger


try:
    import imagej
    imagej_imported = True
except ImportError:
    imagej_imported = False



class ImageJHelper:

    def __init__(self):
        if imagej_imported == False:
            logger.error(f"[ImageJ Helper] ImageJ module failed to import, unable to use {self.__class__.__name__}")
            self._ij = None
            return
        
        try:
            # self._ij = imagej.init("2.14.0", add_legacy=False, mode="headless")
            self._ij = imagej.init("sc.fiji:fiji:2.14.0", add_legacy=False, mode="headless")
            logger.info(f"[ImageJ Helper] ImageJ version: {self._ij.getVersion()}")
        except Exception as ex:
            self._ij = None
            logger.error(f"[ImageJ Helper] Unable to initialize ImageJ: {ex}")


    def _log_uninitialized(self):
        logger.error(f"[ImageJ Helper] ImageJ not initialized")


    def test(self):
        if not self._ij:
            self._log_uninitialized()
            return None

        array = np.random.rand(5,4,3)
        dataset = self._ij.py.to_java(array)
        print(f"Dataset shape: {dataset.shape}")

