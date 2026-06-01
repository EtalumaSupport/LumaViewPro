# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import enum
import logging

import numpy as np

from lvp_logger import logger


try:
    import imagej.doctor
    import imagej
    import scyjava

    imagej_imported = True
    logging.getLogger('scyjava').setLevel(level=logging.INFO)
    logging.getLogger('imagej').setLevel(level=logging.INFO)
except ImportError:
    imagej_imported = False


def init_ij():
    """Initialize ImageJ and return a helper instance."""
    if not imagej_imported:
        logger.error('[ImageJ Helper] init_ij: pyimagej not importable -- ImageJ unavailable')
        return ImageJHelper()

    import imagej.doctor
    import imagej

    # Logged because this runs on a background worker behind a no-cancel wait
    # popup; on a machine without Java it can churn for a long time, and the
    # operator (and the log) otherwise have no record that we are stuck here.
    logger.info(
        '[ImageJ Helper] init_ij: initializing ImageJ (Fiji 2.14.0). First run '
        'downloads Fiji and requires Java; this can take a while.'
    )
    imagej.doctor.checkup()
    helper = ImageJHelper()
    logger.info(f'[ImageJ Helper] init_ij: done (ImageJ available={helper._ij is not None})')
    return helper


class ZProjectMethod(enum.Enum):
    Min = 'min'
    Max = 'max'
    Average = 'avg'
    Median = 'median'
    Sum = 'sum'
    StdDev = 'sd'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.name, cls))


class ImageJHelper:
    def __init__(self):
        try:
            imagej.doctor.checkup()
        except Exception as ex:
            logger.info(f'[ImageJ Helper] Unable to run ImageJ Doctor Checkup: {ex}')

        self._test_dependencies()

        if not imagej_imported:
            logger.error(
                f'[ImageJ Helper] ImageJ module failed to import, unable to use {self.__class__.__name__}'
            )
            self._ij = None
            return

        try:
            self._ij = imagej.init(
                'sc.fiji:fiji:2.14.0', add_legacy=False, mode='headless'
            )  # mode="interactive"
            logger.info(f'[ImageJ Helper] ImageJ version: {self._ij.getVersion()}')
        except Exception as ex:
            self._ij = None
            logger.error(f'[ImageJ Helper] Unable to initialize ImageJ: {ex}')

    def _test_dependencies(self):
        import importlib

        for pkg in ('imglyb', 'jgo', 'jpype', 'labeling', 'numpy', 'scyjava', 'xarray'):
            try:
                mod = importlib.import_module(pkg)
                logger.info(f'Imported {mod.__name__}')
            except Exception as ex:
                logger.error(f'Unable to import {pkg}: {ex}')

    @property
    def available(self) -> bool:
        """True when ImageJ initialized (Java present and Fiji loaded).

        init_ij always returns a helper, even when Java is absent -- the
        helper just has no live ImageJ gateway. Callers must check this
        before running an operation; a False helper produces only generic
        "Failed to create ..." errors deep in the algorithm otherwise.
        """
        return self._ij is not None

    def _log_uninitialized(self):
        logger.error(f'[ImageJ Helper] ImageJ not initialized')

    def zproject(self, images_data: list[np.ndarray], method: ZProjectMethod) -> np.ndarray:
        if not self._ij:
            self._log_uninitialized()
            return None

        if len(images_data) == 0:
            logger.error(f'[ImageJ Helper] zproject -> No images provided')
            return None

        orig_dtype = images_data[0].dtype
        images_to_stack = scyjava.jimport('ij.plugin.ImagesToStack')()
        z_projector = scyjava.jimport('ij.plugin.ZProjector')()

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
