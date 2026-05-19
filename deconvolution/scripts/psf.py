"""Point spread function helpers for the deconvolution prototype.

A point spread function, or PSF, describes how one ideal bright point spreads
out in the microscope image. Deconvolution uses this blur model to estimate
what the sharper original image likely looked like.
"""

from __future__ import annotations

import numpy as np


def gaussian_psf(
    shape: tuple[int, int],
    sigma_px: float,
) -> np.ndarray:
    """Create a normalized 2D Gaussian PSF.

    Plain English:
    - Put a Gaussian bump at the center of a small image.
    - ``sigma_px`` controls how wide the blur is in pixels.
    - Normalize the PSF so all values sum to 1.0; this preserves brightness
      when we later convolve an image with it.
    """
    raise NotImplementedError('Next step: fill in Gaussian PSF generation.')


def airy_disk_psf(
    shape: tuple[int, int],
    radius_px: float,
) -> np.ndarray:
    """Create a normalized Airy-like PSF placeholder.

    Plain English:
    - A real diffraction-limited microscope PSF is closer to an Airy disk than
      a Gaussian.
    - For the first version, this function will provide a simple approximation
      useful for comparing "known PSF" versus "mismatched PSF" behavior.
    """
    raise NotImplementedError('Later step: fill in Airy-like PSF generation.')


def normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Return a floating PSF whose values sum to 1.0.

    Plain English:
    - Deconvolution expects the PSF to represent redistribution of light, not
      creation or loss of light.
    - Dividing by the sum makes the PSF energy-conserving.
    """
    raise NotImplementedError('Next step: fill in PSF normalization.')


def validate_psf(psf: np.ndarray) -> None:
    """Validate that a PSF is usable by deconvolution algorithms.

    Plain English:
    - The PSF must be a 2D numeric array.
    - It cannot contain NaN or infinite values.
    - It cannot be empty or sum to zero.
    """
    raise NotImplementedError('Next step: fill in PSF validation.')
