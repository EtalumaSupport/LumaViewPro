# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import datetime
import enum
import pathlib

import cv2
import numpy as np
import tifffile as tf

from modules.common_utils import ColorChannel
import modules.common_utils as common_utils
import modules.image_utils as image_utils

from fractions import Fraction

from lvp_logger import logger, version

# Pre-built lookup tables for bit-depth conversion (built once at import, ~4 KB each)
# Using the same float math as the original per-pixel conversion ensures identical results.
_LUT_12_TO_8 = np.clip(
    np.arange(4096, dtype=np.float32) / 4095 * 255, 0, 255
).astype(np.uint8)

_LUT_16_TO_8 = (np.arange(65536, dtype=np.float64) / 256).astype(np.uint8)

# Conversion to tifffile's desired datatype references
tifffile_dtypes = {
    'BYTE': 1,
    'ASCII': 2,
    'SHORT': 3,
    'LONG': 4,
    'RATIONAL': 5,
    'SBYTE': 6,
    'UNDEFINED': 7,
    'SSHORT': 8,
    'SLONG': 9,
    'SRATIONAL': 10,
    'FLOAT': 11,
    'DOUBLE': 12,
    'SINGLE': 13,
    'QWORD': 16,
    'SQWORD': 17,
}

def is_color_image(image) -> bool:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return True

    return False


def add_false_color(array, color, output=None):
    src_dtype = array.dtype
    if (not image_utils.is_color_image(array)) and (color in (*common_utils.get_fluorescence_layers(), *common_utils.get_luminescence_layers())):
        if output is not None and output.shape == (array.shape[0], array.shape[1], 3) and output.dtype == src_dtype:
            img = output
            img[:] = 0
        else:
            img = np.zeros((array.shape[0], array.shape[1], 3), dtype=src_dtype)
        if color in ('Blue', 'Lumi'):
            img[:,:,0] = array
        elif color == 'Green':
            img[:,:,1] = array
        elif color == 'Red':
            img[:,:,2] = array

        # For HSL colorspace
        # elif color == 'Lumi':
        #     img[:,:,0] = 215 / 2 # Hue (OpenCV uses range of 0-180, so divide by 2)
        #     img[:,:,1] = array # Luminance
        #     img[:,:,2] = 255 # Saturation
    else:
        img = array

    # For HSL colorspace
    # if color == 'Lumi':
    #     img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)

    return img


def image_file_to_image(image_file):
    logger.info(f'[LVP image_utils  ] Loading: {image_file}')
    if not cv2.haveImageReader(image_file):
        logger.error(f'[LVP image_utils  ] - Image not supported by OpenCV')
        return

    num_images = cv2.imcount(image_file)
    logger.info(f'[LVP image_utils  ] - {num_images} images detected')

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error(f'[LVP image_utils  ] - Unable to load file')
        return

    return image


def get_used_color_planes(image) -> list:
    if not is_color_image(image=image):
        return []

    used_color_planes = []
    for color_plane_idx in range(image.shape[2]):
        image_view = image[:,:,color_plane_idx]
        if np.any(image_view):
            used_color_planes.append(color_plane_idx)

    return used_color_planes


def rgb_image_to_gray(image):

    def _is_grayscale(image):
        shape = image.shape
        if (len(shape) <= 2) or (shape[2] == 1):
            return True

        return False

    def _values_in_one_plane(image):
        used_color_planes = get_used_color_planes(image=image)

        if len(used_color_planes) <= 1:
            return True
        else:
            return False

    if _is_grayscale(image=image):
        return image

    if _values_in_one_plane(image=image):
        return np.amax(image, axis=2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def encode_image(image: np.ndarray, fmt: str = 'png', jpeg_quality: int = 80) -> bytes:
    """Encode a numpy image array to binary image data.

    Args:
        image: 2D (grayscale) or 3D (color) numpy array.
        fmt: Output format — 'png', 'jpeg', or 'tiff'.
        jpeg_quality: JPEG quality (1-100), only used for JPEG format.

    Returns:
        bytes: Encoded image data.

    Raises:
        ValueError: If format is unsupported or encoding fails.
    """
    fmt = fmt.lower()
    if fmt in ('jpg', 'jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ext = '.jpg'
    elif fmt == 'png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # fast compression
        ext = '.png'
    elif fmt in ('tiff', 'tif'):
        params = []
        ext = '.tiff'
    else:
        raise ValueError(f"Unsupported image format: {fmt}")

    success, buf = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError(f"Failed to encode image as {fmt}")
    return buf.tobytes()


def convert_12bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image

    return _LUT_12_TO_8[image]


def convert_12bit_to_16bit(image):
    if image.dtype == 'uint8':
        return image

    new_image = image.copy()
    new_image *= 16
    return new_image


def convert_16bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image

    return _LUT_16_TO_8[image]

@enum.unique
class LvpColormap(enum.Enum):
    GRAY = 'gray'
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'


def color_channel_to_colormap_type(color_channel: str | ColorChannel) -> LvpColormap:
    if isinstance(color_channel, str):
        color_channel = ColorChannel[color_channel]

    lut = {
        ColorChannel.Lumi: LvpColormap.BLUE,
        ColorChannel.Blue: LvpColormap.BLUE,
        ColorChannel.Green: LvpColormap.GREEN,
        ColorChannel.Red: LvpColormap.RED,
        ColorChannel.BF: LvpColormap.GRAY,
        ColorChannel.PC: LvpColormap.GRAY,
        ColorChannel.DF: LvpColormap.GRAY,
    }

    return lut[color_channel]


def get_tiff_colormap(colormap: LvpColormap, dtype):
    if dtype in ('uint8', np.uint8):
        dtype = np.uint8
        step_size = 2**0
    elif dtype in ('uint16', np.uint16):
        dtype = np.uint16
        step_size = 2**8
    else:
        raise NotImplementedError(f"Unsupported dtype for colormap: {dtype}")

    max_value = np.iinfo(dtype).max + 1

    if colormap == LvpColormap.GRAY:
        cmap_array = np.tile(np.arange(0, max_value, step_size, dtype=dtype), (3, 1))
    else:
        cmap_array = np.zeros((3,2**8), dtype=dtype)
        if colormap == LvpColormap.RED:
            cmap_array[0] = np.arange(0, max_value, step_size, dtype=dtype)
        elif colormap == LvpColormap.GREEN:
            cmap_array[1] = np.arange(0, max_value, step_size, dtype=dtype)
        elif colormap == LvpColormap.BLUE:
            cmap_array[2] = np.arange(0, max_value, step_size, dtype=dtype)
        else:
            raise NotImplementedError(f"Unsupported colormap: {colormap}")

    return cmap_array


def write_tiff(
        data,
        file_loc: pathlib.Path,
        metadata: dict,
        ome: bool,
        color: str,
        video_frame: bool = False,
        extratags: list = None,
):
    if extratags is None:
        extratags = []

    kwargs = {}
    # Enable BigTIFF for datasets >3.8 GB to prevent silent corruption at 4 GB boundary
    data_size_bytes = data.nbytes
    use_bigtiff = data_size_bytes > 3.8 * 1024 * 1024 * 1024
    if ome:
        kwargs = {
            'bigtiff': use_bigtiff,
        }
    elif not video_frame:
        if is_color_image(data):
            # For now, prevent 16-bit color images from being converted to ImageJ type
            # such as composite (or bullseye). Could allow this once proper support is added.
            pass
        elif data.dtype == np.uint16:
            kwargs['imagej'] = True

    def _validate_type() -> str:
        type_count = 0
        image_type = None

        if ome:
            type_count += 1
            image_type = 'ome'

        if kwargs.get('imagej', False):
            type_count += 1
            image_type = 'imagej'

        if video_frame:
            type_count += 1
            image_type = 'video_frame'

        if type_count > 1:
            raise ValueError("Tiff must only be one type at most (OME, ImageJ, or Video Frame)")

        return image_type

    image_type = _validate_type()

    support_data = generate_tiff_data(
        data=data,
        metadata=metadata,
        image_type=image_type,
        color=color
    )

    with tf.TiffWriter(str(file_loc), **kwargs) as tif:
        if image_type == 'video_frame':
            tif.write(
                data,
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f"LumaViewPro {version}",
                **support_data['options'],
            )

        elif (image_type is None) and is_color_image(image=data):
            # Handles case where an actual color image is provided (such as the bullseye in engineering mode)
            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f"LumaViewPro {version}",
                **support_data['options'],
            )

        elif (image_type == 'ome') and is_color_image(image=data):

            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f"LumaViewPro {version}",
                **support_data['options'],
            )

        else:

            colormap_type = color_channel_to_colormap_type(color_channel=color)
            colormap_array = get_tiff_colormap(
                colormap=colormap_type,
                dtype=data.dtype,
            )
            # Ref: https://forum.image.sc/t/saving-tiff-stack-with-a-colormap-with-tifffile-library/101788
            if data.dtype == np.uint16:
                support_data['extratags'].append((320, tifffile_dtypes['SHORT'], 0, colormap_array.tobytes(), True))
                colormap_array = None
            elif (data.dtype == np.uint8) and (colormap_type == LvpColormap.GRAY):
                # Note: tifffile doesn't support colormaps with 8-bit image and photometric is 'minisblack', so just disable
                colormap_array = None

            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f"LumaViewPro {version}",
                colormap=colormap_array,
                extratags=support_data['extratags'],
                **support_data['options'],
            )


def generate_tiff_data(data, metadata: dict, image_type: str, color: str,):

    dtype = tifffile_dtypes
    axes = 'YX'

    modality = ''
    if is_color_image(data):
        photometric = tf.PHOTOMETRIC.RGB
        modality = 'RGB'
    elif color in ('BF', 'PC', 'DF'):
        photometric = tf.PHOTOMETRIC.MINISBLACK
        modality = color
    elif color in ('Red', 'Green', 'Blue', 'Lumi'):
        photometric = tf.PHOTOMETRIC.PALETTE
        modality = 'MIF'
    else:
        raise ValueError(f"Unexpected color value ({color}) for tiff data generation")

    # Video frames pass through metadata as-is with no structured fields
    if image_type == 'video_frame':
        options = dict(
            photometric=photometric,
            compression='lzw',
            resolutionunit='CENTIMETER',
            maxworkers=2,
        )
        if data.dtype == np.uint8:
            options['tile'] = (128, 128)
        return {
            'metadata': metadata,
            'extratags': [],
            'options': options,
        }

    # Shared plane metadata for all structured image types
    plane = {
        'PositionX': metadata['plate_pos_mm']['x'],
        'PositionY': metadata['plate_pos_mm']['y'],
        'PositionZ': metadata['z_pos_um'],
        'PositionXUnit': 'mm',
        'PositionYUnit': 'mm',
        'PositionZUnit': 'um',
        'Objective': metadata['objective'],
        'ExposureTime': metadata['exposure_time_ms'],
        'ExposureTimeUnit': 'ms',
        'Gain': metadata['gain_db'],
        'GainUnit': 'dB',
        'Illumination': metadata['illumination_ma'],
        'IlluminationUnit': 'mA',
    }

    # Base metadata shared by all structured types
    tiff_metadata = {
        'axes': axes,
        'SignificantBits': data.itemsize * 8,
        'PhysicalSizeX': metadata['pixel_size_um'],
        'PhysicalSizeXUnit': 'um',
        'PhysicalSizeY': metadata['pixel_size_um'],
        'PhysicalSizeYUnit': 'um',
        'Channel': {'Name': [metadata['channel']]},
        'Plane': plane,
    }

    # ImageJ adds unit, channel modality, and document block
    if image_type == 'imagej':
        tiff_metadata['unit'] = 'um'
        tiff_metadata['Channel']['Modality'] = [modality]
        tiff_metadata['Document'] = {
            'Manufacturer': metadata.get('camera_make', ''),
            'Device': metadata.get('microscope', ''),
            'WellLabel': metadata.get('well_label', ''),
            'WellSite': metadata.get('well_site', ''),
        }
        options = dict(
            photometric=photometric,
            compression='deflate',
            maxworkers=1,
        )
        # Resolution for ImageJ types is in pixels/pixel
        resolution = (1. / metadata['pixel_size_um'], 1. / metadata['pixel_size_um'])
    else:
        # ome and default use same options
        options = dict(
            photometric=photometric,
            compression='lzw',
            resolutionunit='CENTIMETER',
            maxworkers=2,
        )
        resolution = (1e4 / metadata['pixel_size_um'], 1e4 / metadata['pixel_size_um'])

    # Tile setting: 8-bit images use tiles for ImageJ colormap compatibility
    if data.dtype == np.uint8:
        options['tile'] = (128, 128)

    return {
        'metadata': tiff_metadata,
        'extratags': [],
        'options': options,
        'resolution': resolution,
    }


def ms_exposure_to_rational(ms_exposure):
    exposure_seconds = ms_exposure / 1000
    fraction = Fraction(exposure_seconds).limit_denominator(1_000_000)
    # Metadata uses rational number of seconds
    return fraction.numerator, fraction.denominator

def subject_dist_to_rational(distance):
    distance_meters = distance / 1_000_000  # Convert um to m
    fraction = Fraction(distance_meters).limit_denominator(1_000_000)
    return fraction.numerator, fraction.denominator


_scale_bar_cache = {}


def _compute_scale_bar_overlay(height, width, dtype, is_color, objective, binning_size, color):
    """Pre-render scale bar overlay and mask. Returns (overlay, mask, cache_key)."""
    pixel_size_um = common_utils.get_pixel_size(
        focal_length=objective['focal_length'],
        binning_size=binning_size
    )

    # Scale bar should be 1/8 to 1/4 the image length
    min_px = int(width / 8)
    max_px = int(width / 4)
    mid_px = (min_px + max_px) // 2

    mid_um = mid_px * pixel_size_um
    min_um = min_px * pixel_size_um
    max_um = max_px * pixel_size_um

    good_numbers = np.array(
        [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000],
        dtype=float,
    )

    if min_um > good_numbers.max():
        while min_um > good_numbers.max():
            good_numbers *= 10
    elif max_um < good_numbers.min():
        while max_um < good_numbers.min():
            good_numbers = good_numbers / 10

    good_numbers_index = np.absolute(good_numbers - mid_um).argmin()
    scale_bar_length_um = good_numbers[good_numbers_index]
    scale_bar_length_pixels = int(scale_bar_length_um / pixel_size_um)

    scale_bar_thickness_pixels = min(3, int(height / 300))
    scale_bar_bottom_offset = int(height / 40)
    scale_bar_right_offset = int(width / 40)

    transmitted_channels = ('BF', 'PC', 'DF')
    if color in transmitted_channels:
        scale_bar_value = 0
    elif dtype == np.uint8:
        scale_bar_value = 255
    else:
        scale_bar_value = 4095

    x_end = width - scale_bar_right_offset
    x_start = x_end - scale_bar_length_pixels
    y_start = scale_bar_bottom_offset
    y_end = y_start + scale_bar_thickness_pixels

    # Render onto a blank canvas
    if is_color:
        canvas = np.zeros((height, width, 3), dtype=dtype)
        canvas[y_start:y_end+1, x_start:x_end+1, :] = scale_bar_value
    else:
        canvas = np.zeros((height, width), dtype=dtype)
        canvas[y_start:y_end+1, x_start:x_end+1] = scale_bar_value

    text_x_pos = x_start
    text_y_pos = y_end + 5
    font_scale = max(0.75, width / 2000)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    scale_bar_text = f"{scale_bar_length_um}um, {objective['magnification']}x"

    while True:
        text_size, _ = cv2.getTextSize(
            text=scale_bar_text,
            fontFace=font_face,
            fontScale=font_scale,
            thickness=font_thickness
        )
        if text_size[0] < scale_bar_length_pixels:
            break
        font_scale *= 0.75

    cv2.putText(
        img=canvas,
        text=scale_bar_text,
        org=(text_x_pos, text_y_pos),
        fontFace=font_face,
        fontScale=font_scale,
        color=(scale_bar_value, scale_bar_value, scale_bar_value),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=True
    )

    # Build boolean mask of non-zero pixels
    if is_color:
        mask = np.any(canvas != 0, axis=2)
    else:
        mask = canvas != 0

    # For black scale bars (transmitted), the overlay is zeros and mask marks where to write
    # We need to handle this differently: store the value and use it during apply
    return canvas, mask, scale_bar_value


def add_scale_bar(
    image,
    objective: dict,
    binning_size: int,
    color: str = None,
):
    global _scale_bar_cache

    height, width = image.shape[0], image.shape[1]

    MIN_IMAGE_WIDTH_PIXELS = 100
    if width < MIN_IMAGE_WIDTH_PIXELS:
        return image

    dtype = image.dtype
    is_color = is_color_image(image=image)

    cache_key = (height, width, dtype, is_color, objective['focal_length'], objective['magnification'], binning_size, color)

    if _scale_bar_cache.get('key') != cache_key:
        overlay, mask, value = _compute_scale_bar_overlay(
            height, width, dtype, is_color, objective, binning_size, color
        )
        _scale_bar_cache = {'key': cache_key, 'overlay': overlay, 'mask': mask, 'value': value}

    cached = _scale_bar_cache
    mask = cached['mask']

    if cached['value'] == 0:
        # Black scale bar for transmitted channels — set masked pixels to 0
        if is_color:
            image[mask] = 0
        else:
            image[mask] = 0
    else:
        # White scale bar — apply overlay values
        image[mask] = cached['overlay'][mask]

    return image


def add_timestamp(image, timestamp_str: str, in_place: bool = True):
    """Draw a timestamp on the image.

    Args:
        image: Input image array (modified in place by default).
        timestamp_str: Text to draw.
        in_place: If True, modify image directly (no copy). If False,
                  work on a copy and return it. Default True to avoid
                  allocating a full-frame copy.

    Returns:
        The image with timestamp drawn.
    """
    height, width = image.shape[0], image.shape[1]

    dtype = image.dtype

    text_color_bg = (0,0,0)
    font_scale = max(0.75, width/2000)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    text_size, _ = cv2.getTextSize(
        text=timestamp_str,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=font_thickness
    )
    text_w, text_h = text_size

    bottom_offset = int(height/40)
    left_offset = int(width/40)

    top_offset = height - bottom_offset

    if dtype == np.uint8:
        text_intensity = 2**8-1
    else: # 16-bit
        text_intensity = 2**16-1

    if not in_place:
        image = image.copy()
    cv2.rectangle(
        image,
        (left_offset, top_offset),
        (left_offset+text_w, top_offset+text_h),
        text_color_bg,
        -1
    )

    cv2.putText(
        img=image,
        text=f"{timestamp_str}",
        org=(left_offset, int(top_offset + text_h + font_scale - 1)),
        fontFace=font_face,
        fontScale=font_scale,
        color=(text_intensity,text_intensity,text_intensity),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=False
    )

    return image
