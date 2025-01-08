import pathlib

import cv2
import numpy as np
import tifffile as tf

import modules.common_utils as common_utils
import image_utils
import datetime
from fractions import Fraction

from lvp_logger import logger, version

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


def add_false_color(array, color):
    src_dtype = array.dtype
    if (not image_utils.is_color_image(array)) and (color in common_utils.get_fluorescence_layers()):
        img = np.zeros((array.shape[0], array.shape[1], 3), dtype=src_dtype)
        if color == 'Blue':
            img[:,:,0] = array
        elif color == 'Green':
            img[:,:,1] = array
        elif color == 'Red':
            img[:,:,2] = array
    else:
        img = array

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


def convert_12bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image // 16).astype(np.uint8)


def convert_12bit_to_16bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image * 16)


def convert_16bit_to_8bit(image):
    if image.dtype == 'uint8':
        return image
    
    new_image = image.copy()
    return (new_image/256).astype('uint8')


def write_tiff(
        data,
        file_loc: pathlib.Path,
        metadata: dict,
        ome: bool,
        video_frame: bool = False,
        extratags: list = [],
):
    
    # Note: OpenCV and TIFFFILE have the Red/Blue color planes swapped, so need to swap
    # them before writing out to tiff
    use_color = image_utils.is_color_image(data)
    if use_color:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    support_data = generate_tiff_data(data=data, metadata=metadata, ome=ome, video_frame=video_frame)

    if True == ome:
        kwargs = {
            'bigtiff': False
        }
    else:
        kwargs = {}

    with tf.TiffWriter(str(file_loc), **kwargs) as tif:
        if not video_frame:
            tif.write(
                data,
                resolution=support_data['resolution'],
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f"LumaViewPro {version}",
                **support_data['options'],
            )
        else:
            tif.write(
                data,
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=f"LumaViewPro {version}",
                **support_data['options'],
            )
            
    
def generate_tiff_data(data, metadata: dict, ome: bool, video_frame: bool):
    
    dtype = tifffile_dtypes
    
    use_color = image_utils.is_color_image(data)

    if use_color:
        photometric = 'rgb'
        axes = 'YXS'
    else:
        photometric = 'minisblack'
        axes = 'YX'

    """
    To Add:
    ImageNumber
    LensModel

    """
    if True == ome:
        tiff_metadata={
            'axes': axes,
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': metadata['pixel_size_um'],
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': metadata['pixel_size_um'],
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': [metadata['channel']]},
            'Plane': {
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
                'IlluminationUnit': 'mA'
            }
        }
        tiff_extratags = []
        # Metadata seems to be working properly for OME-TIFF's so let's leave it alone for now.

    else:
        if not video_frame:
            """tiff_metadata={
                "CameraMake": metadata['camera_make'],
                "ExposureTime": metadata['exposure_time_ms'],           
                "ISOSpeed": metadata['gain_db'],
                "DateTime": metadata['datetime'],
                "Software": metadata['software'],
                "XPosition": metadata['x_pos'],             
                "YPosition": metadata['y_pos'],
                "SubjectDistance": metadata['z_pos_um'],
                "SubSecTime": metadata['sub_sec_time'],
                "Channel": metadata['channel'],
                "BrightnessValue": metadata['illumination_ma']
            }"""

            tiff_metadata={
            'axes': axes,
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': metadata['pixel_size_um'],
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': metadata['pixel_size_um'],
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': [metadata['channel']]},
            'Plane': {
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
                'IlluminationUnit': 'mA'
            }
        }
            # extratags:
            # Additional tags to write. A list of tuples with 5 items:
            #
            # 0. code (int): Tag Id.
            #
            # 1. dtype (:py:class:`DATATYPE`):
            #    Data type of items in `value`.
            #
            # 2. count (int): Number of data values.
            #    Not used for string or bytes values.
            #
            # 3. value (Sequence[Any]): `count` values compatible with
            #   `dtype`. Bytes must contain count values of dtype packed
            #    as binary data.
            #
            # 4. writeonce (bool): If *True*, write tag to first page
            #    of a series only.
            #
            # Duplicate and select tags in TIFF.TAG_FILTERED are not written
            # if the extratag is specified by integer code.
            #
            # Extratags cannot be used to write IFD type tags.
            #
            # Format: (tag_number, datatype, count, value, write_ifd)
            # For rational number values: (numerator, denominator)
            tiff_extratags = []
            """tiff_extratags = [
            # CameraMake: Tag ID 271, 'ASCII'
            (271, dtype['ASCII'], len(metadata['camera_make']) + 1, metadata['camera_make'], False),
            

            # ExposureTime: Tag ID 33434, 'RATIONAL'
            (33434, dtype['RATIONAL'], 1, ms_exposure_to_rational(metadata['exposure_time_ms']), False),

        
            # ISOSpeed: Tag ID 34867, 'double'
            # Using in place of GainControl (Improper use of GainControl)
            (34867, dtype['DOUBLE'], 1, metadata['gain_db'], False),
            
            # DateTime: Tag ID 306, 'ASCII'
            (306, dtype['ASCII'], len(metadata['datetime']) + 1, metadata['datetime'], False),

            # SubjectDistance: Tag ID 37386, 'RATIONAL'
            (37386, dtype['RATIONAL'], 1, subject_dist_to_rational(metadata['z_pos_um']), False),

            # SubSecTime: Tag ID 37520, 'ASCII'
            (37520, dtype['ASCII'], len(metadata['sub_sec_time']) + 1, metadata['sub_sec_time'], False),

            # Channel: Tag ID 65001, 'ASCII'  **CUSTOM**
            (65001, dtype['ASCII'], len(metadata['channel']) + 1, metadata['channel'], False),

            # BrightnessValue: Tag ID 37393, 'SRATIONAL'
            (37393, dtype['SRATIONAL'], 1, (metadata['illumination_ma'], 1), False)]"""

            """
            # XPosition: Tag ID 65001, 'RATIONAL' 
            # Need to double check units
            (286, dtype['RATIONAL'], 1, (metadata['x_pos'], 1), False),]

            
            # YPosition: Custom Tag ID 65002, 'RATIONAL'
            # Need to double check units
            (287, dtype['RATIONAL'], 1, (metadata['y_pos'], 1), False),

            
            """
            


        else:
            # Video Frame
            # Add further parameters in the future if testing goes well

            """date_time_data = metadata['timestamp'].strftime("%Y:%m:%d %H:%M:%S")
            sub_sec_time = f"{metadata['timestamp'].microsecond // 1000:03d}"

            tiff_extratags = [
            # Tag 306: DateTime (ASCII)
            (306, 'ascii', len(date_time_data) + 1, date_time_data, False),

            # Tag 37520: SubSecTime (ASCII)
            (37520, 'ascii', len(sub_sec_time) + 1, sub_sec_time, False),

            # Tag 37393: ImageNumber (long)
            (37393, 'long', 1, metadata['frame_num'], False)

            ]"""
            tiff_extratags = []
            tiff_metadata = metadata



    options=dict(
        photometric=photometric,
        tile=(128, 128),
        compression='lzw',
        resolutionunit='CENTIMETER',
        maxworkers=2
    )

    if not video_frame:
        resolution = (1e4 / metadata['pixel_size_um'], 1e4 / metadata['pixel_size_um'])

        return {
            'metadata': tiff_metadata,
            'extratags': tiff_extratags,
            'options': options,
            'resolution': resolution,
        }
    
    else:
        return {
            'metadata': tiff_metadata,
            'extratags': tiff_extratags,
            'options': options,
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


def add_scale_bar(
    image,
    objective: dict,
    binning_size: int
):
    height, width = image.shape[0], image.shape[1]

    MIN_IMAGE_WIDTH_PIXELS = 100
    if width < MIN_IMAGE_WIDTH_PIXELS:
        # Don't try to add a scale bar if the image is too small
        return image

    dtype = image.dtype
    is_color = is_color_image(image=image)

    pixel_size_um = common_utils.get_pixel_size(
        focal_length=objective['focal_length'],
        binning_size=binning_size
    )

    # Scale bar should be 1/8 to 1/4 the image length
    scale_bar_length_range_pixels = {
        'min': int(width/8),
        'max': int(width/4),
    }
    scale_bar_length_range_pixels['mid'] = int((scale_bar_length_range_pixels['min'] + scale_bar_length_range_pixels['max']) / 2)

    scale_bar_length_range_um = {k: v*pixel_size_um for k,v in scale_bar_length_range_pixels.items()}

    good_numbers = np.array(
        [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
    )

    # If needed, adjust the good numbers by factors of 10 to keep them 'good'
    if scale_bar_length_range_um['min'] > good_numbers.max():
        while scale_bar_length_range_um['min'] > good_numbers.max():
            good_numbers *= 10
    elif scale_bar_length_range_um['max'] < good_numbers.min():
        while scale_bar_length_range_um['max'] < good_numbers.min():
            good_numbers = (good_numbers / 10)

    # Find the nearest good number to the midpoint target
    good_numbers_diff = np.absolute(good_numbers-scale_bar_length_range_um['mid'])
    good_numbers_index = good_numbers_diff.argmin()
    scale_bar_length_um = good_numbers[good_numbers_index]

    # Convert the calculated value back to pixels
    scale_bar_length_pixels = int(scale_bar_length_um / pixel_size_um)

    scale_bar_thickness_pixels = min(3, int(height/300))
    scale_bar_bottom_offset = int(height/40)
    scale_bar_right_offset = int(width/40)

    if dtype == np.uint8:
        scale_bar_value = 2**8-1
    else: # 12-bit
        scale_bar_value = 2**12-1

    x_end = width - scale_bar_right_offset
    x_start = x_end - scale_bar_length_pixels
    y_start = scale_bar_bottom_offset
    y_end = y_start + scale_bar_thickness_pixels

    if is_color:
        image[y_start:y_end+1,x_start:x_end+1,:] = scale_bar_value
    else:
        image[y_start:y_end+1,x_start:x_end+1] = scale_bar_value

    text_x_pos = x_start
    text_y_pos = y_end + 5
    font_scale = max(0.75, width/2000)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    scale_bar_text = f"{scale_bar_length_um}um, {objective['magnification']}x"

    # Adjust the font scaling until the text string is smaller than the scale bar length
    while True:
        text_size, _ = cv2.getTextSize(
            text=scale_bar_text,
            fontFace=font_face,
            fontScale=font_scale,
            thickness=font_thickness
        )
        text_w, text_h = text_size
        if text_w < scale_bar_length_pixels:
            break

        font_scale *= 0.75

    cv2.putText(
        img=image, 
        text=scale_bar_text,
        org=(text_x_pos, text_y_pos),
        fontFace=font_face,
        fontScale=font_scale,
        color=(scale_bar_value,scale_bar_value,scale_bar_value),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=True
    )

    return image


def add_timestamp(image, timestamp_str: str):
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
