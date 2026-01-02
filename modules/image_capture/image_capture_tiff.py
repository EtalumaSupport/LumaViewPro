
import pathlib

import numpy as np
import tifffile as tf

import image_utils
from modules.image_capture.image_capture_enums import *
from modules.image_capture.image_capture_format_base import ImageCaptureFormatBase

class ImageCapture_Tiff(ImageCaptureFormatBase):
    
    def __init__(self):
        self._name = self.__class__.__name__
        super().__init__(self._name)


    @staticmethod
    def supported_configs() -> tuple[ImageCaptureConfig]:
        return (
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.SINGLE_CHANNEL,
                file_format=ImageFileFormat.TIFF,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.SINGLE_CHANNEL,
                file_format=ImageFileFormat.TIFF,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.TIFF,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.TIFF,
            ),
        )


    def _generate_support_data(
        self,
        image_data,
        metadata: dict,
        color_channel: str,
    ):
        axes = self._lookup_axes(image_data=image_data)

        photometric, modality = self._lookup_photometric_and_modality(
            image_data=image_data,
            color_channel=color_channel,
        )

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
            'SignificantBits': image_data.itemsize*8,
            'PhysicalSizeX': metadata['pixel_size_um'],
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeY': metadata['pixel_size_um'],
            'PhysicalSizeYUnit': 'um',
            'Channel': {
                'Name': [metadata['channel']],
                'Modality': [modality],
            },
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
            },
            'Document': {
                'Manufacturer': metadata.get('camera_make', ''),
                'Device': metadata.get('microscope', ''),
                'WellLabel': metadata.get('well_label', ''),
                'WellSite': metadata.get('well_site', ''), # TODO: provide well site, id of image within the well, ex: 3
            },
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
        # 2025-10-03:
        # - Keeping tile seems to break ImageJ compatibility with tiff colormaps in 16-bit images
        # - Removing tile seems to break ImageJ compatibility with tiff colormaps in 8-bit images
        options=dict(
            photometric=photometric,
            compression='deflate',
            resolutionunit=tf.RESUNIT.MICROMETER,
            maxworkers=1,
        )

        if image_data.dtype == np.uint8:
            options['tile'] = (128, 128)

        resolution = (1 / metadata['pixel_size_um'], 1 / metadata['pixel_size_um'])

        return {
            'metadata': tiff_metadata,
            'extratags': tiff_extratags,
            'options': options,
            'resolution': resolution,
        }
    

    def save(
        self,
        image_data: np.ndarray,
        file_loc: pathlib.Path,
        metadata: dict,
        color_channel: str,
    ):
        kwargs = {}
        if image_utils.is_color_image(image_data):
            # For now, prevent 16-bit color images from being converted to ImageJ type
            # such as composite (or bullseye). Could allow this once proper support is added.
            pass
        elif image_data.dtype == np.uint16:
            kwargs['imagej'] = True

        support_data = self._generate_support_data(
            image_data=image_data,
            metadata=metadata,
            color_channel=color_channel,
        )

        with tf.TiffWriter(str(file_loc), **kwargs) as tif:
            if True == image_utils.is_color_image(image=image_data):
                # Handles case where an actual color image is provided (such as the bullseye in engineering mode)
                tif.write(
                    image_data,
                    resolution=support_data['resolution'],
                    metadata=support_data['metadata'],
                    datetime=metadata['datetime'],
                    software=metadata['software'],
                    **support_data['options'],
                )
            
            else:

                colormap_type = image_utils.color_channel_to_colormap_type(color_channel=color_channel)
                colormap_array = image_utils.get_tiff_colormap(
                    colormap=colormap_type,
                    dtype=image_data.dtype,
                )
                # Ref: https://forum.image.sc/t/saving-tiff-stack-with-a-colormap-with-tifffile-library/101788
                if image_data.dtype == np.uint16:
                    support_data['extratags'].append((320, image_utils.tifffile_dtypes['SHORT'], 0, colormap_array.tobytes(), True))            
                    colormap_array = None
                elif (image_data.dtype == np.uint8) and (colormap_type == image_utils.LvpColormap.GRAY):
                    # Note: tifffile doesn't support colormaps with 8-bit image and photometric is 'minisblack', so just disable
                    colormap_array = None

                tif.write(
                    image_data,
                    resolution=support_data['resolution'],
                    metadata=support_data['metadata'],
                    datetime=metadata['datetime'],
                    software=metadata['software'],
                    colormap=colormap_array,
                    extratags=support_data['extratags'],
                    **support_data['options'],
                )
