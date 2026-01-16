
import pathlib

import numpy as np
import tifffile as tf

import image_utils
from modules.image_capture.image_capture_enums import *
from modules.image_capture.image_capture_format_base import ImageCaptureFormatBase

class ImageCapture_Video(ImageCaptureFormatBase):
    
    def __init__(self):
        self._name = self.__class__.__name__
        super().__init__(self._name)


    @staticmethod
    def supported_configs() -> tuple[ImageCaptureConfig]:
        return (
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.SINGLE_CHANNEL,
                file_format=ImageFileFormat.VIDEO,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.SINGLE_CHANNEL,
                file_format=ImageFileFormat.VIDEO,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.VIDEO,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.VIDEO,
            ),
        )
    

    def _generate_support_data(
        self,
        image_data,
        metadata: dict,
        color_channel: str,
    ):
        photometric, modality = self._lookup_photometric_and_modality(
            image_data=image_data,
            color_channel=color_channel,
        )

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

        # 2025-10-03:
        # - Keeping tile seems to break ImageJ compatibility with tiff colormaps in 16-bit images
        # - Removing tile seems to break ImageJ compatibility with tiff colormaps in 8-bit images
        options=dict(
            photometric=photometric,
            compression='deflate',
            resolutionunit=tf.RESUNIT.MICROMETER,
            maxworkers=1
        )

        if image_data.dtype == np.uint8:
            options['tile'] = (128, 128)

        return {
            'metadata': tiff_metadata,
            'extratags': tiff_extratags,
            'options': options,
        }
    

    def save(
        self,
        image_data: np.ndarray,
        file_loc: pathlib.Path,
        metadata: dict,
        color_channel: str,
    ):
        kwargs = {}
        if (image_data.dtype == np.uint16) and (image_utils.is_color_image(image_data)):
            # For now, prevent 16-bit color images from being converted to ImageJ type
            # such as composite (or bullseye). Could allow this once proper support is added.
            pass
        else: #if image_data.dtype == np.uint16:
            kwargs['imagej'] = True


        support_data = self._generate_support_data(
            image_data=image_data,
            metadata=metadata,
            color_channel=color_channel,
        )

        with tf.TiffWriter(str(file_loc), **kwargs) as tif:
            tif.write(
                image_data,
                metadata=support_data['metadata'],
                datetime=metadata['datetime'],
                software=metadata['software'],
                **support_data['options'],
            )
