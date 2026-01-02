

import pathlib

import numpy as np
import tifffile as tf

import image_utils
from modules.image_capture.image_capture_enums import *
from modules.image_capture.image_capture_format_base import ImageCaptureFormatBase

class ImageCapture_OmeTiff(ImageCaptureFormatBase):
    
    def __init__(self):
        self._name = self.__class__.__name__
        super().__init__(self._name)


    @staticmethod
    def supported_configs() -> tuple[ImageCaptureConfig]:
        return (
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.SINGLE_CHANNEL,
                file_format=ImageFileFormat.OME_TIFF,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.SINGLE_CHANNEL,
                file_format=ImageFileFormat.OME_TIFF,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.OME_TIFF,
            ),
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.OME_TIFF,
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
        tiff_extratags = []
        # Metadata seems to be working properly for OME-TIFF's so let's leave it alone for now.

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
        kwargs = {
            'bigtiff': False,
        }

        support_data = self._generate_support_data(
            image_data=image_data,
            metadata=metadata,
            color_channel=color_channel,
        )

        with tf.TiffWriter(str(file_loc), **kwargs) as tif:

            if True == image_utils.is_color_image(image_data):
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

