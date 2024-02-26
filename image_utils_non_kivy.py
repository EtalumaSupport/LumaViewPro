import pathlib

import tifffile as tf

import modules.common_utils as common_utils


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


def write_ome_tiff(data, file_loc: pathlib.Path, channel: str):
    pixel_size=0.29
    # pixel_size = common_utils.get_pixel_size(focal_length= )

    with tf.TiffWriter(str(file_loc), bigtiff=False) as tif:
        metadata={
            'axes': 'YXS',
            'SignificantBits': data.itemsize*8,
            'PhysicalSizeX': pixel_size,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixel_size,
            'PhysicalSizeYUnit': 'µm',
            'Channel': {'Name': [channel]},
            'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
        }

        options=dict(
            photometric='rgb',
            tile=(128, 128),
            compression='jpeg',
            resolutionunit='CENTIMETER',
            maxworkers=2
        )

        tif.write(
            data,
            resolution=(1e4 / pixel_size, 1e4 / pixel_size),
            metadata=metadata,
            **options
        )
