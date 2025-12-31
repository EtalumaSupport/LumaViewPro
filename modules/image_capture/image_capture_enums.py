
import dataclasses
import enum


@enum.unique
class ImagePixelDepth(enum.Enum):
    BIT8    = '8-bit'
    BIT12   = '12-bit'


@enum.unique
class ImageChannelCount(enum.Enum):
    SINGLE_CHANNEL  = '1-chan'
    THREE_CHANNEL   = '3-chan'


@enum.unique
class ImageFileFormat(enum.Enum):
    TIFF                = 'TIFF'
    OME_TIFF            = 'OME-TIFF'
    IMAGEJ_HYPERSTACK   = 'ImageJ Hyperstack'
    VIDEO               = 'Video'


@enum.unique
class VideoFileFormat(enum.Enum):
    FRAMES  = 'Frames'
    MP4     = 'mp4'


@dataclasses.dataclass(eq=True, frozen=True)
class ImageCaptureConfig:
    pixel_depth: ImagePixelDepth
    channel_count: ImageChannelCount
    file_format: ImageFileFormat
