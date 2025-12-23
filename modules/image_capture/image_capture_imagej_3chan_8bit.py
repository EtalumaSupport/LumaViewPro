
from modules.image_capture.image_capture_enums import *


class ImageCapture_ImageJ_3chan_8bit:
    
    def __init__(self):
        self._name = self.__class__.__name__
        super().__init__(self._name)


    def supported_configs() -> tuple[ImageCaptureConfig]:
        return (
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT8,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.IMAGEJ,
            ),
        )


    def save(
        self,
        image_data,
    ):
        pass
