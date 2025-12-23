
from modules.image_capture.image_capture_enums import *


class ImageCapture_Tiff_3chan_12bit:
    
    def __init__(self):
        self._name = self.__class__.__name__
        super().__init__(self._name)


    def supported_configs() -> tuple[ImageCaptureConfig]:
        return (
            ImageCaptureConfig(
                pixel_depth=ImagePixelDepth.BIT12,
                channel_count=ImageChannelCount.THREE_CHANNEL,
                file_format=ImageFileFormat.TIFF,
            ),
        )


    def save(
        self,
        image_data,
    ):
        pass
