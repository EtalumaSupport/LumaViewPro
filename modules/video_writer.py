
import datetime
import pathlib

import cv2
import numpy as np

import image_utils


class VideoWriter:

    CODECS = [
        0,
        'mp4v',
        'mjpg',
        'ffv1',
    ]

    def __init__(
        self,
        output_file_loc: pathlib.Path,
        fps: float,
        include_timestamp_overlay: bool,
    ):
        self._output_file_loc = output_file_loc
        self._codec = self.CODECS[1] # mp4v
        self._fourcc = self._get_fourcc_code(codec=self._codec)
        self._fps = fps
        self._include_timestamp_overlay = include_timestamp_overlay

        if not output_file_loc.parent.exists():
            output_file_loc.parent.mkdir(parents=True)

        self._video = None


    @staticmethod
    def _get_fourcc_code(codec: str | int):
        if type(codec) == str:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        else:
            fourcc = codec

        return fourcc
    

    @staticmethod
    def _get_image_info(image: np.ndarray) -> tuple:
        is_color = True if image.ndim == 3 else False
        
        if is_color:
            frame_height, frame_width, _ = image.shape
        else:
            frame_height, frame_width = image.shape
        
        return (frame_height, frame_width), is_color
    

    @staticmethod
    def _get_timestamp_str():
        ts = datetime.datetime.now()
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return ts_str


    def _init_video(self, image: np.ndarray):
        (height, width), is_color = self._get_image_info(image=image)

        self._video = cv2.VideoWriter(
            filename=str(self._output_file_loc),
            fourcc=self._fourcc,
            fps=self._fps,
            frameSize=(width, height),
            isColor=is_color,
        )

        # Ref: https://stackoverflow.com/questions/71945367/how-to-properly-use-opencv-videowriter-to-write-monochrome-video-with-float32-so
        # Example for recording 16-bit grayscale video
        # Playback of FFV1 not supported in Windows Media Player with default codecs, but works in VLC
        # self._video = cv2.VideoWriter(
        #     filename=str(self._output_file_loc),
        #     apiPreference=cv2.CAP_FFMPEG,
        #     fourcc=self._fourcc,
        #     fps=self._fps,
        #     frameSize=(width, height),
        #     params=[
        #         cv2.VIDEOWRITER_PROP_DEPTH,
        #         cv2.CV_16U,
        #         cv2.VIDEOWRITER_PROP_IS_COLOR,
        #         0,
        #     ],
        # )


    def add_frame(self, image: np.ndarray):

        # First frame initialization
        if self._video is None:
            self._init_video(image=image)
        
        if self._include_timestamp_overlay:
            ts = self._get_timestamp_str()
            image = image_utils.add_timestamp(image=image, timestamp_str=ts)
        
        self._video.write(image)

    
    def finish(self):
        cv2.destroyAllWindows()
        self._video.release()
