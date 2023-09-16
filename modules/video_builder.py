
import pathlib

import cv2

from lvp_logger import logger


class VideoBuilder:

    CODECS = [
        0,
        'mp4v'
    ]

    def __init__(self):
        self._name = self.__class__.__name__


    def _get_all_images_in_directory(self, directory: pathlib.Path) -> list[pathlib.Path]:
        if not directory.exists():
            raise Exception(f"Directory {directory} does not exist")
        
        if not directory.is_dir():
            raise Exception(f"{directory} is not a directory")
        
        images = []
        for image_path in directory.glob("*.tif*"):
            images.append(image_path)

        return images
    

    def _all_images_same_size(self, image_list: list[pathlib.Path]) -> bool:

        if len(image_list) == 0:
            return False, {}

        frame_shapes = {}
        for image in image_list:
            shape = self._get_frame_size(image=image)

            # Track which images have which frame shapes
            # Useful for troubleshooting/logging
            if shape not in frame_shapes:
                frame_shapes[shape] = []
            
            frame_shapes[shape].append(image)

        # Check if more than one size of image was found
        if len(frame_shapes) > 1:
            return False, frame_shapes
        
        return True, frame_shapes
    

    @staticmethod
    def _get_fourcc_code(codec: str | int):
        if type(codec) == str:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        else:
            fourcc = codec

        return fourcc
    

    def _get_frame_size(self, image: pathlib.Path) -> dict:
        frame = cv2.imread(str(image))
        height, width, _ = frame.shape
        
        return (height, width)

        
    def create_video_from_directory(
        self,
        input_directory: pathlib.Path,
        frames_per_sec: int,
        output_file_loc: pathlib.Path
    ) -> bool:
        
        def _are_valid_inputs():
            if not issubclass(type(input_directory), pathlib.Path):
                logger.error(f"[{self._name}] Expected input directory to be of type pathlib.Path, got {type(input_directory)}")
                return False
            
            if not issubclass(type(output_file_loc), pathlib.Path):
                logger.error(f"[{self._name}] Expected output file location to be of type pathlib.Path, got {type(output_file_loc)}")
                return False
            
            if type(frames_per_sec) not in (int, float):
                logger.error(f"[{self._name}] Invalid type for frames_per_sec, must be int or float")
                return False

            if frames_per_sec <= 0:
                logger.error(f"[{self._name}] Invalid value for frames_per_sec, must be >0")
                return False
            
            return True
            

        if not _are_valid_inputs():
            return False

        
        logger.info(f"""[{self._name}] Starting video creation:
                            Input directory: {input_directory}
                            Output file: {output_file_loc}
                    """)
        
        images = self._get_all_images_in_directory(directory=input_directory)

        if len(images) == 0:
            logger.error(f"[{self._name}] No images found in {input_directory}")
            return False
        
        logger.info(f"[{self._name}] Found {len(images)} images")
        
        valid_image_size_match, frame_sizes = self._all_images_same_size(image_list=images)

        if not valid_image_size_match:
            logger.error(f"[{self._name}] Not all images in {input_directory} have matching dimensions:\n{frame_sizes}")
            return False

        codec = self.CODECS[0] # Set to AVI

        fourcc = self._get_fourcc_code(codec=codec)

        frame_height, frame_width = self._get_frame_size(image=images[0])

        video = cv2.VideoWriter(
            filename=str(output_file_loc),
            fourcc=fourcc,
            fps=frames_per_sec,
            frameSize=(frame_width, frame_height),
            isColor=True
        )

        logger.info(f"[{self._name}] Writing video to {output_file_loc}")
        for image in images:
            video.write(cv2.imread(str(image)))

        cv2.destroyAllWindows()
        video.release()

        logger.info(f"[{self._name}] Video creation complete")

        return True
