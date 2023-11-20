
import os
import pathlib


from modules.protocol import Protocol


class Stitcher:

    def __init__(self):
        pass


    @staticmethod
    def _get_image_filenames_from_folder(path: pathlib.Path) -> list:
        images = path.glob('*.tif[f]')
        image_names = []
        for image in images:
            image_names.append(image.name)

        return image_names

    def _find_protocol_tsv(self, path: pathlib.Path) -> str | None:
        tsv_files = list(path.glob('*.tsv'))
        if len(tsv_files) ==  1:
            return tsv_files[0]
        
        # Return none if no .tsv's are found or if multiple are found
        return None
    

    def load_folder(self, path: str | pathlib.Path):
        path = pathlib.Path(path)

        protocol_tsv = self._find_protocol_tsv(path=path)
        if protocol_tsv is None:
            return False
        
        protocol = Protocol.from_file(file_path=protocol_tsv)

        image_names = self._get_image_filenames_from_folder(path=path)
        print(image_names)
        

if __name__ == "__main__":
    stitcher = Stitcher()
    stitcher.load_folder(pathlib.Path(os.getenv("SAMPLE_IMAGE_FOLDER")))