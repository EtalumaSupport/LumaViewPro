
import pathlib


class Stitcher:

    def __init__(self):
        pass


    def _find_protocol_tsv(self, path: pathlib.Path) -> str | None:
        tsv_files = path.glob('*.tsv')
        if len(tsv_files) ==  1:
            return tsv_files[0]
        
        # Return none if no .tsv's are found or if multiple are found
        return None
    

    def load_folder(self, path: str | pathlib.Path):
        path = pathlib.Path(path)

        protocol_tsv = self._find_protocol_tsv(path=path)
        if protocol_tsv is None:
            return False
        
        self._load_protocol_tsv(path=path)
