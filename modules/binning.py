

BINNING_SIZE_MAP = {
    '1x1': 1,
    '2x2': 2,
    '4x4': 4,
}


def binning_size_str_to_int(text: str) -> int:
    return BINNING_SIZE_MAP.get(text, 1)


def binning_size_int_to_str(val: int):
    for k, v in BINNING_SIZE_MAP.items():
        if v == val:
            return k
        
    return '1x1'
