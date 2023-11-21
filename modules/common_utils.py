
def generate_default_step_name(
    well_label,
    color,
    z_height_idx = None,
    scan_count = None,
    tile_label = None 
):
    name = f"{well_label}_{color}"

    if z_height_idx not in (None, ""):
        name = f"{name}_Z{z_height_idx}"

    DESIRED_SCAN_COUNT_DIGITS = 6
    if scan_count not in (None, ""):
        name = f'{name}_{scan_count:0>{DESIRED_SCAN_COUNT_DIGITS}}'
    
    if tile_label not in (None, ""):
        name = f"{name}_T{tile_label}"

    return name


def get_tile_label_from_name(name: str) -> str | None:
    name = name.split('_')

    last_segment = name[-1]
    if last_segment.startswith('T'):
        return last_segment[1:]
    
    return None


def get_layers() -> list[str]:
    return ['BF', 'PC', 'EP', 'Blue', 'Green', 'Red']


def get_transmitted_layers() -> list[str]:
    return ['BF', 'PC', 'EP']


def to_bool(val) -> bool:
    if 'str' in str(type(val)):
        return True if val.lower() == "true" else False
    else:
        return bool(float(val))


def to_float(val) -> float:
    if 'numpy' in str(type(val)):
        return val.astype(float)
    else:
        return float(val)
    
    
def to_int(val) -> int:
    if 'numpy' in str(type(val)):
        return int(val.astype(float))
    else:
        return int(float(val))
