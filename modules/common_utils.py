
def generate_default_step_name(
    well_label,
    color,
    z_height_idx = None,
    scan_count = None,
    tile_label = None 
):
    name = f"{well_label}_{color}"

    if tile_label not in (None, ""):
        name = f"{name}_T{tile_label}"

    if z_height_idx not in (None, ""):
        name = f"{name}_Z{z_height_idx}"

    DESIRED_SCAN_COUNT_DIGITS = 4
    if scan_count not in (None, ""):
        name = f'{name}_{scan_count:0>{DESIRED_SCAN_COUNT_DIGITS}}'
    
    return name


def get_tile_label_from_name(name: str) -> str | None:
    name = name.split('_')

    segment = name[2]
    if segment.startswith('T'):
        return segment[1:]
    
    return None

def get_z_slice_from_name(name: str) -> int | None:
    name = name.split('_')

    # Z-slice info can either be at segment index 2 (if no tile label is present), or segment index 3 (if tile label is present)
    for segment in (name[2], name[3]):
        if segment.startswith('Z'):
            return segment[1:]
    
    return None


def convert_zstack_reference_position_setting_to_config(text_label: str) -> str:
    LABEL_MAP = {
        'Current Position at Top': 'top',
        'Current Position at Center': 'center',
        'Current Position at Bottom': 'bottom'
    }

    if text_label in LABEL_MAP:
          return LABEL_MAP[text_label]
    
    raise Exception(f"Unknown Z-stack position reference: {text_label}")

                    
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
