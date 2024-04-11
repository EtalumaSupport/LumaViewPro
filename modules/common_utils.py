
import os
import pathlib
import re


def generate_default_step_name(
    well_label,
    color,
    z_height_idx = None,
    scan_count = None,
    tile_label = None,
    custom_name_prefix = None,
    stitched: bool = False,
):
    if custom_name_prefix not in (None, ""):
        name = f"{custom_name_prefix}_{color}"
    else:
        name = f"{well_label}_{color}"
    
    if tile_label not in (None, "", -1):
        name = f"{name}_T{tile_label}"

    if z_height_idx not in (None, "", -1):
        name = f"{name}_Z{z_height_idx}"

    DESIRED_SCAN_COUNT_DIGITS = 4
    if scan_count not in (None, ""):
        name = f'{name}_{scan_count:0>{DESIRED_SCAN_COUNT_DIGITS}}'

    if stitched:
        name = f'{name}_stitched'
    
    return name


def get_tile_label_from_name(name: str) -> str | None:
    name = name.split('_')

    if len(name) <= 2:
        return None

    segment = name[2]
    if segment.startswith('T'):
        return segment[1:]
    
    return None


def get_first_section_from_name(name: str) -> str | None:
    
    # This will retrieve just the filename if the name has parent folders
    name = pathlib.Path(name).name

    name = name.split('_')
    return name[0]


def get_layer_from_name(name: str) -> str | None:
    name = name.split('_')

    return name[1]



def replace_layer_in_step_name(step_name: str, new_layer_name: str) -> str | None:

    # Extract basename in case we are handling protocol with separate folders per channel
    base_name = os.path.basename(step_name)
    # if is_custom_name(name=base_name):
    #     return None

    # This replaces the parent folder when using per-channel folders for protocol runs
    split_name = list(os.path.split(step_name))
    if len(split_name) == 2:
        using_per_channel_folders = True
    else:
        using_per_channel_folders = False
    
    if using_per_channel_folders:
        split_name[0] = new_layer_name
        step_name = str(pathlib.Path(split_name[0]) / split_name[1])

    step_name_segments = step_name.split('_')
    
    # Confirm it's actually a layer before replacing it
    if step_name_segments[1] in get_layers(): 
        step_name_segments[1] = new_layer_name

    return '_'.join(step_name_segments)


def is_custom_name(name: str) -> bool:

    # This will retrieve just the filename if name includes parent folders
    name = pathlib.Path(name).name

    name = name.split('_')

    # All generated names have at least one '_'
    if len(name) <= 1:
        return True
    
    well = name[0]
    well_pattern = r"^[A-Z]{1,2}[0-9]+$"
    if not re.match(pattern=well_pattern, string=well):
        return True

    color = name[1]
    if color not in get_layers():
        return True
    
    return False

def get_z_slice_from_name(name: str) -> int | None:
    name = name.split('_')

    # Z-slice info can either be at segment index 2 (if no tile label is present), or segment index 3 (if tile label is present)
    if len(name) <= 2:
        return None
    
    if name[2].startswith('Z'):
        return name[2][1:]
    
    if len(name) <= 3:
        return None
    
    if name[3].startswith('Z'):
        return name[3][1:]
        
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


def get_fluorescence_layers() -> list[str]:
    return ['Blue', 'Green', 'Red']


def get_opened_layer(lumaview_imagesettings) -> str | None:
    for layer in get_layers():
        layer_is_collapsed = lumaview_imagesettings.ids[f"{layer}_accordion"].collapse

        if not layer_is_collapsed:
            return layer
        
        return None


def to_bool(val) -> bool:
    if 'str' in str(type(val)):
        return True if val.lower() == "true" else False
    elif val in ("", None):
        return False
    else:
        return bool(float(val))


def to_float(val) -> float:
    if 'numpy' in str(type(val)):
        return val.astype(float)
    else:
        return float(val)
    
    
def to_int(val) -> int | None:
    if 'numpy' in str(type(val)):
        return int(val.astype(float))
    elif val in ("", None):
        return -1
    else:
        return int(float(val))


def get_pixel_size(
    focal_length: float
):
    magnification = 47.8 / focal_length # Etaluma tube focal length [mm]
                                        # in theory could be different in different scopes
                                        # could be looked up by model number
                                        # although all are currently the same
    pixel_width = 2.0 # [um/pixel] Basler pixel size (could be looked up from Camera class)
    um_per_pixel = pixel_width / magnification

    return um_per_pixel


def get_field_of_view(
    focal_length: float,
    frame_size: dict
):
    um_per_pixel = get_pixel_size(focal_length=focal_length)
    fov_x = um_per_pixel * frame_size['width']
    fov_y = um_per_pixel * frame_size['height']
    
    return {
        'width': fov_x,
        'height': fov_y
    }


def max_decimal_precision(parameter: str) -> int:
    DEFAULT_PRECISION = 5
    PRECISION_MAP = {
        'x': 4,
        'y': 4,
        'z': 5
    }

    return PRECISION_MAP.get(parameter, DEFAULT_PRECISION)
