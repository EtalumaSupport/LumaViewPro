from typing import TYPE_CHECKING
from lumascope_api import Lumascope

if TYPE_CHECKING:
    from lumaviewpro import MainDisplay
    from modules.sequenced_capture_executor import SequencedCaptureExecutor

lumaview: "MainDisplay" = None
source_path: str = None
settings: dict = {}
sequenced_capture_executor: "SequencedCaptureExecutor"  = None
protocol_callbacks: dict = {}

def set_view(view: "MainDisplay"):
    global lumaview
    lumaview = view

def set_settings(new_settings: dict):
    global settings
    settings = new_settings

def set_sequenced_capture_executor(new_sce: "SequencedCaptureExecutor"):
    global sequenced_capture_executor
    sequenced_capture_executor = new_sce

def set_source_path(new_sp: str):
    global source_path
    source_path = new_sp

def set_protocol_callbacks(callbacks: dict):
    global protocol_callbacks
    protocol_callbacks = callbacks

def get_image_capture_config() -> dict:
    microscope_settings = lumaview.ids['motionsettings_id'].ids['microscope_settings_id']
    output_format = {
        'live': microscope_settings.ids['live_image_output_format_spinner'].text,
        'sequenced': microscope_settings.ids['sequenced_image_output_format_spinner'].text,
    }
    use_full_pixel_depth = lumaview.ids['viewer_id'].ids['scope_display_id'].use_full_pixel_depth
    return {
        'output_format': output_format,
        'use_full_pixel_depth': use_full_pixel_depth,
    }

def get_scope() -> Lumascope:
    return lumaview.scope

def get_settings() -> dict:
    return settings

def get_sequenced_capture_executor() -> "SequencedCaptureExecutor":
    return sequenced_capture_executor

def get_source_path() -> str:
    return source_path

def get_protocol_callbacks() -> dict:
    return protocol_callbacks