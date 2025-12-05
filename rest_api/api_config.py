from typing import TYPE_CHECKING
from lumascope_api import Lumascope

if TYPE_CHECKING:
    from lumaviewpro import MainDisplay

lumaview: "MainDisplay" = None

def set_view(view: "MainDisplay"):
    global lumaview
    lumaview = view

def get_scope() -> Lumascope:
    return lumaview.scope