

from dataclasses import dataclass

@dataclass
class ProtocolStep:
    name: str
    x: float
    y: float
    z: float
    auto_focus: bool
    channel: int
    false_color: bool
    illumination: float
    gain: float
    auto_gain: bool
    exposure: float

    def __post_init__(self):
        if not isinstance(self.auto_focus, bool):
            self.auto_focus 
