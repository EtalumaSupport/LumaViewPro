# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Camera hardware profiles — static specs per camera model.

Each profile captures hardware characteristics that don't change at runtime:
sensor identity, shutter type, pixel size, gain architecture, etc.

Dynamic values (actual gain min/max, exposure limits) are queried from the
SDK at connect time and merged into the profile by the Camera base class.


Adding a new camera model
=========================

1. GET THE DATASHEET
   Obtain the manufacturer's datasheet PDF for the camera. Key specs to
   extract (with field names in parentheses):

   - Sensor manufacturer and model    (sensor)
   - Pixel pitch in micrometers       (pixel_size_um)
   - Native resolution in pixels      (native_resolution)
   - Shutter type: rolling or global  (shutter)
   - Supported pixel formats          (pixel_formats)
     * Basler: typically Mono8, Mono10, Mono10p, Mono12, Mono12p
     * IDS: vendor-specific names like Mono10g40IDS, Mono12g24IDS
     * Note whether Mono8 is natively supported — if not, software
       ConvertTo is required and cannot be skipped
   - Maximum exposure time in ms      (max_exposure_ms)
   - Binning support                  (binning_sizes, binning_modes)
     * Which factors: [1, 2] or [1, 2, 4]
     * Modes: Sum, Average, or both
     * Any restrictions (e.g. H+V must be applied jointly)
   - Gain architecture                (gain)
     * Max analog gain (dB or linear multiplier)
     * Whether digital gain exists
     * GainSelector value: 'All', 'AnalogAll', 'DigitalAll'
   - Auto gain / auto exposure        (has_auto_gain, has_auto_exposure)
     * Whether the camera hardware supports these features
   - Temperature sensors              (has_temperature)
   - AOI alignment constraints        (alignment)
     * Width step (e.g. 4 or 48 pixels)
     * Height step (e.g. 4 pixels)

2. CREATE THE PROFILE
   Add a CameraProfile instance in the "Known camera profiles" section
   below, following the existing examples. Use the datasheet values for
   static fields. Leave dynamic fields (gain.total_min_db, total_max_db,
   exposure_min_us, exposure_max_us) as None — they will be populated
   at connect time by _query_dynamic_capabilities().

3. REGISTER IT
   Add a (substring, profile) entry to the _PROFILES list. The substring
   should match what the SDK returns from the camera's model name query
   (e.g. camera.active.ModelName() for IDS, GetDeviceInfo().GetModelName()
   for Pylon). Use the most specific substring that uniquely identifies
   the model.

4. ADD A DRIVER (if new SDK)
   If this is a new camera SDK (not Pylon or IDS), create a new driver
   file in drivers/ following the pattern of pyloncamera.py or idscamera.py:
   - Inherit from Camera base class
   - Call self._load_profile() in connect() after setting self.model_name
   - Override _query_dynamic_capabilities() to read gain/exposure ranges
     from the SDK
   - Set driver='new_sdk_name' in the profile

5. ADD TESTS
   In tests/test_simulators.py under TestCameraProfiles:
   - Add a test_lookup_known_<model> test that verifies the profile fields
   - Verify key specs match the datasheet

6. VERIFY ON HARDWARE
   Connect the camera and check the log output for:
   - "Loaded profile: <model>" — confirms profile matched
   - "Gain range: X - Y dB" — confirms dynamic query worked
   - "Exposure range: X - Y us" — confirms dynamic query worked
   Run the test suite with the camera connected to verify grab, binning,
   exposure, and gain all work correctly.
"""

from dataclasses import dataclass, field


@dataclass
class GainInfo:
    """Gain architecture for a camera model."""
    analog_max_db: float | None = None      # Max analog gain before digital kicks in
    has_digital: bool = False                # Whether camera has digital gain
    gain_selector: str = 'All'              # GainSelector value: 'All', 'AnalogAll', etc.
    # Dynamic — populated at connect time from SDK
    total_min_db: float | None = None
    total_max_db: float | None = None


@dataclass
class CameraProfile:
    """Static + dynamic hardware profile for a camera model."""

    # --- Static (hardcoded per model) ---
    model_name: str = ''
    sensor: str = ''
    pixel_size_um: float = 0.0              # Pixel pitch in micrometers
    shutter: str = 'rolling'                # 'rolling' or 'global'
    native_resolution: dict = field(default_factory=dict)  # {'width': int, 'height': int}
    pixel_formats: list[str] = field(default_factory=list)
    max_exposure_ms: int = 1_000
    binning_sizes: list[int] = field(default_factory=lambda: [1])
    binning_modes: list[str] = field(default_factory=lambda: ['Sum'])
    alignment: dict = field(default_factory=lambda: {'width': 4, 'height': 4})
    gain: GainInfo = field(default_factory=GainInfo)
    has_auto_gain: bool = False
    has_auto_exposure: bool = False
    has_temperature: bool = False
    driver: str = ''                        # 'pylon', 'ids', 'simulated'
    notes: str = ''

    # --- Dynamic (populated at connect time from SDK) ---
    exposure_min_us: float | None = None
    exposure_max_us: float | None = None


# ---------------------------------------------------------------------------
# Known camera profiles
# ---------------------------------------------------------------------------

# Basler dart — daA3840-45um
_daA3840_45um = CameraProfile(
    model_name='daA3840-45um',
    sensor='Sony IMX334LLR-C',
    pixel_size_um=2.0,
    shutter='rolling',
    native_resolution={'width': 3840, 'height': 2160},
    pixel_formats=['Mono8', 'Mono10', 'Mono10p', 'Mono12', 'Mono12p'],
    max_exposure_ms=1_000,
    binning_sizes=[1, 2, 4],
    binning_modes=['Sum', 'Average'],
    alignment={'width': 4, 'height': 4},
    gain=GainInfo(
        analog_max_db=24.0,         # TODO: verify on hardware
        has_digital=True,
        gain_selector='All',
    ),
    has_auto_gain=True,
    has_auto_exposure=True,
    has_temperature=False,          # TODO: verify on hardware
    driver='pylon',
)

# Basler ace 2 — a2A3536-31umBAS
_a2A3536_31umBAS = CameraProfile(
    model_name='a2A3536-31umBAS',
    sensor='Sony IMX676-AAMR1-C',
    pixel_size_um=2.0,
    shutter='rolling',
    native_resolution={'width': 3536, 'height': 3552},
    pixel_formats=['Mono8', 'Mono10', 'Mono10p', 'Mono12', 'Mono12p'],
    max_exposure_ms=10_000,
    binning_sizes=[1, 2, 4],
    binning_modes=['Sum', 'Average'],
    alignment={'width': 4, 'height': 4},
    gain=GainInfo(
        analog_max_db=30.0,         # Confirmed from Basler docs
        has_digital=True,
        gain_selector='All',
    ),
    has_auto_gain=True,
    has_auto_exposure=True,
    has_temperature=True,
    driver='pylon',
)

# IDS — U3-34L0XCP-M (NO and GL variants share same specs)
_U3_34L0XCP_M = CameraProfile(
    model_name='U3-34L0XCP-M',
    sensor='Sony IMX676-AAMR1-C',
    pixel_size_um=2.0,
    shutter='rolling',
    native_resolution={'width': 3552, 'height': 3552},
    pixel_formats=['Mono10g40IDS', 'Mono12g24IDS'],  # No native Mono8
    max_exposure_ms=2_000,
    binning_sizes=[1, 2],           # Sensor 2x2 only, H+V joint
    binning_modes=['Sum'],
    alignment={'width': 48, 'height': 4},
    gain=GainInfo(
        analog_max_db=None,         # 31.6x max — query dB from SDK
        has_digital=False,
        gain_selector='AnalogAll',
    ),
    has_auto_gain=False,            # Not supported in hardware
    has_auto_exposure=False,        # Not supported in hardware
    has_temperature=False,
    driver='ids',
    notes='No native Mono8 — requires software ConvertTo. '
          'Binning H+V must be applied jointly. '
          'Max gain 31.6x (analog only).',
)

# Simulated camera
_simulated = CameraProfile(
    model_name='SimulatedCamera-1920x1200',
    sensor='Simulated',
    pixel_size_um=2.0,
    shutter='global',
    native_resolution={'width': 1920, 'height': 1200},
    pixel_formats=['Mono8', 'Mono10', 'Mono12'],
    max_exposure_ms=10_000,
    binning_sizes=[1, 2, 4],
    binning_modes=['Sum'],
    alignment={'width': 48, 'height': 4},
    gain=GainInfo(
        analog_max_db=20.0,
        has_digital=False,
        gain_selector='All',
        total_min_db=0.0,
        total_max_db=20.0,
    ),
    has_auto_gain=True,
    has_auto_exposure=True,
    has_temperature=True,
    driver='simulated',
)


# Aptina MT9P031 — Lumascope Classic LS620 / LS560 / LS720 via Cypress FX2
# Native sensor is 2592×1944 but the driver crops/centers a 1900×1900
# window. Gain is hardcoded by driver math (0–42.1 dB). Max exposure is
# MAX_EXPOSURE_ROWS (65535) × _ROW_TIME_MS (0.1124) = 7366 ms.
# See drivers/fx2driver.py + LumaviewClassic/docs/DATASHEET_VERIFICATION.md.
_MT9P031_LS620 = CameraProfile(
    model_name='MT9P031-LS620',
    sensor='Aptina MT9P031',
    pixel_size_um=2.2,
    shutter='rolling',
    native_resolution={'width': 1900, 'height': 1900},
    pixel_formats=['Mono8'],          # 12-bit sensor, FX2 streams top 8 bits
    max_exposure_ms=7_366,            # 65535 rows × 0.1124 ms/row
    binning_sizes=[1],                # driver doesn't wire up sensor binning
    binning_modes=['Sum'],
    alignment={'width': 4, 'height': 4},  # matches set_frame_size() step
    gain=GainInfo(
        analog_max_db=18.06,          # 8× analog = 20*log10(8) = 18.06 dB
        has_digital=True,             # digital stage adds up to 16× more
        gain_selector='All',
        total_min_db=0.0,
        total_max_db=42.1,            # audit-corrected per RR_A legal ranges
    ),
    has_auto_gain=False,              # no hardware AE/AG on MT9P031
    has_auto_exposure=False,
    has_temperature=False,
    driver='fx2',
    notes='Cypress FX2 USB + Aptina MT9P031 sensor. 4 LED channels via '
          'I2C at 0x2A. No hardware auto gain/exposure. No binning. '
          'Mono8 only (top 8 bits of 12-bit ADC). Exposure changes '
          'have a 2-frame pipeline delay. Hardware-validated at 4.5 fps '
          'on LS620 macOS (63/63 frames).',
)


# ---------------------------------------------------------------------------
# Profile registry and lookup
# ---------------------------------------------------------------------------

# Maps model name substrings to profiles. Checked in order — first match wins.
_PROFILES: list[tuple[str, CameraProfile]] = [
    ('daA3840-45um',            _daA3840_45um),
    ('a2A3536-31umBAS',         _a2A3536_31umBAS),
    ('U3-34L0XCP-M',            _U3_34L0XCP_M),   # spec sheet model
    ('U3-34LxXCP-M',            _U3_34L0XCP_M),   # as reported by SDK
    ('SimulatedCamera',         _simulated),
    ('MT9P031',                 _MT9P031_LS620),  # FX2Camera sets model_name='MT9P031-LS620'
    ('LS620',                   _MT9P031_LS620),  # explicit model-name match
    ('LS560',                   _MT9P031_LS620),  # same sensor, same profile
    ('LS720',                   _MT9P031_LS620),  # same sensor, same profile
]

# Default profile for unknown cameras
_DEFAULT = CameraProfile(
    model_name='Unknown',
    pixel_formats=['Mono8'],
    max_exposure_ms=1_000,
    binning_sizes=[1],
    has_auto_gain=False,
    has_auto_exposure=False,
    driver='unknown',
    notes='Fallback profile — camera model not recognized',
)


def lookup_profile(model_name: str | None) -> CameraProfile:
    """Find the best matching profile for a camera model name.

    Matches by substring — e.g. 'daA3840-45um' matches a model_name
    containing that string. Returns a copy so callers can safely
    modify dynamic fields without affecting the registry.
    """
    import copy

    if not model_name:
        return copy.deepcopy(_DEFAULT)

    for key, profile in _PROFILES:
        if key in model_name:
            result = copy.deepcopy(profile)
            result.model_name = model_name  # Use the actual full model name
            return result

    logger = None
    try:
        from lvp_logger import logger as _logger
        logger = _logger
    except ImportError:
        pass

    if logger:
        logger.warning(f'[CAM Prof  ] No profile found for model: {model_name}, using defaults')

    result = copy.deepcopy(_DEFAULT)
    result.model_name = model_name
    return result
