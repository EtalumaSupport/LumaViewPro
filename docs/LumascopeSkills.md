# LumaViewPro — API & Integration Reference

**What is public.** The public surface is defined in code: a member is
public when it carries the `@api` mark (`modules/api_surface.py`), and a
class publishes the fields its `@api_fields` names. A member marked
`@api(in_process=True)` is for Python callers and is kept off the wire.
Three more marks say what a member does, for a host deciding how to carry
it:

- `reads`: it changes nothing.
- `stop`: the member that stops what it starts, written `Class.member`
  (`Operation.stop` for a `start_` member). A toggle (`enable_streaming` /
  `disable_streaming`) starts nothing to stop: it sets a state, and its pair
  sets it back.
- `engineering`: a bench or support member (fan, drive status, LED
  engineering mode, the camera bandwidth test, the Pylon probe, the grab
  benchmark), which a researcher's client does not need.

What a member may raise is its docstring's `Raises:` section: each refusal
and fault by class, and for a refusal the reason a caller branches on; a
member that raises nothing says `Nothing.`. An argument of another type than
the member declares is refused at every member alike (see the arguments
paragraph), so no section lists that refusal.

This document is the reference for that surface, and the test suite holds
the two in lockstep both ways -- every call form here must resolve on the
live API, and every public member must appear here. A member absent from
this document is not API: it may be renamed, moved, or removed in any
release without notice, and code that calls it is unsupported.

## PRE-RELEASE API

The Lumascope SDK API documented in this file is **subject to breaking changes** until `4.0.0` ships. What this document describes is on the trunk now: the `ScopeSession` entry point and the scope's sub-APIs (`scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, `scope.runtime_state`, `scope.protocols`). What is still fluid:

- The REST surface ("REST surface reference"): its routes, handles, jobs, problems and event stream are on the trunk as documented, and may still change before `4.0.0`.
- Protocol-level surfaces may still be renamed or restructured as the wire forms settle.

If you are using this API before stabilization, **contact Etaluma support** so we know to consult you before structural changes. Internal LumaViewPro use does not trigger this requirement.

The warning retires when the first non-`-beta` LumaViewPro release ships (the `4.0.0` git tag on the `4.0.0-beta` lineage). At that point the L2-callable surface freezes for the 4.x major version and every subsequent change is recorded in the [Changelog](#changelog) section below (additive / behavior-change / rename / removal, with version + justification). Until `4.0.0` ships, the API surface stays structurally fluid -- methods may be renamed, moved into sub-APIs, or retired without a deprecation cycle.

---

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination, XYZ stage + turret motion, and camera image acquisition. This document is the integration reference for developers building scripts, headless automation, or external control applications on top of LumaViewPro.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.12–3.13, Windows / macOS / Linux

---

## Architecture

```text
┌─────────────────────────────────────────────────┐
│  Your Application                               │
│  (MATLAB, Python script, LabVIEW, web app)      │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  REST surface  (HTTP/JSON, any language)        │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  ScopeSession session layer  (Python, headless) │
│  └─ executor-routed commands, protocol runner   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Lumascope composition root  (Python)           │
│  ├─ scope.motion        ├─ scope.capabilities   │
│  ├─ scope.illumination  ├─ scope.runtime_state  │
│  ├─ scope.imaging       ├─ scope.protocols      │
│  └─ scope.diagnostics                           │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  modules/  (image_save, coord_transformations,  │
│             composite_builder, autofocus, …)    │
└─────────────────────────────────────────────────┘
```

Each layer wraps the one below. Higher = easier. Lower = more control.

For internal serial-protocol details (firmware updates, bring-up tooling), see [`FirmwareTooling.md`](FirmwareTooling.md) — not intended for application integration, and not part of this API.

---

## Integration Levels

Pick the layer that fits your use case:

| Layer | Interface | Language | Best for |
|---|---|---|---|
| **REST surface** | HTTP (JSON) | Any | External apps, cross-language control |
| **ScopeSession session layer** | Python | Python | Headless scripts, automation, tests |
| **The scope's sub-APIs** (`session.scope.motion`, `.illumination`, `.imaging`, ...) | Python | Python | Direct hardware control inside a session: a plugin, or an application built on `ScopeSession` |

The remainder of this document is organized as the sub-API reference (one section per sub-API), then the modules layer, plugin platform pointers, REST surface, and finally practical patterns + appendices.

### Sentinel-return vs raise contract

Methods on the L2 surface follow one of two contracts, read from what the method's `Raises:` section says: a method whose section lists entries follows the raise contract; one whose section is `Nothing.` raises nothing you handle, and when it is a read that cannot always know its value it answers a sentinel, its own docstring saying when.

- **Hardware-state queries** (capability probes, status reads, getters like `get_led_state`, `get_target_position`, `get_led_states`, `read_motor_fan_rpm`) return a sentinel value -- `None`, `False`, or an empty container -- when the value cannot be read (no hardware, channel not set, firmware does not implement the probe). No exception is raised. The caller branches on the sentinel. The camera's reads are not of this kind: they are the pair below.
- **Camera reads come in a pair** (`scope.camera`). A held read is a property named for the setting -- `gain_db`, `exposure_ms`, `black_level`, `binning`, `frame_size`, `pixel_format`, `auto_gain`, `auto_exposure`, `high_conversion_gain`, `line_noise_reduction` -- and answers what the camera last reported or took, with no read of the camera, so it is safe at any frequency from any thread. It is `None` whenever that value is unknown: no camera, not yet reported, or a write the camera did not answer; there is no sentinel such as `-1`. A fresh read that fails leaves the held value as it was. `active`, whether the camera is connected and active as last found, is a held `bool`. A fresh read is a `read_*` method -- `read_gain_db`, `read_exposure_ms`, `read_black_level`, `read_binning`, `read_frame_size`, `read_pixel_format`, `read_frame_rate`, `read_auto_gain`, `read_auto_exposure` -- that asks the camera now and raises `CameraReadFailedError` (`drivers.exceptions`, reason `camera_read_failed`) when the camera does not answer; its `None` means only that no camera is active, or that the camera reports no such value (a black level, a frame rate). A fresh read the camera answered is the held value from then on. `auto_gain` and `auto_exposure` answer a `CameraAutoMode` (`modules.lumascope_api.camera`): `OFF`, `ONCE` or `CONTINUOUS`, `'Off'`, `'Once'` or `'Continuous'` on the wire. `ONCE` is a one-shot the camera is still running; the camera turns it off itself when done, so the held value says `ONCE` until a fresh read finds it ended. A camera without hardware auto-gain or auto-exposure answers `OFF` unasked. High conversion gain and line-noise reduction have a held read only: no camera reads them back, so each is what the camera last accepted. `image_mode` is the saved mode, since the camera holds only the pixel format the mode chose. What the camera was at a given frame is that frame's own record, not a read made afterwards: a saved file carries its exposure and gain in its metadata, and an in-process caller reads it whole as `last_capture_info['frame_record']`.
- **State-changing operations** (setters like `move_xy`, `led_on`, etc.) return only once the change is made; a change that could not be made raises (a refusal names what was missing or wrong, a fault what broke). A `Raises:` section in the docstring documents the typed exception (`HardwareError` and its children from `drivers.exceptions`; `CaptureError`, `ConfigError` from `modules.exceptions`) that propagates when the underlying SDK call itself fails. Some members still log and show their own notification before re-raising; they are being moved to raise only, so the failure is shown once, by whoever reports it. The typed exception is what L2 callers should catch. **Read the member's own `Raises:` section rather than this paragraph: it is the declaration, and not every setter returns a status.**
- **Camera setting applies** (`set_gain_db`, `set_exposure_ms`, `set_black_level`, and the camera's `set_frame_size`, `set_binning` and `set_image_mode`) are the raise contract, not the True/False one: every setting write either applied or raises `CameraSettingRejected` (`modules.exceptions`), a fault carrying `setting`, `requested`, `reason`, and the `title` and message a person reads, with the driver's exception as its `__cause__` when it raised; nothing is logged or shown before it reaches you, so reporting it is yours. Its `reason` is `camera_rejected` (title 'Camera Setting Not Applied') when the camera refused the value or its driver raised from the write, and `camera_no_answer` (title 'Camera Did Not Answer') when the camera did not answer the write in time: the API sends a write the camera did not answer once more, at once, and raises only when the resend goes unanswered too. A refused write is not resent. After an unanswered one, whether the camera took the value is not known, so the gate that holds a capture until frames carry the new setting forgets the value it was waiting for and settles by frame count. Both are REST `500`. Success is observed by the returned value, which is what the camera actually applied and may differ from the request: the DELIVERED size for the geometry setters, and for `set_gain_db` / `set_exposure_ms` the gain in dB / exposure in ms now in effect (a body snaps or quantizes). A gain or exposure outside the camera's declared range (`capabilities.camera_limits`: `gain_db_min` .. `gain_db_max`, `exposure_ms_min` .. `exposure_ms_max`, read once at connect) is refused before anything reaches the camera with `CameraSettingOutOfRangeError`, a refusal and a `ValueError`, carrying `reason` (`gain_db_out_of_range` / `exposure_ms_out_of_range`), `requested`, `minimum` and `maximum`, and a message naming the range; an end the camera does not declare is `None` and is not checked. A frame width or height outside [`camera_limits.frame_min`, the scope's maximum / the current binning] is refused the same way from `set_frame_size` (`frame_width_out_of_range` / `frame_height_out_of_range`); the scope's maximum is `camera_limits.frame_max`, unbinned whatever binning the camera held at connect: the sensor, or the model's smaller one where its lens images less of the sensor (1700 on the LS560). When it is `None` the maximum is unknown and is not checked. Inside the range every camera delivers exactly the size asked for: it acquires the next window up on its own grid and crops back. The camera's `set_frame_size` floors the size to even sides first (`get_pixel_alignment()`, `{2, 2}` on every camera) and refuses before it stores. With no camera connected each raises `HardwareCommandRefusedError` (`not_connected`, `MissingPart.CAMERA`; see the dispatch paragraph below), not `CameraSettingRejected`. So does every camera command whose camera was declared removed (`scope.camera.camera_removed`) by the time it failed or, for a write, returned, with the command's own fault as its `__cause__` and nothing recorded; an off (`disable_streaming`, `set_auto_gain(False)`, `set_auto_exposure(CameraAutoMode.OFF)`, `lock_auto_gain`) is satisfied instead, as with no camera. A camera is declared removed only on definite evidence that it is gone (its SDK's removal callback, a grab that finds no device, a run of failed grabs, and on the Classic (FX2) body a failed write or stream start after which the device is absent from the USB bus on two probes 0.5 s apart). On the Basler and IDS bodies that happens on another thread: one failed write is that write's `CameraSettingRejected` with the camera still connected, so a write its cable was pulled during can first raise `CameraSettingRejected`, and the next command is refused `not_connected`. On the Classic body the failed command itself asks the bus, so a setting written to, or a stream started on, a removed camera is refused `not_connected` at once, streaming or not. A driver with no confirmation signal is deliberately **not** a rejection and does not raise: it answers `None`, meaning "cannot confirm", not "refused". On the Classic (FX2) body the sensor register write raises rather than reporting a refusal, so a failed apply there is a `CameraSettingRejected` chained to the driver's exception. The camera's `apply` writes the gain, the exposure and the auto-gain change even when one is not taken, and then raises one `CameraSettingRejected` naming each that was not (`setting` comma-separated, `camera_no_answer` when any went unanswered).
- **Hardware-command dispatch** (LED, motion, and camera commands): each command submits to one of the scope's own lanes -- IO for LED and motion, CAMERA for camera -- and blocks until the hardware has it. Every `Lumascope` builds and starts its two lanes, so commands from every caller run one at a time per bus, in order. While a run, a diagnostic or a home holds the scope (or a lane is disabled), the command raises `HardwareCommandRefusedError` (`modules.exceptions`), carrying the machine-readable `reason` (`exclusive_activity_running`) and the refused member. Once `session.shutdown()` has shut the lanes -- the first step of the scope's own `disconnect()`, before the boards are released -- every command raises at once with `reason` `scope_disconnected`, `stop_motion` included. While a control session is held (see "The control session", below) a command that is not its holder's raises it with `reason` `control_session_held`, naming the session in `control_session`. These are the lanes' refusals, every hardware command's, so no member's `Raises:` section lists them, nor the control session's refusal of a write: every member that writes a setting, a protocol file or a recording's pending frames -- `update_settings` and every member that stores a setting, `save_protocol`, `recover_file_writer`, `discard_close_drain`, `manual_recording.discard_pending`, `repair_camera_driver` -- raises `ControlSessionRefusedError` (`control_session_held`) while a control session is held and the call is not its holder's, before it writes anything; the display preferences (`show_tooltips`, `show_step_locations`, `live_view_fps`, `stimulation_enabled`, `focus_score_enabled`) are written by anyone; a support report and a logs zip are made by anyone, since each writes a support artefact and changes no scope state; and a post-processing build is open to anyone. A member's own gate that refuses with `exclusive_activity_running` (a plate change during a run, a home) is in its section. A motion, LED or camera command for hardware the scope does not have raises it too, before anything moves, lights or is written, and its `missing` (a `MissingPart`, `modules.exceptions`) names the part, its words coming from it: `reason` `not_connected` when the model has a motor controller and none is connected -- never came up, or its cable pulled -- (`MissingPart.MOTOR_CONTROLLER`, "Check the USB cable ..."), or when no LED controller is connected (`MissingPart.LED_CONTROLLER`), or no camera ("The camera is not connected.", `MissingPart.CAMERA`: every camera command, the camera's diagnostic runners, `add_frame_listener` and a manual still's Future included); `axis_absent` when the scope does not have the part: `MissingPart.FAN_CONTROL` and `MissingPart.LED_ENGINEERING_MODE` for the diagnostics' fan and LED engineering mode, `MissingPart.X`, `Y` or `Z` for an axis it lacks, `MissingPart.TURRET` for a turret, and `MissingPart.MOTORS` on a manual scope (LS620, LS560), for every motion command, `stop_motion`, `home()` and the acceleration limit included; `MissingPart.led(...)` for an LED the model lacks, named as the command named it ("This microscope has no Red LED.", "... no LED on channel 2."); `MissingPart.layer(...)` for a layer the model lacks that a layer's settings, its focus or a still names ("This microscope has no Lumi layer."). A name that is no axis at all is refused first, on every model, with `ArgumentRefusedError` reason `axis_unknown` (see the arguments paragraph below). An LED off whose end state already holds is not refused: an off of a channel the API holds dark, of an LED the model lacks, or with no LED controller and nothing believed lit returns with nothing written and nothing logged. An LED command whose own write is what finds the LED controller gone (on the Classic the LED shares the camera's USB device, which tells nothing when unplugged) answers as it would have with the controller known gone: refused `not_connected` with the device's fault as its `__cause__`, or an off satisfied when nothing is believed lit; a channel it could not darken stays believed lit. So with no camera does a camera off, with the answer it gives with a camera: `disable_streaming` (None), `set_auto_gain(False, ...)` and `set_auto_exposure(CameraAutoMode.OFF)` (True), and `lock_auto_gain` (an `AutoGainLock` whose `state` is None: no arm can stand). `remove_frame_listener` never needs a camera. There is one form of each command; a caller that must not wait runs it on its own thread.
- **Sentinel-return methods log** at `logger.warning` or `logger.info` per Rule 5; they do **not** fire user notifications (no actionable failure occurred -- the value is just unknown).
- **`camera_connected` is an instantaneous, non-latching poll.** A `False` can be transient (a single flaky connectivity query on an otherwise healthy camera). Consumers may skip work on `False` and re-poll on their next cycle; they must never latch, self-cancel, or tear anything down on it -- one transient `False` on a multi-day run should cost one skipped cycle, not the rest of the session.
- **`scope.camera.camera_removed` is the latched answer.** It is True once the camera's driver has declared it removed (unplugged, off the bus), and stays True until the camera connects again. Unlike a `False` from `camera_connected`, it is never transient, so it is safe to act on. A capture that meets a removed camera is refused `HardwareCommandRefusedError` (`not_connected`), and a run that meets it ends at once, `status='failed'`, `reason='hardware_disconnected'`, in words naming the part gone. False on a scope with no camera at all.
- **Captures return a frame or raise.** `capture_and_wait` and `get_image` never answer None. A capture that got no frame raises `CaptureFailedError` (`modules.exceptions`, a `CaptureError` fault), its `reason` one of `no_frame` (the camera delivered no new frame within its grab window), `deadline_expired` (the scope's state kept changing for longer than the capture's budget), `no_fresh_frame` (`capture_and_wait` only: frames came, but none within the grab's bound counted for a change by its source's rule, `capabilities.frame_rules`; its `setting` names the change as a person says it, `'the focus move'`) or `chunk_rejected` (frames came, none exposed at the `setting` asked, `'exposure'` or `'gain'`), its title and words the reason's. A camera that is not there, absent at the call or removed during it, is refused `HardwareCommandRefusedError` (`not_connected`), and one connected but not streaming `camera_not_streaming`. Nothing inside the API reports either: the raise is the report, yours to show.
- **A captured image is read-only.** The array `capture_and_wait` and `get_image` return refuses a write (`ValueError`), whichever path made it: the camera's own frame, a sum, an 8-bit rendering or a scale-bar copy. It is the instrument's account of the frame; to process it in place, work on `image.copy()`.

If you are writing a new wrapper, the `Raises:` section is the canonical declaration of which contract applies: its entries, or `Nothing.`.

---

## Lumascope composition root

The `Lumascope` class is the **hardware-composition-root**. It constructs and holds the seven sub-APIs (`scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, `scope.runtime_state`, `scope.protocols`), wires them together, and owns lifecycle (connect / disconnect / emergency shutdown). `scope.protocols` holds the two `Protocol` constructors, documented under Running protocols.

**How you reach it:** `session.scope`, the scope `ScopeSession.create` built for the session. The GUI, ScopeSession, and REST surface all go through this class.

### Initialization

```python
from modules.scope_session import ScopeSession

session = ScopeSession.create(settings)                    # real hardware
session = ScopeSession.create(settings, simulate=True)     # simulated: the model is settings['microscope']
session = ScopeSession.create(settings, source_path='/path/to/lvp')  # another data folder
scope = session.scope

scope.labware                             # every plate, by name: WellPlate records, read-only
scope.objectives                          # every objective, by id: Objective records, read-only
scope.scope_models                        # every model, by name: ScopeModel records, read-only
scope.scope_models['LS850T'].has_turret   # one model's record: True, the LS850T has a turret
```

A scope acts on configuration it does not hold: the labware, the stage offset, the turret map, the objective selected on a scope with no turret, and whether the scale bar is drawn are its session's settings, read whenever it acts on them. A `ScopeSession` binds the scope it builds to its settings.

```python
list(scope.objectives)                              # every objective id in the catalogue
scope.objectives['10x Oly'].focal_length_mm         # one objective's record; every field is named with its unit
list(scope.labware)                                 # every plate name in the catalogue
scope.labware['96 well microplate'].spacing_x_mm    # one plate's record: rows, columns, spacing_x_mm, ...
scope.runtime_state.get_labware()                   # the selected plate's record
```

`settings_template` hands out the caller's own copy, and `labware`, `objectives` and `scope_models` each a read-only mapping of frozen records: changing what you were given changes nothing the scope or any other reader sees, and an `Objective`, a `WellPlate` or a `ScopeModel` refuses a change outright. A plate's positions are in mm on the plate: `offset_x_mm` / `offset_y_mm` is the first well's centre and `spacing_x_mm` / `spacing_y_mm` the pitch between centres; `well_diameter_mm` is None where the catalogue states none.

An `Objective` names each field with its unit. Autofocus starts at `autofocus_step_max_um` and refines down to `autofocus_step_min_um` across `autofocus_half_range_um` either side of where it starts; a jog moves `jog_z_fine_um` / `jog_z_coarse_um` on Z and `jog_xy_fine_um` / `jog_xy_coarse_um` on X and Y; `focal_length_mm` sets the image scale; `short_name` is the token step and file names carry. `focus_peak_fwhm_dense_um` is None until the engineering plugin's characterization has measured the objective. The installation's `objectives.json` keeps its own keys (`focal_length`, `DOF`, `AF_min`, ...), and a capture's metadata carries the objective under those keys. An entry missing a field, carrying a key no field declares, or giving a number as anything else refuses bring-up with `InstallationFileError`, naming the file, the objective and the key.

A `ScopeModel` says what a model is built with: `has_focus`, `has_xy_stage` and `has_turret` (the Z, X and Y, and T axes), its boards (`led_board`; `motor_board`, None on a model with no motor axes), its `filterset`, and its `layers`, `LayerRecord`s resolved against the release's layer vocabulary, which a unit has when its board carries no Channels block of its own. A Classic model, which has no motorconfig, also declares its optics, `pixel_size_um` and `lens_focal_length_mm`, and one whose lens images less than its sensor its `max_frame`; each is None where the model declares none. The installation's `scopes.json` keeps its own keys (`Focus`, `XYStage`, `Turret`, `LEDBoard`, ...). A model entry or layer row missing a key, carrying a key no field declares, or giving a key a value it cannot hold refuses bring-up with `InstallationFileError`, naming the file, the model and the key.

Slides and Petri dishes are in `labware.json` but are not labware the API publishes yet: the labware catalogue is the multi-well plates.

`source_path` is the folder holding `data/`; without one the scope uses the installation's own. The scope reads `data/labware.json`, `data/objectives.json`, `data/scopes.json` and `data/motorconfig_defaults.json` from it once, first, before anything starts, and everything that asks about plates, objectives or models reads the scope's copy: the scope's runtime state, protocol construction and validation, autofocus, the run, the session and the GUI. The model catalogue (`scopes.json`'s `Models`) is `scope.scope_models`, a read-only mapping of model name to its `ScopeModel`. A file that is missing, unreadable or not the shape its reader needs (for `scopes.json`, one with no `Models` section) raises `InstallationFileError` (`modules.exceptions`) naming the file and its folder, and nothing is left running. A simulated scope's `sim_model` that the folder's catalogue does not list raises `ScopeModelUnknownError` (reason `model_unknown`, a refusal) before anything starts. The release's layer vocabulary (`scopes.json`'s `LayerOrder`) is the one exception: it is process-wide, read from the installation's own folder the first time any code asks for the layers, and a scope's layers are resolved against it. It is not a `ConfigError`: the installation is at fault, not your settings.

The scope builds and starts its own IO and CAMERA lanes at construction and shuts them when its session shuts down (`session.shutdown()`). A lane hands a finished command's callback to the process's one UI dispatcher (`ScopeSession.set_ui_dispatcher`, below); with none set, the default, callbacks run on the lane's worker.

### Layer identity

Every scope resolves its **layer identity** at construction — what the layers on this unit ARE: stable key name, display name, LED board address, excitation wavelength, plus the unit's filterset. It resolves from the unit's own configuration when one exists, else from the model's entry; a motor-reported model always outranks the selected one (`settings['microscope']`; hardware truth beats a selection), and the selection serves models whose hardware cannot report one (the Classic/FX2 line). A simulated scope reports the model it was DECLARED with, so the simulated motor board below reports `LS560` and the layers resolve exactly as that model's catalogue entry says.

```python
session = ScopeSession.create(settings, simulate=True)   # settings['microscope'] is 'LS560'
scope = session.scope

identity = scope.layer_identity          # immutable snapshot
identity.source                          # 'motorconfig' | 'scopes' | 'unresolved'
identity.model                           # the model it resolved as: the board's report, else the selected model; None if neither
identity.filterset                       # the unit's filterset identity string
for layer in identity.layers:
    layer.key_name                       # stable name: settings keys, protocol Color, filenames
    layer.display_name                   # the unit's name for the layer (e.g. 'BF-Phase' on FX2); None for a fluorescence
                                         # channel a board's Channels block names by its excitation alone
    layer.label                          # what to show: a fluorescence layer's excitation ('405 nm'), else display_name
    layer.modality                       # 'transmitted' | 'fluorescence' | 'luminescence'
    layer.background                     # 'bright' | 'dark': what an overlay is drawn against
    layer.display_color                  # the unit's colour for the layer, '#rrggbb': its board block row's, else the
                                         # release's; the colour used is session.channel_colors(), which applies the override
    layer.led_channel                    # tuple of board addresses; () = drives no LED
    layer.excitation_nm                  # excitation wavelength; None for broadband/LED-less

identity.find('BF')                      # record by stable key name, or None
```

A unit's own channels are its motor board's `Channels` block in `motorconfig.json`, beside its `Filterset`: a mapping from each channel's key to its row, `led_channel` (the LED address, null for none) and, for a fluorescence channel, `excitation_nm`, or for any other, `display_name`, each with an optional `display_color` (`'#rrggbb'`, lower case, brightest component ff). Its keys are the release's. A row this release cannot read is left out, and bring-up reports every such row in one notice, `ChannelRowsSkippedNotice` (`'channel_rows_skipped'`, `skipped`: each key and why). A block that is not a mapping, or that names one LED address twice, leaves the scope no channels, reported as `ChannelBlockUnusableNotice` (`'channel_block_unusable'`). A board carrying the old `Layers` block, or a `Filterset` with no `Channels`, resolves from its model, its channels and filterset both, reported as `ChannelBlockSupersededNotice` (`'channel_block_superseded'`).

A scope's model is fixed for its lifetime: `identity.model`, `scope.capabilities.model` and the model every saved image records are one answer. A model the catalogue lacks is still carried, with no layers. To change the model of a scope that cannot report its own (the FX2 line), save the selection through the session; it applies the next time the scope is brought up, and a motor board that reports its own model still wins then:

```python
session.select_model('LS560')   # saved to settings['microscope']; raises ScopeModelUnknownError if the catalogue lacks it;
                                # a model that is not the one running is reported once as a ScopeModelDeferredNotice
                                # (reason 'scope_model_deferred', solicited) to every outcome listener
session.model_at_next_start     # 'LS560' until the next bring-up; None when the saved model is the one running
```

A scope with no resolvable identity carries the empty `'unresolved'` snapshot: LED commands then raise a named error rather than guessing. Names accepted by `scope.illumination` are the `key_name` values.

A session configures its scope for you: `ScopeSession.create` applies the runtime configuration (frame size, objective, binning, stage offset) from the settings before it returns (see "ScopeSession session layer").

### Connection

```python
scope.unconnected_parts()                 # ('LED controller', 'motor controller', 'camera') -- those not up; () when all are
scope.motor_connected                     # motor board (property)
scope.motion_expected                     # False on a manual scope (LS620, LS560): no motor board to connect
scope.led_connected                       # LED board (property)
scope.camera_connected                    # camera (property)
scope.camera.camera_removed              # True once the camera was declared unplugged (latched until it reconnects)
scope.no_hardware                         # True if all-null (no real hardware found)
```

### Objective management

The objective sets the pixel size stamped into every capture, so the Session owns it: whether it is unknowable, how it is confirmed, and the plain writers. On a scope with a turret, the active objective is the one assigned to the slot in the light path, derived on every read: a turret move changes it, and no copy of it is stored beside the slot map. `session.objective_id` and `get_settings_snapshot()` answer it (None while it is unknown); the stored id is a turretless scope's selection, which a turret scope does not read. Which glass is in which slot is written only by `assign_turret_objective` and `clear_turret_objective`; on a scope with no turret every slot writer, and `confirm_objective` given a `turret_position`, is refused with `HardwareCommandRefusedError` (`axis_absent`) before the slot or the name is judged. With no turret, the active objective is the selected one, and `select_objective` moves the settings store and the scope's runtime state together. Every writer refuses an id that is not exactly a catalogue key with `CatalogueNameRefusedError` (reason `objective_not_in_catalogue`), and None as an argument of another type (`ArgumentTypeRefusedError`), before any write; a stored None read back is `ObjectiveUnknownError` (`none_selected`). The resolved optics (`[Optics   ] objective=... -> N um/px`) are recorded in the log once each time the active objective changes -- after a selection, an assignment or a turret move -- the next time it is read, so before any capture stamps it.

When no one can say which objective is in the light path, it is unknown, never a stored guess: on a turreted scope, before the turret has been homed or moved since bring-up, when its slot has no assignment, or when the assignment is not in the catalogue; with no turret, before anything was selected. `scope.runtime_state.resolve_current_objective()` then raises `ObjectiveUnknownError` (a `ConfigError`; `.reason` is `'slot_unknown'`, `'slot_unassigned'`, `'not_in_catalogue'`, `'none_selected'` or `'turret_undecided'` -- the scope has not yet recorded whether it has a turret, and `.slot` is the slot in the light path or `None`).

The plate decides every well position the program computes, so the Session owns it on the same terms: `select_labware` moves the settings store and the scope's runtime state together, or moves neither. It refuses -- before either store is written -- a name the labware catalogue cannot resolve with `CatalogueNameRefusedError` (reason `labware_unknown`), and settings with no usable `protocol` block to hold the selection with `ConfigurationDefectError` (`settings_not_prepared`). Plate names that were renamed still resolve, so a protocol saved under an old name is accepted rather than refused.

Its return value reports whether the stored NAME changed, not whether the plate did: selecting a renamed plate under its old name while the new name is stored returns `True` and both names refer to the same plate. Both stores are written on every accepted call, including one that reports no change -- the settings key is not evidence about what the scope holds, so a caller that writes it first cannot make the selection skip itself.

A scope with no XY stage is on "Center Plate": it has one field and no wells to move between, so bring-up replaces any other stored plate with it and logs the replacement, every protocol made from the settings is born on it, and `session.select_labware` refuses any other plate there (`HardwareCommandRefusedError`, reason `axis_absent`, `missing` is `MissingPart.XY_STAGE`): a protocol on a plate with wells would image one field under every well's name. A protocol loaded there is put on Center Plate, as before.

A protocol has its own plate, and every position a step holds is stated against it. `session.set_protocol_labware(protocol, plate_key)` puts `protocol` on `plate_key` and returns the catalogue key it took; the scope's plate is `select_labware`'s and does not move. A plate the catalogue does not have raises `CatalogueNameRefusedError` (`labware_unknown`) and the protocol keeps its plate; a renamed plate is stored under its current key. On a scope with no XY stage the protocol takes "Center Plate" whatever was asked, and a different plate asked for is logged as replaced. LumaViewPro's plate list puts the scope and its protocol on the plate picked, the protocol only once the scope has taken it.

```python
session.set_protocol_labware(protocol, '6 well microplate')   # '6 well microplate'
```

A protocol loaded from disk is put on its own plate by one Session member: `session.load_protocol(file_path)` loads the file and selects the plate the file names through `select_labware`, and raises what either raises. A refused selection -- a run, a diagnostic or a recording holds the scope and the file names another plate -- refuses the whole load, and the scope stays on the plate it had. On a scope with no XY stage the protocol takes "Center Plate", through `set_protocol_labware`'s rule. The member sets the plate only: the protocol's period, duration and per-layer settings stay in the protocol, and the settings' stored schedule is not changed. To take a protocol's per-layer settings into the layer controls as well, call `session.apply_channel_settings(protocol)` after the load, as LumaViewPro's own Load does: every layer stops acquiring and stimulating, then each layer the protocol names takes its acquire mode and every value its row holds; a blank value leaves that control as it was, and a layer this release does not know, or this scope does not have, is logged and dropped. A script that loads a protocol only to run it does not need it. `protocol.layer_settings()` returns the rows typed: `Acquire` `'image'` or `'video'`; `Illumination`, `Gain` and `Exposure` floats; `Sum` an int; `Auto_Gain`, `False_Color` and `Stim_Enabled` bools; a blank cell None. A file whose Layer Settings block has a cell of the wrong type, or no `Layer` column, is refused at load with `ProtocolFormatError`, naming the file; a file with no block has its layer settings inferred from its steps: each layer with steps acquires (`'video'` when any of its steps records video), and a cell its first step cannot supply is None. A step value the run cannot use is reported by the step check -- a notice at load, a refusal when a run starts -- in a file with a block or without.

```python
question = session.objective_question()            # None, or ObjectiveQuestion(turret_position, proposed, choices)
if question is not None:
    session.confirm_objective(question.proposed, turret_position=question.turret_position)

session.select_objective('10x Oly')                # True when the objective changed; False for the one held.
                                                   # On a turret scope it turns the turret to the one slot holding it
                                                   # (move_turret, Z parked and restored) and writes no slot;
                                                   # ArgumentRefusedError 'objective_in_several_slots' (offered: the slots)
                                                   # or 'objective_not_on_turret' (offered: what the slots hold);
                                                   # the move's own refusals (an unhomed turret: AxisStateUnknownError)
session.select_labware('384 well microplate')      # True when the plate changed; a retired spelling is accepted and stored under its catalogue key; both stores are written either way
session.assign_turret_objective(2, '10x Oly')      # slot 1-4 (PositionOutOfRangeError otherwise)
session.turret_objective(2)                        # '10x Oly': what the slot holds, None for none;
                                                   # refused as the assignment is, for a scope with no turret or a bad slot
session.clear_turret_objective(2)
session.clear_current_turret_objective()          # clear the slot in the light path;
                                                   # ObjectiveUnknownError while that slot is unknown
```

On a turret scope `confirm_objective` requires `turret_position` (`ArgumentRefusedError`, `turret_position_required`): it records the glass in that slot and never turns the turret. The confirmation records which light path it was given for, and `objective_confirmed` holds `'turret'`, `'turretless'` or None (never confirmed); a scope brought up with the other light path owes the question again, and a turretless question then proposes the objective the scope last selected. A file from before the record holds `true`, which bring-up records as the light path the scope has, or `false`, read as None.

`objective_question()` is a read: it returns a question when no one has confirmed the objective for this scope's light path, or when the slot in the light path on a declared turret model has no assignment. The question names the live slot (`scope.motion.get_turret_slot()`) and proposes only that slot's assignment; an unknown slot alone -- as during every turret move -- owes no question, but on an install whose objective has never been confirmed an unknown slot raises `ObjectiveUnknownError` (`.reason` `'slot_unknown'`) rather than ask about a slot the turret may not be in. It may log one withheld-question line per call while a question is owed and suppressed (no hardware; provisional settings) -- a caller that polls it will see that line per poll. A configured session may still have a question to ask: the factories do not ask it. A turret move onto an unassigned slot does not ask either, so a headless caller asks `objective_question()` after the move.

`current_slot_objective_question()` is the same question for the slot in the light path, asked whether or not one is owed: to change what a slot holds, or to name a slot nothing is assigned to. It proposes the slot's assignment (the catalogue default when the slot holds a name the catalogue lacks), so an unchanged answer leaves the slot as it is, and its answer goes to `confirm_objective` with the slot the question names. It raises `ObjectiveUnknownError` (`'slot_unknown'`) while the slot is unknown. On a scope with no turret it asks which objective is installed: `turret_position` is None, it proposes the selected objective once one has been confirmed (the catalogue default before), and the answer goes to `confirm_objective` with no slot.

Labware / turret-config / stage-offset are runtime-mutable microscope configuration (not live hardware). The `scope.runtime_state` sub-API answers them, reading the session's settings, their one store, on every answer; it holds no copy and has no setter. L2 callers reach it through the composition root the Session exposes: `session.scope.runtime_state.*`. They are changed only through the Session: `session.select_objective`, `session.assign_turret_objective` / `session.clear_turret_objective`, `session.select_labware`, and `session.update_settings('stage_offset.x', ...)`. What a getter returns is a copy: changing it changes no setting.

```python
scope.runtime_state.is_turreted()                      # True when the objective is derived from the turret slot
scope.runtime_state.resolve_current_objective()        # (id, Objective), or raises ObjectiveUnknownError saying why
scope.runtime_state.get_pixel_size_um()                # µm per pixel at the current objective and the camera's binning; None when unknown
scope.runtime_state.get_field_of_view()                # FieldOfView(width_um, height_um) the camera's frame covers; None when unknown

# Turret integration
scope.runtime_state.get_turret_config()                # (in-process) {1: '4x Oly', 2: '10x Oly', 3: None, 4: None}, a copy;
                                                       # a wire client reads a slot as session.turret_objective(position)
scope.motion.is_current_turret_position_objective_set()        # False when the CURRENT turret slot has no configured objective

# Labware + stage offset -- the plate-coordinate inputs
scope.runtime_state.get_labware()                      # the plate the settings select, from the catalogue
scope.runtime_state.get_stage_offset()                 # StageOffset(x_um, y_um): where the plate is mounted on the stage
scope.runtime_state.get_well_label()                   # 'A1' for the current stage XY; '' when the labware has no wells or XY is off its wells

# Stage µm ↔ plate mm, one symmetric pair, on the selected plate at the
# live stage offset; labware_id= and stage_offset= state another (a
# protocol's positions are on its own plate, a run keeps the offset it
# started with). Raises ConfigError on a scope no session bound.
plate = scope.runtime_state.stage_to_plate(60000, 40000)      # PlatePosition(x_mm, y_mm)
stage = scope.runtime_state.plate_to_stage(50.0, 30.0)        # StagePosition(x_um, y_um)
stage = scope.runtime_state.plate_to_stage(50.0, 30.0, labware_id=protocol.labware())

# stage_to_plate bound to the plate and offset of this call (the same two
# arguments): a later change moves none of the positions it converts.
to_plate = scope.runtime_state.plate_transform()
plate = to_plate(60000, 40000)                                # PlatePosition
```

To MOVE to a plate coordinate, give it to `scope.motion.move_to_plate`
(or `start_move_to_plate`, which returns the started moves) rather than
converting first. It takes a `PlateLocation` in mm on the selected plate
at the live offset; an axis that is None is left where it is, a move under
way on it included, and the axes given are asked once and started on one
lane task, so a capture never takes the new X at the old Y. Each is checked
against what the stage can actually reach and refused in plate mm, naming
the coordinate you asked for, before either moves. A Z is refused
(`ArgumentRefusedError`, `plate_frame_axis`): a plate position has no Z.
`move_xy` and `move_z` take stage microns only.

```python
from modules.coord_transformations import PlateLocation
scope.motion.move_to_plate(PlateLocation(x_mm=None, y_mm=180.0, z_um=None))
# PositionOutOfRangeError: Y plate position 180.0 is outside the
# reachable range 1.48 to 81.48.
scope.motion.move_to_well('H12')   # the well's centre on the selected plate
session.go_to_bookmark(('X', 'Y'))  # the axes asked, as save_bookmark stored them
```

`move_to_well` refuses a label the selected plate does not have
(`well_unknown`). `session.go_to_bookmark(axes)` and its started form
`start_go_to_bookmark(axes)` move each axis given to `session.bookmark`, X
and Y in plate mm and Z in um, and leave the others where they are.

---

## ScopeSession session layer

GUI-free session container. All hardware commands route through executor threads for thread safety. Use this for scripts and automation.

**When to use:** You want to write a Python script that controls the microscope without the GUI.

### Setup

For **real hardware** with settings loaded from disk:

```python
from modules.scope_session import ScopeSession

# The user's configuration, prepared as the GUI prepares it: data/current.json
# read, checked against the shipped data/settings.json template, repaired and
# merged with the keys newer releases added. A root with no usable settings
# is refused (ConfigError), never quietly replaced by the template: a script
# has nobody to ask.
settings = ScopeSession.load_user_settings('.')
session = ScopeSession.create(settings=settings, source_path='.')

session.source_path                       # the scope's data folder (source_path above), a pathlib.Path
session.scope.labware                     # the scope's labware catalogue
session.scope.objectives                  # the scope's objective catalogue
```

`source_path` is the data folder `create` builds the scope on; leave it out and the scope uses the installation's own folder. The session's folder and catalogues are always its scope's, never copies, so nothing in one session can disagree about which plates or objectives exist.

The session comes back **configured** and **running**: `create` builds the scope, configures the scope (turret slot keys normalized, the stored objective selected on a scope with no turret, labware selected, the frame, binning and image mode applied), releases the camera start gate — so a capture works at once — and starts the executor lanes: the scope builds and starts its own IO and CAMERA lanes, and the factory builds and starts the FILE lane, the worker pool and the protocol thread around them. Each lane is started once, by whoever built it; starting a running lane again raises `RuntimeError`. `session.shutdown()` is the teardown for everything the factory built (see "Cleanup"): lanes and their threads down, LEDs off, motion stopped, scope disconnected.

`create` takes the host's injections as named keyword arguments; every one of them is optional, and the headless form passes none of them.

```python
session = ScopeSession.create(
    settings=ScopeSession.load_user_settings('.'),
    source_path='.',                        # refused beside scope=: a session's folder is its scope's
    simulate=False,                         # True builds a simulated scope instead of opening hardware
    engineering_mode=False,                 # the session's mode to begin with; a plugin may turn it on
    no_engineering=False,                   # True: the engineering plugin is not loaded
    settings_file=ScopeSession.user_settings_file('.'),  # where accepted writes are saved; None saves nothing
)
```

A display that follows Z, a run's moves and an autofocus sweep's included, subscribes to `scope.motion.add_position_listener` (below): it is told on every move, the give-up restore included, and every poll, with the polled position, and `scope.motion.get_target_position(axis)` answers where the axis is going. No run or autofocus member takes a display callback.

Bring-up asks the motor board which model it is before anything else. When the board reports a model the catalogue (`scopes.json` `Models`) knows and it differs from `settings['microscope']`, the reported model is WRITTEN into the settings dict you passed and logged — hardware truth outranks the stored selection, so your dict can come back changed. A reported model outside the catalogue, or none at all (no motor board), leaves the stored model alone.

To hear what bring-up reports (a camera not found, a partial-hardware warning), pass your outcome listener to the factory: `ScopeSession.create(..., outcome_listener=on_outcome)` registers it before the scope is built. See "Outcomes" below.

**The UI dispatcher.** A host with a UI thread tells the process once how to reach it, before it builds a session: `ScopeSession.set_ui_dispatcher(UiDispatcher(schedule=..., thread=...))`, with `UiDispatcher` from `modules.kivy_utils`. `schedule` has `Clock.schedule_once(func, dt)`'s shape, and `thread` is the thread it delivers on. It is one setting for the process, not for a session: every lane's completion callback, every run event but `frame_captured` of a run whose `RunEvents` says `on_ui_thread`, and every state listener the GUI marshals is handed to it at the moment it is delivered, whichever session or scope it belongs to. A host with no UI thread (a script, a REST server) never sets it, and a callback then runs on the thread that delivers it, its raise reported rather than raised; `set_ui_dispatcher(None)` restores that. An `Operation`'s `result()` or `exception()` made on `thread` while it is not yet done, outside a run's own `run_ended` handler, raises `BlockingCallRefusedError` (`wait_on_ui_thread`). A waiting form, a start followed by its operation's `result()` (`move_xy`, `move_z`, `go_to_step`, `run_application_session`, and each run's `run_` form), raises the same refusal on `thread` before it starts anything, so nothing is left running behind it.

```python
from modules.kivy_utils import UiDispatcher

ScopeSession.set_ui_dispatcher(
    UiDispatcher(schedule=Clock.schedule_once, thread=threading.main_thread())
)                                           # once, before the first ScopeSession.create
```

**Settings a factory needs.** Every dict `create` takes is prepared as the file's is, by the same preparation `load_user_settings` runs: a hand-built dict is completed from the shipped template, so a key it lacks takes the template's value at every depth and no reader hedges for it; a value in it the writer refuses (outside its range, of the wrong kind, not a number) is refused as a write of it would be, `SettingRefusedError` naming the path, since a dict handed to a session is a write of every key at once, where the file's such value is replaced and told; a block of the wrong kind (a `frame` that is not an object) is refused as the argument, `ArgumentRefusedError` (`wrong_kind`) naming `settings.<path>`. A dict the loader above already prepared goes through unchanged. Bring-up adopts the model the hardware reports into `settings['microscope']` whenever the catalogue knows that model, so the microscope key is an input the bring-up may correct. On a scope with no turret an `objective_id` the objective catalogue (`data/objectives.json`) does not hold is replaced by the shipped objective at bring-up, where the catalogue is first available, and told as part of the load's `stored_setting_replaced` notice (below). A stored plate (`settings['protocol']['labware']`) the labware catalogue cannot resolve -- an empty name, a plate it no longer has -- is replaced by the shipped plate at bring-up and told as part of the load's `stored_setting_replaced` notice (below); every other stored setting is kept. A null or a non-string in either place never reaches bring-up: the preparation above replaces it in a file and refuses it in a handed dict, as the writer would. After bring-up, every writer of the selection refuses a plate the catalogue does not have. A turreted scope does not read the stored `objective_id`: its objective is unknown until the turret is homed or moved to a slot, then it is that slot's assignment. `turret_objectives` keys may be JSON strings or ints; the factory normalizes them. A configured session may still owe the objective question (`session.objective_question()`, above); the factories do not ask it. A missing or unusable `labware.json`, `objectives.json`, `scopes.json` or `motorconfig_defaults.json` stops the scope's construction with `InstallationFileError` (see "Initialization").

For **simulated** (no hardware needed, development / CI):

```python
from modules.scope_session import ScopeSession

session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
```

`simulate=` is the one choice between simulated and real hardware, the same for every host: `simulate=True` wires up simulated drivers, `simulate=False` (the default) finds the real ones. Either way the factory configures the scope from settings and releases the start gate, so the session it returns can capture and save. `ScopeSession.load_user_settings(source_path)` reads the user's configuration the way the GUI does (`current.json`, then the shipped template); `source_path` must be an LVP installation root (a `data/` directory with `settings.json`), otherwise it raises `ArgumentRefusedError` (`not_an_installation`) naming the root; a `current.json` it cannot use raises `SettingsFileError`. In a process that has already prepared its settings -- the GUI's -- those are answered and `source_path` is not read. It checks the settings against the release's layer vocabulary, so an unusable `scopes.json` in the installation's own folder raises `InstallationFileError` naming it. Settings are a required argument of `create`, never read from disk behind your back: pass `load_user_settings(...)` or your own dict. `ScopeSession.user_settings_file(source_path)` names the file that read reads first, the installation's `current.json`; a host that keeps accepted writes passes it as `create(settings_file=...)`, as the GUI and the REST server do. It reads and creates nothing.

A camera stream that stops delivering while no removal of the camera has been reported is reported once per stall, unsolicited, to your outcome listener, in every host: `CameraStreamStalledError` (`modules.exceptions`, a fault), when no new frame has come for longer than a frame at the current exposure can take. That includes a camera whose grab stopped when nobody asked it to (a camera SDK that stopped its own grab, a grab thread that died, a start that did not take); a stop you asked for (`disable_streaming()`, a frame-size, format or binning change) is not a stall, and the check starts again from the stream's next frame. A pulled cable can meet the same report first, when the camera's driver reports the removal later than that; the words name the cable and never say the camera is still connected.

A simulated scope can also show a stalled camera stream: pass `sim_camera_stall=SimulatedStall(after_s=30, for_s=20)` (from `drivers.simulated_camera`) to `create`, and 30 s after the scope is built the simulated camera stops delivering frames for 20 s while it stays connected and streaming, as a real camera's link can. The GUI takes the same stall as a launch argument beside `--simulate`: `--sim-camera-stall=30,20`. It is refused on real hardware (`ArgumentRefusedError`, `needs_simulated_scope`) and on a model simulated with an FX2 (LS620, LS560), whose camera is not the simulated one (`needs_simulated_camera`).

A simulated scope's sample is in focus where bring-up leaves the stage. Pass `sim_focal_spread_um=25.0` to `create` and its focus is instead drawn once, within 25 um of there, so each session needs a real autofocus, as on a real scope. The GUI takes it as a launch argument beside `--simulate`: `--sim-focal-spread=25`; without it the GUI's simulated sample is in focus too. It is refused as `sim_camera_stall` is: on real hardware (`needs_simulated_scope`) and on a model simulated with an FX2 (`needs_simulated_camera`).

A simulated session can likewise show a save drive that stops answering: pass `sim_file_stall=SimulatedStall(after_s=60, for_s=120)` to `create`, and 60 s after bring-up the session's file lane is held for 120 s by one write that does not return (`simulated_stuck_write`). A run that ends while its files wait behind it is reported as a stalled file writer (`FileWriterStalledError`) once the write has been stuck for 30 s, with the recovery as its remedy. The GUI takes it as `--sim-file-stall=60,120` beside `--simulate`. It is refused on real hardware (`ArgumentRefusedError`, `needs_simulated_scope`).

Three more of `create`'s arguments arm a fault before bring-up, so what bring-up does with it is what a test sees. `sim_model='LS850T'` builds the simulated scope as that model while the settings declare another, as when a stored file disagrees with the scope attached; bring-up adopts the scope's model. `sim_model_answer=SimulatedModelAnswer('LS999')` (from `modules.lumascope_api.simulation`) has the motor board name that model as its own, whatever it was built as, and `SimulatedModelAnswer(None)` names none (`scope.simulation.answer_model`); a model outside the catalogue, or none, leaves the stored one standing. `sim_refused_stream_starts=1` refuses the camera's first stream start, bring-up's own (`scope.simulation.refuse_stream_start`), on a model simulated with an FX2. Each is refused on real hardware (`needs_simulated_scope`), and the last two by the scope on a board or camera that cannot show them (`not_injectable_on_board`).

### Application startup sequence

```python
started = session.start_application_session()        # home ALL axes, then position turret; returns an Operation
started.result()                                     # None once the motion has ended and the scope is free
session.run_application_session()                    # the same, waited for
session.run_application_session(disable_homing=True) # skip homing; no startup motion at all
```

`start_application_session()` is the single source of truth for the standard startup orchestration the GUI runs on launch: an all-axis `move_home` (firmware homes Z/T/X/Y in one routine; Z-only boards home what they have), then, when the scope has a turret, the T-axis to position 1 (where the home leaves it); the active objective is then slot 1's assignment. Headless / REST callers should use it rather than open-coding the home + turret sequence. It returns at once with an `Operation` (see "How a run ends"), whose `stop()` is `stop_motion` while the motion drives (refused `home_not_stoppable` while the home drives on a board that cannot stop one; the home runs to its end and nothing after it moves) and nothing once it has ended, so a late stop never halts what took the scope after it; `run_application_session()` is the same call, waited for, and refuses a call from an executor's worker, whose own queue or lane the motion may wait on, with `BlockingCallRefusedError` (`blocking_on_worker`; still a `RuntimeError`).

The scope is the startup's from before `start_application_session()` returns until the motion ends, as it is during any home: every other request is refused naming the home (`exclusive_activity_running` / `home_in_flight`), and `controls_locked` is True. The claim is released before the operation is done. A failed home is reported as an outcome and skips the turret move; the operation still settles with None. `disable_homing=True` skips every startup motion: the turret is left where it is, like the stage axes, and no turret position is recorded. Position it yourself after homing.

### Camera capture settings

The image mode, the binning and the frame each have one writer, on the camera (`scope.camera`): it applies the setting to the camera and saves it in the settings only once the camera took it, whoever calls it -- the GUI, REST, a run's restore, a plugin. A refusal or a rejection leaves the settings as they were, so they never describe a camera state that is not in force. The pixel format is set only through the image mode, so a capture's depth and its save policy always agree. Over REST these are `/api/v1/scope/camera/...`. Call them from the camera's own lane or any thread; each waits for the camera.

```python
camera = session.scope.camera
camera.set_image_mode('12bit_scientific')    # saved once the camera took it; SettingRefusedError (out_of_range) for an unknown mode;
                                             # CameraSettingRejected if the camera refuses the format
camera.set_binning(2)                        # the delivered FrameSize(width, height); the framed
                                             # region is kept and divided by the new factor.
                                             # CameraSettingUnsupportedError (a refusal) for a size
                                             # the camera does not list
camera.set_frame_size(960, 600)              # a displayed (post-binning) size at the binning the camera holds;
                                             # returns the FrameSize the camera delivered (the size,
                                             # floored to even sides); CameraSettingOutOfRangeError outside
                                             # the scope's range
camera.frame_at_binning(2)                   # the frame set_binning(2) will ask for; applies nothing;
                                             # refused as set_binning refuses the same size
camera.binning                               # the binning factor the camera took, e.g. 2; None when unknown
```

The saved frame always carries its unbinned region (`frame.native_width` / `frame.native_height`): the shipped settings hold it, and bring-up saves one that gives the frame the camera delivers. A setting the camera changed at bring-up -- a binning it does not offer, a frame above the scope's maximum, high conversion gain or line-noise reduction saved on for a camera without them (`CameraFeatureAbsentNotice`, saved off) -- is saved as the camera runs it and reported once. A conversion gain or line-noise value the camera refuses at bring-up stays saved and is reported once.

With no camera connected, `camera.set_binning`, `camera.frame_at_binning` and `camera.set_frame_size` raise `HardwareCommandRefusedError` (`not_connected`, `MissingPart.CAMERA`) and store nothing. `set_image_mode` stores the mode for bring-up to apply.

The image mode is a save policy, and every camera honours every mode. `'8bit'` reduces each frame to 8 bits. The three full-depth modes keep the payload at the depth the frame has: `'12bit_scientific'` (labelled "Scientific (full depth)") stores it right-aligned, `'12bit_scaled'` ("Scaled (full depth)") left-justifies it to fill its 16-bit container, and `'12bit_false_color_rgb'` ("RGB (full depth)") writes it as false-colour RGB. The ids keep their `12bit_` names: a camera with a 12-bit format is asked for it in a full-depth mode, and an 8-bit camera (LS560/620/720) delivers 8 bits in every mode. Bring-up runs the saved mode as saved.

A sum (`sum_count` > 1) is stored in a uint16 array on every camera, an 8-bit one included, so a full-depth file keeps every count: four 8-bit frames are a 10-bit sum. Under "Scaled (full depth)" a sum is left-justified like any narrow payload (a 4-sum on an 8-bit camera, tagged 10, is shifted by 6). Under "RGB (full depth)" a summed channel is coloured, and an unsummed frame from an 8-bit camera is written as the 8-bit file it is. A hyperstack whose planes mix 8-bit and 16-bit files is built at 16 bits: under scientific each plane keeps its counts; under scaled and RGB each fills its container, as its own file does.

### Reading and persisting configuration

```python
session.get_setting('BF')                      # one layer's settings, by its path, in API names
session.update_settings('BF.sum', 3)           # write one setting, from any thread
session.update_settings('video.max_fps', 30)  # a nested setting, by its dotted path
session.set_live_folder('/data/run7')          # where captures and runs are saved; relative means the installation's
session.set_protocol_filepath('/data/plate.tsv')  # the protocol the next start opens; '' forgets it
session.live_folder_path('ProtocolData/run1')  # a wire caller's name -> its absolute path inside the live folder; LiveFolderPathRefusedError when it leaves
session.live_folder_listing('ProtocolData')    # one level of the live folder: LiveFolderEntry(name, kind, size, modified) each, by its full name
snapshot = session.get_settings_snapshot()     # a consistent copy, taken under the lock
session.get_setting('stage_offset')           # a copy of one setting, by its dotted path; ArgumentRefusedError (not_a_setting) when absent;
                                               # a setting with its own read is read there: ArgumentRefusedError (has_member,
                                               # member naming it), get_setting('binning') for scope.camera.binning
session.scope.settings_template                # every setting there is, with its shipped value
session.scope.camera.set_high_conversion_gain(True)  # the camera takes it, then it is saved; a raise: neither (below)
session.scope.camera.set_line_noise_reduction(True)  # likewise for the line-noise filter
session.set_scale_bar(True)                   # whether captures draw the scale bar: the setting the imaging API reads
session.set_acceleration_limit(80)            # a whole number 1-100, else AccelerationLimitRefusedError (a ValueError); the motors take it, then it is stored; with no motor controller HardwareCommandRefusedError ('not_connected' / 'axis_absent'); nothing stored on a raise
session.save_bookmark(('X', 'Y'))             # the live position as the bookmark: X/Y plate mm, Z um
session.save_all_bookmarks()                  # the live Z as the Z bookmark and every layer's focus
session.settings_are_provisional()            # True while current.json is unread and undecided: nothing is saved
session.retire_rejected_settings()            # resolve it: retire the unreadable file, the session saves again; the retired file's Path
```

The settings dict IS the configuration surface: its storage keys are the
API names, so what you read is what you write. The session owns the dict
and the lock that guards it. Long-running work should take one
`get_settings_snapshot()` at entry and read from that rather than the live
dict.

**A setting keeps the value asked for when the hardware delivers less.**
An LED current is sent as the board's nearest step: `led_on` returns that
step and the illumination API holds it as the lit current, while the
setting keeps the request. A layer's gain or exposure above the attached
camera's maximum is applied as that maximum
(`scope.camera.applied_gain_db_for`, `applied_exposure_ms_for`) and the
apply logs both numbers, while the setting keeps the request, so a camera
that can take the value gets it back.

**`update_settings(path, value)` is the one write.** `path` names one
setting by its keys joined with dots (`'BF.sum'`, `'zstack.step_size'`,
`'video.max_fps'`); the settings that exist, and the kind each holds,
are `session.scope.settings_template`, which holds a block for every
layer the release catalogues: a layer the shipped `settings.json` names
no block for takes its modality's shared one from the file's
`channel_defaults`, which the read consumes, so no template, settings
dict or saved file holds it. Two settings act on the whole
process as they are written: `logging.default.level`, the level of the
main log on every host (applied when the session is created, before its
scope comes up, and on each write; the installation's `debug_mode` puts it
at DEBUG whatever it says), and `logging.autofocus`, whether each
focus score is logged. A write to the shown layer's `gain_db` or
`exposure_ms` (or, while the shown layer is on auto-gain, to the
`protocol.autogain` and `ag_ae_max_exposure_ms` bounds) is on the camera,
and to its `illumination_ma` on its LED while it is lit, when the call
returns; a hidden layer's, or one made while a run, a diagnostic or a home
holds the scope, is stored only, and a run's end puts the layer shown
before it back on the camera from the store. The write is taken under the lock,
or refused with `SettingRefusedError` (from `modules.exceptions`) and
nothing is written. It carries the refused setting's `path`, and for
`has_member` the `member` to call instead; a REST problem carries both.
Its `reason` says why:

| `reason` | Refused when |
|---|---|
| `has_member` | the setting is changed by its own member, the one a client calls, named in `member` by its path from the session (`scope.camera.set_frame_size` for the camera's; over REST a setting with a read of its own is written by `PUT` on that read's URL, `/api/v1/scope/camera/frame_size`, "Routes" below): `microscope` (`select_model`), `live_folder` (`set_live_folder`), `protocol.filepath` (`set_protocol_filepath`), `objective_id` (`select_objective`), `objective_confirmed` (`confirm_objective`), `turret_objectives` (`assign_turret_objective`), `protocol.labware` (`select_labware`), `image_mode` (`scope.camera.set_image_mode`), `binning` (`scope.camera.set_binning`), `frame` (`scope.camera.set_frame_size`), `camera.high_conversion_gain` (`scope.camera.set_high_conversion_gain`), `camera.line_noise_reduction` (`scope.camera.set_line_noise_reduction`), `scale_bar.enabled` (`set_scale_bar`), `motion.acceleration_max_pct` (`set_acceleration_limit`), `bookmark` (`save_bookmark`, `save_all_bookmarks`), a layer's `acquire` (`set_channel_acquire`), `auto_gain` (`set_channel_auto_gain`) and `focus` (`save_channel_focus`) |
| `installation_only` | the setting is read only from the installation's settings file, so no caller can reconfigure the scope it drives: the `rest_api` block, `mode`, `lvp_lock_port`, `profile_trace_output_dir`, and the debugging switches `debug_mode`, `cprofile_enabled`, `profile_trace_enabled`, `tracemalloc_enabled`, `memory_profile_enabled`, `memory_profile_interval_s` and `fx2_debug_wire_enabled` |
| `derived` | the setting is derived from the release when the template is read, never a person's: `step_channel_order`, from the catalogue's layer order. A stored copy that differs is replaced at load and told, and one in a settings dict handed to `ScopeSession.create` is refused |
| `not_a_setting` | no setting has the path |
| `block` | the path names a block of settings (`'video'`); each is written by its own path |
| `wrong_kind` | the value is not the kind the setting holds: true/false, a finite number (int or float), text, or a list. NaN and an infinity are not numbers here, on every number setting, ranged or not. A setting shipped as `null` takes null or the kind it holds when set: text for a layer's `acquire` and `display_color`, a turret slot's objective and `profile_trace_output_dir`; a finite number for a layer's `focus` and `profiling.metrics_interval_s`; `'turret'`, `'turretless'` or true for `objective_confirmed`. A stored value of another kind is replaced at load. A numpy scalar, alone or anywhere in a list or dict (and in a settings dict handed to `ScopeSession.create`), is taken as the Python value it holds and stored as that |
| `out_of_range` | `simulator_tier` not one of `fast`, `firmware`, `realistic`; `zstack.position` not one of `'Current Position at Top'`, `'Current Position at Center'`, `'Current Position at Bottom'`; `logging.default.level` not the name of a log level, in any case (`'debug'` is taken and applied as DEBUG); `video.max_fps` outside 0 to 200 (0 is no cap); `video.max_duration_s` outside 1 to 3600; `tiling_overlap_percent` outside 0 to 50; `image_output_format.live` / `.sequenced` not a format the writer takes; `protocol.period` (minutes) or `protocol.duration` (hours) one a protocol cannot run, the protocol's own range (below); `jpg_quality` not a whole number from 1 to 100; `rest_api.port` not a whole number from 1024 to 65535, in a dict handed to `create` or a stored file (no write reaches it: `installation_only`); `live_view_fps` below 0 (0 is uncapped); `ag_ae_max_exposure_ms.transmitted`, `.fluorescence` or `.luminescence` (the longest exposure auto-gain may drive that channel class to, in ms) not above 0; `zstack.step_size` or `zstack.range` below 0 (0 is not configured: a stack with it is refused when it is built); `protocol.autogain.target_brightness` not above 0 or above 1; `protocol.autogain.min_gain_db` or `max_gain_db` below 0 (each is written alone, so a minimum above the maximum is stored, and refused when auto-gain is armed with them: `ArgumentRefusedError` `gain_range_inverted`); `profiling.handle_trace_obj_sample_every` not a whole number of at least 1; a layer's `exposure_ms`, `video_config.fps` or `video_config.duration` not above 0, its `gain_db` or `illumination_ma` below 0, its `sum` not a whole number of at least 1, its `composite_brightness_threshold` outside 0 to 100, its `video_config.duration` above 3600 s, its `stim_config.frequency` outside 0.5 to 6 Hz, `stim_config.pulse_width` outside 5 to 25 ms, `stim_config.pulse_count` not a whole number from 1 to 300, `stim_config.illumination_ma` below 0. These are the ranges that hold whatever hardware is attached. One hardware limit is refused here too: a layer's `illumination_ma` or `stim_config.illumination_ma` above the attached LED board's maximum (`capabilities.led_max_ma`), since the board refuses to drive it. A layer's gain or exposure is not held to the camera's range at the write: it is stored as written, the intent, and the camera is driven at its own limit when the layer is applied ("What a stored setting applies as") |

`protocol.period` and `protocol.duration` are the schedule a new protocol
starts from. A `current.json` written before the range was enforced can
hold one.

At start-up, every stored setting is held to the writer's rule, and a value
it would refuse -- the wrong kind (NaN included) on any setting, or outside
a range above, a protocol schedule, a `motion.acceleration_max_pct`
outside 1 to 100, a `binning.size` that is not a positive square `NxN` label,
an `image_mode` that is not an image mode, a layer's `acquire` that is not
`'image'`, `'video'` or null, an `objective_id` the catalogue lacks on a scope with no turret, or a plate the
catalogue cannot resolve -- is replaced for
that key alone by the shipped value, and the session reports one notice
`stored_setting_replaced` once the scope is brought up (or once its bring-up
fails), naming each replaced setting, its saved value and the one now in its
place (`replacements`: a list of `(path, saved, used)`).
A settings dict handed straight to `ScopeSession.create` holding an
acceleration limit outside 1 to 100 is refused with `SettingRefusedError`
(`out_of_range`, a `ConfigError`) before anything is commanded.

`set_live_folder(folder)` stores the live folder as it is at start-up: a
folder given relative to the installation is made absolute, and the folder
is created. A folder that is not a path (a NUL byte) is refused with
`SettingRefusedError` (`out_of_range`) and nothing is written.

**A path from a wire caller is a name under the live folder.**
`session.live_folder_path(name)` answers the absolute path `name` names
inside the live folder, and is what a REST bridge passes every path a
caller gives through, so a remote caller reaches the live folder and
nothing beside it: `live_folder_path('ProtocolData/run1')`. It raises
`LiveFolderPathRefusedError` (`modules.exceptions`, a refusal and a
`ValueError`, carrying `name`) with reason `outside_live_folder` when the
name is empty, absolute, carries a drive (`C:x`, `C:\x`) or a network
share, or leads outside the live folder through `..` or a link -- in the
forms either operating system writes, wherever the scope runs -- and with
reason `capture_location_unusable` when the live folder itself is missing
or not a folder; it never creates the live folder. A Python caller on the
scope's own machine passes any path straight to the member it calls.
Every parameter that takes a path is annotated `FilePath`
(`modules.api_surface`, a `str` or `os.PathLike`), so a bridge can tell
which arguments to pass through `live_folder_path` from the signature.

`session.live_folder_listing(name='.')` answers what the folder `name`
holds, one level, sorted by name, as a tuple of `LiveFolderEntry(name,
kind, size, modified)`: `name` the entry's full name under the live
folder, `/`-separated on every host, which any path parameter and a
further listing take as it is; `kind` `'file'` or `'folder'`; `size` in
bytes; `modified` a timezone-aware local `datetime`. `'.'` lists the live
folder itself. An entry gone by the time it is read (a file a run renames
into place, a link to nothing) is not listed; a link that leads outside
the live folder is listed, and refused where its name is used. `name` is refused as
`live_folder_path` refuses it, with reason `not_found` when it names
nothing, and with reason `not_a_folder` when it is a file.

**A path that cannot be used is refused as a request.** A member that
reads or writes a path the caller gives refuses it before it starts, with
`LiveFolderPathRefusedError`: reason `not_found` when nothing is there to
read, `not_a_file` when a folder is where a file is expected,
`not_a_folder` when a file is where a folder is expected, and
`wrong_extension` when a file to write is named without its kind's
extension (`load_protocol('runs')` on a folder is `not_a_file`;
`make_logs_zip` into a file is `not_a_folder`; every post-processing member
given a missing folder is `not_found`; `save_protocol(protocol, 'plate1')`
is `wrong_extension`). A writer given a file's name (`save_protocol`)
writes exactly that name and adds no extension to it, so
the name a Save As dialog checked for a file to replace is the name
replaced. A writer (`save_protocol`,
`make_logs_zip`, `make_support_report`) makes a
missing folder where it was sent. Over REST each refusal is a 422. A fault
(`ProtocolNotLoadedError`, `ProtocolNotSavedError`,
`SupportReportNotSavedError`) is left for a path that is usable and still
failed: a file that is there and cannot be read, a disk that refuses the
write.

Writing into the settings dict you passed to `create` directly skips every check above and the
lock; it is not a supported write.

**Every accepted write is saved** when the session was given a file:
`ScopeSession.create(..., settings_file=path)`. The session saves the whole
settings dict there one second after the first write it has not saved, so
a burst of writes is one save, and once more at `shutdown()`, after the
session's work has finished. Each save goes to a file beside the target
and is then moved over it, so a crash mid-save leaves the previous file
whole. The GUI and `python -m rest` name the installation's
`data/current.json` (`ScopeSession.user_settings_file(root)`), never the
shipped `settings.json` a first run reads from. A session given no file
(the default) saves nothing, which is what a script working on its own
dict wants. A write is accepted and in effect either way. Two conditions
keep writes in memory without saving them, and the session reports each
once as a `SettingsNotSavedNotice` (title `Settings not saved`) carrying
the condition as `reason`: `'settings_provisional'`, the settings are the
shipped template because `current.json` could not be read (see below),
and `'no_hardware'`, no hardware came up at bring-up. A save the drive
refuses is reported as `SettingsFileNotWrittenError`, naming the file and
the operating system's reason; the file is unchanged, and the next write
and the close save again.

**Provisional settings.** When `current.json` cannot be read, the GUI
comes up on the shipped template and asks the user what to do; that
file is the user's only copy of their configuration, so it is left
untouched until they decide, and the session saves nothing meanwhile.
Check `session.settings_are_provisional()` and, once the user (or the
controlling client) has chosen to start over, resolve with
`session.retire_rejected_settings()`: the unreadable file is renamed
aside, never deleted, and the session saves again at its next accepted
write and at its close. `ScopeSession.load_user_settings` raises
`SettingsFileError` for such a file instead, since a script has nobody
to ask.

### Periodic metrics logging (optional)

```python
session.enable_metrics()    # start periodic runtime-health logging
session.disable_metrics()   # stop it (idempotent; shutdown() calls this too)
```

The session owns the metrics-logger lifecycle and one scheduler for every
periodic consumer, plain daemon timers on every host (never a UI clock:
a safety bound armed on a UI loop stops when the UI freezes). The
scheduler existing does not start metrics: they run only after
`enable_metrics()`, so every session, factory-built or not, keeps metrics
off unless the host opts in. `settings['profiling']['metrics_interval_s']`
overrides the default hourly cadence: seconds above zero, or null for the
default; any other value is refused (`SettingRefusedError`). Every host's metrics log the
session's executor queues and futures; a host with a live display passes
it as `enable_metrics(display=...)` (a `DisplayRates`: the GUI passes its
own), and its drawn rate and frame intervals are logged beside the
camera's. `enable_metrics()` raises `AlreadyStartedError`
(`metrics_running`; still a `RuntimeError`) if metrics are already running,
until `disable_metrics()`, which is idempotent. These members
are host-serialized: call them from one thread (the GUI uses its main
thread only).

### Hardware commands from L2

The Session carries no hardware-command forwarders: every command has
exactly one public spelling, on the sub-APIs of the composition root the
Session exposes as `session.scope`. The dispatch contract (each command
submits and blocks; refusal raises `HardwareCommandRefusedError` while a
protocol run owns the executors) is documented once in the contract
section above.

```python
# LED
session.scope.illumination.led_on('Blue', 200)      # returns once the board has been sent the command
session.scope.illumination.led_off('Blue')
session.scope.illumination.leds_off()

# Motion
session.scope.motion.home('ALL')
session.scope.motion.move_z(5000)
session.scope.motion.move_xy_by(dx_um=500)

# Imaging
session.scope.camera.set_gain_db(8.0)                 # dB; blocks; returns the dB in effect
session.scope.camera.set_exposure_ms(50.0)       # ms; blocks; returns the ms in effect
image = session.scope.imaging.capture_and_wait()    # returns frame-valid grab

# A dark frame is never refused. With a channel lit (strictly positive
# current), a frame with no lit pixel is retried until timeout_s in case a
# lit one is coming, then RETURNED, its record saying so:
# last_capture_info['frame_record'].dark is True. Read that to tell a dark
# capture from a lit one; do not re-measure pixels. With nothing
# commanded -- or a channel at 0 mA -- a dark frame is by design and is
# not measured at all (.dark is None). accept_dark=True skips the
# measurement for callers whose dark frames are expected (custom focus
# sweeps, benchmark probes), so .dark is None.
# A blown frame is never refused either: with all_ones_check=True a frame
# at least 98% saturated is grabbed again once and, blown again, RETURNED;
# the record's saturated_fraction is the measured fraction and .blown says
# whether it is blown (both None without the flag); a saved file's metadata
# carries 'saturated_fraction' and 'dark' where they were judged. Neither
# is logged as a
# warning by the capture: a run tells each step's first and counts the
# rest (RunOutcome.captures.saturated / .dark), a still tells its own.
# timeout_s is the retry budget for the content checks (dark floor,
# saturation, chunk verify); leave it 0.0 to judge the first grab only.
# The executor wait is bounded internally.
image = session.scope.imaging.capture_and_wait(timeout_s=2.0)

# Whether a frame counts for a change. Every gain, exposure, black level,
# LED, Z, XY and turret change is recorded, and one function decides
# whether a frame counts for it: judge_frame(frame, mapping) asks it against
# every source's last change (recorded_changes()). Each source runs one rule
# (capabilities.frame_rules): the clock rule, where the camera has a clock
# latch and a measured delay for the source, counts a frame that started
# exposing after the change plus that delay; every move on such a camera
# runs it, from the axis's stop when no delay is measured, so a frame that
# started while the axis could still move never counts; the credit rule
# counts one once the source's frame count was met before it, a move's by
# frames that arrived after its stop.
# A capture judges the frame it returns (a sum's first frame) against each
# source's last change; the answers ride its record
# (last_capture_info['frame_record'].judgments), its saved file
# ('frame_validity' in the metadata) and one line in api.log. A Judgment
# carries counts, frames_after, ms_after_write, ms_after_in_force (motion:
# from the axis settling), under the clock rule frame_start_ticks and
# qualifies_after_ticks, and the change's rule and credits, each frame
# counted toward its wait with who counted it (drain, grab or preview).
# Beside each grab a capture maps the camera's timestamp clock onto the
# host's steady clock (time.monotonic_ns): the record's clock_mapping, a
# ClockMapping (host_ns, camera_ticks, tick_hz, uncertainty_ns, half the
# latch's bracket), None on a camera without a latch
# (capabilities.camera_has_clock_latch: Basler yes, IDS and the FX2 no).
# mapping.camera_ticks_at(host_ns) is the camera's clock at a host time.
mapping = session.scope.imaging.last_capture_info['frame_record'].clock_mapping
for judgment in session.scope.imaging.judge_frame(frame, mapping):   # frame: from a listener
    print(judgment.change.source, judgment.change.rule.rule, judgment.counts)
```

### The control session

One client controls the scope at a time. `session.take_control(name)` takes the control session and returns its id, a string told to the taker and to no one else: it is what a call in the session names, and what ends it. `session.control` reads the live session as a `ControlSession` -- the `name` it was taken with, `taken_by` (the host the take came through) and `taken_at` -- and never its id; None when none is held. While it is held, every write that is not the holder's is refused: a hardware command (`HardwareCommandRefusedError`), a setting or a file (`ControlSessionRefusedError`), a run, a recording, a home or a diagnostic (each taker's own refusal), all with `reason` `control_session_held` and the session's name in `control_session`. The display preferences are written by anyone, and a stop is never refused. `session.status` carries the session as `control`.

The holder acts in its session by its id: in process, inside `with session.acting_in_control(session_id):`, where every call the block makes on its thread is the session's; over REST, by sending the id in a `Control-Session` header on each request. A done-callback the API runs on a thread of its own is outside the block and is refused as any non-holder's call. Inside the session the holder works the scope as anyone does without one: its runs, recordings, homes and diagnostics take the scope one at a time, and a recording still refuses the writes that would falsify its file.

```python
session_id = session.take_control('plate reader')
with session.acting_in_control(session_id):
    session.scope.camera.set_gain_db(5.0)
session.end_control(session_id)
```

A take is refused with `ControlSessionRefusedError` while anything of anyone else's is under way, so no one's work is ever put under a client's control: `control_session_held` while another session is held; `exclusive_activity_running` while a run, a recording, a home or a diagnostic holds the scope, naming it in `holder`; `write_in_flight` while a setting or a file is being written; `work_finishing` while a run or a recording that has ended is still finishing its images, video or file, or while a build is on the post-processing lane, which a capture in the session would overlap. Each is a `409` on the wire; take again when the work is done. During the session's close a take raises `SessionClosingError`.

`session.end_control(session_id)` ends the session: its taker ends it by its id, and the person at the scope takes it back ("Take back control") through `take_back_control`, below. It ends at once, and the run or recording taken in it is stopped: it unwinds under its own claim and holds the scope until it has, as any stopped run does. A run or recording begun after the session ended is not stopped. In process, `session.take_back_control()` ends whatever session is held without naming it, as the GUI's Take back control press and its close do; it is kept off the wire, so a client ends only the session whose id it was told. No record carries the id, `session.control` included: only the taker holds it. A home or a diagnostic taken in it ends on its own, and a write of the session's already under way finishes whole. From then on, a request naming the ended session is refused, `ControlSessionRefusedError` `control_session_ended` (`409`), so a client learns it lost control instead of writing as nobody; an in-process holder is not told, and its next call is anyone's. The session's close ends a held session first.

### Capture

`session.manual_capture.capture()` captures one still and saves it, the same
file the GUI's Capture button makes. The caller names only whether the
shown layer is displayed in false colour and the overlays it wants; the
channel (the lit LED, else the shown channel's layer, `session.shown_channel`;
see "The shown channel"), the folder (`live_folder/Manual`, per channel when
`separate_folder_per_channel`),
the name, summing, format and encoding come from the session's settings. The
channel is read as the frame is taken, and the camera holds that layer's
stored exposure, gain and auto-gain for the frame, whatever came before: a
run, a go-to, a written setting, an LED lit through `scope.illumination`.
When the light changes while the frame is taken, the still is taken again on
the new light's layer, within the first capture's budget. After a raw camera
write (`scope.camera.set_gain_db`, `set_exposure_ms` and the other raw
doors) the written values stand, and the frame's record says so:
`frame_record.held_layer` (the layer whose stored settings the camera held,
else None) is None, the file's `camera_held_a_layer` False. It
returns at once with an `Operation` of a `SavedStill` per file written, the
unmarked file first; an overlay adds a second file from the same frame. Each
`SavedStill` is its `file` and the instrument's record of it, built at the
capture: `channel`, `objective_id`, `plate_x_mm`, `plate_y_mm`, `stage_z_um`,
`well_label`, `significant_bits`, `output_format`, the frame's `FrameRecord`
(`frame`) and the image's `ImageStatistics` (`statistics`: `minimum`,
`maximum`, `mean`, `standard_deviation`, `percentile_1`, `median`,
`percentile_99`, in the array's own counts; saturation and darkness are the
frame record's). A JPG gets the same record, though its file embeds none of
it. On the wire the record is the job's result, its `file` a live-folder name,
and `GET /api/v1/files/<name>` returns the file's exact bytes, `GET /api/v1/previews/<name>`
a downscaled JPEG of it for display. It is done once the
still has let go of the camera, so a second still asked then is not refused
`'capture_in_flight'`. A still has no stop: its `stop()` does nothing.
Each file records where the stage was when the frame was grabbed: the plate
position (`PositionX`, `PositionY`, mm) and Z (`PositionZ`, um), each absent
when the scope did not know it (an axis that has lost its reference, or a
scope with no X and Y).

```python
session.show_channel('Blue')
(still,) = session.manual_capture.capture(
    false_color_on=True, bullseye=False, crosshairs=False,
).result(timeout=30)
print(still.file, still.well_label, still.statistics.maximum, still.frame.exposure_ms)
```

It raises `HardwareCommandRefusedError` with reason `'capture_in_flight'` while an
earlier still is running, `'exclusive_activity_running'` while a run, a
diagnostic or a home holds the scope, and `'scope_disconnected'` when the camera is
closed. While a manual recording runs, a still whose layer the camera does not
hold is refused from its operation with `'exclusive_activity_running'`
(`holder` `'recording'`) before anything is written, since changing the camera
would change the recording's frames; a still on the layer the camera holds is
taken. When it returns, the still is on the camera lane. The operation raises
`ObjectiveUnknownError` when the objective in the light path is unknown
(nothing captured), `CaptureFailedError` (the capture's own reason:
`no_frame`, `deadline_expired`, `no_fresh_frame`, `chunk_rejected`) when no frame passed, and
`HardwareCommandRefusedError` (`not_connected`, `camera_not_streaming`) when
the camera is gone or not streaming.
`session.manual_capture.in_flight` is True from the call until the still
has finished or been refused. The order of the calls decides, not how soon
the lane's worker wakes. A run started after `capture()` returns waits for the
still before it touches the camera when the still is running or was put on an
idle camera lane, so the
still saves under the state it started with and the run begins after it. A
still queued behind other camera work (another command waiting or running
when `capture()` put it there) and not yet running when a run takes the scope
is refused, and its operation raises `HardwareCommandRefusedError` (reason
`'exclusive_activity_running'`).

A frame `capture_and_wait` returns is a bare array: it carries no record of
the instrument -- the objective, the position, the settings the camera held
-- so saving it is the caller's to do as it likes, and LumaViewPro's file
writer is not offered for it. A file with that record is saved by the API
that took the frame: a still by `session.manual_capture.capture` (above), a
z-stack, a scan or a protocol by its run (see "Running protocols").


```python
# The capture derives the dark-floor expectation itself from commanded
# LED state -- there is no illumination fact to pass.
image = session.scope.imaging.capture_and_wait()

# Live-view tap: the latest buffered frame, no new exposure forced
# (the capture calls above always force one). (None, None) when unavailable.
frame, timestamp = session.scope.imaging.get_image_from_buffer()

# Payload bit depth of a frame this scope just produced (8, 12, or for a sum
# the bits it can reach) -- needed to interpret or rescale full-depth payloads
# before saving. The capture knows how many frames it summed; pass only the
# frame, and ask before the scope captures again.
session.scope.imaging.capture_frame_depth(image)
# The value that frame saturates at: one frame's full scale, or N of them for
# a sum (not the tag's power of two, which a sum of blown frames misses when N
# is not a power of two).
session.scope.imaging.capture_frame_full_scale(image)
```

**What a saved file holds.** A fluorescence still is saved mono, its layer
recorded in the TIFF's ImageDescription; the false colour is a display-time
choice, never in the pixels. FIJI, MATLAB ``imread`` and tifffile read mono
2D natively. A file written by a LumaViewPro from before 4.0 may hold a
fluorescence frame as three channels with one populated: it is mono in all
but its shape, where a real colour composite has several non-zero channels.

Full-pixel-depth frames store raw, right-aligned sensor values (a 12-bit
frame is ``0..4095``) and declare the true depth in the OME-TIFF
``SignificantBits`` tag (e.g. ``SignificantBits=12`` inside a 16-bit
container). To render or scale such a file to 8-bit, divide by
``(1 << SignificantBits) - 1`` -- treating the values as full 16-bit will
render a 12-bit frame ~16x too dark. A sum's tag is the bits it can reach
(``SignificantBits=14`` for four 12-bit frames, 10 for four 8-bit ones). An
older file with no payload-depth tag was left-justified into the 16-bit
range: read it at the container's width.

A still's exposure is one frame's (``ExposureTime``); ``FramesSummed`` says how
many frames the sum holds. An OME-TIFF keeps the exposure and the position in
its OME ``Plane``, and every other acquisition fact of the frame in the image's
``MapAnnotation`` with namespace ``etaluma.com/lvp``, one key each, the value
JSON text: ``FramesSummed``, ``Gain`` and ``GainUnit``, ``Illumination`` and
``IlluminationUnit``, ``BlackLevel``, ``Binning``, ``Objective``, the timestamps
and ``FrameID``, ``FrameValidity``, ``SaturatedFraction``, ``Dark``,
``CameraHeldALayer`` and ``CameraHeldLayer``. Bio-Formats (Fiji, QuPath,
OMERO) shows the annotation; a plain or ImageJ TIFF carries the same keys in
its ``Plane`` block. A key the frame did not know is absent, never a stand-in.
A Sum z-projection and a composite state no ``FramesSummed``: a sum of slices,
or a merge of channels each with its own count, is not one capture's sum.
An OME-TIFF from an earlier release carries no annotation, so it states only
its exposure and position.

### Running protocols

```python
runner = session.create_protocol_runner()
protocol = session.load_protocol('my_protocol.tsv')    # and the scope takes the plate it names
session.apply_channel_settings(protocol)   # optional: its Layer Settings into the layer controls, as the GUI's Load does
# ProtocolFormatError (a refusal) names the file and what is wrong with it -- malformed, too large, or a plate this
# installation's labware catalogue does not have; ProtocolNotLoadedError names a file that cannot be read and the OS reason
# or build one from the session's settings, as the GUI's New does, or one with no steps:
protocol = session.new_protocol(tiling='2x2', use_zstacking=True)
protocol = session.create_empty_protocol()                 # for add_step to fill
# save one: whole or not at all. ProtocolNotSavedError (modules.exceptions) names the
# file and the OS reason, and a file already there is unchanged. A run saves its own
# copy in its run folder; when that copy cannot be written the run ends failed_at_start
# (reason run_dir_init_failed) before anything moves.
written = session.save_protocol(protocol, 'my_protocol.tsv')   # -> Path('my_protocol.tsv')
# save_protocol writes exactly the name given, which ends in .tsv (in any case) --
# a name without it is refused (LiveFolderPathRefusedError, wrong_extension) --
# and writes the protocol with the
# session's Layer Settings block (each layer set to acquire, with the values its
# controls hold), which apply_channel_settings puts back; it returns the path written.
# It does not change what LumaViewPro opens at its next start-up.
# save_protocol is the one way to write a protocol file.

# A run captures and saves in the session's image mode and output formats:
# session.scope.camera.set_image_mode(...) chooses the mode before the run starts.
pending = runner.start_single_scan(protocol)
result = pending.result(timeout=300)
print(result.status, result.reason, result.message)

# Or stop it at any time through the operation its call returned. An
# operation whose run has ended never touches the run that is live now.
pending.stop()
```

**Standalone autofocus.** `start_autofocus(channel)` focuses once on one layer at the current stage position, without a protocol. Before a home there is no current position to start from: it raises `ProtocolRunRefusedError` with reason `position_unknown`, as every run does, `start_zstack` and `start_composite` included; while a home is running it raises `exclusive_activity_running` with `holder='home'`. Each of the three reads the settings once, when it is called, and states the position on that copy's plate:

```python
pending = runner.start_autofocus('BF', save_characterization_data=True)
result = pending.result(timeout=120)
if result.af_focus_z_um is None:
    print('no focus found; the stage is back where it started')
else:
    print('focused at', result.af_focus_z_um)
if result.af_data_saved:
    print('focus curve written to', result.af_data_path)
```

The layer is named by the caller and has no default; the GUI names the session's shown channel (`session.shown_channel.layer`). A name that is no layer raises `ArgumentRefusedError` (`layer_unknown`, or `no_layer_selected` for None), before any hardware moves. Everything else -- illumination, gain, exposure, frame size, binning, labware, objective -- comes from the settings store, so this run resolves exactly as the standalone autofocus button does.

The run saves no images and writes no run artifacts, and it leaves the stage at the focus it found. A sweep that finds none (a flat or invalid focus curve, a capture that got no frame, an error) puts the stage back where it started, and the run ends `failed`: `autofocus_failed` (title `Autofocus Failed`), or `autofocus_not_started` (title `Autofocus Not Started`) when its first window already reached past Z's travel, its `message` the autofocus's own words and `af_focus_z_um` None. The autofocus told the failure (below); the run's ending records it and does not tell it again. If putting Z back failed, the run ends `position_lost` instead; a Stop ends it `stopped`, and STOP pressed while it sweeps ends it `stage_stopped`; either leaves Z where the stop left it, since putting it back would be a move made after the person stopped the stage. There is no return-to-position input, and a caller sweeping the same field repeatedly sets its own Z between runs. Why a sweep found none reaches your outcome listener (see "Outcomes") as a fault titled `Autofocus Failed` (`AutofocusFailedError`, `modules.exceptions`), or the capture's own `CaptureFailedError` when every one of its tries got no frame, whose `reason` is one of three: `flat_focus_curve` (the curve has no peak: every focus score was zero or invalid, or the final pass's scores all sit above half their maximum, as over an empty field, and fit no peak inside the measured range: check the sample and the illumination), `out_of_travel` (a window the sweep reached after it moved the stage reaches past Z's travel; it is refused before the stage moves into it, and never narrowed to fit, because a focus beyond the travel would come back as the travel's edge), or `unexpected_error` (the sweep raised; the log has the details). When the first window already reaches past the travel, nothing moves: that is a refusal titled `Autofocus Not Started` (`AutofocusOutOfTravelError`, reason `out_of_travel`, cause `state`): start further from the limit, or narrow the objective's autofocus range. A sweep cut short is not a failure, and `except AutofocusAborted` catches each way it ends: stopped by its caller (`AutofocusStopped`, reason `stopped`) or started after its run had ended (`AutofocusRunEnded`, reason `run_ended`) is quiet, logged and never shown; started under a live run whose LEDs another lease holds is a refusal titled `LEDs In Use` (`AutofocusLedsHeldError`, reason `leds_held`), and the step captures at its fallback focus. A step's autofocus inside a run reports the same fault, and the step captures at the Z the sweep started from; if that Z could not be put back, the run ends instead of capturing at an unknown position. The standalone run hands the illumination back the way it found it rather than forcing every channel dark, because it is a single-field operation at a scope someone is standing at.

`save_characterization_data` is off by default: a caller that does not ask for the focus curve gets no folder. When on, the data lands under `Autofocus Characterization` in the live folder (or a `parent_dir` you state), and the outcome's `af_data_saved` / `af_data_path` report whether the file was written and where.

**Autofocus all steps.** `start_autofocus_all_steps(protocol)` autofocuses at every step of a protocol and writes the focus it finds into that protocol:

```python
pending = runner.start_autofocus_all_steps(protocol)
result = pending.result(timeout=600)
if result.focus_written:
    print('focused Z:', protocol.steps()['Z'].tolist())
else:
    print('protocol unchanged:', result.status, result.reason)
```

The scan visits each step with autofocus on, whatever the step's own autofocus setting, and captures nothing; the protocol's autofocus settings are not changed. When the scan completes, each step's Z becomes the focus found for it, written before the run lets go of the scope, so a caller reading the protocol in `run_ended` or after `result()` sees the focused values. A step whose sweep found no focus keeps its Z. A scan that does not complete (stopped, failed) writes nothing, because it focused only some of the steps. A protocol whose steps changed during the scan -- a different number of them, or a step at a different position, channel or objective -- is left unchanged, and the person is told once as `Focus Not Saved` (`FocusNotWrittenError` in `modules.exceptions`, a refusal); the outcome's `focus_written` says which happened. The run-wide settings (illumination, gain, exposure, frame size) come from the settings store, so this run resolves exactly as the protocol panel's Autofocus All Steps button does.

**Standalone z-stack.** `start_zstack(channel)` captures a stack on one layer, around the current stage position:

```python
pending = runner.start_zstack('BF')
result = pending.result(timeout=600)
print(result.status, result.reason)
```

The stack's range, step size and reference come from the settings store, so this run resolves exactly as the z-stack panel's Acquire button does, and an unconfigured stack behaves the same way there as here. The layer is named by the caller, and a name that is no layer raises `ArgumentRefusedError` (`layer_unknown`, or `no_layer_selected` for None) before any hardware moves.

Unlike `start_autofocus`, the slices are the product: the run saves its images (under `Manual/Z-Stacks` in the live folder, or a `parent_dir` you state) and writes its run artifacts. Autofocus is forced off for the step and cannot be turned on -- refocusing at each slice re-centres the very range the stack is sweeping. Stimulation configs are carried onto the step as stored, enabled or not, matching the button.

`return_to_start` is on by default: a stack ends at whichever end of its range it finished on, which is not where the operator was looking, so the stage goes back to the position the stack was centred on. Pass `return_to_start=False` to leave it where the stack ended.

`start_single_scan()` runs one scan; `start_protocol()` runs the full multi-scan protocol. `start_single_scan(protocol, return_to_position=...)` returns the stage to a `PlateLocation` once the scan ends, however it ends. The location is read on the protocol's plate, since the run moves in its protocol's frame: `session.get_current_plate_position()` states one on the selected plate, which is the protocol's when the protocol was made on it. An axis given None is left where the scan left it. LumaViewPro's own Scan and Run buttons start their runs through these two calls. A run records who started it, from the host it came through rather than from any argument: `'gui'` for LumaViewPro's own controls, `'rest'` for a REST client, `'api'` for code in the process (a script, the SDK, a plugin), as `Caller` (`modules.caller`) names them; a refusal it causes carries that as `holder_trigger`, and a Stop of it ends it with a message naming who stopped it ("Stopped by a REST client."). Whether a run stamps the turret position into its filenames is the session's `engineering_mode` when the run starts. A run reads its image mode (`set_image_mode`) and output formats from the settings when it starts. Both raise `ProtocolRunRefusedError` (`modules.exceptions`) when the run is refused before any state is committed -- already running, files still writing, empty protocol, a validation failure, hardware not connected, a camera that is connected but not streaming (`camera_not_streaming`: its feed was stopped -- `scope.imaging.enable_streaming()` restarts it; a frame-size or binning change in flight is waited out, not refused), an axis whose position is unknown (`position_unknown`: the scope is not homed -- the message names each axis), a home running (`exclusive_activity_running`, `holder='home'`), a video recording live or still finishing (`recording_active`, `holder='recording'`), the stage's lid open (`lid_open`: every run moves X and Y, which the open lid refuses), a `sequence_name` that is a path rather than a name (`sequence_name_invalid`: it holds `/` or `\`, a drive, or is `..`; the name is the protocol file the run saves inside its folder, and a blank one is `unsaved_protocol`), or a live owner holding the illumination. Its `str()` is the sentence written for a person, and an L2 caller branches on its `reason` / `title` / `message` attributes (they map cleanly to a REST status code or a UI message). A run that could not be checked at all -- validating the protocol, or reading whether the hardware is connected, or reading the stage's interlocks, crashed instead of answering -- is not a refusal: it raises `RunCheckFailedError` (`modules.exceptions`, reason `validation_crashed` or `hardware_state_unknown`), a fault with the crash chained as `__cause__`. Neither commits anything, so neither needs unwinding. What a run tells its caller as it goes is its run events, below.

**Waiting forms.** Each run's start has a waiting form, in-process only: `run_single_scan`, `run_protocol`, `run_autofocus`, `run_autofocus_all_steps`, `run_zstack` and `run_composite` take their start's arguments and `timeout_s`, and are `start_X(...).result(timeout_s)`: they start the run and return its outcome however it ended, a stopped run's included. Over REST a start's job with `Prefer: wait` is the waiting form. A wait that runs out raises `RunWaitExpiredError` (`modules.exceptions`, reason `still_running`, a `TimeoutError`, quiet), whose `operation` is the run, still going, to wait on again or to stop. On the UI thread each is refused before it starts anything (`BlockingCallRefusedError`, `wait_on_ui_thread`). `run_composite` alone also raises when the run produced no composite (below).

**Run events.** Every run member -- each `start_` form above, `start_composite`, and each one's `run_` waiting form -- takes an optional `events=RunEvents(...)` (`modules.run_events`), a frozen record with one optional handler per event; a misspelt handler name fails where the record is built. Each event carries its values by value, so a handler never reads the run's own state, which a next run may already own. The engine names no host and no widget: what LumaViewPro's GUI does with an event (draw the step, hold a saved frame on screen, put the layer panel back) is its own handler, and a script or REST server subscribes to the same events.

| Event | Handler arguments | When |
|---|---|---|
| `scan_started` | `(scan_index, scans_remaining, period_s)` | as each scan begins: its index from 0, how many scans remain with this one, the protocol's period in seconds |
| `scan_ended` | `(scan_index, scans_remaining, period_s)` | as each scan ends, a stopped one included: how many remain after it |
| `step_started` | `(step_index)` | as each step begins, step 0 included, with the index read at the step |
| `frame_captured` | `(image, frames_summed, frame_significant_bits)` | for each captured frame whose write was handed to the run's writer, once it was: the frame as the file gets it (a sum when summing), read-only. A frame whose write is refused is not announced |
| `video_progress` | `(progress)`, a `VideoProgress` | as a video step records (`phase='recording'`, `elapsed_s` of `total_s`) and writes (`'writing'`, `percent`), then `'ended'` once after either; a step that ends before it records sends none |
| `run_ended` | `(outcome, run_dir, protocol)` | once the run has let go of the scope and its outcome has settled; `run_dir` is None for a run that saved nothing |
| `files_written` | `(run_dir, files)` | once the run's last image write has landed |

Every event but `frame_captured` is delivered on the thread that sends it (the run's own thread; for `files_written` the file writer's, when the last write lands after the run ends; for a composite's `run_ended` the post-processing lane's, where its merge runs), unless the record says `on_ui_thread=True`: then through the process's UI dispatcher (below), under LumaViewPro's GUI on its UI thread. The run carries the choice, so a script's or a REST client's run in the GUI's process is never held up by the GUI; LumaViewPro's own buttons set it, since their handlers draw. `frame_captured` is called on the run's own thread, which waits for it, so it must not block. A handler's own exception is reported once, as the exception it is, under the event's name (`category`): a refusal stays a refusal and a fault a fault, never a run cleanup failure. During an unattended run a `step_started` or `frame_captured` failure is logged and muted by the run's scope, and a `run_ended` or `files_written` failure is shown. No handler needs to guard itself.

`run_ended` and `files_written` reach a handler only once the run has let go of the scope -- it is no longer in progress and holds no claim -- so a handler can act on the scope at once. `run_ended` comes first for every run but a composite, whose `run_ended` waits for its merge to settle and so can come before or after its `files_written`, whichever the merge and the last write reach first. A run's files stop draining when its last write lands, just before `files_written`, so a new run started from inside `files_written` is not refused `files_writing`. One started from inside `run_ended` can be: the run's writes -- its images, and the rows of its run record, which a run that saves no images still writes -- may still be landing then, and a run does not start over another's unwritten files. A client that chains runs starts the next from `files_written`; a Full Protocol whose folder is handed to post-processing processors, or a build anyone has on the post-processing lane, refuses it `post_processing_running` until that work ends (`session.live_work` lists it as `'post_processing'`). Inside the run's own `run_ended`, its operation's `result()`, `exception()` and `done()` answer at once with the run's outcome; read the outcome there through `result()` only, since `concurrent.futures.wait`, `as_completed` and `asyncio.wrap_future` called inside it never return (the handler's return is what settles the run). Do not wait inside a handler for the run's `files`, or on a different run, a run you just started there included: under the GUI a wait on the UI thread for an operation not yet done is refused (`BlockingCallRefusedError`, `wait_on_ui_thread`), and with no GUI the handler can be running on the thread the wait needs, so it runs out its bound. Start the run there and wait on it from your own thread, or give it a done-callback. `files_written` fires once per run: `files` is `'written'`, or `'incomplete'` when some of the run's images are not on disk -- given up on, never taken by the file writer, failed to save, refused because the save drive was nearly full, or a video step's file that did not finish. It counts images: a capture that produced no image is not a file, and is in the outcome's `captures` instead. When an image is not on disk the person is told once, as the last write lands. A run's outcome is final once the run lets go of the scope: a write that meets the disk floor after that leaves the outcome as it settled (a run that acquired every capture stays `completed`) and is told once, as the run's images not saved (`write_batch_disk_full`), with no fatal popup and no LED command, since the scope is no longer the run's. A fault that ends a run is shown once, the first, after a Stop too; another behind it is logged, and the same fault met again (each queued write meeting the floor) is told once.

```python
from modules.run_events import RunEvents

events = RunEvents(
    step_started=lambda step_index: print('step', step_index),
    run_ended=lambda outcome, run_dir, protocol: print(outcome.status, run_dir),
    files_written=lambda run_dir, files: print('files', files),
)
pending = runner.start_single_scan(protocol, events=events)
```

**How a run ends.** Every run that commits returns an `Operation` (`modules.operation`), the run's one identity: everything a caller asks about its run, and its Stop, go through it. An `Operation` is a `concurrent.futures.Future`, running from the moment it is returned (`cancel()` is False). It is done once the run has ended, no longer holds the scope, and its `run_ended` handler, when it has one, has run: `result(timeout)` then hands back the run's outcome however the run ended (a stopped run's is `aborted` / `stopped`), so the caller's next move or setting needs no second wait and finds whatever its `run_ended` handler did; a bound that expires first raises `TimeoutError` and leaves the run going. `done()`, `add_done_callback`, `concurrent.futures.wait`, `as_completed` and `asyncio.wrap_future` see the same moment. Done-callbacks run on one callback thread, one at a time in the order their operations ended, never on the thread that ended the run, so a callback may itself wait on any operation; under the GUI a callback that touches a widget hands that to the UI thread. A video step's files are not among the run's files to wait for: the run's end waits for each video step to finish writing (its frames, `recording_manifest.json` and MP4), up to 600 s, before it lets go of the scope, so `result()` returns with them on disk; a video still writing at that bound is logged, the run ends without waiting further, and `session.live_work` lists it until it finishes. A run's other files can still be writing after it lets go of the scope, and a next run is refused `files_writing` until they land, then `post_processing_running` while the run's own builds -- its hyperstack, a composite's merge, the protocol's processors -- are on the post-processing lane: the run's operation's `files` is a second `Operation`, done once the run's operation is done, this run's images are on disk or given up on, its hyperstack build (when it saves hyperstacks) has ended, and the run has said everything about them -- the report of a lost image, and its `files_written` -- so a next run is admitted; `files.result(timeout)` hands back what became of the images -- `outcome` (`'written'`, or `'incomplete'` when any image is not on disk), `written`, `not_written`, and `not_written_reason` (`write_batch_save_failed`, `write_batch_disk_full`, `write_batch_abandoned`, `write_batch_not_taken`, `write_batch_video_unfinished`; `None` when every image landed). A hyperstack build's own failure is reported by the build, not here. A `result()` or `exception()` made on the GUI's UI thread for an operation not yet done is refused (`BlockingCallRefusedError`, `modules.exceptions`, reason `wait_on_ui_thread`), since what it waits for is delivered there; one already done answers. A wait with no bound under a host that has stopped delivering events (a GUI that has closed) does not return. A call that is refused raises and returns no operation, so there is never a stale outcome to wait on. Only the run settles its operation: a holder's `set_result` or `set_exception` is refused (`OperationSettleRefusedError`, reason `settled_by_its_producer`), so no code holding an operation -- a run's, a recording's, a move's -- can forge the outcome every other holder reads. That closes the accident and the casual forge, not a determined caller, which can reach the base class as the producers do.

The outcome carries fourteen fields. `status` is one of `completed`, `incomplete`, `aborted`, `failed` or `failed_at_start`: `incomplete` is a run that reached its end without every capture it was asked for (`captures_failed`; the images it did capture are saved, and the person is told once), `aborted` is an ending someone asked for (`stopped`, the run's own Stop; `stage_stopped` when `stop_motion` -- STOP, from any client -- was pressed while it ran, told once as a notice titled `Protocol Stopped`, `RunEndedByStopNotice`; `shutdown` when the session's close stops it), `failed` is one the instrument imposed (`camera_failure` (the camera failed three steps in a row: a capture that got no frame, or a step whose camera setting was not applied -- a manual step's gain or exposure, an Auto_Gain step's arm, a video step's auto-gain -- which is not captured, so no image is saved at a setting its step did not ask for; each is a `captures` failure naming its cause, and the message names the last one's), `disk_space_critical`, `consecutive_scan_failures`, `video_writer_died` (a video step's writer stopped working, so the run was stopped; the frames already written are on disk), `hardware_disconnected` (a part of the scope was disconnected mid-run; a capture that fails after the camera was declared removed ends the run at once), `autofocus_failed` and `autofocus_not_started` (a standalone autofocus run that did not focus, above), `interlock` (the stage's interlock refused a step's move -- the lid was opened mid-run -- so the run stopped at once; its `title` is 'Protocol Aborted -- Stage Interlock' and its `message` carries the refusal's words, which say whether the lid or the stage's power), and `position_lost` -- an axis lost its position mid-run, a step's move that stopped progressing among them, so the run stopped at once and saved no image from that position; home before the next run; a run has no motion time limit of its own), and `failed_at_start` is a run that could not begin after it committed. `reason` is the machine-readable cause, stable enough to branch on; `title` and `message` are the sentences a user reads, and never carry raw exception text. `merged`, `artifact_path` and `merge_reason` describe the composite merge only: a run with no merge reports `merged=False` with an empty `merge_reason`, so `merge_reason` is non-empty only when a merge was owed and produced no file. `af_data_saved` and `af_data_path` describe autofocus characterization data the same way: `af_data_saved` is true only when the data file was actually WRITTEN, and `af_data_path` names it. Both paths, `artifact_path` and `af_data_path`, are `pathlib.Path`s. A run that asked for no data, a sweep that measured none, and a run whose queued write an abort discarded all report `af_data_saved=False` with `af_data_path=None` -- the fields answer "did it land", not "was it requested", so a headless caller never has to go looking on disk to find out. `af_focus_z_um` is the Z a standalone autofocus run chose as focus, or None when it chose none; only `start_autofocus` sets it, and every other run reports None, since a run that autofocuses at several steps has no one focus to report -- `start_autofocus_all_steps` answers in the protocol's Z column instead. `captures` is what the run captured of what it was asked for: `asked` (scans times steps for a run that saves images, 0 for one that saves none), `captured`, and `failed`, one entry per capture that produced no image with its `scan`, `step_index`, `step_name` and `cause`; it is counted once per scan and step, so a scan run again after a transient failure never counts a capture twice. `captures` is `None` only for a run a shutdown settled before it ended. `cleanup_failures` names the steps putting the scope back after the run that did not finish (`Restore LED states`, `Restore camera gain/exposure`, `Return to position`, ...): empty when the LEDs, camera settings and stage were all put back, `None` only for a run a shutdown settled before its cleanup ran. The person is told the same once, as `Protocol cleanup issues`. `Stop autofocus` is an autofocus sweep that did not stop within 30 s of the run telling it to: the run keeps its own ending, the sweep can no longer reach the hardware, and autofocus and new runs are refused `autofocus_running` until it stops; restart LumaViewPro if it does not. `focus_written` is whether `start_autofocus_all_steps` wrote the focus it found into the caller's protocol: `None` for every other run, `False` when the scan did not complete or the protocol's steps changed during it (the protocol is unchanged), `True` when it was written. `focus_failures` lists each step whose autofocus did not focus, a flat curve included, as a `FocusFailure` with its `scan`, `step_index`, `step_name` and `cause` (the autofocus failure's words): a capture protocol captured that step at its fallback focus and an autofocus scan left its focus as it was, the run going on either way. It is counted once per scan and step, so a scan run again that focuses the step clears it, and it is empty when every autofocus focused. A run that would end `completed` with any says so in its `message` ("The run finished; autofocus did not focus 2 steps, which were captured at their fallback focus.", or for an autofocus scan "The scan finished; ... whose focus was not updated."), its status still `completed`; an `incomplete` run names both. Each was told by its autofocus, so the ending tells it no second time.

The two vocabularies are deliberately separate. A run that aborted names why in `reason` and leaves `merge_reason` empty; a run that reached its end but whose merge produced nothing reports its run status (`completed`, or `incomplete` when a channel failed) with the cause in `merge_reason`. `run_composite()` raises `CompositeFailedError` (a `CaptureError`) carrying whichever of the two applies; when its `timeout_s` runs out first it raises `RunWaitExpiredError` (`still_running`, a `TimeoutError`, quiet), as every waiting form does: the run or its merge may still be going, and its `operation` holds the run. A merge that raised is posted to the listeners as that exception's own type, kind and title -- a refusal such as `PostProcessingRefusedError` (`no_data`, nothing usable to merge) as a warning, `RunFilesNotWrittenError` under "Run Images Not Written" -- with `merge_reason` its `reason` (`merge_error` when it carries none, posted under "Composite Failed"); a merge that could not start or produced no file is posted as `CompositeFailedError`. A composite missing a channel merges nothing: its merge fails with `PostProcessingFailedError` (`post_processing_failed`, posted under "Composite Failed", naming the channel), and the shortfall is told as its own `RunIncompleteError` after it. The merge's notice and the shortfall are delivered to the listeners before the outcome settles, so `run_composite()` and the run's `result()` return after them.

`stop()` only asks: the run ends on its own thread, and the operation is done when it has. A run that ends before its end, by a Stop or a fault, halts a stage move it has under way first, before it waits for anything or puts the scope back (a video step's frames still being written included), so the stage stops short of the step it was heading for; on motor firmware with no STOP command the move runs on, and a home on a board that cannot stop one runs to its end (the outcome's `cleanup_failures` then names `Halt the stage`). A Stop through an operation whose run has ended does nothing, to that run or to any other run live by then. `progress()` answers about its own run only, as one `RunProgress` (`modules.sequenced_capture_runner`) read at once: `is_live`, True while the run holds the scope (False once its teardown has finished, while its `run_ended` may still be on its way, so a run can be not live and its operation not yet done); `is_stopping`, True from an accepted Stop until the run's teardown finishes; `is_last_run`, the engine's most recent run; `run_dir`, the run's folder (`None` for a run that saves nothing or never started); and for a progress readout `step_index` (the step executing now, its row from 0), `num_steps`, `scans_remaining` and `period_s` (the scan period in seconds), each `None` once the run is no longer live.

A refusal with reason `files_writing` means the previous run's files are still draining -- the previous run's `files` operation is done once they have landed. Reason `post_processing_running` means a build is on the post-processing lane -- a run's own hyperstack, merge or processors, or a caller's build -- and a capture and a build never overlap, so the run waits until the lane is idle (`session.live_work` lists its `'post_processing'` items until then); a recording start is refused the same way. Reason `files_writing_stalled` means the file writer has stopped making progress entirely (a wedged write, e.g. an unresponsive save drive); waiting will not clear it. Its message says how many unsaved images recovery would lose, and it names its own answer: `refusal.remedy` is a `Remedy` (`modules.exceptions`) whose `member` is `'recover_file_writer'`, with `confirm_text` and `cancel_text` for an offer (the prompt is the refusal's own `title` and `message`). Every refusal has a `remedy` attribute; it is `None` when nothing answers the refusal but waiting or asking differently. The GUI shows a refusal that carries one as a confirmation, and so can any client. A client need not start a run to hear of a stall: once for each stuck write, the session reports `FileWriterStalledError` (`modules.exceptions`, a fault, reason `'files_writing_stalled'`) through the outcome subscription below, carrying the same `remedy` in the same words. Recover with either:

```python
session.apply_remedy(refusal.remedy)   # takes the remedy a refusal named
session.recover_file_writer()          # the same recovery, called by name
```

`session.apply_remedy(remedy)` is the one door from a remedy's name to an action, and returns that action's answer. It takes only the members the Session offers as remedies (`recover_file_writer`, whose answer is how many of the run's images it gave up on, and `repair_camera_driver`, below); any other name is refused with `RemedyUnknownError` (reason `remedy_unknown`), before anything runs.

A Lumascope Classic on Windows that came up on no driver, or on one LumaViewPro cannot use, is reported at bring-up as `CameraNotAvailableError` with reason `camera_driver_unusable`; on an installed Windows build that carries the Classic driver (every build made with a signing certificate) it carries `remedy.member == 'repair_camera_driver'`, and its words say to install the driver from there; elsewhere they say to install a LumaViewPro build that carries it. The installer does not install this driver itself: `session.repair_camera_driver()` does, by running the installed driver step as an administrator on the scope PC's desktop. Windows asks there for an administrator's approval, then whether to install software from Etaluma (unless the person chose to always trust it), so someone at the scope answers whoever called. It returns the bindings after the step (`scope.diagnostics.get_camera_usb_bindings`), reports `CameraDriverRepairedNotice` (reason `camera_driver_repaired`, a notice: restart LumaViewPro so the camera connects), and never touches a camera that is up or a binding that works (a Zadig WinUSB binding is left as it is). Refused with `CameraDriverRepairRefusedError`: `repair_needs_windows`, `repair_needs_installed_build` (a source run, or a build run from `dist`), `repair_needs_driver_package` (an installation built without the Classic driver), `no_classic_bindings` (a scope reading no Classic USB device, such as a simulated one without an FX2), `driver_already_usable`, `repair_declined` (the administrator's approval was not given; ask again), or `driver_not_accepted` (Windows did not take the package: declined at its question, or refused; the step's log says which; ask again). `CameraDriverRepairFailedError` (a fault, reason `camera_driver_repair_failed`) when the step could not start, did not finish in 300 s, ended unsuccessfully, or left a present device on a driver LumaViewPro cannot use; its words name the step's log folder.

Recovery is deliberate data loss: the finished run's outstanding images are given up on (they were never going to finish), and a partial file from the stuck write may remain on disk. Returns how many images were given up on. It is refused with `FileWriterNotStuckError` (reason `file_writer_not_stuck`) while the writer is still making progress -- those files finish on their own -- and with `HardwareCommandRefusedError` while a run or a diagnostic holds the scope.

**Canonical entry points.** Build the runner with `session.create_protocol_runner()`. Build the `Protocol` it runs with `session.load_protocol(file_path)` (from a `.tsv` on disk, with the scope put on the protocol's plate), `session.new_protocol(...)` (from the session's settings, as the GUI's New does) or `session.create_empty_protocol()` (no steps, for `add_step` to fill). Each resolves `data/tiling.json` from the scope's data folder and judges the protocol against the scope's catalogues.

**Tiling grids.** `scope.protocols.tiling_config()` returns the grids this installation offers, read from the same `data/tiling.json`, as a record: `available` lists the labels a protocol's `tiling` accepts (`'1x1'`, `'2x2'`, ...), and `default` is the one to preselect, one of `available`. `available_configs()` and `default_config()` answer the same in-process. The scope reads the file once, at bring-up, with the installation's other catalogues, so a `tiling.json` edited while the program runs is seen at the next start; a file that is missing, unreadable or not JSON, or is not a `data` object of grids with whole-number `m` and `n` and a `metadata.default` naming one of them, refuses the bring-up with `InstallationFileError` (`installation_file_unusable`, a fault: reinstall or restore the file), naming it. The members that build a protocol still read the file as they build, and raise the same for a file broken since. `protocol.tiling()` names the grid a protocol's steps already carry: `'1x1'` when no step is tiled, the grid's label when the tiles form one on offer, and None when they are tiled in a layout no grid offers. `session.apply_tiling` refuses any grid over a tiled protocol (`ProtocolRunRefusedError`, reason `already_tiled`).

```python
grids = session.scope.protocols.tiling_config()
grids.available             # ('1x1', '2x2', '3x3', ...)
grids.default               # '1x1'
```

**Choosing what each layer captures.** `session.set_channel_acquire(channel, mode)` sets whether a layer captures an image (`'image'`), a video (`'video'`) or nothing (None), as the GUI's acquire toggle does; the layers set to acquire are the ones `new_protocol` and `add_step` build steps for and a composite merges. A layer set to acquire stops stimulating. A name that is no layer raises `ArgumentRefusedError` (`layer_unknown`), an unknown mode `SettingRefusedError` (`out_of_range`), and `'image'` or `'video'` for a layer this scope does not have `HardwareCommandRefusedError` (`axis_absent`), each changing nothing (`scope.layer_identity`); setting any layer to acquire nothing is always accepted. A saved setting that has such a layer acquiring -- one written on another model -- is set to acquire nothing when the session comes up, and the change is logged.

```python
session.set_channel_acquire('BF', 'image')
session.set_channel_acquire('Green', None)
```

**Turning a layer's auto-gain on and off.** `session.set_channel_auto_gain(channel, enabled)` does what the GUI's Auto Gain/Exp box does. For the shown layer the camera follows before it returns: turning it on (`True`) starts the camera adjusting, and turning it off (`False`) stops it and keeps what it reached: the gain becomes the layer's stored gain and the exposure its stored exposure, rounded to 0.1 dB and 0.01 ms. The stored exposure is never below the channel's usable floor (0.1 ms for brightfield, phase contrast and darkfield, 1 ms for fluorescence and luminescence), even when the camera went lower on a bright sample. A value the camera did not report is left as it was. For a layer the camera does not hold, only the preference is stored: the camera's adjusting belongs to the layer it holds. Turning it off returns the lock result: its `state` says whether the camera converged, hit the exposure ceiling (`MAXED`), hit the floor (`AT_MINIMUM`), or reported nothing usable (`FAILED`); a `state` of None means auto-gain was not running. `session.channel_auto_gain_lock(layer)` answers that same lock afterwards, until the layer's auto-gain is next turned off: over REST, where `PUT /api/v1/channel_auto_gain` answers the setting's read, `GET /api/v1/channel_auto_gain_lock?channel=BF` is how a client learns it. It records that turning off, not the layer now, so turning auto-gain back on or storing a gain leaves it as it was; it is None until the layer's auto-gain is first turned off in the session. Turning it on returns None. Either waits for the camera. While a run, a diagnostic or a home holds the scope, turning it on is stored only, and turning the shown layer's off raises `HardwareCommandRefusedError` and stores nothing. A name that is no layer raises `ArgumentRefusedError` (`layer_unknown`), and turning it on for a layer this scope does not have `HardwareCommandRefusedError` (`axis_absent`), each changing nothing; turning it off is always accepted.

```python
session.set_channel_auto_gain('BF', True)
lock = session.set_channel_auto_gain('BF', False)
lock.state, session.get_setting('BF.gain_db'), session.get_setting('BF.exposure_ms')
session.channel_auto_gain_lock('BF') == lock          # True
```

**Creating a protocol.** `session.new_protocol(tiling='1x1', use_zstacking=False, period_s=None, duration_s=None)` does what the GUI's New does: one step per layer whose `acquire` is set, at every well of the session's labware, with the current objective, tiled and z-stacked as asked. `period_s` and `duration_s` are seconds; one left out (None) is the stored default's (`settings['protocol']`), and one scan is 0. The GUI passes the schedule on screen. When no layer is set to acquire it raises `ProtocolRunRefusedError` with reason `no_acquiring_layer`, and a `tiling` this installation does not offer raises it with reason `tiling_unknown`; each is raised to you and posted to no listener (whoever catches it reports it), and builds nothing; a labware with no wells gives an empty protocol to fill with `add_step`. With `use_zstacking=True` the steps are stacked as `apply_zstacking` stacks them, and refused as it refuses them: a range or step size not greater than zero (`zstack_not_configured`), a scope with no Z motor (`positions_unreachable`), a slice outside the Z travel (`zslices_outside_travel`). When a layer set to acquire has no saved focus its steps take the current Z, so with Z's position unknown (before a home) it raises `AxisStateUnknownError` naming Z, posted to no listener, and builds nothing; with every acquiring layer focused it builds before a home too. `session.create_empty_protocol()` is the no-step protocol that needs no objective.

**A protocol's schedule.** `protocol.period_s()` and `protocol.duration_s()` are the protocol's own, in seconds, and the run runs them; `protocol.num_scans()` is how many scans a full run takes, the duration over the period counted exactly (one for a period of 0, or a duration shorter than one period). `session.set_protocol_schedule(protocol, period_s=..., duration_s=...)` sets both: a period is 0 (one scan) or at least one second, and a duration is 0 or more. Anything else -- a sub-second or negative period, a negative duration, a value that is not a finite number -- raises `ProtocolScheduleRefusedError` (from `modules.protocol`; a `ProtocolFormatError`, a refusal) naming the value and the rule, and the protocol keeps the schedule it had; nothing is raised to one second. A protocol built with one, or a file carrying one, is refused the same way, the file by name. Post-processing a finished run reads its saved protocol without judging the schedule, which it never uses, so a run saved by a release that allowed a shorter period can still be stitched or projected. `schedule_from_units('period', minutes)` and `schedule_from_units('duration', hours)` convert the units the file and the settings hold, refusing what is not a runnable number.

```python
session.set_protocol_schedule(protocol, period_s=600, duration_s=24 * 3600)
session.set_protocol_schedule(protocol, period_s=0.5, duration_s=3600)  # ProtocolScheduleRefusedError
```

**Reading a protocol's steps.** `protocol.step(i)` returns step `i` as a `StepRecord` (`modules.protocol`), a field per column of the steps table (`'Z-Stack Group ID'` is `z_stack_group_id`), its video config a `VideoConfig` (`duration_s`, `fps`) and its stimulation, by layer, `Stimulation` records (`enabled`, `illumination_ma`, `frequency_hz`, `pulse_width_ms`, `pulse_count`), None where the step's config does not say; `protocol.step_records()` returns every step's record in order, the table in one read, and over REST as one array. In-process, `protocol.steps()` returns a copy of the steps table and `protocol.step_row(i)` a copy of one row; each row `protocol.layer_settings()` returns is a copy too. The dicts in a row's `Video Config` and `Stim_Config` cells are read-only (each a `dict` that refuses writes): a write into one raises `TypeError`, and `copy.deepcopy(cell)` gives a plain dict to edit. A write to a copy changes nothing in the protocol: a protocol changes only through the Session's edits (`session.set_protocol_schedule`, `session.add_step`, ...), which judge each change. A run works on its own copy of the protocol it is handed, so nothing the run writes, such as an autofocus's Z, reaches the caller's protocol either, unless the run is asked to write its focus back (`start_autofocus_all_steps`).

**A protocol's capture root.** `session.set_protocol_capture_root(protocol, text)` stores the root as given, and `protocol.capture_root()` returns it so. `protocol.capture_prefix()` is that root as the prefix of a saved file's name (only letters, digits, `-` and `_` kept): the one prefix a run's images and post-processing's outputs both carry, so `exp/2026:10` names files `exp202610_...`.

**A step's autofocus.** `session.set_step_autofocus(protocol, enabled=True, step_index=2)` turns autofocus on or off for one step, and with no `step_index` for every step; a `step_index` the protocol lacks raises `StepNotFoundError` (`no_such_step`) and changes nothing.

**Adding a step.** `session.add_step(protocol, before_step_index=... | after_step_index=...)` does what the GUI's Add Step does: one step per layer whose `acquire` is set, at the live stage position read on the protocol's plate (`protocol.labware()`, which need not be the session's selected plate; `session.get_current_plate_position()` reads on the selected one). Every step takes the live Z, whatever its layer's saved focus: the focus of the channel being viewed is a better guess for every channel than an older saved one, and no saved focus is changed, since it is where `new_protocol` starts the next protocol. Each step is made with the current objective, in the settings' `step_channel_order`. `channels=('BF', 'Blue')` adds a step for each layer named instead, in that order, with its stored settings, a layer whose `acquire` is None added as an image step, so a script adds without changing the settings; `at=PlateLocation(x_mm=..., y_mm=..., z_um=...)` places the steps there on the protocol's plate instead of at the live position (a layer name that is no layer is `ArgumentRefusedError` `layer_unknown`, a layer this scope lacks `HardwareCommandRefusedError` `axis_absent`). With no `before_step_index` or `after_step_index` the steps follow the last step; giving both raises `StepEditRefusedError`. `use_zstacking=True` makes each added step a z-stack of the stored z-stack settings, which `session.get_zstack_params()` answers (`range` and `step_size` in micrometres, `z_reference` one of `'top'`, `'center'`, `'bottom'`), built as `apply_zstacking` builds one, and leaves the protocol's other steps as they are and where they are; the stack is judged before anything is added, so a stack `apply_zstacking` would refuse (no extent, no Z motor, a slice outside the Z travel) adds nothing, with the same reason. It returns the inserted step names in protocol order, each stack's slices when stacked. When any axis (X, Y, Z, or the turret) does not know its position -- never homed, homing, or lost after a failed home -- it raises `ProtocolRunRefusedError` with reason `step_position_unknown`, naming the axes: the position read keeps answering the last number an axis reported, so a step saved then would record a place the scope no longer vouches for. When no layer is set to acquire it raises `ProtocolRunRefusedError` with reason `no_acquiring_layer`; on a turret scope whose slot is unknown or has no objective assigned, reason `turret_objective_unset`; when the objective is otherwise unknown (a slot assigned an objective that is not in the catalogue, or `objective_id=None` passed below), reason `objective_unknown`. Each is raised to you and posted to no listener; nothing is added.

```python
protocol = session.new_protocol()                   # one step per acquiring layer at every well; refused no_acquiring_layer when none acquires
protocol = session.create_empty_protocol()          # no steps; needs no objective, so it works before one is known
names = session.add_step(protocol, before_step_index=0)   # ['custom0000_BF', ...]
```

**Updating a step.** `session.update_step(protocol, step_index, channel=..., label=None)` does what the GUI's Update Step does: step `step_index` takes `channel`'s settings, the live stage position read on the protocol's plate, as an add does, and the current objective. `label` renames the step; `None` keeps its label. When `channel`'s stim config is enabled the update is a stim edit, so the step keeps the channel it already acquires. It returns the step's name after the update. It is refused for the same reasons, under the same names, as an add -- `step_position_unknown`, `turret_objective_unset`, `objective_unknown`, and `no_acquiring_layer` when the layer the step takes (for a stim edit, the step's own) is not set to acquire -- each raised to you and posted to no listener, with the step unchanged; a `step_index` that is not a step of `protocol` raises `StepNotFoundError`; a `channel` that is no layer raises `ArgumentRefusedError` (`layer_unknown`), and one this scope lacks `HardwareCommandRefusedError` (`axis_absent`), each before anything is read.

```python
name = session.update_step(protocol, 0, channel='Blue')   # 'custom0000_Blue'
```

**Deleting and renaming a step.** `session.delete_step(protocol, step_index)` does what the protocol panel's Delete does: step `step_index` is removed and the steps after it move up one. `session.rename_step(protocol, step_index, name)` does what typing in the step's name field does: the step takes `name` as its label, kept through later channel changes, and it returns the step's name after the rename. Characters a filename cannot carry are removed from `name`; a name with no letter, digit, dash or underscore left raises `StepEditRefusedError` and the step keeps its name. A `step_index` that is not a step of `protocol` -- including any index of a protocol with no steps -- raises `StepNotFoundError`, and nothing changes.

```python
session.rename_step(protocol, 0, 'center')   # 'center_BF'
session.delete_step(protocol, 1)
```

**Tiling and z-stacking a protocol.** `session.apply_tiling(protocol, tiling)` does what the protocol panel's tiling Apply does: every step becomes a grid of tiles, `tiling` one of the labels `tiling_config().available` lists. The tiles are spaced for the configured frame, binning and tile overlap (`tiling_overlap_percent`), and laid out on the protocol's own plate. `session.apply_zstacking(protocol, range_um=..., step_size_um=..., z_reference=...)` does what the z-stack Apply does: every step not already in a stack becomes a stack of slices `step_size_um` apart over `range_um`, its own Z at the stack's `'top'`, `'center'` or `'bottom'`. The slices of every channel one Add (or one New position and tile) made at one place are one z-stack group (`Z-Stack Group ID`): a run focuses the group once, at the first slice in run order that sits at an autofocusing channel's focus, and moves every slice of it, every channel, by what it found. Both leave the steps in the order a run visits them, and report the step check's notice as the other edits do. Each is refused before any step changes. `ProtocolRunRefusedError` covers a grid this installation does not offer, a protocol already tiled (reason `already_tiled`), a step's objective not in the catalogue (reason `objective_not_in_catalogue`), a range or step size not greater than zero, a scope with no motor for the axes the build moves (reason `positions_unreachable`; a Z-only scope cannot be tiled), a tile grid on a scope that cannot report its pixel size, so its spacing is unknown (reason `tiling_scale_unknown`), and a tile or slice outside the stage's travel. `ArgumentRefusedError` covers an unknown `z_reference` (`zstack_reference_unknown`, offering the three), and `CatalogueNameRefusedError` a protocol plate the catalogue does not have.

```python
session.apply_tiling(protocol, '3x3')
session.apply_zstacking(protocol, range_um=20.0, step_size_um=5.0, z_reference='center')
```

**Saving a focus.** `session.save_focus(protocol, channel, step_index=None)` does what the layer panel's Save Focus does: the live Z becomes `channel`'s saved focus (where `new_protocol` puts the layer's steps, `session.saved_focus(channel)`; `add_step` takes the live Z), and, when `step_index` names a step of `channel`, that step's Z. A step of another channel is left alone, and no other step is written: every step `new_protocol` builds for a layer is born at the same focus, so a step that matches the old focus says nothing about whether it should take the new one. It returns a `SavedFocus` (`modules.scope_session`): `z`, the Z saved, and `step_index`, the step that took it or None. `session.apply_focus_to_channel_steps(protocol, layer)` does what Apply to Steps does: the live Z becomes the layer's focus and the Z of every step of `channel`, and it returns how many steps took it. Both are refused before anything is written: a name that is no layer with `ArgumentRefusedError` (`layer_unknown`); a layer this scope lacks with `HardwareCommandRefusedError` (`axis_absent`, `MissingPart.layer`); on a scope with no Z axis with `ProtocolRunRefusedError`, reason `positions_unreachable`; when Z does not know its position (never homed, homing, or lost after a failed home) with `AxisStateUnknownError`, each raised to you and posted to no listener; and a `step_index` that is not a step of `protocol` raises `StepNotFoundError`. `session.save_channel_focus(layer, z_um)` stores a Z you hold, not the stage's, as `channel`'s focus: the Autofocus button saves the Z its run chose this way (`outcome.af_focus_z_um`), and a script saves one it measured. It says only that the number is a Z this scope's focus can reach, not where it came from, and it is refused, with nothing written, for a name that is no layer (`ArgumentRefusedError`, `layer_unknown`), a `channel` this scope does not have (`HardwareCommandRefusedError`, `axis_absent`), a scope with no Z axis (`ProtocolRunRefusedError`, `positions_unreachable`), a Z that is NaN or infinite (`ArgumentRefusedError`, reason `not_a_number`), and a Z outside Z's travel (`PositionOutOfRangeError`). `save_focus` and `apply_focus_to_channel_steps` write the layer's focus through the same path. `update_settings` on a layer's `focus` is refused naming `save_channel_focus`. The live Z they read is `scope.protocols.focus_z(then=...)`, refused as above.

```python
saved = session.save_focus(protocol, 'BF', step_index=0)   # SavedFocus(z=5000.0, step_index=0)
count = session.apply_focus_to_channel_steps(protocol, 'BF')
session.save_channel_focus('BF', outcome.af_focus_z_um)    # a Z you hold, checked against Z's travel

z = scope.protocols.focus_z(then='save the focus')     # the live Z, refused as save_focus refuses it
```

**A step's motor targets.** `scope.protocols.step_targets(protocol, step_index)` is the one conversion a run and a person's navigation share: a `StepTargets` (`modules.lumascope_api.protocols`) of `turret_slot` (the slot the step recorded, its `Turret Slot`; a step that recorded none, from an older file, means the only slot holding its objective; None on a scope without a turret), and `x`, `y` and `z` in stage micrometres, X and Y converted on the protocol's own plate, not the scope's selected one; an axis the scope has no motor for is None, and nothing drives it. Every step move a run makes takes the same conversion, so a run's return to its first step between scans and after a z-stack drives only the axes the scope has too. It raises `StepNotFoundError` for a `step_index` the protocol lacks, `ProtocolRunRefusedError` for a step whose objective this scope cannot put in the light path (`turret_objectives_unassigned` when no turret slot carries it, as after the slot is cleared; `objective_not_mounted` when, without a turret, it is not the mounted objective; 409, the scope's state) or that cannot say which slot it means (`turret_slot_ambiguous`, `turret_slot_reassigned`, below; 409), and `CatalogueNameRefusedError` for a plate the catalogue does not have.

**Which turret slot a step uses.** Every step records the slot it was made or updated at, its `Turret Slot` (-1 without a turret): `add_step` and `update_step` record the slot in the light path, and `new_protocol` records, for each step, the slot in the light path when it holds the step's objective and otherwise the only slot that does. A run and `go_to_step` turn each step to its own slot, so two steps naming one objective in two slots each use their own, and updating one changes no other. A protocol opened with `load_protocol` gets a slot for each step that recorded none (a file from before steps recorded their slot) when only one slot holds its objective. A step that cannot say which slot it means -- its objective is in two or more slots and it recorded none, or the slot it recorded now holds another objective -- does not stop the load; `scope.protocols.turret_slot_questions(protocol)` returns one `TurretSlotQuestion` per objective with such steps (`objective_id`, `step_indexes` the indexes of the steps, `reassigned_step_indexes` those whose slot now holds another objective, `choices` the slots holding it now), and `session.point_steps_at_turret_slot(protocol, objective_id, slot)` answers it, pointing exactly those steps at `slot` and returning their indexes; a slot that does not hold the objective is refused with `ArgumentRefusedError` (`slot_not_holding_objective`, `offered` the slots that do), nothing changed. Until it is answered, a run's start and going to such a step are refused with `ProtocolRunRefusedError`: `turret_slot_ambiguous` or `turret_slot_reassigned` (409), naming the steps and the slots. LumaViewPro asks each question when a protocol is opened. Saving the protocol writes the slots.

```python
for q in scope.protocols.turret_slot_questions(protocol):   # [] on a scope without a turret
    session.point_steps_at_turret_slot(protocol, q.objective_id, q.choices[0])
```

`step_targets` converts X and Y with `scope.runtime_state.plate_to_stage(px, py, labware_id=protocol.labware())`, on the protocol's plate whatever plate is selected; it takes a `stage_offset=` (a `StageOffset`) to convert with a snapshot instead of the live offset, as a run does. A run binds `scope.runtime_state.plate_transform(labware_id=protocol.labware(), stage_offset=offset)` when it starts and states every frame it saves through it, stills and recorded frames alike: the position the stage was read at beside the grab, never the step's planned one, in the frame the run's steps were driven in, whatever plate is selected meanwhile; an axis the scope lacks or has lost its reference on states none.

```python
targets = scope.protocols.step_targets(protocol, 2)   # StepTargets(turret_slot=1, x=..., y=..., z=...)
stage = scope.runtime_state.plate_to_stage(60.0, 40.0, labware_id=protocol.labware())
to_plate = scope.runtime_state.plate_transform(labware_id=protocol.labware(), stage_offset=offset)   # bound now
```

**Going to a step.** `session.go_to_step(protocol, step_index)` does what a click on a step does, as one task on the scope's IO lane: it asks every axis once whether it knows its position, turns the turret to the step's own slot (above), starts X, Y and Z towards the step's targets together (`scope.protocols.step_targets`, above), puts the step's values into its layer's live settings (the layer acquiring as the step does, its stimulation from the step's; the layer's saved focus is left as it is, since `new_protocol` starts from it), and applies the step's LED preview: the step's channel at its current when `protocol_led_on` is set, every channel dark when not. While the stage travels it puts the step's layer on the camera (see "Putting a layer on the camera", below), and it returns once the camera holds that layer's exposure, gain and auto-gain and X, Y and Z have arrived, waiting off the IO lane so the lane takes other work meanwhile; a move that does not arrive raises `MoveNotCompletedError`, with the layer and the preview already the step's, and a setting the camera refuses raises `CameraSettingRejected` once the stage has arrived. `session.start_go_to_step(protocol, step_index)` does the same, the step's layer on the camera before it returns, but returns without waiting for the stage, with one `Operation` for the X, Y and Z moves it started: done once every one has ended, its `result()` None when all arrived and the first `MoveNotCompletedError` in X, Y, Z order when one did not, its `stop()` the moves' stop. It is the form for a click, where a second click re-targets the stage at once; it waits on the camera, so on the UI thread it is refused `wait_on_ui_thread` and a GUI calls it from a worker. A setting the camera refuses raises `CameraSettingRejected` with the moves under way. Only the axes the scope has are moved: a manual scope moves nothing, does the rest and returns an operation already done; a motorized scope whose motor controller is not connected is refused with `HardwareCommandRefusedError` (`not_connected`) before anything changes, and so is a scope with no camera connected (`MissingPart.CAMERA`), and a step whose preview would light on a scope with no LED controller connected (a dark preview needs none, so with `protocol_led_on` off the stage still moves); once `session.shutdown()` has shut the lanes it raises `scope_disconnected`. A repeat of the step the session last went to (a re-click) does everything but the preview, so a channel lit or put out in between stays as it was; when another layer was shown in between, the step's layer is shown again, lit or dark as the shown channel was. A run transition forgets it. It is refused before anything changes: a `step_index` that is not a step of `protocol` with `StepNotFoundError`; a step whose objective this scope cannot put in the light path with `ProtocolRunRefusedError`; a step whose colour is no layer with `ArgumentRefusedError` (`layer_unknown`); a step on a layer this scope lacks with `HardwareCommandRefusedError` (`axis_absent`); a stimulation naming no layer with `ArgumentRefusedError` (`layer_unknown`); an axis that does not know its position with `AxisStateUnknownError` (raised to you and posted to no listener); a scope held by a run or a diagnostic with `HardwareCommandRefusedError`; a step on another layer than the shown one while a manual recording runs with `HardwareCommandRefusedError` (`exclusive_activity_running`, `holder` `'recording'`), since a recording keeps one layer. A step outside an axis's travel raises `PositionOutOfRangeError` from that axis's move; the axes before it have moved and nothing else changes.

```python
session.go_to_step(protocol, 2)                       # turret, X, Y, Z, the layer, the LED, the camera; returns on arrival
going = session.start_go_to_step(protocol, 3)         # the same, started; going.result() is the moves' outcome
```

**Putting a layer on the camera.** `session.show_channel(channel)` is the one public way a layer's settings reach the camera: it shows the layer and puts its stored exposure, gain and auto-gain on the camera, and waits. Nothing is written when the camera already holds the layer at its stored settings, so a repeat costs no frames. A stored auto-gain arms the live auto loop, capped to the layer's channel class; a camera without hardware auto-gain applies the layer manually. Bring-up applies BF this way (skipped with no camera, or on a scope with no BF layer), `go_to_step` the step's layer, a write to the shown layer's settings the shown layer, and a run's end the layer shown before it. `scope.camera.apply` writes numbers that are no layer's: after it the camera holds none, and a still's record says so.

**A step the run will refuse.** Every step cell is read as its column's type when a protocol is loaded or built: numbers for X, Y, Z, Illumination, Gain and Exposure, whole numbers for Sum, Z-Slice and the group IDs, True or False for Auto_Focus, Auto_Gain, False_Color, Custom Step and Auto_Named, `'image'` or `'video'` for Acquire, and a parsable config for Video Config and Stim_Config. A cell of the wrong type -- text in a number column, `maybe` in a True/False column, an empty position -- refuses the load with `ProtocolFormatError`, naming the file, the step and the column; nothing is guessed for it. Every text cell is the text in the file: a step named, or a well labelled, `NA`, `None` or `null` loads as that text. A file no LumaViewPro writer produces -- a NUL in a cell, a quote that does not close, a row whose cell count is not the header's, a header naming one column twice -- refuses the load with `ProtocolFormatError`, naming the file. A `Protocol` constructed in memory (`Protocol(config=...)`) is refused the same way, and the protocol's own writers refuse a value of the wrong type with `StepEditRefusedError`. A step index the protocol lacks is refused with `StepNotFoundError` by every writer and by `protocol.step(i)`; its words say how many steps there are ("The protocol has 6 steps.") and name no index, which is zero-based where a person counts from 1, and it carries `index`, the index refused as it was given, and `num_steps`. Both are `ProtocolError`s and refusals (`modules.protocol`): the reporter shows them as a warning, under "Step Not Changed" and "No Such Step", with no traceback. A step whose cells are the right type can still hold a value the run gate rejects (an exposure of 0, an objective not in the catalogue, an illumination above the board's maximum), and an add or update composed from live settings can too. Neither is refused: the load and the edit succeed, so the step can be fixed, and `session.load_protocol`, `add_step`, `update_step`, `delete_step` and `rename_step` report one notice, reason `protocol_steps_invalid`, listing the validator's lines. The run start refuses the protocol until every step is valid. A step's exposure, illumination, gain, sum, video rate and length and each stimulation config it carries are judged by the layer settings' own ranges (the `out_of_range` row of `update_settings`' refusals), so a step holds nothing a write of the layer could not store. A Layer Settings value outside its layer setting's range is not lenient: `session.load_protocol` refuses a file whose Layer Settings block holds one (a file with no block, v5, has its rows inferred from its steps, which stay lenient), and `session.apply_channel_settings(protocol)` a protocol built in memory, before any layer changes, with `ProtocolRunRefusedError`, reason `values_out_of_range`, naming each value ("BF Exposure: it must be above 0, not 0.0"); the file is fixed outside the app. `go_to_step` / `start_go_to_step` refuse a step holding such a value the same way, before any move starts. A step on a layer this scope does not have is refused at load and when a run starts, reason `layer_not_on_scope`, naming the layers and the steps.

### Video steps and recordings

A protocol step with `Acquire` = `video` records through the session's recording engine for the step's configured duration. This is the supported video path for L2 / headless callers; the GUI's manual Record button is a GUI-hosted convenience on the same engine.

Per recording (one per well per scan), the run produces:

- a frames folder (`<step>_video/`) of per-frame TIFFs, numbered in capture order;
- `recording_manifest.json` in that folder -- the measured truth: delivered frame count, measured frame rate, per-frame timestamps, and the recording's end reason. Downstream consumers (including Create Video's `auto` rate) read the manifest, not the configured rate;
- one variable-frame-rate MP4 per recording;
- when the recording is saved as frames, each frame's TIFF carries its own record of when it arrived: the plate position (`plate_pos_mm`, `x_pos`, `y_pos`) and Z (`z_pos_um`) as the scope tracked them, each absent when the scope did not know it; `stage_moving`, true when any axis was moving or homing at delivery; and `channel`, the channel that lit the frame. These are the same keys a still capture writes, and the frame is false-coloured as its recorded channel;
- after the run completes, one OME-TIFF hyperstack per (well, scan): `T` = frame capture order, `C` = channel, per-plane `DeltaT` from the frames' own timestamps. Hyperstacks build at run completion on every host -- headless and REST runs included, no GUI involved. Nothing waits on that build, so its outcome is told as a notification: "Saving Hyperstacks" as it starts, replaced by "Hyperstacks Saved" with the count and folder, or by what went wrong -- including a build that saved only some of its stacks, which names the ones missing and where the rest are.

A manual frames recording with the hyperstack output format on builds one OME-TIFF hyperstack from its frames, `T` = frame order, with each plane's position from the frame's own record (an axis the scope did not know on any frame is omitted from every plane, as the OME writer refuses to invent one). It is built only when one channel lit the whole recording: the stage and the LEDs stay open to L2 callers while a manual recording runs, and a channel change mid-recording leaves frames no single T x C cube can hold, so the build refuses, the `Hyperstack Not Built` notification carries the builder's reason, and the frames stand with their true per-frame channels.

Rate and duration come from the run's settings snapshot at start: `video.max_fps` (0 = uncapped; the effective rate is measured, not assumed) and `video.max_duration_s`. Mid-run settings edits do not affect a run in flight.

Recording starts are guarded like protocol starts: `RecordingRefusedError` (`modules.exceptions`) mirrors the `ProtocolRunRefusedError` shape -- its `str()` is the sentence written for a person -- with machine-readable `reason` codes `recording_active` (another recording is live, or still finishing), `exclusive_activity_running` (a protocol run or other exclusive activity holds the session's activity claim), `post_processing_running` (a build is on the post-processing lane), `camera_inactive`, `camera_exposure_unknown`, `insufficient_disk`, `capture_location_unusable` and `falsifying_change_in_flight` (a change that would falsify the recording, such as a frame-size write, is still in flight). Nothing is started when it raises.

**A manual recording from a script.** `session.manual_recording.start(false_color_on=False)` opens a recording of the live feed and returns its `Operation` once it is recording. The rate, the frames-or-MP4 choice and the duration cap (`video.max_duration_s`) come from the settings at that moment, and it is saved under `Manual/` in the live folder (`save_folder` names it). It raises `RecordingRefusedError` (the codes above), `ObjectiveUnknownError`, `CaptureFailedError` (a `CaptureError`; `recording_not_started`, the camera refused the frame listener); nothing is started when it raises. Its frames are named for the lit LED's channel, else the shown channel's layer, read once at the start. `stop()` closes the recording and returns; frames still queued keep writing on their own. `is_recording` is True until the stop; `is_busy` until the frames, the MP4 close and any hyperstack build are done; `end_reason` says why the last recording ended (`'user_stop'`, `'duration_elapsed'`, `'camera_stalled'`, ...). The `Operation` is done once everything is written and a new recording may start: `result()` hands back that recording's `end_reason`, and its done-callbacks run then, on the callback thread. Its `stop()` stops that recording as the controller's `stop()` does, and does nothing once that recording has stopped, so it never stops a recording started after it. What went wrong in the writing (dropped frames, a hyperstack not built, a finish that failed) is reported through the outcome subscription, not raised from `result()`.

While a manual recording holds the scope -- live or still draining its frames -- it refuses, for every caller, what would falsify its file: the camera's `set_frame_size`, `set_binning` and `set_image_mode` (before anything is saved), `motion.move_turret`, the Session's objective writers (`select_objective`, `assign_turret_objective`, `clear_turret_objective`, `clear_current_turret_objective`) and `select_labware`. Each raises `HardwareCommandRefusedError` with `reason='exclusive_activity_running'` and `holder='recording'`. A home of any axis is refused the same way, since a home holds the whole scope. X/Y/Z moves, LED, gain and exposure stay open to an API caller; a recording's feed-death bound follows an exposure raised mid-recording. The LumaViewPro GUI locks its whole control surface during a live recording, Record/Stop excepted.

Both refusal errors say busy-with-what: `holder` carries what holds the microscope at refusal time (`'protocol'`, `'recording'`, `'diagnostic'` or `'home'` for the exclusive-activity owner, `'autofocus'` for a sweep that did not stop when its run ended, None for refusals that are not holder-shaped), and `holder_trigger` carries who started the run behind that holder -- the run holding the scope, or, for an `autofocus_running` refusal, the run that dispatched the sweep: `'gui'`, `'rest'` or `'api'` (`Caller`, `modules.caller`). The two are different axes: `holder` is the activity, `holder_trigger` who started it. A recording holder has no starter recorded -- its kind is the whole answer, and `holder_trigger` is None. File-drain refusals (`files_writing*`) carry the just-finished run's starter so a poller can report whose files are draining. Both fields are published: a REST problem for `ProtocolRunRefusedError`, `RecordingRefusedError`, `DiagnosticRefusedError` or `HardwareCommandRefusedError` carries `holder` and `holder_trigger`. While a control session is held by another client, each of these refuses with `control_session_held`: `control_session` carries the session's name, `holder_trigger` the host its taker came through, and `holder` stays an activity kind, None when nothing runs. The `message` names a run by its kind in words ("The Z-stack run is using the microscope"); branch on `holder`, `holder_trigger` and `control_session`, show `message`.

**Opening hyperstacks in Fiji:** the container is OME-TIFF; channel color travels as OME `Channel.Color`. Open via `Plugins > Bio-Formats > Importer` with **Color mode = Composite** (the choice persists per user through that dialog). A plain `File > Open` renders ImageJ's default LUTs, not the file's channel colors.

**Run-state semantics:** `session.live_work` lists a `'protocol'` item while a run is under way -- protocol runs, single scans, z-stacks, autofocus scans, and the standalone Autofocus button's run included -- from its claim's grant until its cleanup has ended, and for a run started inside `session.diagnostic_claim()` beside the `'diagnostic'` item, whose holder stays the diagnostic (so a refusal still names the diagnostic). The short post-run file-drain window (files still writing after the run finished) lists no run and a `'run_files'` item instead, its `left` counting the writes still to land, and `session.controls_lock.run_lockout` stays True through it, so a poller that must wait for the disk to settle waits for that item to go. A live video recording is not a run: it is a `'recording'` item. Over REST all of it is `GET /api/v1/status`.

### Post-processing

Builds over a folder of captured images are session members, the same ones LumaViewPro's Post-Processing buttons call:

```python
pp = session.post_processing

pp.stitch(folder, mode='quality')            # one of pp.stitch_modes(); one mosaic per tiled group
pp.zproject(folder, method='Max')            # one of pp.projection_methods()
pp.composite(folder)                         # output format and per-channel blend thresholds
                                             # are the user's configuration, as a run's merge uses
pp.video(folder, frames_per_sec=None,        # None: a recording plays at its own measured rate
         timestamp_overlay=False)
pp.enhance(target)                           # Quick Enhance: one image, or every image in a folder;
                                             # derived files are written beside their sources;
                                             # PostProcessingRefusedError ('unreadable') when target
                                             # is a file that is not an image
pp.count_cells(folder, method=method)        # a CellCountMethod; writes results.csv into the folder

from modules.post_processing_api import ZProject, Stitch, Video
pp.build(folder, zproject=ZProject(method='Max'),  # several builds in one call, in the
         stitch=Stitch(mode='quality'),            # order that keeps the data honest
         composite=True, video=Video())

def on_progress(percent, text):              # optional on every member: percent done, and a
    print(percent, text)                     # status line or None; called on the build's thread
pp.stitch(folder, on_progress=on_progress)
```

Each member blocks until its build finishes and returns a published record whose `message` says what was made, in words written for a person: `stitch`, `zproject`, `composite` and `video` return a `BuildResult` -- `new_count`, the files made; `output_root`, the folder; `artifact_paths`, each file made, where it landed; `degraded_outputs`, a `DegradedOutput(path, algorithm, fallback_from, fallback_reason)` for each file a fallback algorithm made -- a video built from a manual recording names its file as a protocol folder's build does; `excluded_inputs`, an `ExcludedInput(kind, count)` for each kind of file in the folder the build did not take -- a stitch takes no composites, because a stitched composite is built by Composite from the stitched channels, so stitch first and run Composite on the folder after, or both in one call with `build`; a build's own earlier outputs are not among them, and the message names the same files with that route; a recording's frames, which only Video takes, are never among them either, and a stills build over a folder whose only images are a recording is refused `no_data`, its message saying Video builds it; `enhance` returns an `EnhanceResult` -- `output_folder`, and `created`, an `EnhancedFile(source_path, output_path, recipe_path)` for each derived file; `count_cells` returns a `CellCountResult` -- `results_path` and `counted`, the images counted. Every path in them is a `pathlib.Path`. A folder that is not there, or a file given as the folder, is refused as any path is (`LiveFolderPathRefusedError`, `not_found` or `not_a_folder`), asked on the build's lane when the build starts. A build does not start while something holds the scope -- a run, a recording, a diagnostic, a home -- or while a finished run's files are still being written, since it would share the host's cores and disk with that work: it raises `PostProcessingRefusedError` with `reason='scope_held'` (its message naming the holder) or `'files_writing'`, both refusals of the scope's state (REST 409), asked on the build's lane when the build's turn comes, so a build queued behind another is refused if a hold began while it waited. A run or a recording is not started while a build is on the lane (`post_processing_running`), so a build never shares the host with a capture; a run's own builds (its hyperstack, its composite merge) run on the same lane as these members, after the run lets go of the scope, and the next run waits for them. It raises `PostProcessingRefusedError` (`modules.exceptions`) when the folder is there and cannot yield the output (no images, no groups the build can combine, or a setting it cannot use: an unknown stitch mode or projection method, a playback rate below 1). A build stays inside the folder it was given: a run record that is malformed, or that names its protocol outside the run folder, or a run protocol with a step cell that is neither blank nor its column's type, is refused as `protocol_data_unreadable`, its message saying which; a post-processing record the build cannot read is moved aside as `protocol_post_record.tsv.unreadable` (`.unreadable_001` and on when that name is taken, so no earlier copy is replaced) and a new record is started; a protocol whose steps would name an output outside the folder is refused as `output_outside_folder` before any output is written. A group whose images cannot be used -- a frame that is missing or will not decode (a write cut short), frames of different sizes, or pixel types that cannot share one output -- is not built and leaves no file behind, and the build goes on with the other groups; so is a video none of whose frames could be read. So is a group missing a frame its run recorded -- one deleted from the folder since, or a capture that failed during the run -- named with the frame and why: a group is built from every frame its run recorded or not at all. So is a group missing a derived image the folder's post-processing record names -- a projection deleted since it was made -- named with the image and the build that makes it again; a group every one of whose derived images is gone was removed, not damaged: it is not built, the message names it ("2 group(s) not built: their Z-Projection images were removed from the folder"), and the build is complete. A group spanning scans (a stills hyperstack, a timelapse video, and the run-end hyperstack) is not built for one missing scan; its frames stay on disk. A frame the run recorded for a group the build would not have combined (one image) fails nothing. It raises `PostProcessingFailedError` when the build did not produce everything asked of it, its message naming each group it could not build and why; its `produced_paths` lists what was written, and its `excluded_inputs` what the build did not take, as a complete build's result does. A cell count refuses a folder with nothing it could read before touching `results.csv`, and replaces an existing `results.csv` only with a complete new one.

A derived file's name says how it was made, oldest build first: a stitch of Max projections is `..._zproj_max_stitched.tiff`, a Max projection of Fast Preview mosaics `..._stitched_FastPreview_zproj_max.tiff`. A composite's name, and a video's, ends in a short code of the settings that decided its pixels (a composite's format and each channel's colour and blend threshold; a video's frame rate, timestamp overlay and colour), so a rebuild with new settings makes a new file beside the old one, and a rebuild with the same settings makes nothing new, unless the file was deleted since: an output the record names and the folder lacks is built again. A folder holding two derivations of the same images (Max and Sum projections of one stack, Quality and Fast Preview mosaics of one scan) builds from each on its own: a stitch makes a mosaic of each projection, never one of both. A group that still holds two images for one place -- one tile position in a stitch, one slice in a projection, one scan in a timelapse video, one plane in a composite or a hyperstack -- is not built: the build fails that group, its message naming the files and why (two acquisitions, where steps Added together make one output, or two derivations an earlier release made without recording how, to be built again with this release), and builds the rest. The folder's `protocol_post_record.tsv` (version 4) keeps each derived file's `Derivation`: its builds in order with every setting, `ZProject(method=max)>Stitched(mode=quality)`. A record an earlier release wrote is read with each setting unknown (`ZProject(?)`), and an output built from one carries the bare token (`..._zproj_stitched.tiff`); a release from before version 4 sets a version-4 record aside as unreadable.

`build` runs the builds whose settings records it is given -- `ZProject(method)`, `Stitch(mode)`, `Video(frames_per_sec, timestamp_overlay)`, each defaulting as its own member does, and `composite=True`, a composite having no settings of its own; one left None is not run -- in the one order that keeps the data honest: z-project, stitch, composite, video. A projection is made of raw slices, never of mosaics, and a stitched composite from the stitched channels, never by stitching composites. Each build takes only what the build before it in the call carried forward -- what that build made, and the images it did not consume, such as an ordinary Add step's images beside a tiled stack -- so the call makes the chain's product and its inputs, not every combination the folder allows: z-project, stitch and composite of a 2x2 tiled, 3-slice, 2-channel scan make 8 projections, 2 stitched projections and 1 composite, where the three members run one by one make 36 files. An image already in the folder from an earlier build is not the chain's input. Every chosen build's settings are checked before any build starts (`invalid_setting`, the build's `operation` named; a call that chooses no build is refused with `operation` `Post-Processing`), and the call runs as one task on the post-processing lane, so no other build touches the folder between its steps; `work()` names the build it is on ("Building 2 of 3: Stitch"). It returns a `BuildsResult`: `built`, a `BuildStep(operation, result)` for each build run in order, its `result` the `BuildResult` that build's own member returns; `skipped`, a `SkippedBuild(operation, message)` for each build that had nothing to do in the folder (a video on a single-scan run, the build's own `no_data`), the next build taking what the skipped one was given; and `message`. Any other refusal or failure stops the call and is raised as that build's own outcome, unchanged, its `operation` naming it; the files the earlier builds made are in the folder and its record, and the same call made again rebuilds nothing already made and resumes at the build that stopped once its cause is gone. One that does not go -- a raw slice missing, which fails its projection every time -- stops the call there on every run: build that folder with the members one by one. A folder of a manual recording takes `video` alone.

A cell-count method is a `CellCountMethod` record (`modules.cell_count_method`). `CellCountMethod()` is the method LumaViewPro's panel starts from, which `pp.default_cell_count_method()` also answers (over REST `GET /api/v1/post_processing/default_cell_count_method`); build the one you need, or change one with `dataclasses.replace`:

```python
import dataclasses
from modules.cell_count_method import CellCountFilters, CellCountMethod, FilterBounds

method = CellCountMethod(
    filters=CellCountFilters(area=FilterBounds(min=20, max=None)),  # square microns; None is open
)
method = dataclasses.replace(method, threshold=30)
pp.count_cells(folder, method=method)

pp.save_cell_count_method(method, 'methods/cells.json')   # the file LumaViewPro's panel saves
method = pp.load_cell_count_method('methods/cells.json')
```

Over REST a method is sent as an object of every one of its fields, as `GET /api/v1/post_processing/default_cell_count_method` answers it: start from that answer and change what you need.

To see what a method keeps before counting a folder, `pp.preview_cell_count(image, method, significant_bits=..., pixel_size_um=...)` counts one image you hold, as `count_cells` counts each file, and answers a `CellCountPreview`: `image`, the image with the kept regions outlined, and `counts`, a `CellCounts` holding the values of that image's `results.csv` row (`num_cells`, `total_object_area`, `area_unit`, `total_object_intensity`). `significant_bits` is the image's payload depth and `pixel_size_um` the scale it states (microns per pixel, or `None`), which the method's `pixels_per_um` overrides, as in the count. It counts on the caller's thread, not the post-processing lane, so it answers while a build or a run goes on, and it hands back an image array, so it is in-process only.

A form that bounds the area and perimeter with sliders asks `pp.cell_count_size_scale(method, shape=image.shape, pixel_size_um=...)`, the range the panel offers: a `CellCountSizeScale` holding `pixels_per_um`, the scale the count measures at (`None` for pixels), and for `area` and `perimeter` a `SizeSliderStops`: the slider `positions` of its stops, the `sizes` they stand for (stretched at the low end, so small sizes get most of the travel), and `reach`, the largest size the image holds. In Python, `size_at(positions)` and `positions_of(sizes)` map between the two. With no `shape` (no image yet) a position stands for its own size. A `shape` that is not an image's rows and columns, each at least 1, or a `pixel_size_um` that is not a positive number, is refused (`PostProcessingRefusedError`, `invalid_setting`), as `preview_cell_count` refuses that scale.

Each image is measured at the scale it states: a LumaViewPro capture carries its pixel size, so its areas and perimeters are in square microns and microns. An image that states no scale (a PNG, a capture saved without one) is counted in pixels. `results.csv` has one row per image with `file`, `time`, `num_cells`, `total_object_area`, `area_unit` (`um2` or `px2`, that row's unit) and `total_object_intensity`. Setting `pixels_per_um` overrides the scale for every image, for example on external images whose scale you know. The area and perimeter filters are in microns, so an image counted in pixels with either one set is not counted: the count goes on with the rest and raises `PostProcessingFailedError` naming it.

| Field | Meaning |
|---|---|
| `pixels_per_um` | `None` (the default) to use each image's own scale; or a positive number of camera pixels per micron, which overrides it for every image |
| `fluorescent_mode` | `True` (the default) for bright objects on a dark background; `False` inverts the image first |
| `threshold` | brightness threshold, percent of full scale; 20 by default |
| `filters.area`, `filters.perimeter` | a `FilterBounds(min, max)` in square microns and microns; open (`None`) by default. A set bound refuses an image counted in pixels |
| `filters.sphericity` | a `FilterBounds`, 0 to 1 |
| `filters.intensity_min` / `intensity_mean` / `intensity_max` | a `FilterBounds` for each region's minimum, mean and maximum intensity, percent of full scale; 0 to 100 by default |

A method the count cannot use cannot be built: `CellCountMethod` and `CellCountFilters` raise `PostProcessingRefusedError` with `reason='method_invalid'` as they are built, the message naming the field (a `pixels_per_um` that is not a positive number, a `fluorescent_mode` that is not `True` or `False`, a threshold or bound that is not a number, a filter whose `min` is above its `max`), and a method sent over REST is refused the same before the count is queued. A method file is the nested JSON LumaViewPro has always written (`context`, `segmentation`, `filters`, and a `metadata` entry marking it a method file); `load_cell_count_method` refuses a file it cannot read or that is not a method file (`method_unreadable`) and one holding a method the count cannot use (`method_invalid`, naming the field by its place in the file), the message naming the file. Both run on the caller's thread, not the post-processing lane. A method file saved before LumaViewPro read each image's scale (`metadata.version` `'1'`) carries the old fixed default of 1.0 pixels per micron: the load drops it, so the method uses each image's own scale, and says so once.

`graph_options` and `graph`, below, read a `results.csv` with its `time` column as times (the count's own time format). They read any CSV in UTF-8 (the count writes its file in UTF-8 on every host): every cell is the text written, a column whose every cell is a number is a column of numbers (whole numbers when every cell is one), and any other column is text, so a cell such as `NA` is never read as a missing number; a graph takes numbers and times as X and numbers as Y. A file that cannot be graphed (not UTF-8, a NUL, a quote that does not close, a row of the wrong length, a repeated column name, a time the count did not write, no column of numbers) is refused: `PostProcessingRefusedError` with `operation='Graphing'`, `reason='results_unreadable'`, its message naming the file. A time X takes the trendlines `'Linear'`, `'Quadratic'` and `'Exponential'`; two number columns also take `'Power'` and `'Logarithmic'`. A fit the values cannot give is refused, such as a missing value, a 0 under a log, or too few distinct X values for the kind (`PostProcessingRefusedError`, `operation='Trendline'`, `reason='fit_impossible'`, naming the values and how many).

The graph LumaViewPro's Object Plotting popup draws is the session's, so any client makes the same one. `session.post_processing.graph_options(results, x=None, y=None)` answers a `GraphOptions`: the columns of the results file each axis can take (`x_axes`, `y_axes`) and the trendline kinds the axes given so far take (`trendlines`): every kind until an axis rules some out, so a time X offers no `'Power'` or `'Logarithmic'` before a Y is chosen. A client of its own fills the same form. `graph(results, path, x=..., y=..., trendline=None, x_label=None, y_label=None, title='')` draws column `y` against column `x`, with the trendline over it when a kind is named, and writes it to `path` as a PNG, answering a `GraphResult` (`message`, `path`); each label defaults to its column's name, and a missing folder is made where `path` sends it. It runs on the post-processing lane like the builds; over REST both are `/api/v1/post_processing/graph_options` and `/api/v1/post_processing/graph`, the paths under the live folder. `draw_graph`, with `graph`'s arguments but no `path`, returns the matplotlib `Figure` itself, for a GUI in the same process to show; it does not cross the wire. A `path` that does not end in `.png` is refused `wrong_extension` and nothing is written; no results file, or no `x` or `y`, is refused `graph_incomplete` ("choose a results file and an X and a Y axis first"); a column that axis cannot take is refused `axis_not_plottable`, naming the columns it can; the reader's and the fit's refusals are as above. A folder or file the disk will not write is `PostProcessingFailedError` (`operation='Graphing'`), naming the file and the system's reason.

Which images make one output: a composite, a z-projection, a video and a hyperstack each build from one channel group, the steps' `Channel Group ID`. Every step of one `add_step` call shares one group, and a generated protocol (`new_protocol`) gives every channel, tile and z-slice of one position one group; so two adds at one stage position give two composites, never one made from both. A stitch is not grouped by it: it joins one color's tiles across positions, added together or not. A still-image group that holds two images for one input -- two of one channel, or two different transmitted channels in a composite, or two stills of one scan in a timelapse -- is not built: the build's `PostProcessingFailedError` names the images, and the other groups are built. That is what a run folder from before steps carried their group yields when one position was added to more than once, since its steps all read `-1`. A composite run (`run_composite`) with two transmitted channels set to capture is refused at its start (`ProtocolRunRefusedError`, `composite_two_transmitted_layers`), before any capture; `update_step` refuses giving a step a channel another step of its add takes (`channel_already_in_add`).

The builds run on the session's own post-processing lane, one at a time in the order asked. A protocol run writes its images on a different lane, so a long build never delays a run's writes. The tiling config and the turret are the session's own: a caller passes neither.

### Run state and locks

The session derives all run and lock state from its owners: the activity
claim, the recording, the run's file writer. These are the members a
GUI-quality client binds its widget state to -- the records LVP's own GUI
writes into its kv properties at each run-state edge:

```python
session.controls_lock            # in-process; over the wire status.lock. ControlsLock(run_named,
                                 # control_named, recording): run_named what holds the run surface as a
                                 # sentence a person reads ('The Z-stack run is in progress.', 'A diagnostic
                                 # is in progress.', 'A home is in progress.', "A protocol's files are still
                                 # being written."), None while nothing does; control_named the control
                                 # session held ('plate reader has remote control.'), None while none is;
                                 # recording a manual recording is live (not its file drain); .run_lockout
                                 # (run_named or control_named set), .controls_locked (that, or recording)
                                 # and .named (the sentence shown: the session's, else run_named, else the
                                 # recording's) read from them. A display that pauses for a run reads
                                 # run_named, so it keeps showing the scope under an idle control session
session.held_by_other(run)       # the scope is held by anything but `run` (the run's operation);
                                 # False for the run itself, so its Stop stays live; None: held at all
session.last_run()               # the last run's operation, live or ended, whoever started it (the GUI,
                                 # a script, the SDK, a client); None before any run. On the wire a POST
                                 # answering that run's one job, whoever reads it: its progress, outcome
                                 # and job_ended (told at the read for a run already ended); its per-step
                                 # events stay with its starter. A deleted finished job answers 404 on its
                                 # id. A GUI run's operation settles on the GUI's thread, so a client
                                 # watching one through an application quit sees the stream close, not
                                 # its job_ended
session.protocol_files_stalled   # the drain's write in flight has stopped progressing, judged by the
                                 # same threshold that refuses a new run (files_writing_stalled)
session.protocol_files_stuck_write  # the write in flight, named for a stall report
session.manual_recording.is_recording  # a manual recording is LIVE (not its file drain)
session.live_work                # in-process; over the wire status.live_work. Everything the session is
                                 # still doing, one read: .work is a tuple of
                                 # WorkItem(kind, name, left, percent, discardable) -- what holds the scope, a run starting
                                 # or ending, a recording's file, a run's video, a run's images, its post-run
                                 # builds, post-processing builds running and queued, a support report or logs
                                 # zip, a still being saved; left counts frames, images or builds still to
                                 # finish (poll it: the count changes between edges, the listener fires
                                 # only on them), percent is a build's last progress, discardable marks the video
                                 # frames discard_close_drain drops; .closing while the close runs,
                                 # .closed once the session shut down
session.status                   # what the scope is doing, one read, the first a client makes: Status(live_work,
                                 # lock, control, run_refused, axes, parts, camera_streaming) -- live_work as
                                 # above; control the control session held, or None; lock as
                                 # session.controls_lock, ControlsLock(run_named, control_named, recording)
                                 # with run_lockout, controls_locked and named read from those, the Session's policy
                                 # for a person's controls and not a refusal; run_refused the refusal a run
                                 # would get now for the scope's state alone, sent as a refused call's problem
                                 # less type, status and instance, or None (None does not promise a run starts);
                                 # axes as scope.motion.axis_positions(); parts as bring_up_record().parts;
                                 # camera_streaming as scope.imaging.is_streaming(); each read from its owner in
                                 # turn, not under one lock, so two fields can be a moment apart
session.discard_close_drain()    # the escape from a wait on video frames: drop every queued frame of a recording's
                                 # file and a run's video; each file finishes with what was written

def on_run_state():              # called on EVERY run-state transition: an activity taking
    print(session.controls_lock) # or releasing the scope, a manual recording going live, going
                                 # to its drain and finishing, a run's video step finishing,
                                 # a run's files all landing, the close beginning
                                 # (live_work.closing) and ending (live_work.closed): stop
                                 # your own listeners when it begins;
                                 # re-read the derivations (level semantics, no payload)
session.add_run_state_listener(on_run_state)
session.remove_run_state_listener(on_run_state)   # stop being told
```

### Outcomes

A call that fails or is refused raises to you; that exception is the call's outcome, and it is yours to report. Outcomes that no caller waits on -- a camera stream that stops, a run that ends short of its captures, a recording the disk floor stopped, a finished run's file writer that stalls, a notice that a capture was saved without its position -- reach you through the session's outcome subscription instead:

Every exception the scope raises is a `LumascopeError` (`lib.lumascope_error`), so one `except LumascopeError` catches them all, and every one is still an `Exception` (or the built-in it always was, such as `ValueError`), so no `except` you already have changes what it catches. Each says what it is by one kind among its bases: a refusal (`Refusal`, `modules.exceptions`: declined, nothing broke), a notice (`Notice`), a quiet outcome (`Quiet`) or a fault (`Fault`, `lib.lumascope_error`: something failed); `notification_center.outcome_of(exception)` reads it for you. `ConfigError`, `ProtocolError`, `CaptureError`, `MoveNotCompletedError`, `HomingFailedError` and `HardwareError` name a subject, not a kind: catch them to catch every outcome about it, but none is raised itself (constructing one raises `TypeError`), so what you receive is always the child that says what happened, with its `reason` and `title`. A `TimeoutError` from the scope means only that your own wait ran out (`wait_until_finished_moving`'s `timeout_s`, `run_composite`'s `merge_timeout_s`), and is quiet; when the scope's own bound on a call runs out with the call still queued or running, it raises the fault `LaneCallStillRunningError` (`still_running`, `.member`), never a `TimeoutError`: the call may still finish. A program or installation defect is `ConfigurationDefectError` (a `ConfigError` and a fault, its `reason` naming the broken contract, such as `settings_not_prepared` or `scope_not_composed`), never a refusal: nothing you send changes it.

A fault in the hardware itself is a child of `HardwareError` (`drivers.exceptions`), each with its own `reason` and `title`, so `except HardwareError` catches every one and a REST caller branches on the reason (500):

| Class | `reason` | What failed |
|---|---|---|
| `BoardNoReplyError` | `board_no_reply` | A controller board did not answer in time; it may answer the next ask |
| `BoardReplyUnreadableError` | `board_reply_unreadable` | The board answered, and the answer could not be read |
| `BoardReportedError` | `board_reported_error` | The board answered that it could not do it: a firmware error, a refused status, a handshake it did not complete |
| `BoardPortFailedError` | `board_port_failed` | The serial port to the board failed or is not open: a cable or power state |
| `MotionNotCompletedError` | `motion_not_completed` | An axis did not home, arrive or settle in its time (a move or home answers it as the `__cause__` of `MoveFaultError` or `HomingFaultError`) |
| `LedFirmwareWedgedError` | `led_firmware_wedged` | The LED controller's firmware is stuck; power-cycle the unit |
| `CameraReadFailedError` | `camera_read_failed` | A read or probe of a camera value failed |
| `CameraWriteFailedError` | `camera_write_failed` | A camera setting write failed (a setter answers it as the `__cause__` of `CameraSettingRejected`) |
| `CameraNoAnswerError` | `camera_no_answer` | The camera did not answer a setting write in time; it is still on the bus and may answer it sent again |
| `ConfigReadError` | `config_unread` | The board's per-unit config could not be read, or a value in it is not a number |
| `PartAbsentError` | `part_absent` | A driver was asked for a part it does not have; the API refuses such a command first (`axis_absent`, `not_connected`), so only a caller below the API meets it |

The words of all but `LedFirmwareWedgedError` and `ConfigReadError` carry the driver's or SDK's own text, so the person is shown the title and the words go to the log.

```python
def on_outcome(n):               # runs on the thread that reported it: return promptly, never wait on a run or the scope
    print(n.kind, n.title, n.message, n.reason, n.shown, n.outcome_id)

session = ScopeSession.create(settings=settings, outcome_listener=on_outcome)   # hears bring-up too
# or, for a client that comes later:
session.add_outcome_listener(on_outcome)
session.remove_outcome_listener(on_outcome)    # shutdown() removes it too
```

Each call carries one `Notification` (`modules.notification_center`):

| Field | Meaning |
|---|---|
| `kind` | `OutcomeKind`, a string enum: `'refusal'` (declined; nothing broke), `'fault'` (something failed), `'notice'` (information; nothing failed), or `'unclassified'` (a notification posted without declaring its kind; these are being moved to declared kinds). A fifth, `'quiet'` (nothing failed and nothing was declined, such as a stop sent to a run that has already ended), is never delivered here, because a quiet outcome is never shown; a caller that receives one as its answer reads it from the exception's type (`notification_center.outcome_of`) |
| `title`, `message` | The heading and the sentence, written for the person |
| `reason` | The outcome's machine-readable code, stable enough to branch on (a refusal's, a fault's or a notice's). Every outcome the scope raises declares one, and its `title` is its type's (`tests/guards/test_every_outcome_joins_the_root.py`); empty only for an exception from outside the scope's own, such as a Python error |
| `remedy` | A `Remedy` when the outcome has one action that answers it: `session.apply_remedy(n.remedy)` takes it |
| `solicited` | True when it answers a request a person or caller just made |
| `fatal` | True for a fault that ends what was running |
| `shown` | Whether the scope says this is for display now. False when it was muted: during a scan, a protocol, an autofocus scan of every step, or a run under a diagnostic's claim (non-fatal outcomes; a standalone autofocus, composite or z-stack is shown, whoever started it); during a standalone autofocus, composite or z-stack, a non-fatal outcome identical to one already shown in that run (same category, title and message); an outcome you did not ask for whose category and title were shown less than 10 s ago, a fatal one included (an answer to a request is never muted this way, and showing it restarts the 10 s); or during shutdown. You receive muted outcomes too; a display shows only `shown` ones |
| `outcome_id` | One per outcome. An outcome delivered muted and later shown (because someone asked for it) arrives twice with the same id, `shown` False then True; keep the first of an id to count each outcome once |
| `wall_time`, `timestamp` | Wall-clock seconds, and a monotonic time for ordering within the process |
| `severity`, `category`, `operation_key` | The log level, the subsystem, and the operation a notice-then-outcome pair is about |

A refusal raised to you carries `cause` beside `reason` (`RefusalCause`, `modules.exceptions`, a string enum): whether the same request could succeed later with you changing nothing. `'state'` -- the scope's state refused it (a run is live, an axis is not homed, the drive is unplugged), so the same request may succeed once that changes; a REST server answers 409. `'request'` -- as sent, it cannot succeed on this scope (a bad argument, a value outside a fixed rule or this hardware's range, hardware this scope does not have, a handle that no longer names anything live), so change it before sending it again; a REST server answers 422. A client that retries refusals retries only `'state'`. The cause is declared on the refusal's type, per reason where one type has both (`ProtocolRunRefusedError`, `HardwareCommandRefusedError`, `LiveFolderPathRefusedError`), never decided by the code that raises it. A type that declares none cannot be imported (`tests/guards/test_every_refusal_declares_its_cause.py`).

Every argument you pass to an API member is first held to the type the member's signature declares, for every caller alike -- a script, the GUI, the engineering plugin, one member calling another. A value of another type is refused before the member runs with `ArgumentTypeRefusedError` (`modules.exceptions`, a `TypeError`), reason `wrong_argument_type`, cause `request`, its `member`, `argument` and `declared` naming the member, the parameter and the type it takes. Every member refuses it, so no member's `Raises:` section lists it. A number is taken by what it means: an `int` where a `float` is declared, and a numpy scalar (`np.float64`, `np.int64`, `np.bool_`) where the Python number or bool is, each reaching the member as the plain Python value. A `float` is not an `int`, even `2.0`, and a `bool` is never a number. None is taken only where the signature says `| None`, and a `FilePath` takes a `str` or any `os.PathLike`. The check is of the type alone: NaN is a float, and a name's spelling, a number's range and a dictionary's contents are the member's to judge, as below. Over REST a JSON value of the wrong type is refused by the server's own check first, as `invalid_request` (422).

A number you pass is a finite number: NaN or an infinity is refused with `ArgumentRefusedError` (`modules.exceptions`, a `ValueError`), reason `not_a_number`, its `argument` naming the parameter and `value` holding what you gave, before any range is checked and before anything is commanded or stored. This holds for the camera's gain, exposure, frame size, black level and auto-gain target, a stored gain or exposure asked through `applied_gain_db_for` / `applied_exposure_ms_for`, an LED current, a move's position or distance (`ignore_limits=True` included), plate and stage coordinates, a well position, a z-stack's range and step size, a focus Z, and the capture and wait times. A finite number outside a range is refused by the range's owner as before.

A wait you pass is from 0 to 30 days: `wait_until_finished_moving`'s `timeout_s`, the run waiting forms' `timeout_s` (`run_single_scan` and the rest), `Operation.result` / `exception`'s `timeout`, `capture_and_wait` and `get_image`'s times, `enter_led_engineering_mode` and the `send_diagnostic_command` forms' `timeout_s`, `run_camera_bandwidth_test`'s cutoff, and the simulator's `delay_reply` and `reboot`. A negative wait, or one past 30 days, is refused with `ArgumentRefusedError` reason `out_of_range`, `limits` holding the range, before anything starts: a run waiting form has started no run, and a capture whose sum would wait longer than 30 days in all is refused the same way. The bound is the same on every host; past it, a wait the operating system cannot hold used to raise `OverflowError` out of the wait on one platform and not another.

An axis is named exactly `'X'`, `'Y'`, `'Z'` or `'T'`, in capitals. Every member that takes an axis -- the moves, the position and state reads, `get_axis_limits`, the limit switches, the precision mode, `jog_step` and `read_motor_drv_status` -- refuses any other name (`'x'`, `'Q'`) with `ArgumentRefusedError` reason `axis_unknown`, its `offered` holding the names the member takes, before the scope's hardware is asked. A home takes `'Z'`, `'T'` or `'ALL'`; `jog_step` takes `'X'`, `'Y'` or `'Z'`. An axis this model lacks is a name: a command for it is refused as absent hardware, and a read answers as for a missing axis (`get_axis_state` `'unknown'`, a position None). The generic movers refuse `'T'` with reason `turret_moves_by_slot`: the turret moves only through `move_turret(slot)`. A plate move given a Z is `plate_frame_axis`, refused at the door. `set_motor_fan_duty` takes a whole percentage and refuses one outside 0 to 100 (`fan_duty_out_of_range`), with nothing sent. The camera diagnostics refuse a count or time outside what they take (`out_of_range`, its `limits` the ends, which are included): `run_grab_lifecycle_benchmark` takes 1 to 10,000 cycles, 0 to 60,000 ms between them and a slow threshold of 0 to 600 s; `run_pylon_diagnostic_probe` takes 0 to 3,600 s. A stage offset handed to `stage_to_plate`, `plate_to_stage`, `plate_transform` or `step_targets` is a `StageOffset`, and an axis of it that is no finite number is `not_a_number`, naming it (`stage_offset.x_um`). The auto-gain settings handed to `set_auto_gain` or the camera's `apply` are read before the member acts: a key left out is `missing_key` and a value that is no number `not_a_number`, each naming the key, and a key the write does not take is `key_unknown`, its `offered` holding the keys it takes: `target_brightness`, `min_gain_db`, `max_gain_db`, `max_exposure_ms` and `min_exposure_ms`. Each is the request's (`cause` `REQUEST`): it fails the same way until the caller changes it.

A plate or an objective is named by its catalogue key (`list(scope.labware)`, `list(scope.objectives)`); a plate renamed since a file named it is still found under its old name and stored under its key. Every member that takes one -- `select_labware`, `set_protocol_labware`, `select_objective`, `confirm_objective`, `assign_turret_objective` -- refuses a name the catalogue does not hold with `CatalogueNameRefusedError` (`modules.exceptions`, a `ConfigError`), reason `labware_unknown` or `objective_not_in_catalogue`, its `argument` naming the catalogue owner's parameter (`plate_key`, `objective_id`) and its `offered` holding the catalogue's keys, before anything is stored. The run's `ProtocolRunRefusedError` uses the same reason, `objective_not_in_catalogue`, for a protocol step's objective; `argument` is on the catalogue refusal only. A turret slot that is not a whole number 1 to 4 is refused with `PositionOutOfRangeError` by every member that takes one. Each is the request's (`cause` `REQUEST`).

`new_protocol` builds one step per acquiring layer at every well of the session's plate, then tiles them as `apply_tiling` does and stacks them as `apply_zstacking` does, so New's grid and stack are ordered, grouped and refused as Apply's are. What it refuses is `ProtocolRunRefusedError`, raised to you and posted to no listener (over REST, answered 422 and shown to no one): a grid this installation does not offer (`tiling_unknown`), a stored `tiling_overlap_percent` outside 0 to 50 (`overlap_out_of_range`, here and in `apply_tiling`), a tile past the X or Y travel (`tiles_outside_travel`), a grid on a scope with no X/Y motor (`positions_unreachable`), and the stack's refusals as `apply_zstacking`'s. Each is the request's.

The subscription is process-wide: a listener hears every session in the process, and the mute and dedup state is shared. A listener that raises is logged at ERROR with its traceback and the others are still told. A session that already exists has finished bring-up; what bring-up reported is heard only by a listener given to `create`. What it found is held for a client that comes later:

```python
record = session.bring_up_record()         # modules.lumascope_api.bring_up.BringUpRecord
for part in record.parts:                  # 'motor', 'led', 'camera', in that order
    print(part.part, part.up, part.expected, part.cause, part.detail)
record.missing                             # the parts this scope's model has and lacks
record.part('led').cause                   # ArgumentRefusedError (part_unknown) for a name not in parts; None, or why: 'not_detected', 'port_in_use', 'not_responding',
                                           # 'connect_failed', 'no_driver'; for the camera
                                           # 'camera_port_in_use', 'camera_not_detected', 'camera_driver_unavailable'
                                           # (none found, and a camera driver did not load on this installation; the
                                           # detail names it and why), 'camera_firmware_missing' (a Classic's firmware
                                           # file is not installed), 'camera_driver_unusable' (on Windows, a Classic is
                                           # bound to no driver or one LumaViewPro cannot use; the detail names the
                                           # device and its driver; its report carries the remedy
                                           # repair_camera_driver on an installed Windows build), 'camera_not_initialized'
                                           # (a camera found and not opened, one another application holds included);
                                           # on an LED board that came up, 'safety_off_failed'; on a camera that came up
                                           # and whose stream did not start, 'stream_not_started' (it stays after a
                                           # later enable_streaming() that streams)
record.substitution('binning')             # Substitution(setting, saved, used), or None; also 'frame'
record.settings_set_aside                  # SettingsSetAside(path, reason), path a pathlib.Path, while the app runs on the shipped
                                           # template because the user's file could not be used; else None
```

Bring-up reports these once, as outcomes, to the listener given to `create`: a camera that did not come up is a fault under its own heading (`reason` as above), an LED board missing on a scope whose other parts came up is `LedBoardUnavailableError` (`reason` the cause), a part the model has and lacks is listed in one `PartialHardwareError` (`'partial_hardware'`, each part with its cause), a refused connect-time LEDs-off is `LedSafetyOffNotTakenError`, a camera that came up and whose stream did not start is the start's own fault (`CameraStreamStartFailedError`, `'camera_stream_start_failed'`, for a START an FX2 refused; the scope comes up, its camera connected and not streaming, and `scope.imaging.enable_streaming()` tries again), and when nothing came up the one outcome is the notice `NoHardwareDetectedNotice` (`'no_hardware'`). A camera that came up without part of its connect-time configuration (a user set, the pixel format, the image orientation, ...) runs with the rest applied and is reported once as the notice `CameraConfigNotAppliedNotice` (`'camera_config_not_applied'`, `not_applied`: the parts by name). A saved binning the camera cannot take is substituted and reported once as a notice (`'binning_substituted'`); the record holds the saved value beside the one that ran. A saved frame larger than the scope delivers at the binning applied is refitted to its maximum, reported once as the notice `FrameRefittedNotice` (`'frame_refitted'`) and recorded the same way (`substitution('frame')`), and the session stores the frame that ran, so the next bring-up has nothing to refit. A saved layer gain or exposure above the camera's range is kept as saved and applied at the camera's limit, and bring-up reports every such value in one notice `LayerSettingCappedNotice` (`'layer_setting_capped'`, `capped`: a list of `(layer, setting, stored, applied)`); the next bring-up on a camera that reaches the value applies it as saved. The session stores the binning that ran in the saved one's place, with the frame the camera delivered (the two are one geometry). The image mode is never substituted: every camera honours every mode. A manual scope's missing motor board is expected and reported nowhere.

### Holding the scope for a diagnostic

A script that drives the hardware directly -- a characterization, a
bench measurement, anything that homes, moves, lights or grabs outside a
run -- holds the scope for its duration, so a run or a recording cannot
start in the middle of it:

```python
from modules.exceptions import DiagnosticRefusedError

try:
    with session.diagnostic_claim():   # the claim, released when the block ends, even on a raise
        ...                            # drive the hardware
except DiagnosticRefusedError as e:    # a run, a recording or another diagnostic holds the scope
    print(e.reason, e.holder, e.message)
```

While the diagnostic holds the claim it counts as holding the whole scope:
`live_work` lists a `'diagnostic'` item, `controls_lock.run_lockout` and
`controls_lock.controls_locked` read True, a run start is refused
(`ProtocolRunRefusedError`, `exclusive_activity_running`), a recording start
is refused (`RecordingRefusedError`, `holder='diagnostic'`), and an objective
or labware change raises `HardwareCommandRefusedError`. `live_work` lists no
`'protocol'` item for it: a diagnostic is not a run.

A diagnostic that needs autofocus runs the public one under its own claim,
`runner.start_autofocus(channel, claim=held)` with the `held` the block yields.
That run acts inside the diagnostic: it is not refused by it and cannot end
it, and its non-fatal outcomes are muted (`shown` False) -- the diagnostic
gets the run's outcome from its handle and decides what to show. When the block ends with such a run still live, the release waits for
the run first; if it is still live after the wait, the claim stays held,
released when the run ends. A block that ended cleanly then raises
`ClaimStillLentError` (`claim_still_lent`, a fault; still a `RuntimeError`);
a block that raised keeps its own exception, and the kept claim is reported
once to the outcome listeners instead.

### Support report

Two members make the ZIP Etaluma support asks for. Each blocks until the ZIP
is written into the folder you name, which is made if it is missing, and
returns where it is. The folder is required: the session chooses none. The
GUI and the command-line report name the user's Desktop (the home folder on
a machine with no Desktop); a script names any folder it likes:

```python
import pathlib
from modules.exceptions import SupportReportNotSavedError

try:
    saved = session.make_support_report(        # minutes: boards, motors, camera, then the files
        output_dir=pathlib.Path.home() / 'Desktop',   # required; any folder
        include_bandwidth_test=False,           # True adds a camera frame-delivery timing
        on_progress=lambda pct, msg: print(pct, msg),
    )
    saved = session.make_logs_zip(output_dir='/data/support')   # seconds: logs, data folder, recent
                                                                # protocols, video receipts; no hardware
except SupportReportNotSavedError as e:         # no ZIP; the failure's own words, chained from it
    print(e.report, e)                          # 'support report' or 'logs zip'

saved.path      # the ZIP
saved.title     # 'Support Report Saved' or 'Logs Zip Saved'
saved.message   # the folder it is in, and the address to send it to
                # Both copy the data folder with the value of every "api_key"
                # in every current.json* (the live file, a rejected one, a
                # .saving a power cut left) blanked, in any section, a
                # plugin's own included, and every other byte kept; a copy
                # that fails fails the ZIP, and a file the settings saver
                # renamed away mid-copy is named in data/NOT_COPIED.txt.

health = session.plugin_health()   # modules.plugins.PluginHealth, or None before load_plugins()
health.namespaces                  # each namespace's NamespaceHealth: loaded, last_runtime_errors
health.not_loaded                  # PluginNotLoaded(name, version, reason) for each that did not load

session.plugin_api_level           # int: what this host does that a plugin may rely on ("Plugin API level")
session.app_version                # the application's version, '4.0.0-beta36'; None where the build has none
session.app_runtime                # how it was launched: 'source' (Python, a checkout or pip install), 'bundle' (a packaged build not installed) or 'installed'
```

The full report holds the scope for a diagnostic around its hardware steps
(see above); started while a run holds the scope, it skips those steps and
says so in the ZIP. A step that fails is written into the ZIP and the report
goes on: only a ZIP that could not be written raises. Neither member is
cancellable.

Both ZIPs carry `bring_up.json` (the session's `bring_up_record()`, each
part with `cause_words` beside its `cause`) and `plugins.json` (each plugin
namespace's health, and `not_loaded`: the plugins that did not load and
why). A session whose host never called `load_plugins()` writes
`{"plugins": null, "why": "no plugin registry on this host"}`.

#### Plugins

```python
session.load_plugins()     # the installed plugins (entry points 'lvp.plugins'), then the built-ins
session.unload_plugins()   # each plugin's unregister, last loaded first; shutdown() does it too
```

A session has plugins only when its host asks: `load_plugins()`, once,
after `create` (a second call raises `AlreadyStartedError`, `plugins_loaded`,
still a `RuntimeError`, and loads nothing). A script or a test that never calls it has none, whatever
is installed on the machine. Each plugin's `register` is handed the session
as its `ctx`; a plugin that does not load is reported and the rest load. An
installed plugin claiming a built-in's name keeps it, and the built-in is
the one reported as not loaded. No plugin is told of a settings change
(level 9): a spec that names `subscribes_to` keys is reported as not
loaded, naming the level its `on_settings_changed` hook needs.

A post-processing plugin that opts in (`auto_run_on_protocol_complete`) is
handed the folder of every Full Protocol run, whoever started it, once the
run's images are written and its hyperstack build has ended. It runs on the
post-processing lane, isolated from the run: the run's `files` does not wait
for it, and the scope takes the next run while it works. On that lane it may
call the post-processing members; a blocking call to a camera, motion or
file member raises. A run whose images are not all written hands its folder
to no processor.

`unload_plugins()` is for a host whose close must stop plugin work before
anything else, as the GUI's does before it stops runs;
`shutdown()` calls it first for every other host, and a second call does
nothing.

#### Plugin API level

`session.plugin_api_level` says what
this LumaViewPro does that a plugin may rely on, within its major version.
A version cannot say it: `version.txt` names the promoted release, and many
trunk commits share one beta number. A commit that changes behaviour a
plugin relies on raises the level by one. A plugin that needs a level checks
it at its own entry and refuses below it, naming both levels; a host older
than the level has no `plugin_api_level`, which reads as 0
(`getattr(session, 'plugin_api_level', 0)`). `PluginSpec.requires_lvp_version`
still states the major version.

| Level | Added |
|---|---|
| level 1 | `capabilities.camera_limits.analog_gain_db_max`; `capabilities.camera_reports_temperature`, and `get_camera_temperatures_degc` answering `{}` only with no sensor, `None` with no camera, and raising on a failed read; `save_camera_state` / `restore_camera_state` covering pixel format, frame size, binning and black level. |
| level 2 | `imaging.get_black_level_range()`: the range `set_black_level` accepts for the current pixel format. |
| level 3 | With no LED board: `led_on` and a lighting transition raise `not_connected`, the LED state reads (`get_led_state`, `get_led_states`, `save_led_state`) answer `None`, and `capabilities.led_max_ma` / `led_channels` are `None`; `led_on` for an LED the model lacks raises `axis_absent`. |
| level 4 | With no camera: every camera command raises `not_connected` (`MissingPart.CAMERA`) but an off whose end state holds; `get_gain_db`, `get_exposure_ms`, `get_width` and `get_height` answer `None`; after `disconnect()` the camera cache holds the no-camera values and `get_camera_temperatures_degc` / `get_camera_link_info` answer `None`. |
| level 5 | A plugin's `register`, `unregister` and `on_settings_changed` are handed the session (see "Plugins"); `on_settings_changed` is told changes made since load; engineering mode is `session.engineering_mode`, which the plugin turns on itself, and every run and still reads it (the runs' and the still's `engineering_mode` argument is gone); auto-run fires for every Full Protocol starter, on the post-processing lane, after the hyperstack build. |
| level 6 | `imaging.set_binning_size`, `set_pixel_format`, `set_conversion_gain_mode` and `set_line_noise_reduction`, and the Session's `set_image_mode`, `set_high_conversion_gain` and `set_line_noise_reduction`, return `None`: every failure raises, a camera gone before the body with `not_connected`; a toggle's mode on a camera without it raises `CameraSettingUnsupportedError`, its off is satisfied. |
| level 7 | `lvp_logger.git_revision()` takes no argument: the commit the running code was built from, read where the code lives (links resolved), so a launch from a folder of links into a clone names the clone's commit; from a GitHub ZIP's or an installer's `.git_archival.txt` first; a frozen build asks git nothing, so an installer built by a build script below v4 answers `None`. |
| level 8 | Every `Refusal` and `Quiet` type states its cause (`RefusalCause`, `modules.exceptions`), as `cause` or, where its reasons differ, as `causes` keyed by reason: a plugin's own refusal type that states neither is refused at import, and one built with a reason its `causes` lacks raises `TypeError`. |
| level 9 | The session saves every accepted settings write itself (`ScopeSession.create(settings_file=...)`), and `save_settings`, `SettingsSaveRefusedError` and `on_settings_changed` are gone; `PluginSpec.subscribes_to` is accepted empty, and a spec naming keys is not loaded, its reason naming level 8 as the last that told a plugin of a settings change. |
| level 10 | LED state is by address: `get_led_state` (a layer name or an address) answers an `LedReading` (`address`, `name`, `state`, `illumination_ma`), `get_led_states` a tuple of them for every address of the board, `save_led_state` an `LedSnapshot`; the LED listener is called with `(address, name, state, illumination_ma)`; `get_led_state` of a layer the identity gives no LED raises `axis_absent`; the drivers' `ch2color`, `color2ch`, `available_colors` and `led_ma` are gone, and the EL-0940 board addresses 0-7; an LED write the active lease does not permit raises `exclusive_activity_running`, and `led_on` always returns the commanded current. |
| level 11 | A run returns its `Operation` (`start_move_absolute`, `start_move_relative` and `start_home` return one too), and `ProtocolRunner`'s run members start it: `run_single_scan`, `run_protocol`, `run_autofocus`, `run_autofocus_all_steps` and `run_zstack` are `start_single_scan`, `start_protocol`, `start_autofocus`, `start_autofocus_all_steps` and `start_zstack`. |
| level 12 | A capture returns a frame or raises: `capture_and_wait` and `get_image` raise `CaptureFailedError` (`no_frame`, `deadline_expired`, `chunk_rejected` with its `setting`) for no frame and `HardwareCommandRefusedError` (`not_connected`, `camera_not_streaming`) for a camera gone or not streaming, and never answer None; `capture_failure_cause` and the failure flags of `last_capture_info` (`not_ready`, `drain_failed`, `deadline_expired`, `chunk_rejected`) are gone; `run_camera_bandwidth_test` answers `num_frames_failed` and `failed_by_reason` where it answered `num_frames_none` and `num_frames_error`. |
| level 13 | A frame listener is called `cb(frame)` with the camera's read-only `Frame` record (`image`, `seq`, `timestamp`, `arrived_ns`, `significant_bits`, `chunks`), where it was called `cb(image, timestamp, chunks)`; the image a capture returns (`capture_and_wait`, `get_image`) is read-only. |
| level 14 | `imaging.judge_frame(frame)` answers whether a frame counts for each source's last recorded change (a `Judgment` per change, carrying its `Change` and the `Credit` of each frame counted toward it), and `frame_validity.count_frame` takes who counted (`by=Counter`); a capture's `frame_record` carries `judgments`, and its saved file `frame_validity`. |
| level 15 | `capabilities.camera_has_clock_latch`, and a capture's `frame_record` carries `clock_mapping`: the camera's clock mapped onto the host's steady clock beside the grab (`ClockMapping`, `modules.lumascope_api.frame_record`), None on a camera without a latch. |
| level 16 | A camera timing record (`data/camera_timing/<model>.json`) is schema 2: its `validity` group holds, per source and direction, an `apply_delay_us` or `delivered_frames` spread (`p50`, `p95`, `max`) over its `trials` with the `provenance` of the run that measured it, and `FrameValidity.check_timing_record(record)` (`modules.frame_validity`) refuses any other with `ValueError`, as bring-up does with `InstallationFileError`; `FrameValidity.load_camera_timing` is gone, and frame validity drains at its own counts. |
| level 17 | A capture's `frame_record` carries `saturated_fraction` (measured when `all_ones_check` is asked) and `dark` (judged when an LED is lit and `accept_dark` is not), each None when not measured, and answers `blown`; `last_capture_info` no longer carries `dark_saved`. A blown or dark frame is not logged as a warning by the capture; a run's outcome counts them (`captures.saturated`, `captures.dark`, each a tuple of `TakenCapture`). |
| level 18 | A capture's `frame_record` carries `held_layer`, the layer whose stored settings the camera held (None after a raw camera write), and its saved file `camera_held_a_layer` (and `camera_held_layer` when one was). A manual still has the camera hold its name's layer for its frame, and is refused while a manual recording runs when that would change the camera. |
| level 19 | The camera is its own object, `scope.camera` (`CameraAPI`; in-process also `session.camera`): every camera setter, read, hold, apply, save and restore moved there from `scope.imaging`, and the Session's `set_frame_size`, `set_binning_size`, `set_image_mode`, `set_high_conversion_gain`, `set_line_noise_reduction` and `frame_at_binning` with them, under the same names. Each storing setter saves what the camera took, whoever calls it, and `restore_camera_state` saves the binning, frame, unbinned region and image mode it puts back; the unsaved `imaging.set_frame_size`, `set_binning_size`, `set_pixel_format`, `set_conversion_gain_mode` and `set_line_noise_reduction` are gone, and a pixel format is chosen only through `set_image_mode`. `update_settings`' `has_member` refusal names the camera's members by their path, `scope.camera.<name>`. |
| level 20 | The camera's reads are the pair: a held property (`gain_db`, `exposure_ms`, `black_level`, `binning`, `frame_size`, `pixel_format`, `auto_gain`, `auto_exposure`, `high_conversion_gain`, `line_noise_reduction`), `None` whenever unknown (`active` a held `bool`), and a fresh `read_*` (`read_gain_db`, `read_exposure_ms`, `read_black_level`, `read_binning`, `read_frame_size`, `read_pixel_format`, `read_frame_rate`) that raises `CameraReadFailedError` when the camera does not answer; the `get_*` getters, `get_live_camera_settings` and the `*_cached` properties are gone, and a frame whose exposure or gain cannot be read fails its capture. `set_binning` and `set_auto_exposure` replace `set_binning_size` and `set_auto_exposure_time`, and `apply` replaces `apply_layer_camera_settings`. `save_camera_state`, `restore_camera_state`, `last_capture_info` and `saturated_fraction` are in-process engineering members, so each refuses an argument of another type (`restore_camera_state(None)` included). |
| level 21 | `runtime_state.get_pixel_size_um()` and `runtime_state.get_field_of_view()`: the effective pixel size and the `FieldOfView(width_um, height_um)` at the current objective and the camera's held binning and frame, each `None` when it cannot be known; a plugin reads the scale here instead of computing it from an objective and a binning. |
| level 22 | Each setting the Session owns has its own read: `model`, `objective_id`, `turret_objective(position)`, `labware`, `scale_bar_enabled`, `acceleration_limit_pct`, `layer_acquire(layer)`, `layer_auto_gain(layer)` and `bookmark` (a `PlateLocation`), beside the existing `saved_focus(layer)` and the camera's reads; `runtime_state.get_turret_config()` and `imaging.scale_bar_config` are in-process. `camera.set_binning` and `camera.set_frame_size` answer the delivered `FrameSize`, not a dict. `get_setting` of a setting with its own read is refused `ArgumentRefusedError` (`has_member`), its `member` naming the read: `get_setting('scale_bar.enabled')` for `scale_bar_enabled`, `get_setting('image_mode')` for `scope.camera.image_mode`. |
| level 23 | A camera timing record's `apply_delay_us` is one number, 0 or more: the least start over its `trials`, on the camera's clock and after the change was in force, of the first frame showing the change, which the sensor's delay never exceeds; `delivered_frames` stays a `p50`, `p95`, `max` spread. `FrameValidity.check_timing_record` refuses a delay given as a spread. |
| level 24 | Each frame-validity source runs one of two rules, published as `capabilities.frame_rules` (a `FrameRule` per source: `rule` `Rule.CLOCK` or `Rule.CREDIT`, and under the clock rule its `delay_us` and `provenance`): the clock rule where the camera has a clock latch and its timing record holds a measured delay, the credit rule (the source's frame count) everywhere else. `imaging.judge_frame(frame, mapping)` takes the camera's clock mapping (a capture's `frame_record.clock_mapping`); a `Change` carries its `rule`, and a `Judgment` its `frame_start_ticks` and `qualifies_after_ticks` under the clock rule, in its `fields()` too. `imaging.recorded_changes()` answers each source's last `Change`. |
| level 25 | The diagnostics answers are records or published reads (W3c part E): `get_camera_link_info` answers a `CameraLink`, `get_motor_info` a `MotorBoardInfo` and `get_led_info` a `LedBoardInfo` (no `connected`: that is `led_connected`), each field `None` when unknown, an FX2 LED controller's and a missing motor board's firmware version included; `get_camera_temperatures_degc` is a `dict[str, float]`; `get_camera_diagnostic_info`, `get_camera_profile_info` and `get_system_info` are gone, their facts read from `capabilities` (`camera_model`, `camera_limits`, `pixel_size_um`, and new `camera_sensor`, `camera_shutter`, `camera_sdk_version`) and the camera's reads, the live exposure floor from `scope.camera.read_exposure_ms_min()`, which raises `CameraReadFailedError` when the camera does not answer. |
| level 26 | `session.controls_lock` answers a `ControlsLock` (`run_named`, `recording`, and `run_lockout` and `controls_locked` read from them): which of a person's controls the Session locks now, the one read every lock read reads; a plugin greys its own controls from it. |
| level 27 | The Session's ten run-state reads are gone: `is_protocol_running`, `run_in_progress`, `exclusive_activity`, `protocol_files_draining`, `protocol_files_pending`, `run_lockout`, `run_lockout_named`, `recording_active`, `controls_locked` and `motion_enabled`. `controls_lock` answers the lock, `live_work` what holds the scope and a finished run's files still to write, and a stage gesture reads `capabilities.has_xy_stage` beside the lock. |
| level 28 | A `Change` carries `earliest_ns`, the earliest moment it could have reached the hardware: a camera or LED write's start, or for a move once in force the start of the last read that saw its axis still moving (else its `written_ns`). A delay measured from it is never short; the clock rule still judges from `written_ns`, or a move's `in_force_ns`, so the pair never accepts a frame early. |
| level 29 | `session.plugin_run_in_flight()`: a context manager entered when a plugin's run starts and left when its worker ends, on any threads; while it is held, `live_work` lists a `'plugin_run'` item (`'An engineering plugin run'`, `left` its count) and `in_flight_named` names it; it holds no claim and refuses nothing. |
| level 30 | On a camera with a clock latch (`capabilities.camera_has_clock_latch`), every move (`xy_move`, `z_move`, `turret`) runs the clock rule in `capabilities.frame_rules`: at its measured delay, or at 0 from the axis's stop when its timing record holds none or the camera has no record, its `provenance` then `{'measured': 'none: a frame started after the stop'}`; a frame that started exposing before the stop never counts for the move. A camera reports a clock latch only when its frames carry their start (the `Timestamp` chunk); a Basler camera whose chunk did not turn on has no `camera_timestamp_tick_hz` and no latch. |
| level 31 | A move judged by the credit rule (`xy_move`, `z_move`, `turret` on a camera without a clock latch) counts its frames from its stop, not its drive: `motion_settled` places its `Change` in the frame stream there (`at_seq` the frames delivered at the stop, `credits` cleared, the source pending again), and no frame that arrived before it is credited. While the axis moves, `pending_sources` and `frames_until_valid()` answer the move's whole count, not 0 then 1. A later stamp of the axis settled starts the count again. |
| level 32 | A move is asked for by what it moves: `move_xy(x_um, y_um)` / `start_move_xy` and `move_xy_by(dx_um, dy_um)` / `start_move_xy_by` move X and Y as one request, every axis given refused before either drives, an axis given as None left as it is and not read; `move_z(z_um)` / `start_move_z` and `move_z_by(dz_um)` / `start_move_z_by` move Z, a move to a place approaching from below and a move by a distance not, unless `overshoot_enabled` (None, the API's approach, by default) says otherwise. `move_absolute`, `move_relative`, `start_move_absolute` and `start_move_relative` are gone, with no alias. |

### Configuration queries

```python
session.saved_focus('Blue')              # the Z saved as a layer's focus; FocusNotSavedError when none was
                                         # ever saved (a step built for that layer takes the current Z)
session.channel_acquire('Blue')            # what a layer captures: 'image', 'video' or None (set_channel_acquire)
session.channel_auto_gain('Blue')          # whether a layer's auto-gain is on (set_channel_auto_gain; turning it
                                         # off also stores the layer's gain and exposure, read by get_setting)
session.channel_auto_gain_lock('Blue')     # what the camera reached when that layer's auto-gain was last turned
                                         # off: the AutoGainLock set_channel_auto_gain returned, else None
session.model                            # the saved model, brought up at the next start (select_model)
session.objective_id                     # the active objective's key; None while it is not known (select_objective)
session.labware                          # the current plate's catalogue key (select_labware)
session.scale_bar_enabled                # whether captures draw the scale bar (set_scale_bar)
session.acceleration_limit_pct           # the motors' acceleration limit, percent (set_acceleration_limit)
session.bookmark                         # PlateLocation(x_mm, y_mm, z_um): the saved bookmark (save_bookmark)
session.scope.runtime_state.resolve_current_objective()  # (id, Objective) of the active objective; ObjectiveUnknownError when unknown
session.capture_settings_snapshot()      # (in-process) settings snapshot with objective_id set to the active objective,
                                         # for composing a capture or run; not for saving
session.get_current_plate_position()     # PlateLocation(x_mm, y_mm, z_um): where the scope is on the selected plate, focus
                                         # included, the record start_single_scan's return_to_position takes;
                                         # None on an axis this scope has no motor for;
                                         # CatalogueNameRefusedError when the stored plate is not in
                                         # the catalogue; HardwareCommandRefusedError('not_connected') when this
                                         # model has a motor controller and none is connected;
                                         # AxisStateUnknownError when an axis does not know its position
                                         # (before a home) -- never a converted number nobody read
session.get_setting('protocol.autogain') # auto-gain config
session.get_setting('Blue.stim_config')  # one layer's stim settings
session.channel_colors()                 # every layer's colour, (r, g, b) weights 0-1: what captures, recordings,
                                         # builds and the view are coloured in, read by each as it starts
session.update_settings('Red.display_color', '#ff00ff')  # override a layer's colour; None gives the unit's back
                                         # (its board block's, else the catalogue's).
                                         # '#rrggbb', lower case, brightest component ff, else SettingRefusedError
```

Building a protocol from the session's settings:

```python
# One step per acquiring layer at every well of the session's plate -- no GUI involved.
protocol = session.new_protocol()

# Tiling and z-stacking are ARGUMENTS, not stored settings. Neither survives
# a restart, so there is nothing for the session to read them from: state
# what you want.
protocol = session.new_protocol(tiling='2x2', use_zstacking=True)
```

A layer whose focus was never saved (`saved_focus(channel)` raises
`FocusNotSavedError`) is imaged at the stage's Z, Z's position from
`scope.motion.axis_positions()`; `new_protocol` refuses the build while Z does
not know it (before a home). A scope with no Z motor takes the plate
position's Z. A layer with a saved focus keeps it.

The GUI's New builds through the same member, supplying those two from its
own controls, so a scripted protocol and one made on the screen are built
identically.

### Cleanup

`session.shutdown()` first finishes what the session is doing, then tears down. From its first step nothing new takes the scope: a run, a recording, a home or a diagnostic asked for while it runs raises `SessionClosingError` (`modules.exceptions`, `reason` `session_closing`, a refusal); work inside an activity already under way -- a run's video step -- is not refused, and neither is a post-processing build. Plugins unload first, so a plugin's own work stops through its `unregister`; a run that finishes afterwards and would hand its folder to a plugin's processor is reported as `PluginProcessorSkippedError`, naming the run. The live run is stopped (its outcome `aborted` / `shutdown`) and a live recording is stopped. Then the close waits, without a time limit, until `session.live_work` lists nothing: the run's images and post-run builds, the recording's file, post-processing builds running and queued, a support report or logs zip, a still. While it waits, any caller may call `session.discard_close_drain()` to drop the video frames still queued (the file is finished with what was written), and a finished run's file writer judged stuck (`protocol_files_stalled`) is recovered by the close itself: its unwritten images are given up on, counted in the run's files and reported. The close logs what it is waiting for every few seconds. It may be called from any thread, a run's `run_ended` handler included; a second call on another thread waits for the first and returns. Once it has returned, `session.live_work.closed` is True and nothing takes the scope again: a start is refused by its own checks where they come first (a run's `hardware_disconnected`, a recording's `camera_inactive`), and otherwise `session_closing` when it takes the scope -- a home, a diagnostic, or a run that passed its checks before the close returned.

```python
# Full teardown of everything the session constructed: the FILE lane, the
# worker pool and the session's threads always. On a scope the FACTORY built
# also: LEDs off, motion stopped, scope disconnected,
# which shuts its IO and CAMERA lanes. Reading that scope afterwards:
# motor_connected is False, imaging.is_streaming() is False,
# diagnostics.get_motor_info().model is None. A second shutdown() logs one
# info line and does nothing.
session.shutdown()
```

If a part does not shut down cleanly, `shutdown()`
still runs every teardown step, then raises `ScopeDisconnectError`
(`modules.exceptions`): its `parts` name each part that failed, in teardown
order (`'motor stop'`, `'LED board'`, `'motor board'`, `'camera'`), `causes`
holds each part's error, and the first is its `__cause__`. Everything that
could be torn down has been, so a second `shutdown()` completes. Neither
returns a value.

---

## scope.motion

Axes available depend on the scope — always check `scope.capabilities.axes`.

```python
# Homing (required before movement)
scope.motion.home()                              # home everything the board has (axis='ALL' default)
scope.motion.home(axis='Z')                      # Z only
scope.motion.home(axis='T')                      # turret only (parks Z at 0, homes T, restores Z)
# home() waits with no time limit of its own: the board bounds every home,
# and the driver stops waiting only for a board that has stopped answering;
# either ends it in HomingFaultError. home() and move_turret() are waiting
# forms, refused on the UI thread (BlockingCallRefusedError, 'wait_on_ui_thread');
# start_home() is the form for a button.
# Each blocks until the home is done and returns only when it established
# the reference. An axis other than 'Z', 'T' or 'ALL' is refused,
# ArgumentRefusedError reason 'axis_unknown'. A home that could not
# raises: HardwareCommandRefusedError (reason 'not_connected' or 'axis_absent':
# no motor controller, or no Z or turret to home; or 'lid_open' /
# 'stage_unpowered': the stage's interlock refused the home before anything
# moved; nothing moved, and every axis keeps the state it had) or
# HomingFailedError, never raised itself, whose child says how the home
# ended (.home, .reason, .axes the axes left unknown): HomingFaultError, a
# fault titled 'Homing Failed' (.reason 'error' | 'unread' | 'lid_open' (the
# lid was opened while the home moved), chained from the driver's error when
# there is one), or HomingStoppedError, a notice titled 'Homing Stopped'
# (.reason 'stopped', stop_motion ended it). A stop_motion that lands after
# home() is called but before the board starts homing ends the home
# 'stopped' with .axes empty: nothing moved, every axis keeps its state.
# Once the board is homing, a stop_motion ends the home only on a board
# that can take a STOP mid-home (the LS720's); on the EL-0940 it raises
# HardwareCommandRefusedError reason 'home_not_stoppable', nothing is sent, and
# the home runs to its end -- for a turret home, its Z park and restore too.
# A home holds the scope as a run does, from when it is asked until it ends:
# every other request -- a move, an LED, a camera setting, a run, a
# recording, a diagnostic -- raises HardwareCommandRefusedError (or the
# start's own refusal) naming the home, nothing is queued to run after it,
# and run_lockout / controls_locked read True. A second home -- by any
# caller, on any axis -- raises reason 'home_in_flight'; a home asked while
# a run, a recording or a diagnostic holds the scope raises
# 'exclusive_activity_running' naming that holder. A home asked from inside
# the activity that holds the scope (a diagnostic) runs as its work. The
# guard is the Session's: a Lumascope built without a ScopeSession has no
# activity claim, and its homes are not refused for one another.
# Post-processing stays open during a home, as during a run.
homing = scope.motion.start_home(axis='ALL')     # same home, not waited: refused or started
                                                 # before it returns; an Operation, done once
                                                 # the home has let go of the scope, with what
                                                 # home() returns or raises. homing.stop() is
                                                 # stop_motion while the home drives (refused
                                                 # 'home_not_stoppable' on a board that cannot stop
                                                 # a home), and nothing once it has ended
scope.motion.has_homed()                         # True if the stage/focus axes know where they are
scope.motion.position_is_known('T')              # turret-specific
scope.motion.axes_without_position()             # {axis: 'unknown' | 'homing'}; {} when all known

# Homing is REQUIRED, not advisory. A commanded move on an axis whose
# position is unknown raises AxisStateUnknownError instead of driving --
# there is no reference frame for it to be absolute in. An axis is
# unknown before its first successful home, and again after a home
# fails, the board disconnects mid-move, or a move stalls out. So a
# headless or REST caller homes first and stops on the raise:
#
#   try:
#       scope.motion.home('ALL')
#   except (HomingFailedError, HardwareCommandRefusedError):
#       ...  # do not command moves; the reference frame is not established
#
# A home reads every axis's position from the board before it marks the
# axis known. When the mechanics succeeded but an axis's position could
# not be read, the home raises HomingFailedError (reason 'unread', naming
# the axis), that axis is UNKNOWN, and its last cached number is never
# re-labelled as a position. The home itself shows no popup: the caller
# that waited on it reports it.
#
# has_homed() / position_is_known(axis) answer from that same live state, so they
# report False after a fault revokes a reference that was previously
# good -- not merely "a home once succeeded".

# Position queries (µm for XYZ, 1–4 for turret). Read cache, no serial I/O.
scope.motion.axis_positions()                    # {axis: AxisPosition(state, position)} in ONE snapshot; position is None unless the axis is IDLE or MOVING -- the read for a caller writing a position into a file
scope.runtime_state.plate_transform()            # (x_um, y_um) -> PlatePosition, BOUND to the plate and offset of this call. For a caller converting many positions over time (a recording, one per frame): every frame is stated in one frame of reference
scope.motion.get_target_position('Z')            # the commanded target, moving or arrived; the polled position when none was reached (before a first move, after a home, a STOP or a fault); 0.0, read by nothing, for an axis UNKNOWN because its bring-up read failed
scope.motion.get_target_positions()              # {axis: target} for every axis the scope has a position for
scope.motion.get_actual_position('Z')            # hardware position via serial (slow; use sparingly)
# A position with no hardware behind it is None: an axis the scope does not
# have, and every axis with no motor board installed (a manual scope, a board
# that never came up, after the session's shutdown()). A present axis keeps answering its
# number, its reference lost or not; a pulled cable leaves the board installed.
# A position the controller does not report is never answered with a number:
# get_actual_position raises HardwareError, and HardwareCommandRefusedError
# ('not_connected') when the installed controller is not connected. A move that needs a position it cannot
# read -- a relative move's base, or Z for the backlash approach -- is refused
# with HardwareCommandRefusedError ('position_unread'); nothing is driven.

# Stop + tuning
scope.motion.stop_motion()                       # stop all in-flight moves (the app-level abort for the move_* family)
# It halts every axis on every firmware: the board's STOP where the firmware
# has one, else each axis's actual position written as its target, which is
# what that STOP does on the board. On firmware without STOP an axis caught
# at speed comes to rest a little past where the stop was asked (X at full
# speed: about 2 mm, measured on an LS850T).
# A STOP the board did not take raises MotorStopFailedError (title 'Motor Stop
# Failed', chained from the driver's error): the stage may still be moving.
# While a home drives on a board that cannot stop one, it raises
# HardwareCommandRefusedError reason 'home_not_stoppable' ("This microscope
# cannot stop a home; the home runs to its end."; see home() above).
# A press -- taken, failed or refused -- withholds every move asked before
# it that has not yet driven, from any client and any queue: it ends
# MoveStoppedError 'stopped' where the stage is. It also ends the live run,
# outcome aborted / stage_stopped (see "How a run ends"), and the startup
# motion's later legs.
session.set_acceleration_limit(50)               # motor acceleration cap, percent of max (a whole number 1-100, else AccelerationLimitRefusedError)
scope.motion.set_precision_mode('Z', True)       # per-axis precision mode on the motor board

# A move is asked for by what it moves, in µm, and returns once it has
# arrived, or raises MoveNotCompletedError saying why it did not.
# X and Y are one request: every axis given is refused before either drives,
# so a refusal moves neither; an axis given as None is left as it is, and
# with neither given the move is done at once.
scope.motion.move_xy(x_um=60000, y_um=40000)
scope.motion.move_xy_by(dx_um=500)               # added to X's board target; Y untouched

# Z on its own: a move to a place approaches it from below, so the backlash
# is always taken the same way; a move by a distance does not.
# overshoot_enabled=True or False is a bench script's own choice.
scope.motion.move_z(5000)
scope.motion.move_z_by(-10)

# Started moves: return once the board has taken the command, for work done
# while the stage travels. Each is an Operation (see "How a run ends"), done
# once its axes have been set at rest, or given up, and the position
# listeners told: result() returns None on arrival and raises what the
# waited form would have raised. stop() is stop_motion while a move still
# has its axis, and nothing once it has ended -- a stop sent late never
# halts the move that took the axis; the board's STOP halts every axis.
xy = scope.motion.start_move_xy(x_um=60000, y_um=40000)
z = scope.motion.start_move_z(5000)
xy.result()
z.result()
scope.motion.start_move_z_by(100).result()

# A move that does not complete raises MoveNotCompletedError, never raised
# itself, one object with .axis, .reason and .title. Its child says which
# kind: MoveStoppedError, a notice titled 'Move Stopped', for 'stopped' and
# 'superseded'; MoveFaultError, a fault, for every other reason, after which
# the axis is UNKNOWN: 'driver_failed' (the board did not take the command; chained
# from the driver's error), 'stalled' / 'board_lost' / 'position_unread' /
# 'status_unread' (the motion monitor gave the axis up -- 'stalled' when its
# position held one value for 2 s short of its target, and the stage was then
# halted; a move of any length that keeps moving is never given up;
# 'position_unread' when the board said it arrived but never said where,
# 'status_unread' when it never said whether it arrived, each for the same 2 s;
# the move raises the very object the monitor reported; 'position_unread' also
# when a home, which reads every axis's position, could not read a moving
# axis's), 'monitor_ended' (the
# monitor itself stopped while the axis moved; every later move is refused,
# HardwareCommandRefusedError 'monitor_ended', until a new session is brought up),
# 'faulted' (set UNKNOWN by something else while it moved, with no stop before
# it), 'stopped' (a stop landed on it: stop_motion, the
# stage's own stop in refusing a move, or the close's: a move in flight when
# the scope is closed ends arrived if it got there first, else 'stopped', its
# position then unknown; a close whose STOP the board did not take ends it
# 'stopped' too, and the failed STOP is the close's ScopeDisconnectError), 'superseded'
# (another move or a home took the axis before it arrived; the axis keeps
# what that command gives it). The outcome is fixed when the move ends: a
# move that arrived reads arrived however late result() is called, whatever
# moved or stopped the axis since. A caller's own result(timeout) that
# expires first raises TimeoutError and leaves the move going.

# Interlocks. A stage that guards motion with its own inputs refuses a move
# while one is open: HardwareCommandRefusedError, .reason 'lid_open' (title
# 'Lid Open'; it refuses X and Y moves and every home, a Z-only home
# included, and Z moves still go) or
# 'stage_unpowered' (title 'No Stage Power'; it refuses a home). The refused
# axis did not move and its position stays known; an axis the stage stopped
# in refusing it ends where it stopped, its position known, as after a Stop.
# A lid opened while an X or Y move is under way stops all three axes: every
# move in flight ends MoveStoppedError reason 'stopped', each axis where
# it stopped, its position known; interlocks() says why.
# stop_motion works with the lid open. A board with no interlocks (every
# EL-0940 scope) never raises these.
scope.motion.interlocks()                        # frozenset of open interlocks, e.g. {'lid_open'}; empty when none

# Jog step under the active objective (z_coarse / z_fine for Z, xy_coarse / xy_fine for X, Y);
# ObjectiveUnknownError when the objective is unknown -- no step is guessed
step = scope.motion.jog_step('Z', coarse=True)
scope.motion.move_z_by(-step)

# Status
scope.motion.is_moving()                         # any axis moving?
scope.motion.wait_until_finished_moving()        # block until the axes moving now stop; raises MoveFaultError
                                                 # if one ended UNKNOWN (MoveStoppedError when a stop
                                                 # ended its move, as the close's does), or MoveWaitExpiredError ('still_moving',
                                                 # a TimeoutError, quiet) if timeout_s ran out first
scope.motion.position_is_known('Z')              # False until homed: an absolute move would refuse

# Limit switches -- why a move stopped short. Reaching a limit is reported,
# not raised, so a move that ran out of travel and one that arrived look the
# same until you ask.
scope.motion.get_limit_switch_status('X')        # (left, right); 1 engaged, 0 clear; HardwareError when the board does not answer;
                                                 # HardwareCommandRefusedError 'not_connected' / 'axis_absent'
                                                 # for a switch the scope lacks -- never (0, 0)

# Turret
scope.capabilities.has_turret                    # turret presence probe
scope.motion.move_turret(2)                      # turret position 1-4; any other slot raises
                                                 # PositionOutOfRangeError rather than driving
scope.motion.get_turret_slot()                   # slot in the light path, or None when not known
# The turret has no encoder: the slot is the one the last move_turret or home
# left it in, recorded only when that command returned without error and no
# stop_motion landed on it. None before the first, while one runs, after one
# fails (MoveNotCompletedError), and after the turret's position is lost.
# No other move takes the turret: it moves only by slot, through move_turret.
# Which slot a protocol step uses is the step's own (its Turret Slot, below):
# two steps naming one objective in two slots each go to their own.

# Stage
scope.motion.get_axis_limits('Z')                # AxisLimits(min_um=0.0, max_um=14000.0), frozen
```

The limits are a frozen `AxisLimits` record: they are the bound a move is
refused against, so an edit raises `dataclasses.FrozenInstanceError`.
`to_dict()` gives a working copy.

**Axes: ask which axes the scope has before asking their limits.**
`scope.capabilities.axes` names the axes this scope has a motor for.
`get_axis_limits(axis)` answers the motion board's travel for an axis
the scope has; an axis it does not have is refused as every command for
it is (`HardwareCommandRefusedError` reason `axis_absent`, or
`not_connected` when the motor controller is out of reach), so read
limits only for the axes `capabilities.axes` names. The turret's `'T'`
answers None: it moves by slot, not by distance.

**Z overshoot:** firmware moves below target then approaches from below, eliminating leadscrew backlash for consistent focus. The leg below is part of the move: Z reads `'moving'` from the first target write, `wait_until_finished_moving()` waits through the leg and the approach, and a frame grabbed during either is not valid.

**Axis state model:**

```python
scope.motion.get_axis_state('Z')          # 'idle', 'moving', 'homing', or 'unknown'
```

**Position listeners** (push-based):

```python
def on_position(axis: str, at: AxisPosition):
    # at is what axis_positions()[axis] answers now, from the same snapshot:
    # at.state, and at.position, the polled position (not the target) -- None
    # while the axis is homing or unknown, never the last number it reported
    print(f"{axis}: {at.position} um ({at.state})")

scope.motion.add_position_listener(on_position)
scope.motion.remove_position_listener(on_position)
```

---

## scope.illumination

Channels available depend on the scope — the layers it lights are the records in `scope.layer_identity.layers` whose `led_channel` is not empty.

**A channel is named, never numbered.** `'BF'`, `'PC'`, `'DF'`, `'Blue'`,
`'Green'`, `'Red'` — the name is the portable identity. The number behind it is
a driver detail and is NOT portable: an FX2 board carries four channels and an
RP2040 board six, so the same integer means different LEDs, or none, depending
on the board. Ask `scope.layer_identity.layers` for what a given scope can actually drive.
Passing an integer still works today as a legacy compatibility form, but it is
not the supported contract and will not be part of the REST surface.

**LEDs the scope cannot drive**: `led_on` asks, in order, whether the channel is one at all, whether an LED controller is connected, and whether this model has the LED. A name that is no layer raises `ArgumentRefusedError` (`layer_unknown`) offering the catalogue, and a number outside the installed board's table `led_channel_unknown` offering the board's numbers, at the door, whatever the scope is doing. A current outside 0 to the board's maximum is `illumination_out_of_range`, its `limits` holding both. With no LED controller connected it raises `HardwareCommandRefusedError` (`not_connected`, `MissingPart.LED_CONTROLLER`). An LED this model does not have -- a layer it lacks (an LS560's `'Red'`), a layer that drives no LED (`'Lumi'`), or a number no layer of the model drives (an LS560's `2`) -- raises `HardwareCommandRefusedError` (`axis_absent`, `MissingPart.led(...)`), and nothing lights; it never maps to a substitute channel. A model this release does not know (its layer identity unresolved) takes the board's numbers. `led_off` of an LED the scope does not have is an idempotent no-op — a channel the scope does not have is already off.

**Luminescence** (`Lumi`): not an LED channel. In luminescence mode, all LEDs must be off — the image captures emitted light only.

```python
scope.illumination.led_on('Blue', 200)                 # Blue LED at 200 mA
scope.illumination.led_off('Blue')
scope.illumination.leds_off()                          # turn off all LEDs

# Channel mapping. Numbers are a DRIVER detail -- these exist to read the
# board's own wire vocabulary, not to address channels from L2.
scope.illumination.color2ch('Blue')                    # 0  (or None if the scope doesn't have this color)
scope.illumination.ch2color(0)                         # 'Blue'

```

**Safety limits** (enforced by firmware on RP2040 boards): per-channel max 1000 mA, board total max 3000 mA. FX2 boards have their own per-channel cap declared in the camera profile.

**The current is the board's step, not your number.** Each board drives its LEDs in steps: whole mA on the EL-0940 LED board, and about 3.29 mA (one step of 840 mA over 255) on the FX2 scopes (LS560, LS620, LS720). `led_on` lights the channel at the nearest step, a half going up; a request above zero but below one step lights at one step, never dark; 0 mA leaves the channel on at 0 mA, which the board receives as off. `led_on` returns that commanded current, and it is what `get_led_state`, the LED listeners, a frame's record and a saved file's `Illumination` report: `led_on('BF', 1.0)` on an LS620 returns about 3.29, and the file says so. It is the current the board was commanded, not a measurement. The layer setting and a protocol step's `Illumination` keep the value you wrote, so a protocol stays the same when it moves between scopes; `api.log` names both whenever they differ.

### State queries — read from the API, never the driver

`scope.illumination` (`IlluminationAPI`) holds the authoritative LED state, by LED address; its `get_led_state()` / `get_led_states()` read from that store. A driver holds no LED state and names no address: which layer an address lights is the unit's layer identity's (`ch2color` / `color2ch`), read at the moment of the reading.

Each read answers an `LedReading`: the `address`, the layer `name` the identity gives it (`None` when it names none -- the address is still reported, never hidden), its `state` (`LedState.LIT`, `DARK` or `UNKNOWN`; `'lit'`, `'dark'`, `'unknown'` on the wire) and its `illumination_ma` (the commanded current when lit, else `None`).

```python
scope.illumination.get_led_state('Blue')               # LedReading(address=0, name='Blue', state=LedState.LIT, illumination_ma=200.0)
scope.illumination.get_led_states()                    # one LedReading per address of the board (0-7 on the EL-0940 board, 0-3 on FX2),
                                                       # named or not: the read of an address by its number
scope.illumination.save_led_state('mine')              # an LedSnapshot(tag, readings) for restore_led_state
```

A layer name the unit's identity gives no LED (`'Lumi'`, or `'Red'` on an LS560) is refused `axis_absent`, as `led_on` refuses it.

With no LED board installed -- none came up, or after `session.shutdown()` -- the three answer `None`: there is no board whose state to describe, which is not the same answer as every channel off. Shutdown darkens the board and leaves nothing believed lit. A frame captured then records no LED lit.

### The shown channel — which layer is on the camera and lit

The session holds the channel the scope shows: `session.shown_channel` is a `ShownChannel(layer, lit)`. `session.show_channel(channel)` makes `channel` the shown channel -- every other LED dark, `channel` lit at its stored current only if the shown channel was lit, then its camera settings applied. `session.light_shown_channel(lit)` lights or darkens it, and leaves the camera on its stored settings. Both refuse before turning anything off, so a refused call leaves the LEDs as they were. While a manual recording runs, showing another layer is refused (`exclusive_activity_running`, `holder` `'recording'`): a recording keeps one layer. `go_to_step` and a run's steps move it too, and `add_shown_channel_listener` (the REST `shown_channel` event) hears each change as `(layer, lit)`.

```python
session.show_channel('Blue')          # ShownChannel(layer='Blue', lit=False): Blue's camera settings, every LED dark
session.light_shown_channel(True)     # Blue lit at its stored illumination_ma
session.show_channel('Green')         # Green lit, Blue dark: the light follows the shown channel
```

### Exclusivity — a run holds the LEDs

LED writes carry no owner name: no string grants the right to drive an LED.

```python
scope.illumination.led_on('BF', 200)
scope.illumination.led_off('BF')
scope.illumination.leds_off()                          # unconditional off (shutdown / cleanup)
```

**A run holds the LEDs, and a write from outside it raises.** While a protocol run, an autofocus run, a diagnostic or a home holds the scope, an `led_on`, `led_off`, `restore_led_state` or `leds_off` from any other caller raises `HardwareCommandRefusedError` (`reason='exclusive_activity_running'`, with `holder` naming what holds the scope), and the LED does not change. `leds_off()` ignores any held lease, so inside the holder's own work and whenever nothing holds the scope it turns every LED off; the session's shutdown drains the LEDs past the claim.

### Save / restore — the autofocus pattern

Preserve the user's LED state while a subsystem does its own work, then restore:

```python
# User has Red on at 150 mA. Autofocus needs BF:
snapshot = scope.illumination.save_led_state('autofocus')        # capture current state
scope.illumination.led_on('BF', 100)
# ... autofocus runs: changes Z, captures frames, evaluates focus ...
scope.illumination.restore_led_state(snapshot)                   # Red back on at 150 mA, BF off
```

`save_led_state(tag)` returns an `LedSnapshot` (the tag is for logs; a REST client gets it as JSON and can send it back); `restore_led_state(snapshot)` turns off the addresses the snapshot does not have lit and re-lights the ones it does. While a run holds the LEDs, the restore is refused like any other write.

### Listeners — push-based notifications

Prefer listeners over polling. Listeners fire on every LED state change (enable, disable, illumination change) with no serial I/O cost:

```python
def on_led(address: int, name: str | None, state: LedState, illumination_ma: float | None):
    print(f"{name or address} {state} {illumination_ma}mA")

scope.illumination.add_led_listener(on_led)
# ... later ...
scope.illumination.remove_led_listener(on_led)
```

Use polling only when you specifically need the current value at a moment in time (e.g., settling a UI field to match hardware after a reconnect). For "did anything change?" questions, always use listeners.

---

## scope.imaging

Camera capture lives on the `scope.imaging` sub-API, and the camera's
settings, reads and their saving on `scope.camera` (over REST
`/api/v1/scope/camera/...`).
Their published members are listed under `ImagingAPI` and `CameraAPI` in the
API reference. A camera setting has one door, on the camera: a setting that
is saved (the binning, the frame, the image mode, high conversion gain and
line-noise reduction) is saved by it, whoever calls it.

```python
# Streaming control. connect() returns the camera CONFIGURED but NOT
# grabbing; capture/get_image need a live feed, so start it first. The
# Session factories release the gate at bring-up, after initialize; a bare
# Lumascope you constructed yourself needs the explicit call.
scope.imaging.enable_streaming()  # start the live feed and return; it does not wait for a
                                  # frame (idempotent; restarts a stopped feed); a start the
                                  # FX2 refuses raises CameraStreamStartFailedError, the
                                  # camera left connected and not streaming
scope.imaging.disable_streaming() # stop the feed; capture_and_wait is then refused
                                  # camera_not_streaming
image = scope.imaging.capture_and_wait()  # the first frame after a start: a frame that
                                  # arrives after this call
scope.imaging.is_streaming()      # True while acquiring (queries the driver)
```

```python
# Raw frame grab (no validity wait — use capture_and_wait instead in most cases)
image = scope.imaging.get_image()
image = scope.imaging.get_image(force_to_8bit=False)   # keep native 12/16-bit
# Returns numpy.ndarray, or raises: CaptureFailedError ('no_frame', or
# 'chunk_rejected' with verify_chunk_targets) for no frame, and
# HardwareCommandRefusedError ('not_connected') for a camera that is gone.
# It runs on the camera lane, as capture_and_wait does: while a run, a
# diagnostic or a home holds the scope it is refused
# ('exclusive_activity_running') unless you are that holder.
#
# Shape is (H, W) 2D mono for mono-native cameras and (H, W, 3) RGB
# for color-native cameras (see scope.capabilities.is_color_native).
# Layer false-color is NOT applied here -- apply at the display /
# encode boundary via image_utils.mono_to_rgb_falsecolor(img, color).
# Dtype is uint8 with force_to_8bit=True (default) or for 8-bit
# cameras; uint16 with force_to_8bit=False for 12/16-bit cameras
# (see scope.capabilities.buffer_bit_depth).

# Frame-validity capture — PREFERRED for all real captures.
# Waits for all pending changes (LED, gain, exposure, motion) to settle,
# drains stale frames, returns a valid frame. Each change is waited on by
# its source's rule (scope.capabilities.frame_rules), and the grab keeps
# only a frame that imaging.judge_frame counts for every change it does not
# exclude, taking the next otherwise. Raises CaptureFailedError
# for no frame ('no_frame', 'deadline_expired', 'no_fresh_frame',
# 'chunk_rejected') and
# HardwareCommandRefusedError for a camera gone ('not_connected') or not
# streaming ('camera_not_streaming').
# The capture honors invalidation across its WHOLE window: a state change
# landing after the drain — during the grab itself — is detected, and the
# capture re-drains, re-derives its expectations, and grabs again, so the
# returned frame always reflects the state you last commanded. A capture
# contended by state changes is therefore slower than an uncontended one.
# The settle-and-recheck work is bounded by a deadline sized from the
# pending work at entry (frames to settle, exposure, sum window): if
# invalidation keeps arriving faster than frames can settle it, the
# capture raises CaptureFailedError('deadline_expired') in bounded seconds,
# instead of holding indefinitely. The deadline suspends
# while commanded stage motion is still physically settling — a capture
# issued during a long move waits for the move, as it should.
# The dark-floor expectation is DERIVED from commanded LED state: a
# channel counts as lit only at strictly positive current, so a channel
# commanded at 0 mA is dark by design, as are luminescence captures and
# any capture with nothing commanded. With a channel lit, a frame with
# essentially no lit pixel is retried until timeout_s -- which heals a
# stale pre-LED frame when a lit one is on its way -- and then RETURNED,
# its record's .dark True (last_capture_info['frame_record']), and its
# saved file's metadata 'dark' True. It is never refused:
# pixel content cannot distinguish an LED that failed from a genuinely
# dark sample, so darkness is reported, not acted on. accept_dark=True
# (keyword-only, default False) skips the measurement entirely for the
# callers whose dark frames are expected: custom focus sweeps (an
# out-of-focus fluorescence plane can carry no signal) and benchmark
# probes; those captures' records carry dark None.
image = scope.imaging.capture_and_wait()
image = scope.imaging.capture_and_wait(
    force_to_8bit=True,
    accept_dark=False,                     # True skips the darkness measurement
    all_ones_check=True,                   # measure saturation (frame_record.blown)
    sum_count=4,                           # SUM 4 frames (not an average); a
                                           # sum is uint16 on every camera and
                                           # saturates at 65535
    sum_delay_s=0.05,                      # delay between sum frames
    exclude_sources=(),                    # sources not to wait for; the frame can then
                                           # predate that change (see Frame validity)
)

# Exposure (milliseconds) + gain (dB)
scope.camera.set_exposure_ms(exposure_ms=50)
scope.camera.read_exposure_ms()                 # read from the camera now; None with no camera;
                                                   #   raises CameraReadFailedError when the camera does not answer
scope.camera.read_exposure_ms_min()             # the shortest exposure the camera takes now, read from it: on IDS
                                                   #   it rises above capabilities.camera_limits.exposure_ms_min
                                                   #   as the pixel clock, frame rate and frame size change
scope.camera.set_gain_db(gain_db=10.0)
scope.camera.read_gain_db()                       # read from the camera now; None with no camera;
                                                   #   raises CameraReadFailedError when the camera does not answer
scope.camera.read_frame_size()                    # FrameSize(width, height), read from the camera now
scope.camera.read_pixel_format()                  # e.g. 'Mono12', read from the camera now

# `set_exposure_ms` warns + logs a stack trace at < 0.005 ms (the
# common L1 failure is typing 0.05 thinking microseconds and getting
# a black image).
#

# Every camera-settings setter in this section dispatches to the camera
# lane and BLOCKS until applied (returns the body's own result). While a
# protocol run or recording owns the hardware, these raise
# HardwareCommandRefusedError instead of interleaving with the run --
# same refusal contract as the motion and LED commands.

# Batched settings (gain + exposure + auto-gain in one call)
scope.camera.apply(
    gain_db=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None,
    channel='BF',        # names the channel in api.log; optional, defaults to '(unspecified)'
)

# Auto-exposure: the camera's own exposure control, where the body has one.
# Blocks until the camera has applied it. Check
# caps.camera_supports_auto_exposure first -- not every body offers it.
# The mode is CameraAutoMode.CONTINUOUS or OFF ('Continuous' or 'Off' over
# REST, PUT scope/camera/auto_exposure {"mode": ...}); ONCE is refused,
# ArgumentRefusedError 'auto_mode_unknown' -- a one-shot is auto_gain_once's.
scope.camera.set_auto_exposure(CameraAutoMode.CONTINUOUS)

# Auto-gain: the continuous toggle and the setpoint
scope.camera.set_auto_gain(True, settings={'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0})
scope.camera.update_auto_gain_target_brightness(0.5)   # live setpoint tweak while auto-gain runs

# A camera without the mode (caps.camera_supports_auto_gain /
# _auto_exposure False: the IDS U3-34L family, the LS620's FX2) is never asked.
# Turning the mode on, or setting its target, raises
# CameraSettingUnsupportedError (reason 'auto_gain_unsupported',
# 'auto_exposure_unsupported' or 'auto_gain_target_brightness_unsupported');
# turning it off is that camera's state already, so it returns without a write.
# camera.apply with auto_gain=True applies it as manual there --
# see applied_auto_gain_for below.

# Camera-model-specific tuning knobs. Probe support first:
# scope.capabilities.camera_supports_conversion_gain_mode / _line_noise_reduction.
scope.camera.set_high_conversion_gain(True)        # applied, then saved; returns None
scope.camera.set_line_noise_reduction(True)        # same contract
# On a camera without the mode, turning it on raises
# CameraSettingUnsupportedError (reason 'conversion_gain_mode_unsupported' /
# 'line_noise_reduction_unsupported'); turning it off is that camera's state
# already, so it is stored without a write. A camera that refuses raises
# CameraSettingRejected; with no camera, HardwareCommandRefusedError
# ('not_connected'). Nothing is stored on a raise.

# Black level: the camera's own offset parameter, in the camera's own units
# (Basler: a model-specific step, 0.0625 DN at 12 bits on the daA3840; IDS:
# DN of the current pixel format; FX2: the sensor's Row Black Target, fixed).
# Recorded with every frame (FrameRecord.black_level) and in saved metadata;
# a failed read at capture raises CameraReadFailedError and fails the capture.
scope.camera.read_black_level()                   # read from the camera now; None when no camera or none reported;
                                                   #   raises CameraReadFailedError when the camera does not answer
scope.camera.get_black_level_range()              # (minimum, maximum) set_black_level accepts, live, for the
                                                   #   current pixel format; None when no camera or none settable;
                                                   #   raises CameraReadFailedError when the read fails
scope.camera.set_black_level(value=4.0)           # value in effect; probe
                                                   #   scope.capabilities.camera_supports_black_level first;
                                                   #   raises CameraSettingRejected when the camera refuses
                                                   #   or fails the write, CameraReadFailedError when its
                                                   #   probe or range read fails

# Frame rate: the camera's own figure for what its current settings allow
# (Basler: the resulting acquisition frame rate; IDS: AcquisitionFrameRate's
# maximum; the simulator: its pacing). The rate frames actually reach the
# host at is get_delivered_rate, below.
scope.camera.read_frame_rate()                    # fps; None when no camera or none reported (FX2);
                                                   #   raises CameraReadFailedError when the camera does not answer
scope.camera.read_auto_gain()                     # CameraAutoMode from the camera now; OFF unasked on a
scope.camera.read_auto_exposure()                 #   camera without the loop; raises CameraReadFailedError

# Delivered rate: what the camera delivered to the host over the last
# second, counted from the frames every driver stores, in every host -- the
# same whether a display is drawing, paused or absent. bytes_per_s is those
# frames' payload on the link (the camera's own per-frame figure: Basler's
# grab-result payload, IDS's delivered data size, the FX2's frame bytes),
# not the transport's protocol overhead, so a packed pixel format reads
# smaller than its frames take in memory.
rate = scope.imaging.get_delivered_rate()          # DeliveredRate
rate.frames_per_s                                  # frames per second
rate.bytes_per_s                                   # link payload bytes per second
                                                   #   both 0 when nothing was delivered in the
                                                   #   last window (no camera, not streaming)

# Frame size (camera.frame_size is the held size, None when unknown;
# camera.read_frame_size() asks the camera and raises if it does not answer)
delivered = scope.camera.set_frame_size(2048, 2048)  # saved with the region it implies
# Returns the DELIVERED FrameSize(width, height): the size asked for, on every
# camera (it acquires the next window up on its grid and crops back).
# Only a size within one grid step of the camera's own maximum comes
# back smaller: no window on its grid holds it.
# Raises CameraSettingRejected (modules.exceptions) when a live camera
# refuses the apply; raises HardwareCommandRefusedError (not_connected)
# when no camera is connected.
# Base geometry code on the returned FrameSize, never on the request.
scope.camera.frame_size                           # FrameSize(width, height) the camera delivers -- held, no driver I/O
scope.capabilities.camera_limits.frame_max         # FrameSize, the scope's unbinned frame ceiling: the sensor,
                                                   # or the model's smaller one (LS560: 1700); None if unknown
scope.camera.get_pixel_alignment()                # {'width','height'} deliverable frame-size granularity: {2, 2}, even sides (H.264), on every camera

# Binning
scope.camera.set_binning(2)
# The delivered FrameSize(width, height) when applied; raises
# CameraSettingRejected when a live camera refuses. set_image_mode raises it
# the same way. Success is observed by the return value, rejection by the
# typed raise -- a dropped return cannot silently record a rejected apply.
scope.camera.read_binning()                       # read from the camera now; None with no camera;
                                                   #   raises CameraReadFailedError when the camera does not answer
scope.capabilities.camera_binning_sizes            # e.g. (1, 2, 4)
scope.camera.set_image_mode('12bit_scientific')    # the camera's pixel format for the mode, then saved; raises CameraSettingRejected on refusal
scope.capabilities.camera_pixel_formats            # e.g. ('Mono8', 'Mono12')


# Held reads, no driver I/O: what the camera last reported or took, None
# whenever that is unknown (before the camera's first report, with no
# camera, after an auto cycle whose read-back failed)
scope.camera.gain_db
scope.camera.exposure_ms
scope.camera.black_level
scope.camera.binning
scope.camera.pixel_format
scope.camera.auto_gain                            # a CameraAutoMode: OFF, ONCE (a one-shot still
scope.camera.auto_exposure                        #   running) or CONTINUOUS; an auto-gain arm, its
                                                  #   lock and a one-shot switch both
scope.camera.high_conversion_gain                 # these two have no driver read: the write
scope.camera.line_noise_reduction                 #   the camera took
scope.camera.image_mode                           # the saved mode; the camera holds only its pixel format
scope.camera.active                               # bool: connected and active, as last found

# The longest exposure the camera may be using now (ms), no driver I/O: the
# armed auto-gain's exposure ceiling while continuous auto-gain is on (the
# sensor maximum when it was given none), else the held exposure, or the
# sensor maximum while that is unknown. Bound a wait for the next frame by
# this, not by the held value, which holds the one from before auto-gain was
# armed. None when none of them is known.
scope.camera.longest_exposure_ms

# Scale bar overlay (burned into frames the imaging paths return when enabled;
# skipped while the objective is unknown, with one warning each time it becomes unknown)
# Whether it is drawn is the session's setting (session.set_scale_bar); the
# colour is the imaging API's own.
scope.imaging.set_scale_bar_color('Red')
scope.imaging.scale_bar_config                     # (in-process) {'enabled': bool, 'color': str | None};
                                                   # a wire client reads session.scale_bar_enabled
```

The acquisition frame-rate cap lives on the camera driver and clamps frame production regardless of sensor-readout capability. It is a driver-level control with no public API member -- described here only to explain the behavior. Used by the manual-record path to match user-requested video FPS, and by characterization tools to bound capture rate during long-running probes. No-op on drivers that do not implement the underlying setter (warning logged). Distinct from `set_exposure_ms` (per-frame integration time) and from any host-side throttling.

### Dynamic camera capabilities

Cameras advertise their real limits at connect time. Use these to size UI sliders and clamp auto-exposure / auto-gain:

```python
limits = scope.capabilities.camera_limits           # None if no camera connected
limits.exposure_ms_max                              # ms
limits.gain_db_max                                  # dB
limits.exposure_ms_min                              # ms, None if the camera declares no floor
limits.gain_db_min                                  # dB, None if the camera declares no floor
```

A floor is `None` when the camera declares none, not `0`: a missing floor is not a floor of zero, and nothing checks against it. A run whose steps ask for a gain or exposure outside these limits is refused before it starts (reason `camera_setting_out_of_range`), naming each step; the stored values are left as they are.

These are derived from the camera's profile, which is populated at connect via `_query_dynamic_capabilities()` — live SDK queries for Pylon / IDS, hardcoded-from-datasheet for FX2. Per-camera values observed in practice: LS620 FX2 = 42.144 dB gain (128x) / 1000 ms exposure cap; Pylon/IDS ranges are driver-reported.

### What a stored setting applies as

A saved per-channel gain or exposure is the user's committed intent and outlives whichever camera is attached, whether it came from the file or was just written through `update_settings` (a typed 1500 ms on a 1000 ms camera is stored as 1500). A body that cannot reach the value is driven to its own maximum instead, and the stored value is left alone — reconnect a capable camera and the intent applies again. Ask the API what a stored value actually becomes rather than comparing against a cap yourself:

```python
applied = scope.camera.applied_gain_db_for(48.0)     # stored=48.0 applied=20.0 capped=True
applied = scope.camera.applied_exposure_ms_for(500.0)
applied.stored                                        # what the user set
applied.applied                                       # what this camera is given
applied.capped                                        # True when the body is holding it down
```

The same holds for a stored auto-gain preference. A camera without hardware auto-gain runs manual whatever the layer stored, and the preference is kept for a camera that has the mode:

```python
applied = scope.camera.applied_auto_gain_for(True)   # IDS: stored=True applied=False capped=True
```

All three return an `AppliedCameraSetting`. For gain and exposure, an unknown cap (no camera, or a driver that publishes none) narrows nothing, so `applied == stored` and `capped` is `False`. This is the only place the cap is applied: the per-layer apply path sends `applied` to the driver, so the value a caller reads back is what the sensor is actually at.

Note the difference from `set_gain_db`, `set_exposure_ms` and `set_frame_size`, which are **explicit requests** and raise `CameraSettingOutOfRangeError` for a value outside the camera's range, the same on every camera. Capping belongs to re-applying something already stored; a direct request for an out-of-range value is an error, not something to silently narrow.

### Camera listeners

```python
def on_camera(param: str, value: float):
    print(f"Camera {param} = {value}")

scope.camera.add_camera_listener(on_camera)       # fires on set_gain_db / set_exposure
scope.camera.remove_camera_listener(on_camera)
```

### Live frame listeners

Sync per-frame handlers fire on every successful camera grab (Pylon `PylonImageGrab` thread / IDS grab loop / simulated pump). This is the canonical entry point for live image-processing plugins (see `ctx.plugins.live_processing`) and the manual-record path.

```python
def on_frame(frame):
    # Runs on the SDK callback thread. MUST NOT block. Heavy work
    # belongs on an executor.
    queue_write(frame.image, frame.timestamp)

scope.imaging.add_frame_listener(on_frame, name='my_recorder')
# ...
scope.imaging.remove_frame_listener(on_frame)
```

- **The frame record.** A listener is called with one argument, the camera's `Frame`, the record its frame store built, shared with every other reader:
  - `image`: the pixels, cropped to the camera's window;
  - `seq`: the arrival number, which only goes up (order frames by it, not by a clock);
  - `timestamp`: the host's wall-clock time at arrival, for what a person reads;
  - `arrived_ns`: the host's `time.monotonic_ns()` at arrival, a steady time;
  - `significant_bits`: the payload depth of these pixels (12 for Mono12 in a uint16 container);
  - `chunks`: the camera's per-frame metadata (`ExposureTime`, `Gain`, `FrameID`, `Timestamp` on a Basler), or `None` for a camera that stamps none (IDS, FX2, the simulator).
- **Read-only frame.** The record is read-only: `frame.image` refuses a write (`ValueError`) and `frame.chunks` is a read-only mapping. Never set the image writeable again (`flags.writeable = True`): every other holder would see the change. Draw on your own copy (`frame.image.copy()`). Keeping the record is safe: no later frame changes it.
- **Budget.** Each handler must complete within ~24 ms (anchored to a 30 fps target, half the inter-frame window). Over-budget invocations log a WARNING. After 30 consecutive over-budget hits, the handler is auto-removed and reported once as a `FrameHandlerRemovedError` warning (reason `over_budget`).
- **A handler that raises.** The first error of a run of failures is logged with its traceback; the rest are counted. A handler that raises on 30 consecutive frames is removed the same way (reason `raised`). A frame it handles without raising resets the count.
- **With no camera connected** `add_frame_listener` raises `HardwareCommandRefusedError` (`not_connected`, `MissingPart.CAMERA`; `scope_disconnected` once the session's `shutdown()` has shut the lanes) and registers nothing.
- **A registration the camera refuses** raises `FrameListenerNotRegisteredError` (a `CaptureError`, `reason='frame_listener_refused'`), chained from the driver's error; nothing is left registered, so the call can be retried.
- **Removal.** Once `remove_frame_listener` returns, no new call reaches the handler (a call already running completes), even if the camera driver fails to unregister it.
- **Re-entrancy.** A handler will not be re-entered on the same thread; the driver's fire-site is single-threaded.
- **Plugin authors**: use `ctx.plugins.live_processing.register(spec, handler)` rather than calling `add_frame_listener` directly. The registry forwards through to this API and surfaces the plugin name in the budget-violation log.

### Listener callback signatures (overview)

The seven listener families each pass a different callback signature -- register a callable matching the row for the listener you subscribe to:

| Listener | Register via | Callback signature | Event record |
|---|---|---|---|
| Motion / position | `scope.motion.add_position_listener` | `on_position(axis: str, at: AxisPosition)` | `PositionChanged(axis, at)` |
| LED / illumination | `scope.illumination.add_led_listener` | `on_led(address: int, name: str \| None, state: LedState, illumination_ma: float \| None)` | `LedReading(address, name, state, illumination_ma)` |
| Camera params | `scope.camera.add_camera_listener` | `on_camera(param: str, value: float)` | `CameraChanged(setting, value)` |
| Live frame | `scope.imaging.add_frame_listener` | `on_frame(frame: Frame)` | `FrameDelivered(frame)` |
| Run state | `session.add_run_state_listener` | `on_run_state()` -- no payload; re-read the session derivations (see Run state and locks above) | `Status`, read at the edge but for `camera_streaming`, asked just after |
| Outcomes | `ScopeSession.create(outcome_listener=...)` or `session.add_outcome_listener` | `on_outcome(n)` -- one `Notification` (see Outcomes above) | `Notification` itself |
| Settings | `session.add_settings_listener` | `on_settings(paths: tuple[str, ...])` -- the dotted paths one write stored | `SettingsChanged(paths)` |

Each event declares its record, published in the reference below: its fields name the callback's arguments in order, so a host that carries an event elsewhere (the REST server's stream) builds the record from the arguments rather than keeping its own table of signatures. A run's events do the same: `RunEvents.scan_started` delivers `ScanStarted(scan_index, scans_remaining, period_s)`, `scan_ended` `ScanEnded(...)`, `step_started` `StepStarted(step_index)`, `frame_captured` `FrameCaptured(image, frames_summed, frame_significant_bits)`, `video_progress` the `VideoProgress` it passes, `run_ended` `RunEnded(outcome, run_dir, protocol)`, `files_written` `FilesWritten(run_dir, files)`. Callbacks keep their signatures; a Python caller need not build any record.

The four `scope.*` listeners each have a matching `remove_*_listener(callback)`. The frame listener additionally takes a `name=` kwarg and carries the read-only frame + 24 ms budget contract documented above; the other three are lightweight state-change notifications.

The last three rows are not registered on the scope. **Settings** is registered on the session: each write to the settings store -- by the GUI, a REST client, a script, or the camera storing what it applied -- tells the paths it stored, all of them once its write has let go of the store, on the writer's thread; `session.remove_settings_listener(callback)` removes it. On the REST stream it is the `settings` event, so a client beside the GUI hears the GUI's changes and the GUI redraws a client's. **Run state** is registered on the **session**: it takes no payload and is level-synced -- registering calls it once immediately, so a subscriber never misses a transition that happened before it subscribed, and `session.remove_run_state_listener(callback)` removes it (a callback never added is a no-op). **Outcomes** are registered on the session, or given to the factory to hear bring-up, and have a matching `session.remove_outcome_listener(callback)`.

### Camera info

```python
scope.camera_connected                             # bool property (mirror of motor_connected / led_connected)
scope.diagnostics.get_camera_temperatures_degc()        # temperature sensors; {} no sensor, None no camera, a failed read raises
scope.capabilities.camera_model                    # the camera's model, serial number (camera_serial_number) and
                                                   # timestamp clock (camera_timestamp_tick_hz), read at connect
scope.capabilities.camera_sensor                   # 'Aptina MT9P031', and camera_shutter 'rolling' or 'global', from
                                                   # the camera's datasheet entry; None for a camera none describes
scope.capabilities.camera_sdk_version              # the vendor SDK's version, read at connect; None without one
scope.capabilities.camera_limits                   # the gain and exposure ranges, read at connect (CameraLimits)
scope.capabilities.pixel_size_um                   # the installation's pixel size
```

### Frame validity

Frame validity is the single source of truth for "is the next frame still what I asked for?" Every hardware state change invalidates pending frames. `capture_and_wait()` drains stale frames until all sources settle, detects any invalidation that lands mid-grab (re-draining and re-grabbing so the result reflects the newest commanded state), bounds the whole settle-and-recheck loop with a deadline (loud None when invalidation outruns it), and verifies the returned frame's own chunk metadata (exposure / gain) against the requested values on cameras with chunk support -- the saved frame proves its own settings.

When continuous auto-gain is armed, `capture_and_wait()` first locks it: the camera's auto loop is switched off, the exposure and gain it landed on become the requested values, and the frame is then verified against them like any manual setting. The frame is returned and saved with its achieved values, an exposure pinned at the layer class's ceiling or floor included. A live-view arm is re-armed after the capture; a protocol step's arm stays locked until the next step arms again.

A caller that switches auto-gain off and wants to keep what it achieved calls `scope.camera.lock_auto_gain()`, which returns the same result object: `state` (None when no arm was standing), `exposure_ms` and `gain_db` (the achieved values), and `stored_exposure_ms`, the exposure to store as the manual setting -- the achieved value floored to the channel class's usable floor, decided by the API so every client stores the same value. Limit states and a failed lock under a live-view arm also reach the user through the notification center; a protocol step's arm is logged only.

```python
# Leave auto-gain and keep what it achieved as the manual setting
lock = scope.camera.lock_auto_gain()
if lock.state is not None:
    scope.camera.set_gain_db(lock.gain_db)
    scope.camera.set_exposure_ms(lock.stored_exposure_ms)
    print(f"{lock.state.value}: achieved {lock.exposure_ms} ms, stored {lock.stored_exposure_ms} ms")
```

```python
scope.imaging.frame_is_valid                       # True if next frame is valid
scope.imaging.frames_until_valid()                 # 0 = ready, >0 = keep draining
```

For a sum, measure against `capture_frame_full_scale(image)`, the value N blown frames reach, rather than a bit depth: `capture_and_wait(all_ones_check=True)` already checks each summed frame against its own depth before summing.

A sum carries the bits it can reach: N frames of b bits reach N x (2^b - 1), so four 12-bit frames are a 14-bit sum, up to 16, where its 16-bit container saturates. Summing is for a brighter image, so every 8-bit rendering of a sum -- a JPG, LumaViewPro's display and overlays -- is drawn against one frame's white: four frames look four times brighter, and white wherever the sum passes one frame's full scale. A full-depth file keeps every count.

A whole-frame mean cannot answer this question: an evenly lit field at 70% of full scale and a field that is 70% blown white and 30% black report the same mean. Any caller deciding whether an operating point is usable needs the pixel count.

Invalidation is automatic: every API setter invalidates the frames its change makes stale. The sources that invalidate frames are:

```text
led        — LED turn on/off or illumination change
gain       — gain change
exposure   — exposure change
black_level — black level change
auto_gain  — continuous auto-gain armed (settles against the lit scene)
pixel_format, frame_size, binning — geometry or format change
conversion_gain_mode, line_noise_reduction — sensor noise-mode change
z_move     — Z axis motion
xy_move    — X or Y axis motion
turret     — turret move
```

`exclude_sources` in `capture_and_wait()` skips the wait for a source, so the frame returned can predate that source's change -- a frame already on its way when the change went out. That fits a frame you only show; never pass it for a frame you measure or record. Autofocus waits out each Z move. Earlier builds, the 4.0.0 beta included, excluded `z_move`, so a step could score the previous step's frame at its own Z. Each sweep step now waits for a frame made after its move stopped: on a camera with a clock latch, one that started exposing after the stop; on a camera without one, the frame after the `z_move` count (2 frames) that arrived after the stop. A step costs about one frame more than in earlier builds on a camera with a clock, and two more on one without.

---

## scope.diagnostics

Hardware diagnostic probes and identity getters live on the `scope.diagnostics` sub-API. Per-call (no persistent state); meant for tech-support reports, bench tooling, and bring-up scripts that want one-shot snapshots of camera / motor / LED state.

```python
scope.diagnostics.get_motor_info()         # MotorBoardInfo, as the board reported at connect: model (its own
                                           #   report, 'LS850'), serial_number, firmware_version ('original' for
                                           #   the original firmware), firmware_date, command_set: 'text' (the
                                           #   EL-0940's INFO, FULLINFO, ACTUAL_R, ...) or None (no text commands,
                                           #   e.g. the LS720's TMCM-6110). Each field None when unknown, all of
                                           #   them with no board.
scope.capabilities.model                   # the model the scope runs as: the board's, else the selection
scope.diagnostics.get_led_info()           # LedBoardInfo: firmware_version, firmware_date, command_set:
                                           #   'v2' (INFO, SELFTEST, I2CSCAN, LEDREAD),
                                           #   'legacy' (INFO only: firmware before v2),
                                           #   None (no text commands, e.g. an FX2 scope). Each field None
                                           #   when unknown; whether a board is there is scope.led_connected
scope.capabilities.pixel_size_um           # raw per-installation um/pixel, or None if the scope cannot report it (unknown camera / no declared optics). The effective um/pixel at the current objective and binning is scope.runtime_state.get_pixel_size_um().
scope.capabilities.lens_focal_length_mm    # tube-lens focal length, mm, or None if the scope cannot report it
```

```python
# Camera temperature sensors. Returns dict {sensor_name: degC}, each name the
# camera's own ('FpgaCore' on a Basler body, 'Device' on a one-sensor IDS body); empty only
# for a camera with no temperature sensor (capabilities.camera_reports_temperature
# False); None when no camera is connected, after the session's shutdown() too. A read that fails on an active camera
# raises HardwareError: neither {} nor None ever stands for a failure.
temps = scope.diagnostics.get_camera_temperatures_degc()

# The camera's link, live: a CameraLink (transport 'USB3' | 'GigE' | 'USB2',
# link_speed, link_speed_unit, packet_size_bytes, inter_packet_delay),
# each None where the camera does not report it (packet size and delay are
# GigE's). The speed is in the unit the camera declares, never converted:
# Basler's varies by model; the FX2's is 'Mbps'. A camera that declares no
# unit for its speed reports both None.
# None when no camera is connected, after the session's shutdown() too; a failed read
# raises HardwareError.
link = scope.diagnostics.get_camera_link_info()

# Which driver Windows has bound to each USB identity of a Lumascope Classic,
# the read the camera's connect checks, so it answers when the camera did not
# come up ('camera_driver_unusable' in the bring-up record). A tuple of
# UsbIdentityBindings, the bootloader's (USB\VID_04B4&PID_8613) then the
# running firmware's (USB\VID_04B4&PID_EA17): hardware_id, name, needs (the
# drivers LumaViewPro can use it on), usable (False when a present device is
# on none of them) and devices, each a UsbDeviceBinding (instance_id,
# present, service -- None for no driver -- provider, inf, version, date,
# problem_code). None on macOS and Linux, which bind no driver to a USB
# device, and on a simulated scope with no FX2. A failed read raises
# CameraUsbLookupFailedError.
bindings = scope.diagnostics.get_camera_usb_bindings()

# Throughput and latency characterization. Both run through the
# PRODUCTION capture path, so what they measure is what a real run gets;
# both take an optional progress_cb and return a dict of results. These
# occupy the camera for their duration. While a run, a diagnostic or a
# home holds the scope they are refused (exclusive_activity_running),
# raised; do not start one while a recording is live.
results = scope.diagnostics.run_camera_bandwidth_test(num_frames=200, timeout_s=60.0)
# num_frames is 1 to 100,000; outside it is refused (ArgumentRefusedError,
# out_of_range) before the camera is touched.
# num_frames_failed counts the grabs that raised CaptureFailedError, and
# failed_by_reason counts them by reason; a camera absent at the call or
# removed mid-test is refused (not_connected), raised, not counted.
results = scope.diagnostics.run_grab_lifecycle_benchmark(num_cycles=100, vary_settings=False)

# Cross-host / cross-camera / cross-firmware diagnostic probe.
# Captures camera identity, current config, temperatures, and stream
# stats deltas over duration_s, stamped with the active camera SDK
# (Basler pylon, IDS peak, ...). Writes JSON to data/camera_probe/.
# A driver that does not implement the probe returns the driver's
# {'supported': False, ...} shape unchanged. Does NOT change grab state.
probe = scope.diagnostics.run_pylon_diagnostic_probe(
    duration_s=3.0, drain_camera_side_errors=True,
)

# Motor-board driver / fan diagnostics (already on
# DiagnosticsAPI pre-Phase-5; documented here for completeness).
status = scope.diagnostics.read_motor_drv_status('Z')       # int register or None
rpm = scope.diagnostics.read_motor_fan_rpm()              # RPM or None
spin = scope.diagnostics.motor_fan_spin_times()            # {'spin_up_s', 'spin_down_s'}, or None when
# no motor board drives a fan: how long after set_motor_fan_duty a read_motor_fan_rpm
# sees the new speed
scope.diagnostics.set_motor_fan_duty(50)                   # None once written; HardwareCommandRefusedError
# 'axis_absent' (MissingPart.FAN_CONTROL) for a motor controller without fan
# control, MissingPart.MOTORS on a manual scope; ArgumentRefusedError for a duty
# that is not a number from 0 to 100; HardwareError when the board answers the
# write with an error or does not answer.

# LED engineering-mode handshake (FACTORY / Y / Q with post-Q drain).
# Use these in place of open-coded send_diagnostic_command sequences.
scope.diagnostics.enter_led_engineering_mode(timeout_s=5.0)   # None once entered; HardwareError when
# the handshake fails; 'not_connected' with no LED controller, 'axis_absent'
# (MissingPart.LED_ENGINEERING_MODE) for one without the handshake (Classic).
currents = scope.diagnostics.read_led_currents_ma()   # {channel: mA or None}; v2, in engineering mode
scope.diagnostics.exit_led_engineering_mode()           # likewise; HardwareError if the board is not back
```

---

## scope.capabilities

`scope.capabilities` is a `ScopeCapabilities` dataclass populated at connect time. **Use this to learn what the connected hardware can do** — don't hardcode axis lists, LED channel counts, or camera caps.

```python
caps = scope.capabilities

# Motion
caps.axes                       # ('X', 'Y', 'Z', 'T') on LS850T; ()         on LS620
caps.has_focus                  # True if Z is motorized
caps.has_xy_stage               # True if X/Y are motorized
caps.has_turret                 # True if the turret axis is present
caps.model                      # e.g. 'LS850T'; the layer identity's model, '' if none

# LED
caps.led_channels               # e.g. (0, 1, 2, 3) for FX2 scopes; (0..7) for RP2040; None when the scope came up without its LED board
caps.led_colors                 # the model's layers that drive an LED, from its layer identity
caps.led_max_ma                 # per-channel current cap; None when the scope came up without its LED board
caps.has_firmware_stim          # firmware-timed stim support on the LED board

# Optics -- resolved from the first real source, never a hardcoded default:
#   motorconfig.json Optics (LS820/850/850T) -> scopes.json Optics (Classic)
#   -> camera SDK-reported pitch (pixel size only) -> None
caps.pixel_size_um              # um/pixel, or None if the scope cannot report it
caps.lens_focal_length_mm       # tube lens focal length mm, or None if unavailable

# Camera
caps.camera_model               # 'MT9P031-LS620', 'daA3840-45um', etc.; None if no camera or it did not say
caps.camera_serial_number       # the camera's serial number, or None
caps.camera_timestamp_tick_hz   # the camera's frame-timestamp clock in Hz, or None (no timestamps)
caps.camera_has_clock_latch     # the host can latch that clock: each capture's frame_record.clock_mapping
caps.is_color_native            # True for color-native sensors; False for mono-native (default)
caps.buffer_bit_depth           # the buffers' width: 8 for an 8-bit-only camera (the FX2), else 16 (uint16; holds 10/12/16-bit data)
caps.camera_max_significant_bits  # the deepest data depth the camera delivers, by its driver (12 for a Mono12 camera, Basler or IDS; 8 for the FX2); None with no camera
# scope.imaging.significant_bits  # the depth of the frames it delivers now, following set_image_mode; HardwareCommandRefusedError (not_connected) with no camera
caps.camera_supports_auto_gain
caps.camera_supports_auto_exposure
caps.camera_supports_conversion_gain_mode
caps.camera_supports_line_noise_reduction
caps.camera_supports_black_level  # set_black_level offered; get_black_level reads on any camera that reports one
caps.camera_pixel_formats       # e.g. ('Mono8',) or ('Mono8', 'Mono12')
caps.camera_binning_sizes       # e.g. (1, 2, 4)
caps.camera_limits              # CameraLimits: gain_db_min/_max, analog_gain_db_max, exposure_ms_min/_max,
                                # frame_min, frame_max (FrameSize, unbinned); read once at connect, None with
                                # no camera. A limit the camera does not declare is None and is not checked.
caps.camera_limits.frame_max   # FrameSize unbinned, in pixels: the smallest of the sensor the profile
                                # documents, the camera's own maximum and the model's (data/scopes.json
                                # MaxFrame); None if none of them is known
caps.camera_limits.analog_gain_db_max  # dB: the most gain applied before a digital stage (above it the
                                # camera multiplies digitised values). The profile's analog maximum, or the
                                # camera's live maximum where it has no digital stage; None if neither is
                                # known. The whole range, analog and digital, is camera_limits.gain_db_max
caps.camera_reports_temperature # the camera has a temperature sensor (probed at connect); False:
                                # get_camera_temperatures_degc answers {}
```

Important consequences:

- **`camera_limits` is `None` when no camera is connected**, and its `frame_max` `None` when none of its sources answered. Check it before using it as a `scope.camera.set_frame_size(w, h)` target, and divide by the binning in force; `set_frame_size` raises `HardwareCommandRefusedError` (`not_connected`) when no camera is connected. With a live camera it returns the DELIVERED geometry and raises `CameraSettingRejected` if the apply is refused.
- **LED channel count varies by scope, and not only by driver family.** An LS620 (FX2 driver) exposes 4 channels (`BF`, `Blue`, `Green`, `Red`); an **LS560, same driver family, exposes 2** (`BF`, `Green`); RP2040-based scopes expose 6 (`BF`, `PC`, `DF`, `Blue`, `Green`, `Red`). Don't iterate over a hardcoded list — iterate over the layers in `scope.layer_identity.layers` that have an `led_channel`.
- **Some scopes have no motor at all.** LS560/LS620 have `caps.axes == ()`. Calling `scope.motion.move_xy(x_um=…)` against such a scope raises `HardwareCommandRefusedError` (`axis_absent`, `MissingPart.MOTORS`), as a Z-only scope does for X (`MissingPart.X`); hide motion controls based on `caps.has_xy_stage`, `caps.has_focus` and `caps.has_turret`.
- **Travel limits come from `scope.motion.get_axis_limits(axis)`**, read-only, for present axes; check `caps.has_xy_stage` (or `axis in caps.axes`) before asking about X/Y. A run whose steps lie outside these limits is refused before it starts (reason `positions_outside_travel`), naming each step and the axis; X/Y are judged on the protocol's plate at `scope.runtime_state.get_stage_offset()`, the offset the run moves with. A run whose steps need an axis the scope does not have is refused as `positions_unreachable`: steps at different places on that axis, or, on a scope without X and Y, steps with no X/Y whose wells or tiles differ, since each would be one field saved under a different name.
- **A step may have no position on an axis.** It is recorded on a scope that lacks the axis: blank in the `.tsv`, NaN in `protocol.steps()`'s `X`, `Y` or `Z` column (pandas' missing value, so the column stays a number column), None in `protocol.step(step_index)`'s `x`, `y` or `z` and null over REST. X and Y are both or neither; a file or frame with one of them is refused (`ProtocolFormatError`). On a scope that has the axis such a step says nothing about where to go, so a run, `step_targets` (and Go To Step), `apply_tiling` and `apply_zstacking` refuse it as `step_position_missing` (`ProtocolRunRefusedError`), naming the steps and the axes; nothing moves or changes. A protocol file with a blank position is refused by LumaViewPro releases before this one, which read every position as a number.
- **What a scope without an axis records.** Nothing for that axis, everywhere a position is kept: `session.get_current_plate_position()` answers None for it (a manual LS620 or LS560 answers `PlateLocation(None, None, None)`, a Z-only LS820 X and Y None), so `add_step` and `new_protocol` record no X/Y without an XY stage -- the plate's well centres are not places such a scope reaches; the step keeps its well name -- and no Z without a Z axis, where no focus is asked for either. `update_step` keeps a step's own position on an axis the scope lacks, so a protocol made on a stage scope keeps its X/Y through an update on one without. `new_protocol(tiling=...)` other than `'1x1'` on a scope without X/Y is refused `positions_unreachable`, as `apply_tiling` is; `session.save_focus` on a scope without Z is refused `positions_unreachable`; `session.save_bookmark(axes)` naming an axis the scope has no motor for is refused `HardwareCommandRefusedError` (`axis_absent`). Nothing is written by any of them.

---

## Future I/O

Trigger devices, USB-to-IO trigger boards and external sync are not in LumaViewPro 4.0.x; they will get their own sub-API, documented here, when they ship.

---

## scope.runtime_state

Mutable counterpart to `scope.capabilities`: the runtime-mutable user configuration (labware, objective, turret, stage).

Firmware versions are what each board reported at connect: `scope.diagnostics.get_motor_info().firmware_version` / `scope.diagnostics.get_led_info().firmware_version`.

---

## Images, positions and scale

A composite is a run (`run_composite`) or a build from a folder's files
(`session.post_processing.composite(folder)`), and focus is a run
(`start_autofocus`, `start_autofocus_all_steps`); see "Running protocols" and
"Post-processing". What a saved file holds is described under "Capture".

### Coordinate transformations

Stage and plate positions convert through `scope.runtime_state.stage_to_plate` and `plate_to_stage` (above): plate millimetres from the plate's top-left, stage micrometres, on the selected plate and live offset unless `labware_id=` or `stage_offset=` names another.

### Image scale

```python
scope.runtime_state.get_pixel_size_um()   # µm per pixel at the current objective and the camera's binning
scope.runtime_state.get_field_of_view()   # FieldOfView(width_um, height_um): what the camera's frame covers
```

The scale every saved image carries, published so no caller recomputes it. Each answers `None` when it cannot be known: no objective is known (`resolve_current_objective` says why), there is no camera, the camera has not reported its binning (or, for the field of view, its frame), or the scope cannot report its optics (unknown camera, no declared optics). There is deliberately no fallback: a guessed pixel size is written into every image and cannot be told from a measured one, so a `None` here means no scale bar, no field of view and no `PhysicalSizeX`. `scope.capabilities.pixel_size_um` is the sensor's raw pitch, before the objective and the binning.

### Protocol (`modules.protocol`)

`session.load_protocol(file_path)` loads multi-step acquisition sequences. See the [Headless protocol run](#headless-protocol-run) pattern for a runnable example.

```python
from modules.protocol import Protocol
```

---

## plugin platform reference

Plugin platform spec and live-processing tutorial both live alongside LumaViewPro.

- **Design**: `docs/PLUGIN_API_DESIGN_2026-05-09.md` — the locked platform spec (PluginSpec, namespaces, registry contracts, loading sequence).
- **Plugin tutorial**: `docs/PluginTutorial.md` — the plugin shape and lifecycle, worked for `ctx.plugins.post_processing`.
- **Namespaces (4.x)**: `ctx.plugins.ui`, `ctx.plugins.post_processing`, `ctx.plugins.live_processing`, `ctx.plugins.rest`. A plugin's `ctx` is the session hosting it (see "Plugins").

A worked plugin example ships in `etaluma-engineering/`; see its `pyproject.toml` `entry_points` for how a plugin declares itself.

A plugin runs in LumaViewPro's process and may call any member, marked or not; the API reference below lists what is API. An unmarked member can change in any release, so LumaViewPro names the oldest version of a plugin it works with (`MINIMUM_PLUGIN_VERSIONS` in `modules/plugins`) and refuses an older one when it discovers it.

---

## REST surface reference

The REST server puts the API reference below on HTTP: every member marked for the wire, nothing else, in the forms `modules/wire_encoding.py` declares. A client in any language reaches what a Python caller reaches, is refused in the same words with the same reason, and hears the same events.

### Serving it

Start it from the installation's folder:

```bash
python -m rest --simulate      # the simulated scope; omit --simulate for the attached hardware
```

It brings the scope up as LumaViewPro does -- the user's settings, the startup home, the plugins -- and serves on `127.0.0.1` at the port in `rest_api.port` (shipped 8000), printing the URL. It takes the same single-instance lock as LumaViewPro, so it will not start while LumaViewPro or another server drives the scope. A failed startup home is reported in the log and the server still serves, so a client can home again. Ctrl-C or SIGTERM closes it with its clients connected: a new connection is refused, and a member or a new live view asked on an open one is refused (`503`, `server_closing`), while jobs, handles, files and the event stream are still served, and a live view already open keeps showing frames; the session closes, and the stream carries what that close reports -- a run's end, the LEDs turned off -- then `closing`, and ends, as every live view does; a client still waiting on a job then -- a `GET /api/v1/jobs/<id>` long-poll, or a call inside its `Prefer: wait` -- is answered what the job is at that moment, so no wait holds the exit; the process exits `0`, or `1` when the session did not close cleanly. A second signal while it closes is logged and changes nothing. It runs from a source or pip install.

LumaViewPro serves the same API beside its window when `rest_api.enabled` is set in the installation's settings file (shipped `false`): on `127.0.0.1` at `rest_api.port`, on the window's own session, once the startup home has settled. A port already taken is reported once (`RestServerNotStartedError`, `rest_server_not_started`) and LumaViewPro runs on without the server. Closing the window closes the server in the same order as a signal does above -- calls refused, the session closed with the stream open, then `closing` -- and the server refuses calls before the window takes back a held control session, so no client takes control during the close.

To serve a session your own program has brought up, the server is an application (`rest.app.build_app`) served with uvicorn:

```python
import uvicorn
from modules.scope_session import ScopeSession
from rest.app import build_app

session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
session.run_application_session()          # homes the scope, as the GUI does at launch
uvicorn.run(build_app(session), host='127.0.0.1', port=8000)
session.shutdown()                          # once uvicorn has stopped
```

Serve it on `127.0.0.1` only: it has no credentials, so anyone who can reach the port drives the scope. It needs `fastapi` and `uvicorn` (both in `requirements.txt`). `GET /api` answers the versions served (`{"versions": ["v1"]}`); the OpenAPI 3.1 description of every route is at `/api/v1/openapi.json`, and an interactive reference at `/docs`. Each route's description lists what its member says it may refuse or fail with -- its Python `Raises:` section -- under the status each reason answers (a refusal `422` or `409` by its reason's cause, a fault `500`), with `x-refusals` on each status mapping the problem class to its reasons; an entry that names no reason lists every reason its class can carry. The `4XX` and `5XX` ranges stay for what no section names.

The description says what each route answers, read from the member's declared return type in the form it crosses in. A member route's `200` is its answer: each record is one named schema under `components/schemas` (`AxisPosition`, `Status`, `RunOutcome`, ...) with its keys and their types, a handle is `Handle`, an answered path `PathAnswer`; its `202` is `Job`. A client reads a record's keys there, not from this page. A record's schema is open: a key it does not name may be sent beside the ones it does. Where a member declares only a bare `dict` or `list`, the schema is an object or array that says nothing of its keys or items; those are typed one by one by their owners. `/api/v1/files/...`, `/api/v1/previews/...`, `/api/v1/live`, `/api/v1/live.jpg` and `/api/v1/events` are declared by what they send (a file, a JPEG, MJPEG, a JPEG, an event stream), not as JSON.

### Routes

The Session is the root: its member `m` is `/api/v1/m`, and a sub-object is a path segment, so `session.scope.motion.move_xy` is `POST /api/v1/scope/motion/move_xy`. Names are the Python names.

- A read is `GET`: a property or published field, and a method marked as a read, one that changes nothing (`GET /api/v1/scope/motion/axis_positions`). Its encoded answer is the whole body.
- A read method's arguments are its query string, one key a parameter (`GET /api/v1/scope/motion/get_axis_limits?axis=Z`). The text is converted to the parameter's type: `2` to a number, `true` / `false` to a bool (`1` / `0`, `yes` / `no` and `on` / `off` are taken too), a whole number like `2.0` where an integer goes. A list is its key repeated (`exclude_sources=a&exclude_sources=b`); a record is one key a field, `<parameter>.<field>` (`stage_offset.x_um=3&stage_offset.y_um=4`), and a record given in part is refused. A parameter left out takes the member's own default; a query string has no `null`, so a parameter whose default is `None` is given `None` by leaving it out, and an empty value (`labware_id=`) is the empty string.
- A setting is one URL: its read's. `GET` reads it and `PUT` writes it, with the setter's arguments as a JSON object by parameter name, and the answer to a `PUT` is the read's answer after the write, the value the scope took (`PUT /api/v1/scope/camera/binning` with `{"size": 2}` answers `2`; `PUT /api/v1/scope/camera/frame_size` with `{"width": 960, "height": 600}` answers the `FrameSize` delivered). A setting kept per layer or per slot takes its key in the `PUT` body and in the `GET`'s query (`PUT /api/v1/channel_acquire` with `{"channel": "BF", "mode": "image"}`, then `GET /api/v1/channel_acquire?channel=BF`). The setter itself has no route: `POST /api/v1/scope/camera/set_binning` is `404`. Each setting with a URL: `model` (`select_model`), `objective_id` (`select_objective`), `turret_objective` (`assign_turret_objective`), `labware` (`select_labware`), `scale_bar_enabled` (`set_scale_bar`), `acceleration_limit_pct` (`set_acceleration_limit`), `channel_acquire` (`set_channel_acquire`), `channel_auto_gain` (`set_channel_auto_gain`), `saved_focus` (`save_channel_focus`), and on the camera `image_mode`, `binning`, `frame_size`, `high_conversion_gain`, `line_noise_reduction`, `gain_db`, `exposure_ms`, `black_level` and `auto_exposure`.
- Any other method is `POST`. Its arguments are a JSON object by parameter name; a method with none takes an empty body or `{}`. A parameter with no default must be present, one left out takes the member's own default, and `null` is `None`.
- The answer to a finished call is `200` with its encoded return value as the whole body, `null` for `None`.
- Arguments are checked before the member runs: a key the member does not take, a value that is not of the parameter's type (in a body, a string where a number goes), a number that is not finite, a missing required one, or a query string on a route that takes none (a property, a `PUT` or a `POST`) is refused `422` (`invalid_request`, its `errors` naming each argument and why). A read asked with `POST`, a setting with `POST`, or a call with `GET`, is `405`.
- A read is answered like any call: one that outlives its wait is a `202` job (below). No read hands out a running call or a live object, so a read is safe to repeat; a camera's fresh read (`read_gain_db`) asks the camera and keeps what it answered as the held value, and changes nothing on the camera.
- Units are the API's: positions in um (the turret's in slots), exposure in ms, illumination in mA. A duration is a float of seconds, its name ending `_s`, and a `datetime` crosses as ISO 8601 with its offset; no member hands out or takes a `timedelta`. An axis's state and position (`AxisPosition`) cross as `{"position": ..., "state": ...}`. `state` is one of `unknown` (not homed, or its reference lost), `idle`, `moving` or `homing`, and `position` is `null` unless the state is `idle` or `moving`. `axis_positions` and `Status.axes` are an object keyed by axis name (`"X"`, `"Y"`, `"Z"`, `"T"`), holding only the axes this scope has (no `"T"` without a turret). A `position` event carries one axis, `{"axis": "X", "at": {"position": ..., "state": ...}}`. `get_axis_limits` answers `{"min_um": ..., "max_um": ...}`, or `null` for an axis with no travel limits (the turret).
- A path, sent or answered, is a name in the live folder, `/`-separated (`"ProtocolData/run1"`). A path parameter takes that name; a name that leaves the live folder is refused (`outside_live_folder`). An answered path is `{"name": ..., "host_path": ...}`, `name` null when it is outside the live folder. The live folder itself is `"."`: `GET /api/v1/live_folder_listing`, or with `?name=.`, lists its top level, `{"output_dir": "."}` writes a support report into it, and an answered path naming it has `name` `"."`. An empty name, and one that starts with `/` or a drive, is refused (`outside_live_folder`).

### Live objects: handles

A live object -- a protocol, a protocol runner -- crosses as a handle, `{"handle": "1", "type": "Protocol"}`. One object has one id, whichever call handed it out. Its members are at `/api/v1/handles/<type>/<id>/<member>` (`GET /api/v1/handles/Protocol/1/period`), and a parameter that takes it is sent its id (`{"protocol": "1"}`). `GET /api/v1/handles` lists every handle held, oldest first, so a client that lost an answer finds its object there. `DELETE /api/v1/handles/<type>/<id>` forgets the id, never the object. An operation in flight -- a run, a move, a home, a still -- is not a handle: it is a job (below). The session's protocol runner is every client's, and its id is kept (`409`, `handle_shared`). An id not held is `404`. Past 1000 handles held, a call that could hand out another is refused before it runs (`503`, `overloaded`, with `Retry-After`). A handle's id is a string of decimal digits, counted from `1` when the server starts, and its type is a Python class name, so neither needs percent-encoding in a path. An id is never reused while the server runs; a restarted server counts from `1` again.

### Anything slow is a job

Every call runs on its own thread, and its route waits for it as long as the client says it will: `Prefer: wait=<seconds>` (RFC 7240), 2 s when it says nothing, at most 60 s; the wait applied is echoed in `Preference-Applied`. `Prefer: respond-async` asks for the answer as soon as it can be given, a wait of none, and wins over a `wait` beside it (`Preference-Applied: respond-async`). A call still running then answers `202 Accepted` with `Location: /api/v1/jobs/<id>`, `Retry-After`, and the job:

```json
{"id": "1", "member": "handles/ProtocolRunner/2/start_single_scan", "status": "running",
 "requested": "2026-10-09T11:47:56.043-07:00", "ended": null, "progress": null}
```

`status` goes `running` -> `completed` | `failed`; `progress` is `{"percent", "detail"}` from a member that reports how far it has got, and a run's job's is the run's `RunProgress` (`is_live`, `is_stopping`, `step_index`, `run_dir`, ...). `GET /api/v1/jobs/<id>`, itself waiting by `Prefer: wait`, reads it; once it has ended it carries `result` (what a `200` would have carried) or `error` (the problem). Reading a failed job is `200`. A result the wire cannot carry -- a NaN or an infinity, which JSON has no form for -- is a `500` fault problem naming the value, answered where the call ends, so a job holding it reads as `failed`. A member that hands back an operation (an `Operation`: a run, a move, a home, a still) is answered once it has returned -- once the operation is dispatched -- whatever the client's `Prefer: wait`: `202 Accepted` with the operation's own job, its `Location` and `Retry-After`, and no other job made for the call. To wait for its end, read the job with `Prefer: wait`, or call the waiting form in Python. `GET /api/v1/jobs` lists the jobs held, newest first. `DELETE /api/v1/jobs/<id>` on a finished job forgets it (`204`); on a running operation's job it is the operation's `stop()`, answered `202` with the job, which ends as the stop leaves it -- a stopped run `completed` with its outcome `aborted` / `stopped`, a stopped move or home `failed` with its stop notice -- and is kept until a client forgets it. As in Python, a stop once the operation's work has ended does nothing, and so does one asked of a still, which has no stop: `202`, and the still ends as it would have. A stop the scope refuses is the answer, as the stop's own refusal (a home on a board that cannot stop one: `409`, `home_not_stoppable`). A call held as a job because it outlived its wait holds no operation, and `DELETE` on it while it runs is `409`, `not_cancellable`; its activity's own stop member reaches it. A parameter that takes an operation is sent its job's id, as a handle's is sent its id (`GET /api/v1/held_by_other?run=1`); a job not held is `404`, and one that is no operation, a call held as a job, is `422`, `invalid_request`. Past 256 calls running, a call is refused before it runs (`503`, `overloaded`). A job's id is likewise a string of decimal digits, counted from `1` when the server starts.

A `POST` or `PUT` that may be sent twice -- a client that lost its answer to a dropped connection cannot tell whether the call ran -- carries `Idempotency-Key: <any string the client chooses>`. Sent again with the same key, from any connection, the request runs nothing and is answered what the first one was answered: the same status and body (a started operation's same job), with `Idempotent-Replayed: true`; a retry that arrives while the first still runs waits for that answer. The same key with another member or body is refused (`422`, `idempotency_key_reused`); a body's keys in another order are the same body. A key is held for 24 hours, at most 10,000 of them, the oldest let go first; a request the server refused before it ran (`503`) holds none. A read (`GET`) runs again whatever key it is sent.

### Problems

Every answer that is not a result is an RFC 9457 problem, `application/problem+json`:

```json
{"type": "urn:lumascope:problem:axis_unknown", "title": "Not an Axis",
 "detail": "axis must be one of X, Y, Z, T; 'x' is not one.", "status": 422,
 "instance": "urn:uuid:3faa1296-...", "kind": "refusal", "reason": "axis_unknown",
 "remedy": null, "argument": "axis", "offered": ["X", "Y", "Z", "T"]}
```

`title` and `detail` are the words a Python caller gets; `reason` is the code to branch on (`type` is `urn:lumascope:problem:<reason>`, or `about:blank` for an exception from outside the scope's own); `kind` is the outcome's (`refusal`, `quiet`, `notice`, `fault`); `remedy` is the action that answers it, sent back as `POST /api/v1/apply_remedy {"remedy": ...}`; `instance` names the request. A refusal also carries the fields its type publishes, such as `argument` and `offered` for a refused argument, so a client corrects its request without reading the words. Every exception the scope raises declares its title and reason. One from outside the scope's own -- a Python error such as `ValueError` -- is titled by its exception class's name (`"ValueError"`), with `str()` of the exception as `detail`, `reason` `null`, `type` `about:blank` and `kind` `fault`. A client branches on a 500 by its status, and on `reason` only where one is given.

| Status | When |
|---|---|
| `409` | A refusal the scope's state gave (`cause` `'state'`): the same request may succeed once that changes, so it is the one to retry. Also a move or home a stop ended (`kind` `notice`, `reason` `stopped` or `superseded`): nothing failed. |
| `422` | A refusal of the request as sent (`cause` `'request'`): change it before sending it again. Also `invalid_request`, a body that does not fit the member. |
| `500` | A fault: the call failed. Also `still_running`: the scope's own bound on the call ran out, and the call may still finish. |
| `404` | `not_found`: no such route, handle, job or file. |
| `405` / `415` | `method_not_allowed`; `unsupported_media_type` (a body that is not JSON). |
| `503` | `overloaded`, with `Retry-After`; `server_closing`, while the server closes, with no `Retry-After`: it will not answer again; `no_frame_yet`, a live view the camera sent no frame for, with `Retry-After`. |

### The event stream

`GET /api/v1/events` is one `text/event-stream` for everything the scope tells its listeners. The first event is `status`, the `Status` record, with the current `id:`; each event after it has the next `id:`, its name as `event:`, and its record, wire-encoded, as `data:`.

| Event | Data |
|---|---|
| `position` | `PositionChanged`: an axis's state and position, as `axis_positions` answers |
| `led` | `LedReading` |
| `camera` | `CameraChanged`: gain or exposure |
| `run_state` | `Status`, as `status` reads at the edge; `camera_streaming` is asked just after, off the thread that made the transition, and the events keep the edges' order |
| `outcome` | `Notification`: an outcome nobody asked for, once per `outcome_id` (one a request was refused with is its answer's problem) |
| `scan_started`, `scan_ended`, `step_started`, `video_progress`, `run_ended`, `files_written` | The run event's record (`ScanStarted`, ...), with `run`, `{"job": "<id>"}`, the job of the run that sent it, as its call answered it (`null` for `run_composite`, which waits for its run and hands out no job). `run_ended` leaves out the run's own copy of its protocol. |
| `job_ended` | The job, as `GET /api/v1/jobs/<id>` reads it once it has ended: every job, an operation's, a run's (after its `run_ended`) and a call's held as a job |

Each event's data is a named schema in the description, the event's own (`PositionEvent`, `RunEndedEvent`, ...), and `status`'s is `Status`; the `/api/v1/events` route's description lists which is which. A client reads an event's keys there.

A client that reconnects with `Last-Event-ID` is sent the events it missed. When the server no longer holds them -- more than 1024 events have passed, the id is not one this server sent (it has restarted since), or a slow reader has fallen that far behind on an open stream -- it is sent `reset` (data `{}`) and then `status`, both with the current `id:`; the events in between are gone, and `status` is where the client starts again. A comment is sent every 15 s on a quiet stream, and the first event carries `retry: 1000`, how many milliseconds to wait before reconnecting.

A client that cannot hold a stream open polls instead: `GET /api/v1/events/held?after=<id>` answers `{"events": [{"id", "event", "data"}, ...], "next": <id>}`, the events held after `<id>` -- exactly what a stream resumed from that id is sent, `reset` and `status` once they are no longer held, ending with `closing` once the server closes -- and the id to ask after next. With no `after` it answers `status` and its id, where a poll begins. An event's `id` is a number here, the same number as the stream's `id:`. When the server closes, the last event is `closing` (data `{}`) and the stream ends; a stream opened after it, or resumed from `closing`'s id, is sent only `closing`, and nothing is sent after it.

### The live view

What the camera sees, for a browser or any MJPEG reader:

- `GET /api/v1/live` -- `multipart/x-mixed-replace; boundary=frame`, one `image/jpeg` part per frame, each with its `Content-Length`. Open it in a browser's `<img src=...>`.
- `GET /api/v1/live.jpg` -- one frame. With nobody watching `/live`, it waits up to 2 s for the camera's next frame; none (the camera is not streaming, say) answers `503` `no_frame_yet` with `Retry-After: 1`.

Each frame is half the camera's frame width unless `max_width` (pixels, a whole number above 0) says otherwise; it is never enlarged. Each part, and the snapshot, carries `X-Frame-Ordinal`, a count of the frames the server has received while anyone watched (it rises by one per frame and never resets, so a jump between two parts is frames this client skipped), and `X-Frame-Timestamp`, when the camera delivered it. A client is sent the newest frame whenever it is ready for one, so a slow client skips frames -- its ordinals jump -- rather than falling behind. A camera that is not connected is refused before any frame (`not_connected`). A camera removed while watched leaves the stream holding; the frames resume when it returns.

The live view is display: 8-bit grey, lossy, with no metadata, no scale bar and none of the screen's display adjustments. Data is a capture, a file in the live folder.

### Files

`GET /api/v1/files/<name>` downloads a file in the live folder by the name a listing (`GET /api/v1/live_folder_listing?name=ProtocolData`) or an answered path gives. It comes as an attachment under its own name, its type from its extension, and a `Range` request is answered with that part (`206`), so a large download resumes. A name that is no file is `404`; one that leaves the live folder is refused (`outside_live_folder`). In the URL the name is percent-decoded once, as any URL path is, and then read as a name in the live folder. A client percent-encodes each segment as a URL path segment needs (a space as `%20`, `%` as `%25`, `#` and `?` always) and keeps the `/`s between segments. An encoded `%2F` is decoded to `/` and so separates segments as `/` does, and an encoded `..` is refused as `..` is (`outside_live_folder`). In a JSON argument a name is never percent-decoded: `"a%20b"` names a file called `a%20b`.

`PUT /api/v1/files/<name>` writes the request's body as a new file in the live folder under that name (`session.put_live_folder_file`), named as for a download, the folders on its way made. It answers `201` with the file's path. A file already at the name is refused, `LiveFolderPathRefusedError` `file_exists` (`409`), so an upload never replaces a file a capture wrote; a name that leaves the live folder is `outside_live_folder`; a write the disk refuses is `LiveFolderFileNotWrittenError` (`500`), with nothing left at the name. It is a write like any other: while a control session is held, only a request carrying its `Control-Session` header writes, and any other is refused `control_session_held`.

`GET /api/v1/previews/<name>` answers a saved image in the live folder, named as for a download, downscaled to a JPEG for display: no wider than `max_width` pixels, half the image's width when not given, never wider than the image. It is shown as the live view is -- 8-bit grey, lossy, no metadata; a colour file is the mean of its channels -- and rendered from the file's own depth and white, so a summed still is as bright as it is shown. The data is the file (`/files`) and the still's record. A name that is no file is `404`, one that leaves the live folder `outside_live_folder`, and a file that is no image `InputFramesUnusableError` `unreadable`.

### Request ids

Every answer carries `X-Request-ID`, which a problem also names as its `instance`, and every request is one line of `lumaviewpro_rest_api.log` under that id, with its method, path, status and time.

### A walkthrough

Against the simulator, with `curl`:

```bash
API=http://127.0.0.1:8000/api/v1
curl -s $API/status                                   # what the scope is doing
curl -s -X POST $API/scope/motion/move_z -H 'Content-Type: application/json' \
     -d '{"z_um": 100.0}'                             # 409 position_unknown until homed
curl -s -X POST $API/scope/motion/home -H 'Prefer: wait=60'
curl -s -X POST $API/scope/motion/move_z -H 'Content-Type: application/json' \
     -d '{"z_um": 100.0}'                             # null
curl -s -X POST $API/scope/motion/move_xy -H 'Content-Type: application/json' \
     -d '{"x_um": 60000.0, "y_um": 40000.0}'          # null: X and Y as one request
curl -s -X POST $API/scope/illumination/led_on -H 'Content-Type: application/json' \
     -d '{"channel": "Blue", "illumination_ma": 50.0}'   # 50.0
curl -s -X POST $API/scope/illumination/leds_off

# One scan of a one-step protocol
curl -s -X PUT $API/channel_acquire -H 'Content-Type: application/json' -d '{"channel": "BF", "mode": "image"}'   # "image"
curl -s -X POST $API/create_empty_protocol            # {"handle":"1","type":"Protocol"}
curl -s -X POST $API/add_step -H 'Content-Type: application/json' -d '{"protocol": "1"}'
curl -s -X POST $API/create_protocol_runner           # {"handle":"2","type":"ProtocolRunner"}
curl -s -X POST $API/handles/ProtocolRunner/2/start_single_scan -H 'Content-Type: application/json' \
     -d '{"protocol": "1", "sequence_name": "walk"}'  # 202, the run's job: {"id":"1", ..., "progress":{"run_dir":...}}
curl -s -H 'Prefer: wait=60' $API/jobs/1              # once it has ended, its result: the RunOutcome
                                                      # (curl -s -X DELETE $API/jobs/1 while it runs stops it: 202)
curl -s "$API/live_folder_listing?name=ProtocolData/20261009_114756"
curl -s -O -J $API/files/ProtocolData/20261009_114756/custom0000_BF_0000.tiff
curl -s -o preview.jpg "$API/previews/ProtocolData/20261009_114756/custom0000_BF_0000.tiff?max_width=400"
curl -s -X PUT --data-binary @layout.csv $API/files/Uploads/layout.csv   # 201, its path

curl -s -N $API/events                                # the stream, until you stop it
```

---

## Common patterns

### Basic capture

```python
from modules.scope_session import ScopeSession

# create() builds the scope and brings it up from the user's settings: the
# objective, plate, offset and scale bar a capture uses are those settings.
session = ScopeSession.create(ScopeSession.load_user_settings('.'))
scope = session.scope
scope.motion.home()                        # returns once the home has established the reference

# A still is taken with its layer's stored settings, so set them there.
session.update_settings('BF.exposure_ms', 50)
session.update_settings('BF.gain_db', 5.0)

scope.motion.move_xy(x_um=60000, y_um=40000)
scope.motion.move_z(5000)

scope.illumination.led_on('BF', 100)
stills = session.manual_capture.capture(false_color_on=False).result(timeout=30)
scope.illumination.leds_off()               # stills[0].file: the still under live_folder/Manual
session.shutdown()
```

### Multi-channel composite

A composite is a run kind, not a loop you write. One step per channel set
to acquire an image, each at its own stored focus, followed by the merge --
through the same engine as every other run, so it produces a run directory
and an execution record like any scan.

```python
from modules.exceptions import CompositeFailedError

runner = session.create_protocol_runner()

# Which channels take part, their illumination, exposure, gain and blend
# thresholds all come from the settings snapshot. Set them there rather
# than passing them here, so a headless caller and a GUI click run the
# same composite.
try:
    outcome = runner.run_composite(sequence_name='composite')
    print(f'merged composite: {outcome.artifact_path}')
except CompositeFailedError as failed:
    print(f'no composite: {failed.reason}')
```

`run_composite` blocks until the merge settles and returns the run's
outcome, the merged file's path on it, so a missing artifact cannot be
mistaken for a success. A composite is merged from every channel it was
asked for or not at all: a channel that captured nothing leaves no
composite, and `run_composite` raises `CompositeFailedError` with reason
`post_processing_failed`, its message naming the channel; the run's
outcome (from `start_composite(...).result()`) is `status='incomplete'`
with the failed channels in `captures.failed`. To
launch one without waiting, call `runner.start_composite(...)`, which
returns the run's operation once the run is committed and refuses as
`run_composite` does; its `result(timeout)` gives the outcome once the
merge settles, or raises `TimeoutError` when the bound expires. With no
`parent_dir` the run lands under `Manual/Composites` in the live folder,
where the Composite button puts it.

It raises `ProtocolRunRefusedError` when the run is refused before
anything is committed -- fewer than two channels set to acquire an image,
a rival run holding the scope, or files still draining -- and
`CaptureError` when the run happened but produced no composite. The
refusal's `str()` is the sentence written for a person, as for any run.

**`CaptureError.reason` is a failure code, not a refusal.** It names what
went wrong after the run committed (`write_batch_timeout` -- the run's
images did not finish writing within the merge's bound --
`write_batch_abandoned` -- some were given up on by a writer recovery or a
shutdown -- `write_batch_not_taken` -- some never reached the file writer,
which was stuck or had stopped taking work -- `write_batch_save_failed` --
saving some failed on disk -- `write_batch_disk_full` -- some were refused
because the save drive was nearly full -- `write_batch_video_unfinished` --
a video step's file did not finish --, `merge_failed`,
`aborted`, the composite builder's own codes such as `no_data`,
`excluded_inputs` or `post_processing_incomplete`, ...) and is not a member of the refusal family: a refusal means
nothing changed, while a `CaptureError` means the run ran and did not
produce the artifact.

A composite of a run folder's files already on disk is
`session.post_processing.composite(folder)` (see "Post-processing").

### Z-stack

A z-stack is a run: it saves each slice with its record and returns the
stage to where the stack was centred.

```python
session.update_settings('zstack.range', 2000)      # µm, centred on the current Z
session.update_settings('zstack.step_size', 50)    # µm
runner = session.create_protocol_runner()
result = runner.start_zstack('BF').result(timeout=600)
print(result.status, result.captures)
```

### Well-plate scan

A scan of chosen wells is a protocol with a step added at each, from the
session's own layer settings:

```python
session.set_channel_acquire('BF', 'image')             # the layers a step is added for
protocol = session.create_empty_protocol()
for well in ('A1', 'A2', 'A3'):
    session.scope.motion.move_to_well(well)          # the well's centre on the selected plate
    session.add_step(protocol)                        # one step per acquiring layer, here
# or, without moving: session.add_step(protocol, channels=('BF',), at=PlateLocation(x_mm=14.38, y_mm=11.24, z_um=5000.0))
result = session.create_protocol_runner().start_single_scan(protocol).result(timeout=600)
print(result.status, result.captures)
```

### Headless protocol run

```python
from modules.scope_session import ScopeSession

session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)    # simulated, configured, executors running; '.' must be an LVP root. simulate=False for hardware

protocol = session.load_protocol('./my_protocol.tsv')

runner = session.create_protocol_runner()
pending = runner.start_single_scan(protocol)
result = pending.result(timeout=300)
print(result.status, result.reason, result.message)

session.shutdown()          # the factory built this scope, so shutdown() disconnects it
```

---

## Simulated mode

Use for development, CI, and unit tests without hardware.

```python
session = ScopeSession.create(settings, simulate=True)   # brought up and streaming
scope = session.scope

# All API calls work identically:
scope.illumination.led_on('Blue', 200)
scope.motion.move_z(5000)
image = scope.imaging.capture_and_wait()
```

A scope brought up through `ScopeSession.create()` is already streaming. A bare `Lumascope` you build yourself has no settings, so a capture raises `ConfigurationDefectError` (`scope_not_composed`, a `ConfigError`); build the scope through `ScopeSession.create()` instead (above), which composes and brings it up.

### Faults on the simulated scope

`scope.simulation` presents the faults a real board or camera can, so code under test meets them through the production driver path and asserts what the API answers: the same exception, reason and outcome as on hardware. A fault is armed for the next command, not the one in flight. The members are in-process only (Python, MATLAB calling Python), and each refuses on a real scope (`ArgumentRefusedError`, `needs_simulated_scope`).

Which board answers a member depends on how the model is simulated. The **fast** tier (`simulator_tier` `fast`) runs simulated boards in place of the drivers, a move and a home taking milliseconds; it is the tier a machine without the firmware runtime falls back to, with a warning in the log. The shipped settings choose the **realistic** tier, where a home and a move take the stage's time. The **firmware** and **realistic** tiers run the production EL-0940 drivers over the board firmware in simulation; a **Classic** model (the LS720's TMCM-6110; the LS620 and LS560's FX2) runs its production drivers over a simulated device on every tier. A board that cannot produce a fault refuses it (`not_injectable_on_board`, naming the board), and an argument it cannot take is refused as `argument_not_on_board`. `board` is `'motor'` or `'led'` (`fault_unknown` for a board the model lacks).

| Member | The fault | Fast | Firmware | Classic |
|---|---|---|---|---|
| `drop_replies(board, count=1, command=None)` | the next `count` replies do not come; `count=None`, every one until `clear()` | motor: by `command`'s first word, or any | either board, the next replies whatever the command (`command` and `count=None` refused) | refused (`silence`) |
| `delay_reply(board, seconds)` | the next reply comes `seconds` late; past the driver's wait it is no reply, though the board did what it was told | refused | either board | refused |
| `silence(board)`, `unsilence(board)` | no reply from now on | motor | refused (`unplug`) | TMCM-6110; FX2 (`unsilence` refused: its stream does not resume) |
| `unplug(board)`, `replug(board)` | the cable pulled and put back | motor | either board | TMCM-6110; FX2 (`replug` refused) |
| `reboot(board, booting_for_s=0.0)` | the firmware restarts; found again after the boot | refused | either board | refused |
| `fail_home(axis, how)` | a home that fails: `'switch_never_trips'`, `'stall'` or `'absent'` | refused | every `how` | TMCM-6110, `'switch_never_trips'` |
| `hold_axis(axis, at_fraction=0.5)` | the next move halts part of the way and never arrives; returns an event set once it has halted | motor | refused (the axis arrives) | refused |
| `interlock(lid_open=None, powered=None)` | the stage interlock's inputs | refused | refused | TMCM-6110 |
| `answer_model(model)` | the motor board names `model` as its own (None: no model), its axes staying the ones it was built with | motor | refused | refused |
| `remove_camera()` | the camera leaves the bus, for the session | camera | camera | FX2 |
| `refuse_camera_write(node, answer, count=1)` | the next `count` writes of `node` (`'gain'`, `'exposure'`, `'auto_exposure'`, `'frame_size'`, `'binning'`, `'pixel_format'`, `'black_level'`, `'auto_gain'`, `'auto_gain_once'`, `'auto_gain_target_brightness'`) answer `'false'`, `'none'`, `'true'` (not for `'frame_size'`, whose write answers the window it delivers) or `'raise'`; with `'removed'` False as the camera leaves the bus during the write; with `'no_answer'` `CameraNoAnswerError`, the camera still on the bus. The API sends an unanswered write once more, so `count=2` makes one go unanswered | camera | camera | refused |
| `refuse_camera_read(node, answer, count=1)` | the next `count` reads of `node` (`'gain'`, `'exposure'`, `'frame_size'`, `'binning'`, `'pixel_format'`, `'black_level'`, `'frame_rate'`, `'minimum_exposure'`, `'auto_gain'`, `'auto_exposure'`) fail: `'raise'`; `'sentinel'`, the driver's failed-read value (-1 for the gain, the exposure and the binning, nothing for the frame size; not for the other nodes, whose drivers raise); or `'removed'`, raising as the camera leaves the bus during the read | camera | camera | refused |
| `stall_stream(after_s, for_s)` | no frame for `for_s` s, starting in `after_s` s | camera | camera | refused (`silence('led')`) |
| `delay_delivery(frames)` | each frame reaches the host once `frames` more have been made after it, as through a link or driver queue that has fallen behind | camera | camera | refused |
| `refuse_stream_start(count=1)` | the next `count` stream starts are refused, the camera staying on the bus; the start raises `CameraStreamStartFailedError` (`camera_stream_start_failed`) to its caller | refused (the simulated camera is made like a Pylon body, whose driver logs a start that fails) | refused | FX2 |
| `bind_camera_usb_driver(hardware_id, service)` | Windows binds one USB identity of the FX2 (`USB\VID_04B4&PID_8613` or `USB\VID_04B4&PID_EA17`) to `service` (`'libusbK'`; None for no driver), as a Zadig run would; the camera already up keeps running, `diagnostics.get_camera_usb_bindings` reads the new binding, and the next connect checks it | refused | refused | FX2 |
| `refuse_led_write(count=1)` | the LED board's next `count` writes are refused, the board staying on the bus (an FX2 LED command is three writes); the command raises the LED board's `BoardPortFailedError` (`board_port_failed`) to its caller, and the LED stays connected | refused | refused | FX2 |

The fast LED board refuses every link fault: it has no link to fault. The fast motor board runs no firmware, so its home cannot fail; the firmware's axis reaches its target whatever is injected, so a move cannot be held there. `injected()` lists the faults injected since the last `clear()` as `InjectedFault` records (`member`, `arguments`), one-shot faults included after they are spent. `clear()` takes every fault back: drops, holds, refused writes and stalls lifted, boards plugged in and answering, hardware faults removed; a board that found its cable pulled stays disconnected until it is reconnected, as a real one does. A removed camera, an FX2 unplugged or silenced, and a reboot under way stay as they are, as on a real scope.

The simulated camera is made like a Basler body: a written setting reaches its register at once, and the `SENSOR_PIPELINE_FRAMES` frames it had started come out under the old settings. Each frame carries the chunks a Basler stamps (`FrameID`, `Timestamp` on the camera's 1 GHz clock at its exposure start, and `ExposureTime` and `Gain` from the register as it is tagged, so a stale frame's chunk already names the new value). `exposed_under(frame)` answers what a delivered frame was made under, a `SensorExposure` (`frame_id`, `started_ticks`, `exposure_us`, `gain_db`, `black_level`, `illumination_ma`, `x_um`, `y_um`, `z_um`, `z_end_um`, `depth_of_field_um`, `pixel_size_um`), so a test judges a frame by what its pixels show; it refuses a frame it holds no record for (`exposure_not_recorded`) and the FX2. The light scales with the LED current. The specimen lies on the stage: a frame shows the window centred where the stage's X and Y stand, at the current objective's image scale (the `PhysicalSizeX` every saved image carries), so a move of D um shifts the image D over that pixel size, toward lower column and row as X and Y grow, and two frames whose fields overlap on the stage overlap in their images. The sample does not repeat, a turret move or an objective change rescales it, and the simulated FX2 sensor images the same sample. With no objective selected, or on a model that does not declare its optics, it is drawn at a nominal 2 um per pixel, the scene's own scale, which no image records; a scope with no XY stage images it at 0, 0.

The two stalls a launch can ask for are `create`'s `sim_camera_stall` (which calls `stall_stream`) and `sim_file_stall` (above); the GUI's `--sim-camera-stall` and `--sim-file-stall` pass them.

---

## Protocol file format

Tab-separated file defining multi-step acquisition sequences.

```text
LumaViewPro Protocol
Version	9
Period	1.0
Duration	0.002778
Labware	96 well microplate
Capture Root

Steps
Name	X	Y	Z	Auto_Focus	Color	...
A1_BF	60000	40000	5000	False	BF	...
A1_Green	60000	40000	5000	False	Green	...
```

**Step fields:**

| Field | Type | Description |
|---|---|---|
| Name | string | Step label (e.g. `A1_BF`) |
| X, Y, Z | float | Position in µm |
| Auto_Focus | bool | Run autofocus at this step |
| Color | string | `Blue`, `Green`, `Red`, `BF`, `PC`, `DF`, `Lumi` |
| False_Color | bool | Apply false-color mapping |
| Illumination | float | LED current in mA |
| Gain | float | Camera gain in dB |
| Auto_Gain | bool | Enable auto-gain |
| Exposure | float | Exposure time in ms |
| Sum | int | Frame averaging count (≥1) |
| Objective | string | Must match `data/objectives.json` |
| Well | string | Well label (e.g. `A1`); empty for an added step |
| Tile | string | Tile label within a tiled position; empty when untiled |
| Z-Slice | int | Slice index within a z-stack; -1 when not stacked |
| Custom Step | bool | Added at a manual position rather than generated from the plate |
| Tile Group ID | int | The tiled group a step's tile belongs to; -1 when untiled |
| Z-Stack Group ID | int | The z-stack a step's slice belongs to; -1 when not stacked |
| Acquire | string | `image` or `video` |
| Video Config | JSON | A video step's duration and rate |
| Stim_Config | JSON | Per-channel stimulation settings |
| Auto_Named | bool | Whether Name is still the generated one |
| Label | string | The step's base text: a typed name, or `custom<NNNN>` for an added step; empty means the Well |
| Channel Group ID | int | The add a step came from (one value per `add_step` call; one per position in a generated protocol); post-processing builds one output per group. -1: recorded before the column (version 8 and older) |
| Turret Slot | int | The turret slot the step was made or updated at, which a run and step navigation turn to; -1 on a scope without a turret, or recorded before the column (filled at load where one slot holds the step's objective) |

A file of version 2 to 8 loads, its missing columns filled (`Channel Group ID` and `Turret Slot` -1); saving it writes version 9. A file newer than the reader is refused (`ProtocolFormatError`). `Protocol.COLUMNS[Protocol.CURRENT_VERSION]` is the canonical column list.

---

## Color channel reference

A scope's layers are the records of `scope.layer_identity.layers`, in display order, each with its `key_name`, `modality` and `display_color`. The release has seven; a scope lists the ones it has (an LS850T has no `Lumi`):

```python
[layer.key_name for layer in scope.layer_identity.layers]
# the release's seven: ['BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi']
# BF  -- brightfield (white LED)            transmitted
# PC  -- phase contrast                     transmitted
# DF  -- darkfield                          transmitted, dark background
# Blue  -- blue-EMISSION fluorescence       (stock excitation 405 nm)
# Green -- green-EMISSION fluorescence      (stock excitation 488 nm)
# Red   -- red-EMISSION fluorescence        (stock excitation 589 nm)
# Lumi  -- luminescence (all LEDs off, sensitive mode)
```

(`common_utils.ColorChannel` is gone in 4.0: a layer is a catalogue record, not an enum member.)

**A channel name is the colour you SEE, not the colour that excites it.** `Blue` is a blue-emitting dye excited by 405 nm violet light, `Green` a green-emitting dye excited by 488 nm, `Red` a red-emitting dye excited by 589 nm. The layer's `excitation_nm` field carries the excitation wavelength, and the UI marks it `Ex` for the same reason: a bare "Green 488 nm" invites reading 488 as the emission, which writes silently wrong metadata into your own data.

**Fluorescence excitation wavelengths depend on the installed filterset** — the stock filterset is 405 / 488 / 589 nm, but OEM customers may have custom filtersets at different wavelengths.

**Not every scope has every channel, and two models in the same family differ.** Always check the scope's layers (`scope.layer_identity.layers`) before using a color — an LS620 exposes `{'BF', 'Blue', 'Green', 'Red'}`, while an **LS560 exposes only `{'BF', 'Green'}`**. Phase contrast on those models is brightfield with a mechanical phase slider installed, not a separate illumination channel.

---

## API reference

The API is exactly the members below, and nothing else: each carries `@api` in the source, or is a data attribute its class names in `@api_fields` (`modules/api_surface.py`). An unmarked member stays callable from Python but is not API: it can change in any release, and it is never put on a wire. A member labelled (in-process) is API for a caller in the same process (Python, MATLAB calling Python) and is kept off the wire, because it hosts a session, takes or returns a function (a listener's registration among them: a wire client hears events on the stream), hands out an image array or a table (a wire client gets frames from the live view and data as files, steps through `num_steps` and `step(step_index)`), is a method of a class that crosses the wire as a record (a wire client holds the record's data, and a record has no address), takes a live object no wire member hands out (a wire client has no handle to send for it), or is held back by policy. `tests/guards/test_the_api_is_declared.py` holds this index equal to the marks, and every type a member hands out or takes in published. How each type crosses a wire is declared once, in `modules/wire_encoding.py`: scalars as themselves, `datetime` as ISO 8601, an enum by its value or name, a record (a dataclass, a NamedTuple, or a class `@api_fields(..., record=True)` declares one) as an object of its fields and wire properties, a path as its live-folder name and host path, a running call as a job, any other object as a handle; `tests/guards/test_every_wire_edge_has_a_form.py` fails a wire member whose type has none. A record carries every field on a wire, null where a value is unknown, never a shape that varies. In the same process a record answers `to_dict()`, its data as plain Python for a caller that needs a dict (MATLAB's `struct()` takes a dict, not a dataclass): a nested record becomes a dict and an enum its wire form, while a path, a handle and a running call stay the live objects, a NaN stays a NaN, and a key keeps its type.

### ScopeSession

Reached through `ScopeSession.create(settings, ...)`; every L2 caller starts here.

- `acceleration_limit_pct`
- `acting_in_control` (in-process)
- `add_outcome_listener` (in-process)
- `add_run_state_listener` (in-process)
- `add_settings_listener` (in-process)
- `add_shown_channel_listener` (in-process)
- `add_step`
- `apply_focus_to_channel_steps`
- `apply_channel_settings`
- `apply_remedy`
- `apply_tiling`
- `apply_zstacking`
- `app_runtime`
- `app_version`
- `assign_turret_objective`
- `bookmark`
- `bring_up_record`
- `capture_settings_snapshot` (in-process)
- `channel_colors`
- `clear_current_turret_objective`
- `clear_turret_objective`
- `confirm_objective`
- `control`
- `controls_lock` (in-process)
- `create` (in-process)
- `create_empty_protocol`
- `create_protocol_runner`
- `current_slot_objective_question`
- `delete_step`
- `diagnostic_claim` (in-process)
- `disable_metrics` (in-process)
- `discard_close_drain`
- `enable_metrics` (in-process)
- `end_control`
- `engineering_mode`
- `get_current_plate_position`
- `get_setting`
- `get_settings_snapshot`
- `get_zstack_params` (in-process)
- `go_to_bookmark`
- `go_to_step`
- `held_by_other`
- `labware`
- `last_run`
- `channel_acquire`
- `channel_auto_gain`
- `channel_auto_gain_lock`
- `light_shown_channel`
- `live_folder_listing`
- `live_folder_path` (in-process)
- `live_work` (in-process)
- `load_plugins` (in-process)
- `load_protocol`
- `load_user_settings` (in-process)
- `make_logs_zip`
- `make_support_report`
- `manual_capture`
- `manual_recording`
- `model`
- `model_at_next_start`
- `new_protocol`
- `objective_id`
- `objective_question`
- `plugin_api_level`
- `plugin_health`
- `plugin_run_in_flight` (in-process)
- `point_steps_at_turret_slot`
- `post_processing`
- `protocol_files_stalled`
- `protocol_files_stuck_write`
- `protocol_size_advisory`
- `put_live_folder_file` (in-process)
- `recover_file_writer`
- `remove_outcome_listener` (in-process)
- `remove_run_state_listener` (in-process)
- `remove_settings_listener` (in-process)
- `remove_shown_channel_listener` (in-process)
- `rename_step`
- `repair_camera_driver`
- `retire_rejected_settings`
- `run_application_session` (in-process)
- `save_all_bookmarks`
- `save_bookmark`
- `save_focus`
- `save_channel_focus`
- `save_protocol`
- `saved_focus`
- `scale_bar_enabled`
- `scope`
- `select_labware`
- `select_model`
- `select_objective`
- `set_acceleration_limit`
- `set_channel_acquire`
- `set_channel_auto_gain`
- `set_live_folder` (in-process)
- `set_protocol_filepath` (in-process)
- `set_protocol_capture_root`
- `set_protocol_labware`
- `set_protocol_schedule`
- `set_scale_bar`
- `set_step_autofocus`
- `set_ui_dispatcher` (in-process)
- `settings_are_provisional`
- `show_channel`
- `shown_channel`
- `shutdown` (in-process)
- `source_path`
- `start_application_session` (in-process)
- `start_go_to_bookmark`
- `start_go_to_step`
- `status`
- `take_back_control` (in-process)
- `take_control`
- `turret_objective`
- `unload_plugins` (in-process)
- `update_settings`
- `update_step`
- `user_settings_file` (in-process)

### Lumascope

Reached through `session.scope`.

- `camera`
- `camera_connected`
- `capabilities`
- `diagnostics`
- `illumination`
- `imaging`
- `labware`
- `layer_identity`
- `led_connected`
- `motion`
- `motion_expected`
- `motor_connected`
- `no_hardware`
- `objectives`
- `protocols`
- `runtime_state`
- `scope_models`
- `settings_template`
- `simulation` (in-process)
- `unconnected_parts`

### MotionAPI

Reached through `session.scope.motion`.

- `add_position_listener` (in-process)
- `axes_without_position`
- `axis_positions`
- `get_actual_position`
- `get_axis_limits`
- `get_axis_state`
- `get_limit_switch_status`
- `get_target_position`
- `get_target_positions`
- `get_turret_slot`
- `has_homed`
- `home`
- `interlocks`
- `is_current_turret_position_objective_set`
- `is_moving`
- `jog_step`
- `move_to_plate`
- `move_to_well`
- `move_turret`
- `move_xy`
- `move_xy_by`
- `move_z`
- `move_z_by`
- `position_is_known`
- `remove_position_listener` (in-process)
- `set_precision_mode`
- `start_home`
- `start_move_to_plate`
- `start_move_to_well`
- `start_move_xy`
- `start_move_xy_by`
- `start_move_z`
- `start_move_z_by`
- `stop_motion`
- `wait_until_finished_moving`

### IlluminationAPI

Reached through `session.scope.illumination`.

- `add_led_listener` (in-process)
- `ch2color`
- `color2ch`
- `get_led_state`
- `get_led_states`
- `led_off`
- `led_on`
- `leds_off`
- `remove_led_listener` (in-process)
- `restore_led_state`
- `save_led_state`

### CameraAPI

Reached through `session.scope.camera`.

- `active`
- `add_camera_listener` (in-process)
- `applied_auto_gain_for`
- `applied_exposure_ms_for`
- `applied_gain_db_for`
- `apply`
- `auto_exposure`
- `auto_gain`
- `binning`
- `black_level`
- `camera_removed`
- `exposure_ms`
- `frame_at_binning`
- `frame_size`
- `gain_db`
- `get_black_level_range`
- `get_pixel_alignment`
- `high_conversion_gain`
- `image_mode`
- `line_noise_reduction`
- `lock_auto_gain`
- `longest_exposure_ms`
- `pixel_format`
- `read_auto_exposure`
- `read_auto_gain`
- `read_binning`
- `read_black_level`
- `read_exposure_ms`
- `read_exposure_ms_min`
- `read_frame_rate`
- `read_frame_size`
- `read_gain_db`
- `read_pixel_format`
- `remove_camera_listener` (in-process)
- `restore_camera_state` (in-process)
- `save_camera_state` (in-process)
- `set_auto_exposure`
- `set_auto_gain`
- `set_binning`
- `set_black_level`
- `set_exposure_ms`
- `set_frame_size`
- `set_gain_db`
- `set_high_conversion_gain`
- `set_image_mode`
- `set_line_noise_reduction`
- `update_auto_gain_target_brightness`

### ImagingAPI

Reached through `session.scope.imaging`.

- `add_frame_listener` (in-process)
- `capture_and_wait` (in-process)
- `capture_frame_depth` (in-process)
- `capture_frame_full_scale` (in-process)
- `disable_streaming`
- `enable_streaming`
- `frame_is_valid`
- `frames_until_valid`
- `get_delivered_rate`
- `get_image` (in-process)
- `get_image_from_buffer` (in-process)
- `is_streaming`
- `judge_frame` (in-process)
- `last_capture_info` (in-process)
- `recorded_changes` (in-process)
- `remove_frame_listener` (in-process)
- `saturated_fraction` (in-process)
- `scale_bar_config` (in-process)
- `set_scale_bar_color`
- `significant_bits`

### DiagnosticsAPI

Reached through `session.scope.diagnostics`.

- `enter_led_engineering_mode` (in-process)
- `exit_led_engineering_mode` (in-process)
- `get_camera_link_info`
- `get_camera_temperatures_degc`
- `get_camera_usb_bindings`
- `get_led_info`
- `get_motor_info`
- `motor_fan_spin_times` (in-process)
- `read_led_currents_ma` (in-process)
- `read_motor_drv_status` (in-process)
- `read_motor_fan_rpm` (in-process)
- `run_camera_bandwidth_test` (in-process)
- `run_grab_lifecycle_benchmark` (in-process)
- `run_pylon_diagnostic_probe` (in-process)
- `set_motor_fan_duty` (in-process)

### ProtocolsAPI

Reached through `session.scope.protocols`.

- `focus_z`
- `step_targets`
- `tiling_config`
- `turret_slot_questions`

### RuntimeState

Reached through `session.scope.runtime_state`.

- `get_field_of_view`
- `get_labware`
- `get_pixel_size_um`
- `get_stage_offset`
- `get_turret_config` (in-process)
- `get_well_label`
- `is_turreted`
- `plate_to_stage`
- `plate_transform` (in-process)
- `resolve_current_objective`
- `stage_to_plate`

### SimulationAPI

Reached through `session.scope.simulation`.

- `answer_model` (in-process)
- `bind_camera_usb_driver` (in-process)
- `clear` (in-process)
- `delay_delivery` (in-process)
- `delay_reply` (in-process)
- `drop_replies` (in-process)
- `exposed_under` (in-process)
- `fail_home` (in-process)
- `hold_axis` (in-process)
- `injected` (in-process)
- `interlock` (in-process)
- `reboot` (in-process)
- `refuse_camera_read` (in-process)
- `refuse_camera_write` (in-process)
- `refuse_stream_start` (in-process)
- `refuse_led_write` (in-process)
- `remove_camera` (in-process)
- `replug` (in-process)
- `silence` (in-process)
- `stall_stream` (in-process)
- `unplug` (in-process)
- `unsilence` (in-process)

### ScopeCapabilities

Reached through `session.scope.capabilities`.

- `axes`
- `buffer_bit_depth`
- `camera_binning_sizes`
- `camera_has_clock_latch`
- `camera_limits`
- `camera_max_significant_bits`
- `camera_model`
- `camera_pixel_formats`
- `camera_reports_temperature`
- `camera_sdk_version`
- `camera_sensor`
- `camera_serial_number`
- `camera_shutter`
- `camera_supports_auto_exposure`
- `camera_supports_auto_gain`
- `camera_supports_black_level`
- `camera_supports_conversion_gain_mode`
- `camera_supports_line_noise_reduction`
- `camera_timestamp_tick_hz`
- `frame_rules`
- `has_firmware_stim`
- `has_focus`
- `has_turret`
- `has_xy_stage`
- `is_color_native`
- `led_channels`
- `led_colors`
- `led_max_ma`
- `lens_focal_length_mm`
- `model`
- `pixel_size_um`

### LayerIdentity

Reached through `session.scope.layer_identity`.

- `filterset`
- `find` (in-process)
- `layers`
- `model`
- `refuse_unless_on_scope` (in-process)
- `source`

### PostProcessingAPI

Reached through `session.post_processing`.

- `build`
- `cell_count_size_scale`
- `composite`
- `count_cells`
- `default_cell_count_method`
- `draw_graph` (in-process)
- `enhance`
- `graph`
- `graph_options`
- `load_cell_count_method`
- `preview_cell_count` (in-process)
- `projection_methods`
- `save_cell_count_method`
- `stitch`
- `stitch_modes`
- `video`
- `zproject`

### ManualCaptureController

Reached through `session.manual_capture`.

- `capture`
- `in_flight`

### ManualRecordingController

Reached through `session.manual_recording`.

- `discard_pending`
- `elapsed_s`
- `end_reason`
- `is_busy`
- `is_draining`
- `is_recording`
- `pending_writes`
- `save_folder`
- `start`
- `stop`

### ProtocolRunner

Reached through returned by `session.create_protocol_runner()`.

- `run_autofocus` (in-process)
- `run_autofocus_all_steps` (in-process)
- `run_composite` (in-process)
- `run_protocol` (in-process)
- `run_single_scan` (in-process)
- `run_zstack` (in-process)
- `start_autofocus`
- `start_autofocus_all_steps`
- `start_composite`
- `start_protocol`
- `start_single_scan`
- `start_zstack`

### Protocol

Reached through returned by `session.load_protocol` and `session.new_protocol`.

- `CURRENT_VERSION`
- `PROTOCOL_FILE_HEADER`
- `capture_prefix`
- `capture_root`
- `duration_s`
- `estimate_write_mb`
- `labware`
- `layer_settings`
- `num_scans`
- `num_steps`
- `period_s`
- `size_advisory`
- `step`
- `step_list_revision`
- `step_records`
- `step_row` (in-process)
- `steps` (in-process)
- `tiling`
- `validate_steps` (in-process)

### RunEvents

Reached through built by the caller and passed as `events=` to a `ProtocolRunner` run member.

- `files_written`
- `frame_captured`
- `on_ui_thread`
- `run_ended`
- `scan_ended`
- `scan_started`
- `step_started`
- `video_progress`

### Operation

Reached through returned by a `ProtocolRunner` run member that started a run; a run's `Operation.files`; returned by `scope.motion.start_move_xy`, `start_move_xy_by`, `start_move_z`, `start_move_z_by` and `start_home`, `session.start_go_to_step`, `session.start_application_session`, `session.manual_capture.capture` and `session.manual_recording.start`. A `concurrent.futures.Future`: its `result`, `exception`, `done` and `add_done_callback` are the Future's.

- `files`
- `progress`
- `stop`

### RunProgress

Reached through returned by `Operation.progress` for a run.

- `is_last_run`
- `is_live`
- `is_stopping`
- `num_steps`
- `period_s`
- `run_dir`
- `scans_remaining`
- `step_index`

### RunOutcome

Reached through a run's `Operation.result` and returned by `ProtocolRunner.run_composite`; delivered to `RunEvents.run_ended`'s handler.

- `af_data_path`
- `af_data_saved`
- `af_focus_z_um`
- `artifact_path`
- `captures`
- `cleanup_failures`
- `focus_failures`
- `focus_written`
- `merge_reason`
- `merged`
- `message`
- `reason`
- `status`
- `title`

### AppliedCameraSetting

Reached through returned by `scope.imaging`'s `applied_*_for` members.

- `applied`
- `capped`
- `stored`

### ArgumentRefusedError

Reached through a REST problem, whose members carry it.

- `argument`
- `limits`
- `member`
- `offered`

### ArgumentTypeRefusedError

Reached through a REST problem, whose members carry it.

- `argument`
- `declared`
- `member`

### AutoGainLock

Reached through returned by `session.set_channel_auto_gain` and `scope.camera.lock_auto_gain`, and answered by `session.channel_auto_gain_lock`.

- `ceiling_ms`
- `exposure_ms`
- `floor_ms`
- `gain_db`
- `resume_after_capture`
- `settings`
- `state`
- `stored_exposure_ms`

### AxisLimits

Reached through returned by `session.scope.motion.get_axis_limits`.

- `max_um`
- `min_um`

### AxisPosition

Reached through each value of `scope.motion.axis_positions()`.

- `position`
- `state`

### BringUpRecord

Reached through returned by `session.bring_up_record`.

- `missing`
- `part` (in-process)
- `parts`
- `settings_set_aside`
- `substitution` (in-process)
- `substitutions`

### BuildResult

Reached through returned by `session.post_processing.stitch`, `zproject`, `composite` and `video`.

- `artifact_paths`
- `degraded_outputs`
- `excluded_inputs`
- `message`
- `new_count`
- `output_root`

### BuildStep

Reached through each item of `BuildsResult.built`.

- `operation`
- `result`

### BuildsResult

Reached through returned by `session.post_processing.build`.

- `built`
- `message`
- `skipped`

### CameraLink

Reached through returned by `scope.diagnostics.get_camera_link_info`.

- `inter_packet_delay`
- `link_speed`
- `link_speed_unit`
- `packet_size_bytes`
- `transport`

### CameraChanged

Reached through the event `scope.camera.add_camera_listener` delivers.

- `setting`
- `value`

### CameraLimits

Reached through `session.scope.capabilities.camera_limits`.

- `analog_gain_db_max`
- `exposure_ms_max`
- `exposure_ms_min`
- `frame_max`
- `frame_min`
- `gain_db_max`
- `gain_db_min`

### CaptureFailedError

Reached through a REST problem, whose members carry it.

- `setting`

### CaptureTally

Reached through `RunOutcome.captures`.

- `asked`
- `captured`
- `dark`
- `failed`
- `missing`
- `saturated`

### CatalogueNameRefusedError

Reached through a REST problem, whose members carry it.

- `argument`
- `offered`

### CellCountFilters

Reached through a `CellCountMethod`'s `filters`.

- `area`
- `intensity_max`
- `intensity_mean`
- `intensity_min`
- `perimeter`
- `sphericity`

### CellCountMethod

Reached through `session.post_processing.default_cell_count_method()` and `load_cell_count_method`, or built by the caller; taken by `count_cells` and `save_cell_count_method`.

- `filters`
- `fluorescent_mode`
- `pixels_per_um`
- `threshold`

### CellCountPreview

Reached through returned by `session.post_processing.preview_cell_count`.

- `counts`
- `image`

### CellCountResult

Reached through returned by `session.post_processing.count_cells`.

- `counted`
- `message`
- `results_path`

### CellCounts

Reached through a `CellCountPreview`'s `counts`.

- `area_unit`
- `num_cells`
- `total_object_area`
- `total_object_intensity`

### CellCountSizeScale

Reached through returned by `session.post_processing.cell_count_size_scale`.

- `area`
- `perimeter`
- `pixels_per_um`

### Change

Reached through each `Judgment`'s `change`, from `scope.imaging.judge_frame`.

- `at_seq`
- `credits`
- `earliest_ns`
- `in_force_ns`
- `needs_in_force`
- `rule`
- `skip`
- `source`
- `written_at`
- `written_ns`

### ClockMapping

Reached through `scope.imaging.judge_frame`'s `mapping`, and a capture's `frame_record.clock_mapping`.

- `camera_ticks`
- `host_ns`
- `tick_hz`
- `uncertainty_ns`

### ControlSession

Reached through `session.control`.

- `name`
- `taken_at`
- `taken_by`

### ControlSessionRefusedError

Reached through a REST problem, whose members carry it.

- `control_session`
- `holder`
- `holder_trigger`

### ControlsLock

Reached through `Status.lock`.

- `control_named`
- `controls_locked`
- `named`
- `recording`
- `run_lockout`
- `run_named`

### Credit

Reached through each value of a `Change`'s `credits`.

- `by`
- `seq`

### DegradedOutput

Reached through each item of `BuildResult.degraded_outputs`.

- `algorithm`
- `fallback_from`
- `fallback_reason`
- `path`

### DeliveredRate

Reached through returned by `scope.imaging.get_delivered_rate`.

- `bytes_per_s`
- `frames_per_s`

### DiagnosticRefusedError

Reached through a REST problem, whose members carry it.

- `control_session`
- `holder`
- `holder_trigger`

### DisplayRates

Implemented by a host with a live display, and passed to `session.enable_metrics(display=...)`.

- `display_fps` (in-process)
- `frame_interval_percentiles_ms` (in-process)

### EnhanceResult

Reached through returned by `session.post_processing.enhance`.

- `created`
- `message`
- `output_folder`

### EnhancedFile

Reached through each item of `EnhanceResult.created`.

- `output_path`
- `recipe_path`
- `source_path`

### ExcludedInput

Reached through each item of `BuildResult.excluded_inputs` and `PostProcessingFailedError.excluded_inputs`.

- `count`
- `kind`

### FailedCapture

Reached through each item of `CaptureTally.failed`.

- `cause`
- `scan_index`
- `step_index`
- `step_name`

### FieldOfView

Reached through `RuntimeState.get_field_of_view`.

- `height_um`
- `width_um`

### FilterBounds

Reached through each filter of a `CellCountFilters`.

- `max`
- `min`

### FilesWritten

Reached through the event `RunEvents.files_written` delivers.

- `files`
- `run_dir`

### FocusFailure

Reached through each item of `RunOutcome.focus_failures`.

- `cause`
- `scan_index`
- `step_index`
- `step_name`

### FrameCaptured

Reached through the event `RunEvents.frame_captured` delivers.

- `frame_significant_bits`
- `frames_summed`
- `image`

### FrameDelivered

Reached through the event `scope.imaging.add_frame_listener` delivers.

- `frame`

### FrameRecord

Reached through each `SavedStill`'s `frame`, from `session.manual_capture.capture`.

- `binning_size`
- `black_level`
- `camera_model`
- `camera_tick_hz`
- `camera_timestamp_ticks`
- `captured_at`
- `clock_mapping`
- `dark`
- `exposure_ms`
- `frame_id`
- `frame_significant_bits`
- `frames_summed`
- `gain_db`
- `held_layer`
- `illumination_ma`
- `judgments`
- `saturated_fraction`

### FrameRule

Reached through each value of `ScopeCapabilities.frame_rules`, and a `Change`'s `rule`.

- `delay_us`
- `provenance`
- `rule`

### FrameSize

Reached through `CameraLimits.frame_min` and `CameraLimits.frame_max`.

- `height`
- `width`

### GraphOptions

Reached through returned by `session.post_processing.graph_options`.

- `trendlines`
- `x_axes`
- `y_axes`

### GraphResult

Reached through returned by `session.post_processing.graph`.

- `message`
- `path`

### HardwareCommandRefusedError

Reached through a REST problem, whose members carry it.

- `control_session`
- `holder`
- `holder_trigger`

### HeldClaim

Reached through returned by `session.diagnostic_claim`.

- `holds`

### ImageStatistics

Reached through each `SavedStill`'s `statistics`, from `session.manual_capture.capture`.

- `maximum`
- `mean`
- `median`
- `minimum`
- `percentile_1`
- `percentile_99`
- `standard_deviation`

### InjectedFault

Reached through returned by `scope.simulation.injected`.

- `arguments`
- `member`

### Judgment

Reached through each value of `scope.imaging.judge_frame(frame)`.

- `change`
- `counts`
- `frame_seq`
- `frame_start_ticks`
- `frames_after`
- `ms_after_in_force`
- `ms_after_write`
- `qualifies_after_ticks`

### LayerRecord

Reached through returned by `scope.layer_identity.find` and `layers`.

- `background`
- `display_color`
- `display_name`
- `excitation_nm`
- `id`
- `key_name`
- `label`
- `led_channel`
- `modality`

### LedBoardInfo

Reached through returned by `scope.diagnostics.get_led_info`.

- `command_set`
- `firmware_date`
- `firmware_version`

### LedReading

Reached through returned by `scope.illumination.get_led_state` and `get_led_states`, and the event `scope.illumination.add_led_listener` delivers.

- `address`
- `illumination_ma`
- `name`
- `state`

### LedSnapshot

Reached through returned by `scope.illumination.save_led_state`; passed to `restore_led_state`.

- `readings`
- `tag`

### LiveFolderEntry

Reached through each item returned by `session.live_folder_listing`.

- `kind`
- `modified`
- `name`
- `size`

### LiveWork

Reached through `Status.live_work`; in-process also `session.live_work`.

- `closed`
- `closing`
- `work`

### MotorBoardInfo

Reached through returned by `scope.diagnostics.get_motor_info`.

- `command_set`
- `firmware_date`
- `firmware_version`
- `model`
- `serial_number`

### NamespaceHealth

Reached through each item of `PluginHealth.namespaces`.

- `last_runtime_errors`
- `loaded`
- `namespace`

### Notification

Reached through delivered to `session.add_outcome_listener`'s callback.

- `category`
- `fatal`
- `kind`
- `message`
- `operation_key`
- `outcome_id`
- `reason`
- `remedy`
- `severity`
- `shown`
- `solicited`
- `source`
- `timestamp`
- `title`
- `wall_time`

### Objective

Reached through `session.scope.objectives`, and returned by `session.scope.runtime_state.resolve_current_objective`.

- `autofocus_half_range_um`
- `autofocus_step_max_um`
- `autofocus_step_min_um`
- `depth_of_field_um`
- `description`
- `focal_length_mm`
- `focus_peak_fwhm_dense_um`
- `id`
- `jog_xy_coarse_um`
- `jog_xy_fine_um`
- `jog_z_coarse_um`
- `jog_z_fine_um`
- `magnification`
- `numerical_aperture`
- `short_name`
- `working_distance_mm`
- `z_settle_timeout_ms`
- `z_settle_tolerance_um`

### ObjectiveQuestion

Reached through returned by `session.objective_question`.

- `choices`
- `proposed`
- `turret_position`

### PartStatus

Reached through returned by `BringUpRecord.part`, `missing` and `parts`.

- `cause`
- `describe` (in-process)
- `detail`
- `expected`
- `missing`
- `part`
- `up`

### PluginHealth

Reached through `session.plugin_health`.

- `namespaces`
- `not_loaded`

### PlateLocation

Reached through returned by `session.get_current_plate_position` and `session.bookmark`; taken by `scope.motion.start_move_to_plate` and `move_to_plate`, and by `ProtocolRunner.start_single_scan`'s and `run_single_scan`'s `return_to_position`.

- `x_mm`
- `y_mm`
- `z_um`

### PlatePosition

Reached through returned by `session.scope.runtime_state.stage_to_plate`.

- `x_mm`
- `y_mm`

### PluginNotLoaded

Reached through `PluginHealth.not_loaded`.

- `name`
- `reason`
- `version`

### PluginRuntimeError

Reached through `NamespaceHealth.last_runtime_errors`.

- `exc_type`
- `hook`
- `message`
- `namespace`
- `plugin_name`

### PluginStatus

Reached through `NamespaceHealth.loaded`.

- `name`
- `namespace`
- `version`

### PositionChanged

Reached through the event `scope.motion.add_position_listener` delivers.

- `at`
- `axis`

### PostProcessingFailedError

Reached through a REST problem, whose members carry it.

- `excluded_inputs`

### ProtocolRunRefusedError

Reached through a REST problem, whose members carry it.

- `control_session`
- `holder`
- `holder_trigger`

### ProtocolSizeAdvisory

Reached through returned by `session.protocol_size_advisory` and `Protocol.size_advisory`.

- `message`
- `num_steps`
- `projected_mb`

### RecordingRefusedError

Reached through a REST problem, whose members carry it.

- `control_session`
- `holder`
- `holder_trigger`

### Remedy

Reached through `Notification.remedy`; passed to `session.apply_remedy`.

- `cancel_text`
- `confirm_text`
- `member`

### RunEnded

Reached through the event `RunEvents.run_ended` delivers.

- `outcome`
- `protocol`
- `run_dir`

### RunFiles

Reached through a run's `Operation.files` result.

- `not_written`
- `not_written_reason`
- `outcome`
- `written`

### SavedFocus

Reached through returned by `session.save_focus`.

- `step_index`
- `z`

### SavedStill

Reached through each value of `session.manual_capture.capture`'s `Operation` result.

- `channel`
- `file`
- `frame`
- `objective_id`
- `output_format`
- `plate_x_mm`
- `plate_y_mm`
- `significant_bits`
- `stage_z_um`
- `statistics`
- `well_label`

### ScanEnded

Reached through the event `RunEvents.scan_ended` delivers.

- `period_s`
- `scan_index`
- `scans_remaining`

### ScanStarted

Reached through the event `RunEvents.scan_started` delivers.

- `period_s`
- `scan_index`
- `scans_remaining`

### ScopeModel

Reached through `session.scope.scope_models`.

- `filterset`
- `has_focus`
- `has_turret`
- `has_xy_stage`
- `id`
- `layers`
- `led_board`
- `lens_focal_length_mm`
- `max_frame`
- `motor_board`
- `pixel_size_um`

### SettingRefusedError

Reached through a REST problem, whose members carry it.

- `member`
- `path`

### SettingsChanged

The event `session.add_settings_listener` delivers.

- `paths`

### SettingsSetAside

Reached through `BringUpRecord.settings_set_aside`.

- `path`
- `reason`

### ShownChannel

Reached through `session.shown_channel`, returned by `session.show_channel` and `light_shown_channel`, and the event `session.add_shown_channel_listener` delivers.

- `layer`
- `lit`

### SimulatedModelAnswer

Reached through built by the caller and passed as `sim_model_answer=` to `ScopeSession.create`.

- `model`

### SizeSliderStops

Reached through a `CellCountSizeScale`'s `area` and `perimeter`.

- `positions`
- `positions_of` (in-process)
- `reach`
- `size_at` (in-process)
- `sizes`

### SkippedBuild

Reached through each item of `BuildsResult.skipped`.

- `message`
- `operation`

### StageOffset

Reached through returned by `session.scope.runtime_state.get_stage_offset`.

- `x_um`
- `y_um`

### StagePosition

Reached through returned by `session.scope.runtime_state.plate_to_stage`.

- `x_um`
- `y_um`

### Status

Reached through `session.status`.

- `axes`
- `camera_streaming`
- `control`
- `live_work`
- `lock`
- `parts`
- `run_refused`

### StepRecord

Reached through returned by `Protocol.step` and each item of `Protocol.step_records`.

- `acquire`
- `auto_focus`
- `auto_gain`
- `auto_named`
- `channel_group_id`
- `color`
- `custom_step`
- `exposure`
- `false_color`
- `gain`
- `illumination`
- `label`
- `name`
- `objective`
- `stim_config`
- `sum`
- `tile`
- `tile_group_id`
- `turret_slot`
- `video_config`
- `well`
- `x`
- `y`
- `z`
- `z_slice`
- `z_stack_group_id`

### StepStarted

Reached through the event `RunEvents.step_started` delivers.

- `step_index`

### StepTargets

Reached through returned by `scope.protocols.step_targets`.

- `turret_slot`
- `x`
- `y`
- `z`

### Stimulation

Reached through each value of `StepRecord.stim_config`.

- `enabled`
- `frequency_hz`
- `illumination_ma`
- `pulse_count`
- `pulse_width_ms`

### Stitch

Taken by `session.post_processing.build`.

- `mode`

### TakenCapture

Reached through each item of `CaptureTally.saturated` and `CaptureTally.dark`.

- `blown`
- `dark`
- `scan_index`
- `step_index`
- `step_name`

### TurretSlotQuestion

Reached through returned by `scope.protocols.turret_slot_questions`.

- `choices`
- `objective_id`
- `reassigned_step_indexes`
- `step_indexes`

### Substitution

Reached through returned by `BringUpRecord.substitution` and `substitutions`.

- `saved`
- `setting`
- `used`

### SupportReportSaved

Reached through returned by `session.make_support_report` and `session.make_logs_zip`.

- `message`
- `path`
- `report`
- `title`

### TilingConfig

Reached through returned by `scope.protocols.tiling_config`.

- `available`
- `available_configs` (in-process)
- `default`
- `default_config` (in-process)

### UiDispatcher

Reached through built by a GUI host and passed to `ScopeSession.set_ui_dispatcher`.

- `schedule`
- `thread`

### UsbDeviceBinding

Reached through `UsbIdentityBindings.devices`.

- `date`
- `inf`
- `instance_id`
- `present`
- `problem_code`
- `provider`
- `service`
- `version`

### UsbIdentityBindings

Reached through returned by `session.scope.diagnostics.get_camera_usb_bindings`.

- `devices`
- `hardware_id`
- `name`
- `needs`
- `usable`

### Video

Taken by `session.post_processing.build`.

- `frames_per_sec`
- `timestamp_overlay`

### VideoConfig

Reached through `StepRecord.video_config`.

- `duration_s`
- `fps`

### VideoProgress

Reached through delivered to `RunEvents.video_progress`'s handler.

- `elapsed_s`
- `percent`
- `phase`
- `total_s`

### WellPlate

Reached through `session.scope.labware`, and returned by `session.scope.runtime_state.get_labware`.

- `columns`
- `get_dimensions` (in-process)
- `get_well_index` (in-process)
- `get_well_position` (in-process)
- `has_wells` (in-process)
- `height_mm`
- `name`
- `offset_x_mm`
- `offset_y_mm`
- `rows`
- `spacing_x_mm`
- `spacing_y_mm`
- `standard`
- `well_diameter_mm`
- `width_mm`

### WorkItem

Reached through `LiveWork.work`.

- `discardable`
- `kind`
- `left`
- `name`
- `percent`

### ZProject

Taken by `session.post_processing.build`.

- `method`

---

## Changelog

Post-freeze record of every change to the L2-callable surface (methods documented above). Entries land in the SAME commit as the underlying change. Pre-`4.0.0` (the freeze trigger) is fluid by design and not recorded here; consult `LVP_4.0.0_CHANGELOG.md` for pre-freeze prose history.

Entry format:

| Version | Date | Type | Method / surface | Change |
|---|---|---|---|---|

- **Type**: `additive` (new optional param / new return-dict key / new method) — no version bump beyond patch.
- **Type**: `behavior-change` — requires minor version bump.
- **Type**: `rename` / `removal` — requires deprecation cycle (old name retained with `FutureWarning`, retired in next major) OR a major version bump.

(No entries yet -- 4.0.0 has not shipped. Pre-freeze structural changes do not appear here.)
