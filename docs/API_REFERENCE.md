# LumaViewPro API reference

<!-- Generated from the code; do not edit. Regenerate: LVP_WRITE_API_REFERENCE=1 python3 -m pytest tests/guards/test_the_api_is_declared.py -k reference -->

Every member and field of the API, rendered from the source: a member is API when it carries `@api`, a field when its class names it in `@api_fields` (`modules/api_surface.py`). Anything else is not API, whatever its name, and can change in any release. A member marked *in-process* is API for a caller in the same process (Python, MATLAB calling Python) and is kept off the wire. How the API fits together, and how to use it, is `LumascopeSkills.md`.

A member's **Raises** is its contract: each outcome class it raises, with the reasons it quotes. **Refusals** adds the refusals of the doors the member enters (a hardware lane, a guarded write); for a member on the wire they are **Refusals by REST status**, under the HTTP status each answers. Every outcome class, with each reason's cause and status, is under [Outcomes](#outcomes).

## Contents

- [`ScopeSession`](#ScopeSession)
- [`ManualCaptureController`](#ManualCaptureController)
- [`ManualRecordingController`](#ManualRecordingController)
- [`PostProcessingAPI`](#PostProcessingAPI)
- [`Lumascope`](#Lumascope)
- [`CameraAPI`](#CameraAPI)
- [`DiagnosticsAPI`](#DiagnosticsAPI)
- [`IlluminationAPI`](#IlluminationAPI)
- [`ImagingAPI`](#ImagingAPI)
- [`MotionAPI`](#MotionAPI)
- [`RuntimeState`](#RuntimeState)
- [`SimulationAPI`](#SimulationAPI)
- [`AppliedAutoGain`](#AppliedAutoGain)
- [`AppliedExposure`](#AppliedExposure)
- [`AppliedGain`](#AppliedGain)
- [`AutoGainLock`](#AutoGainLock)
- [`AxisLimits`](#AxisLimits)
- [`AxisPosition`](#AxisPosition)
- [`BringUpRecord`](#BringUpRecord)
- [`BuildResult`](#BuildResult)
- [`BuildStep`](#BuildStep)
- [`BuildsResult`](#BuildsResult)
- [`CameraLimits`](#CameraLimits)
- [`CameraLink`](#CameraLink)
- [`CaptureTally`](#CaptureTally)
- [`CapturedStill`](#CapturedStill)
- [`CellCountFilters`](#CellCountFilters)
- [`CellCountMethod`](#CellCountMethod)
- [`CellCountPreview`](#CellCountPreview)
- [`CellCountResult`](#CellCountResult)
- [`CellCountSizeScale`](#CellCountSizeScale)
- [`CellCounts`](#CellCounts)
- [`Change`](#Change)
- [`ChannelIdentity`](#ChannelIdentity)
- [`ChannelRecord`](#ChannelRecord)
- [`ChannelSettings`](#ChannelSettings)
- [`ClockMapping`](#ClockMapping)
- [`ControlSession`](#ControlSession)
- [`ControlsLock`](#ControlsLock)
- [`Credit`](#Credit)
- [`DegradedOutput`](#DegradedOutput)
- [`DeliveredRate`](#DeliveredRate)
- [`DisplayRates`](#DisplayRates)
- [`EnhanceResult`](#EnhanceResult)
- [`EnhancedFile`](#EnhancedFile)
- [`ExcludedInput`](#ExcludedInput)
- [`ExposureChanged`](#ExposureChanged)
- [`FailedCapture`](#FailedCapture)
- [`FieldOfView`](#FieldOfView)
- [`FilesWritten`](#FilesWritten)
- [`FilterBounds`](#FilterBounds)
- [`FocusFailure`](#FocusFailure)
- [`Frame`](#Frame)
- [`FrameCaptured`](#FrameCaptured)
- [`FrameDelivered`](#FrameDelivered)
- [`FrameRecord`](#FrameRecord)
- [`FrameRule`](#FrameRule)
- [`FrameSize`](#FrameSize)
- [`FrameTime`](#FrameTime)
- [`GainChanged`](#GainChanged)
- [`GraphOptions`](#GraphOptions)
- [`GraphResult`](#GraphResult)
- [`HeldClaim`](#HeldClaim)
- [`ImageStatistics`](#ImageStatistics)
- [`InjectedFault`](#InjectedFault)
- [`Judgment`](#Judgment)
- [`Labware`](#Labware)
- [`LedBoardInfo`](#LedBoardInfo)
- [`LedReading`](#LedReading)
- [`LedSnapshot`](#LedSnapshot)
- [`LiveFolderEntry`](#LiveFolderEntry)
- [`LiveWork`](#LiveWork)
- [`MotorBoardInfo`](#MotorBoardInfo)
- [`NamespaceHealth`](#NamespaceHealth)
- [`Notification`](#Notification)
- [`Objective`](#Objective)
- [`ObjectiveQuestion`](#ObjectiveQuestion)
- [`Operation`](#Operation)
- [`Outcome`](#Outcome)
- [`PartStatus`](#PartStatus)
- [`PlateLocation`](#PlateLocation)
- [`PlatePosition`](#PlatePosition)
- [`PluginHealth`](#PluginHealth)
- [`PluginNotLoaded`](#PluginNotLoaded)
- [`PluginRuntimeError`](#PluginRuntimeError)
- [`PluginStatus`](#PluginStatus)
- [`PositionChanged`](#PositionChanged)
- [`Protocol`](#Protocol)
- [`ProtocolRunner`](#ProtocolRunner)
- [`ProtocolSizeAdvisory`](#ProtocolSizeAdvisory)
- [`Remedy`](#Remedy)
- [`RunEnded`](#RunEnded)
- [`RunEvents`](#RunEvents)
- [`RunFiles`](#RunFiles)
- [`RunOutcome`](#RunOutcome)
- [`RunProgress`](#RunProgress)
- [`SavedFocus`](#SavedFocus)
- [`SavedStill`](#SavedStill)
- [`ScanEnded`](#ScanEnded)
- [`ScanStarted`](#ScanStarted)
- [`ScopeCapabilities`](#ScopeCapabilities)
- [`ScopeModel`](#ScopeModel)
- [`SensorExposure`](#SensorExposure)
- [`SettingsChanged`](#SettingsChanged)
- [`SettingsSetAside`](#SettingsSetAside)
- [`ShownChannel`](#ShownChannel)
- [`SimulatedModelAnswer`](#SimulatedModelAnswer)
- [`SimulatedStall`](#SimulatedStall)
- [`SizeSliderStops`](#SizeSliderStops)
- [`SkippedBuild`](#SkippedBuild)
- [`StageOffset`](#StageOffset)
- [`StagePosition`](#StagePosition)
- [`Status`](#Status)
- [`StepRecord`](#StepRecord)
- [`StepStarted`](#StepStarted)
- [`Stimulation`](#Stimulation)
- [`Stitch`](#Stitch)
- [`Substitution`](#Substitution)
- [`SupportReportSaved`](#SupportReportSaved)
- [`TakenCapture`](#TakenCapture)
- [`TilingConfig`](#TilingConfig)
- [`TurretChanged`](#TurretChanged)
- [`TurretSlotQuestion`](#TurretSlotQuestion)
- [`TurretStatus`](#TurretStatus)
- [`UiDispatcher`](#UiDispatcher)
- [`UnfitArgument`](#UnfitArgument)
- [`UsbDeviceBinding`](#UsbDeviceBinding)
- [`UsbIdentityBindings`](#UsbIdentityBindings)
- [`Video`](#Video)
- [`VideoConfig`](#VideoConfig)
- [`VideoProgress`](#VideoProgress)
- [`WorkItem`](#WorkItem)
- [`ZProject`](#ZProject)
- [`modules.notification_center`](#modules.notification_center)
- [`rest.app`](#rest.app)
- [Closed sets](#closed-sets)
- [Outcomes](#outcomes)
- [Routes](#routes)
- [Events](#events)
- [Saved-file facts](#saved-file-facts)

<a id="ScopeSession"></a>
## `ScopeSession`

Made by [`ScopeSession.create`](#ScopeSession.create); every member below is on the wire under `/api/v1/` unless it is in-process.  
Taken by: [`rest.app.build_app`](#rest.app.build_app).

Owns the shared, GUI-independent state for one microscope session.

**Fields**

- <a id="ScopeSession.engineering_mode"></a>`engineering_mode`: `bool` -- (no description)
- <a id="ScopeSession.manual_capture"></a>`manual_capture`: `ManualCaptureController` -- (no description)
- <a id="ScopeSession.manual_recording"></a>`manual_recording`: `ManualRecordingController` -- (no description)
- <a id="ScopeSession.post_processing"></a>`post_processing`: `PostProcessingAPI` -- (no description)
- <a id="ScopeSession.scope"></a>`scope`: `Lumascope` -- (no description)

<a id="ScopeSession.acceleration_limit_pct"></a>
### `ScopeSession.acceleration_limit_pct`

```python
acceleration_limit_pct: int  # property
```

Route: `GET /api/v1/acceleration_limit_pct`.

The motors' acceleration limit, as a percent of the firmware's maximum.

The one `set_acceleration_limit` stored, which bring-up gives the
motor controller.

**Raises**

Nothing.

<a id="ScopeSession.acting_in_control"></a>
### `ScopeSession.acting_in_control`

```python
def acting_in_control(self, session_id: str) -> Iterator[ControlSession]
```

Marks: in-process.

Act in the control session named by its id for the length of a `with` block.

Every call the block makes on this thread is the session's, and so
is the lane work it hands on. A callback the API runs on a thread of
its own -- an operation's done-callback -- runs outside the block,
and is refused as any non-holder's call while the session is held.

**Yields**

The session.

**Raises**

- `ControlSessionRefusedError`: `'control_session_ended'`, no session is held, or the live one has another id. The block did not run.

**Refusals**

- [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_ended`

<a id="ScopeSession.add_outcome_listener"></a>
### `ScopeSession.add_outcome_listener`

```python
def add_outcome_listener(self, listener: Callable[[Notification], None]) -> None
```

Marks: in-process; event: [`Notification`](#Notification).

Hear every outcome the scope reports from now on.

`listener(notification)` is called once per delivery, on the thread
that reported it, so it must return promptly and must not wait on the
scope. Each `Notification` carries its `kind`, words, `reason`,
`remedy`, `outcome_id` and `shown`: an outcome muted when it
happened (an unattended run, the dedup window, shutdown) arrives with
`shown` False, and arrives again with the same `outcome_id` and
`shown` True if it is later shown.

Bring-up has happened by the time a session exists; pass the listener
to `create(outcome_listener=...)` to hear it. The outcomes heard are
every session's in this process, as the notification centre is one
per process. `shutdown()` removes the listener.

**Raises**

Nothing.

<a id="ScopeSession.add_run_state_listener"></a>
### `ScopeSession.add_run_state_listener`

```python
def add_run_state_listener(self, listener: Callable[[], None]) -> None
```

Marks: in-process; event: [`Status`](#Status).

Register a run-state transition listener and level-sync it.

The immediate call is the level republish: transitions are
edges, and a listener registered after a grant would otherwise
never see it. A wire client is sent the session's `status` at
each edge, in the order the edges happen: every field read at the
edge but the camera's, which is asked just after
(`status_at_edge`); a field that changes with no edge (an axis,
a part, the camera) is as of that edge.

**Raises**

Nothing.

<a id="ScopeSession.add_settings_listener"></a>
### `ScopeSession.add_settings_listener`

```python
def add_settings_listener(self, listener: Callable[[tuple[str, ...]], None]) -> None
```

Marks: in-process; event: [`SettingsChanged`](#SettingsChanged).

Tell `listener` `(paths)` each time a write stores settings, once it has let go of the store.

The paths one hold stored, together: a member that writes several
settings tells them once. Whoever wrote them -- the GUI, a client, a
script, the camera storing what it applied. Called on the writer's
thread, outside the settings lock, so a GUI listener schedules its
drawing.

**Raises**

Nothing.

<a id="ScopeSession.add_shown_channel_listener"></a>
### `ScopeSession.add_shown_channel_listener`

```python
def add_shown_channel_listener(self, listener: Callable[[str, bool], None]) -> None
```

Marks: in-process; event: [`ShownChannel`](#ShownChannel).

Tell `listener` `(channel, lit)` each time the shown channel changes.

It is called on the thread that changed it, so a GUI listener
schedules its drawing.

**Raises**

Nothing.

<a id="ScopeSession.add_step"></a>
### `ScopeSession.add_step`

```python
def add_step(
    self,
    protocol: Protocol,
    *,
    channels: Iterable[str] | None = None,
    at: PlateLocation | None = None,
    before_step_index: int | None = None,
    after_step_index: int | None = None,
    use_zstacking: bool = False,
) -> list[str]
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/add_step`.

Add a step to `protocol` from this session's settings and live position.

The entry point a caller with no GUI uses to do what Add Step
does: one step per layer whose `acquire` is set, at the live
stage position on the protocol's plate, with the current objective,
in the settings' channel order. The protocols API performs the add
and refuses when nothing would be added; this composes its inputs
from the session the same way the GUI's handler does.

`channels` names the channels to add a step for, in that order,
instead: each takes its stored settings, and one whose `acquire`
is None is added as an image step, so a caller adds without
changing the settings. `at` is where on the protocol's plate the
steps are, instead of the live position: X and Y in mm, Z in um.

`use_zstacking` makes each added step a z-stack of the stored
z-stack settings, as `new_protocol` does for its steps; the
protocol's other steps are left as they are.

Returns the inserted step names, in protocol order: each stack's
slices when stacked.

**Raises**

- `AxisMovingError`: `'axis_moving'`, with no `at`, an axis is moving, so the live position is a place it is passing through. Nothing is added.
- `ProtocolRunRefusedError`: `'step_position_unknown'`, an axis does not know its position; `'no_acquiring_layer'`, no layer acquires. With `use_zstacking`, also `'zstack_not_configured'`, `'positions_unreachable'` or `'zslices_outside_travel'`, as `apply_zstacking` refuses the stored stack. Logged and notified once; nothing is added.
- `ObjectiveUnknownError`: the objective in the light path is unknown, as `runtime_state.resolve_current_objective` says why (`'slot_unassigned'`, the turret's current slot has no objective). Nothing is added.
- `ConfigurationDefectError`: `'layers_unresolved'`, `channels` is given and this scope's layers could not be resolved. Nothing is added.
- `ArgumentRefusedError`: `'channel_unknown'`, a name in `channels` is not a channel.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no layer named in `channels`; `'not_connected'`, with no `at`, the model's motor controller is not connected, so there is no live position. Nothing is added.
- `StepNotFoundError`: `'no_such_step'`, an impossible `before_step_index` / `after_step_index`.
- `StepEditRefusedError`: `'step_value_refused'`, both given.

**Refusals by REST status**

- 409: [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `step_position_unknown`, `no_acquiring_layer`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unassigned`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `zstack_not_configured`, `positions_unreachable`, `zslices_outside_travel`; [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`; [`StepNotFoundError`](#StepNotFoundError): `no_such_step`; [`StepEditRefusedError`](#StepEditRefusedError): `step_value_refused`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="ScopeSession.app_runtime"></a>
### `ScopeSession.app_runtime`

```python
app_runtime: AppRuntime  # property
```

Route: `GET /api/v1/app_runtime`.

How the application was launched: `source`, `bundle` or `installed`.

`source`: Python running a checkout or pip install. `bundle`: a
packaged build the installer did not install. `installed`: a
packaged build in the folder the installer wrote, whose data lives in
the user's Documents folder.

**Raises**

Nothing.

<a id="ScopeSession.app_version"></a>
### `ScopeSession.app_version`

```python
app_version: str | None  # property
```

Route: `GET /api/v1/app_version`.

The application's version, `4.0.0-beta36`; None where the build has none.

`version.txt`'s first line, through its one reader. A build with no
readable `version.txt` answers None rather than a name it does not
have.

**Raises**

Nothing.

<a id="ScopeSession.apply_channel_settings"></a>
### `ScopeSession.apply_channel_settings`

```python
def apply_channel_settings(self, protocol: Protocol) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/apply_channel_settings`.

Put a protocol's Layer Settings into this session's layer controls.

The other half of `save_protocol`, and what the GUI's Load does
once `load_protocol` has accepted the plate. Every layer stops
acquiring and stimulating; each layer the protocol names then takes
its acquire mode and every value its row holds. A blank value leaves
that layer's control as it was. A layer this release does not know
(not in `common_utils.get_layers()`), or this scope does not have,
is logged and dropped.

**Raises**

- `ProtocolFormatError`: A protocol built in memory carries a Layer Settings cell that is not of its column's type; no layer is changed. A loaded protocol was refused at load.
- `ProtocolRunRefusedError`: `'values_out_of_range'`, a row it would apply holds a value outside its layer setting's range, each named by layer, column and bound; no layer is changed. A loaded protocol was refused at load.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolFormatError`](#ProtocolFormatError): `protocol_invalid`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `values_out_of_range`

<a id="ScopeSession.apply_focus_to_channel_steps"></a>
### `ScopeSession.apply_focus_to_channel_steps`

```python
def apply_focus_to_channel_steps(self, protocol: Protocol, channel: str) -> int
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/apply_focus_to_channel_steps`.

Save the live Z as `channel`'s focus and as the Z of its every step.

Returns how many steps took it.

**Raises**

- `ProtocolRunRefusedError`: `'positions_unreachable'` -- this scope has no Z axis. Nothing is written.
- `AxisStateUnknownError`: Z lost its reference. Nothing is written.
- `AxisMovingError`: `'axis_moving'`, Z is moving, so its number is a place it is passing through. Nothing is written.
- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no `channel`.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved. Nothing is written.

**Refusals by REST status**

- 409: [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `positions_unreachable`; [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`

<a id="ScopeSession.apply_remedy"></a>
### `ScopeSession.apply_remedy`

```python
def apply_remedy(self, remedy: Remedy) -> int | tuple[UsbIdentityBindings, ...]
```

Route: `POST /api/v1/apply_remedy`.

Take the action a refusal named as its remedy, and return its answer.

The one place a remedy's name becomes an action: the GUI's offer and
a REST caller answer a refusal through this, never by resolving the
name themselves. A name arrives as data -- from a refusal another
layer built, or from a caller sending it back -- and a `Remedy`
holds only a `RemedyName`, so only the members listed here are
reachable through it. The member still decides whether the remedy
applies now, and refuses if not.

**Returns**

The remedy's own answer: `recover_file_writer` answers how many of the run's images it gave up on, `repair_camera_driver` the bindings after the repair.

**Raises**

- `HardwareCommandRefusedError`: `'exclusive_activity_running'`, as `recover_file_writer`.
- `FileWriterNotStuckError`: as `recover_file_writer`.
- `CameraDriverRepairRefusedError`: `'repair_needs_windows'` and the other reasons `repair_camera_driver` names.
- `CameraDriverRepairFailedError`: as `repair_camera_driver`.
- `CameraUsbLookupFailedError`: as `repair_camera_driver`.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`FileWriterNotStuckError`](#FileWriterNotStuckError): `file_writer_not_stuck`
- 422: [`CameraDriverRepairRefusedError`](#CameraDriverRepairRefusedError): `repair_needs_windows`
- 500: [`CameraDriverRepairFailedError`](#CameraDriverRepairFailedError): `camera_driver_repair_failed`; [`CameraUsbLookupFailedError`](#CameraUsbLookupFailedError): `camera_usb_lookup_failed`

<a id="ScopeSession.apply_tiling"></a>
### `ScopeSession.apply_tiling`

```python
def apply_tiling(self, protocol: Protocol, tiling: str) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/apply_tiling`.

Expand every step of `protocol` into the tile grid `tiling`, as Apply does.

The tiles are spaced for the configured frame, binning and tile
overlap, laid out on the protocol's own plate.

**Raises**

- `ProtocolRunRefusedError`: `tiling` is not a grid this installation offers, the protocol is already tiled, a step's objective is not in the catalogue, the scope has no X/Y motor, or a tile falls outside the stage's travel. Nothing changes.
- `CatalogueNameRefusedError`: `'labware_unknown'`, the protocol's plate is not in the catalogue. Nothing changes.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`

<a id="ScopeSession.apply_zstacking"></a>
### `ScopeSession.apply_zstacking`

```python
def apply_zstacking(
    self,
    protocol: Protocol,
    *,
    range_um: float,
    step_size_um: float,
    z_reference: ZStackReference,
) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/apply_zstacking`.

Expand every step of `protocol` not already in a stack into a z-stack.

`z_reference` says where each step's Z sits in its stack.

**Raises**

- `ProtocolRunRefusedError`: `range_um` or `step_size_um` is not greater than zero, the scope has no Z motor, or a slice falls outside the Z travel. Nothing changes.
- `ArgumentRefusedError`: `'not_offered'`, `z_reference` is not a `ZStackReference`. Nothing changes.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `not_offered`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`

<a id="ScopeSession.assign_turret_objective"></a>
### `ScopeSession.assign_turret_objective`

```python
def assign_turret_objective(self, slot: int, objective_id: str) -> None
```

Marks: sets [`turret_objective`](#ScopeSession.turret_objective); enters: `guarded_write`.  
Route: `PUT /api/v1/turret_objective`.

Bind `objective_id` to the turret's `slot`.

Binding the objective the slot already holds is a no-op.

**Raises**

- `HardwareCommandRefusedError`: `'axis_absent'`, the scope has no turret, said before `slot` or `objective_id` is judged; or a run, a diagnostic or a recording holds the scope (`'exclusive_activity_running'`). Nothing is written.
- `PositionOutOfRangeError`: `slot` is not a slot number 1-4.
- `ObjectiveUnknownError`: `'none_selected'`, `objective_id` is None.
- `CatalogueNameRefusedError`: `'objective_not_in_catalogue'`, `objective_id` is not exactly a catalogue key.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `none_selected`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `objective_not_in_catalogue`

<a id="ScopeSession.bookmark"></a>
### `ScopeSession.bookmark`

```python
bookmark: PlateLocation  # property
```

Route: `GET /api/v1/bookmark`.

The bookmark: X and Y in plate millimetres on the selected plate, Z in micrometres.

What `save_bookmark` and `save_all_bookmarks` stored, in the
frame the go-to-bookmark moves read it in.

**Raises**

Nothing.

<a id="ScopeSession.bring_up_record"></a>
### `ScopeSession.bring_up_record`

```python
def bring_up_record(self) -> BringUpRecord
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/bring_up_record`.

What bring-up found, substituted and set aside, for a client that asks later.

Which parts came up and, for each that did not, why; what bring-up
used in place of a saved setting the camera could not take, the
saved value beside it; and the settings file set aside with its
reason, while the app runs on the shipped template. A client given
to `create(outcome_listener=...)` heard these as outcomes; this is
the same facts held, read-through from the scope and the settings
store, for one that connects afterwards.

**Raises**

Nothing.

<a id="ScopeSession.capture_settings_snapshot"></a>
### `ScopeSession.capture_settings_snapshot`

```python
def capture_settings_snapshot(self) -> dict
```

Marks: in-process; reads (changes nothing).

A settings snapshot for composing a capture or a run.

`get_settings_snapshot`, refused while the active objective
(`runtime_state.resolve_current_objective`) is unknown: a capture
stamps its scale, so it is never composed without one.

**Raises**

- `ObjectiveUnknownError`: The active objective is unknown: `'slot_unknown'`, `'slot_unassigned'` or `'not_in_catalogue'` on a turret scope; `'none_selected'` with no turret.

**Refusals**

- [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`, `slot_unassigned`, `not_in_catalogue`, `none_selected`

<a id="ScopeSession.capture_still"></a>
### `ScopeSession.capture_still`

```python
def capture_still(
    self,
    channel: str | None = None,
    *,
    force_to_8bit: bool = True,
    accept_dark: bool = False,
    exclude_sources: tuple[FrameValiditySource, ...] = (),
    all_ones_check: bool = False,
    timeout_s: float = 0.0,
    sum_count: int = 1,
    sum_delay_s: float = 0.0,
    sum_iteration_callback: Callable[[], None] | None = None,
) -> CapturedStill
```

Marks: in-process.

Capture a frame-valid image and hand it back with its record.

Given `channel`, the camera holds that layer's stored exposure,
gain and auto-gain for the grab, as `apply_layer_camera` puts them
on, in the capture's own camera-lane task; given None, it captures
with the camera as set. The capture's arguments are
`scope.imaging.capture_still`'s. In-process: the image does not
travel the wire.

**Returns**

CapturedStill: The image and the instrument's record of its frame.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a layer; `'not_a_number'` or `'out_of_range'`, a time is not a wait from 0 to 30 days, or the wait for the capture and every frame of its sum together is longer. Nothing is captured.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no `channel`; `'not_connected'`, naming the camera, with none connected or one removed during the capture; `'camera_not_streaming'`, its feed is stopped; `'exclusive_activity_running'`, a run, a diagnostic or a home holds the scope and this call is not made under it.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved. Nothing is captured.
- `CaptureFailedError`: no frame passed, with the capture's own reason (`scope.imaging.capture_and_wait`).
- `CameraSettingRejected`: given `channel`, the camera refused one of its settings; nothing was captured.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `camera_not_streaming`, `exclusive_activity_running`, `not_on_this_scope`, `not_connected`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`, `not_a_number`, `out_of_range`
- [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`
- [`CaptureFailedError`](#CaptureFailedError): `no_frame`, `deadline_expired`, `chunk_rejected`, `no_fresh_frame`, `record_read_failed`, `recording_not_started`, `unknown_save_encoding`
- [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`

<a id="ScopeSession.channel_acquire"></a>
### `ScopeSession.channel_acquire`

```python
def channel_acquire(self, channel: str) -> AcquireMode | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/channel_acquire`.

What `channel` captures: an `AcquireMode`, or None (nothing).

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`

<a id="ScopeSession.channel_auto_gain"></a>
### `ScopeSession.channel_auto_gain`

```python
def channel_auto_gain(self, channel: str) -> bool
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/channel_auto_gain`.

Whether `channel`'s auto-gain is on, as `set_channel_auto_gain` set it.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`

<a id="ScopeSession.channel_auto_gain_lock"></a>
### `ScopeSession.channel_auto_gain_lock`

```python
def channel_auto_gain_lock(self, channel: str) -> AutoGainLock | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/channel_auto_gain_lock`.

What the camera reached the last time `channel`'s auto-gain was turned off.

The lock `set_channel_auto_gain(channel, False)` returned, the last
time it was called: its `state` says whether the camera converged,
hit the exposure ceiling or floor, or reported nothing usable, and is
None when no auto-gain was running or the camera did not hold the
layer. It describes that turning off, not the layer now: a later
turning on, a step's values or a stored gain leave it as it was.
None when the layer's auto-gain has not been turned off since the
session started.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`

<a id="ScopeSession.channel_colors"></a>
### `ScopeSession.channel_colors`

```python
def channel_colors(self) -> dict[str, tuple[float, float, float]]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/channel_colors`.

Every layer of the release and the colour it is shown and written
in, as red, green and blue weights, each 0 to 1.

The one read of a channel's colour: every capture, recording, build
and view takes it from here, the run and each build as it starts.
A layer's `display_color` setting, a `'#rrggbb'` colour at full
brightness, overrides its colour; null is no override. Without one,
a layer this unit carries takes the unit's colour (its record's,
which a board's Channels block may give), and any other layer the
release's, so a folder captured on another unit still has one.

**Raises**

- `InstallationFileError`: the installation's `scopes.json` is unusable (`layer_record.release_layers`).

**Refusals by REST status**

- 500: [`InstallationFileError`](#InstallationFileError): `installation_file_unusable`

<a id="ScopeSession.clear_current_turret_objective"></a>
### `ScopeSession.clear_current_turret_objective`

```python
def clear_current_turret_objective(self) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/clear_current_turret_objective`.

Leave the turret slot in the light path unassigned.

The counterpart of `assign_turret_objective` for the slot in the
light path: that says which objective is installed in a slot, this
says none is known to be in the current one. The slot is the API's,
never the caller's guess of it.

**Raises**

- `HardwareCommandRefusedError`: `'axis_absent'`, the scope has no turret; or a run, a diagnostic or a recording holds the scope and the slot has an assignment (`'exclusive_activity_running'`). Nothing is written.
- `ObjectiveUnknownError`: `'slot_unknown'`, the slot in the light path is unknown, so there is no slot to clear.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`

<a id="ScopeSession.clear_turret_objective"></a>
### `ScopeSession.clear_turret_objective`

```python
def clear_turret_objective(self, slot: int) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/clear_turret_objective`.

Leave the turret's `slot` unassigned.

Logged for every host: with that slot in the light path the active
objective is now unknown, and a support bundle can only explain a
capture refused for that afterwards if the clear is in the record.

**Raises**

- `HardwareCommandRefusedError`: `'axis_absent'`, the scope has no turret, said before `slot` is judged; or a run, a diagnostic or a recording holds the scope and the slot has an assignment (`'exclusive_activity_running'`). Nothing is written.
- `PositionOutOfRangeError`: `slot` is not a slot number 1-4.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`

<a id="ScopeSession.confirm_objective"></a>
### `ScopeSession.confirm_objective`

```python
def confirm_objective(self, objective_id: str, slot: int | None = None) -> bool
```

Marks: enters: `lane_call`, `guarded_write`.  
Route: `POST /api/v1/confirm_objective`.

Answer the objective question: this objective is in the light path.

On a turret scope, assigns the objective to `slot`,
which it requires: the answer records the glass in a slot and never
turns the turret. With no turret, selects it (`select_objective`),
and `slot` is refused. Records that a person has
confirmed the objective for this light path, turret or turretless,
so a scope brought up with the other asks again. Returns whether the
active objective changed.

**Raises**

- `ArgumentRefusedError`: `'slot_required'`, no `slot` on a turret scope. Nothing is written.
- `HardwareCommandRefusedError`: `'axis_absent'`, a `slot` on a scope with no turret. Nothing is written.
- `CatalogueNameRefusedError`: `'objective_not_in_catalogue'`, `objective_id` is not exactly a catalogue key. Nothing is written.
- `ObjectiveUnknownError`: `'none_selected'`, `objective_id` is None.
- `PositionOutOfRangeError`: `slot` is not a slot number 1-4.

**Refusals by REST status**

- 409: [`ObjectiveUnknownError`](#ObjectiveUnknownError): `none_selected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `slot_required`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `objective_not_in_catalogue`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ScopeSession.control"></a>
### `ScopeSession.control`

```python
control: ControlSession | None  # property
```

Route: `GET /api/v1/control`.

The control session held now, None when none is.

**Raises**

Nothing.

<a id="ScopeSession.controls_lock"></a>
### `ScopeSession.controls_lock`

```python
controls_lock: ControlsLock  # property
```

Marks: in-process.

Which of a person's controls the Session locks now, and why: its editing policy, not a refusal.

The run surface is held by a run by its kind, a diagnostic as one (a
characterization and the support report share its claim), a home,
a finished run's files still writing, or a control session; the
whole surface also by a live manual recording. The one rule every
lock read reads; a wire client reads it in `status`. It locks for
the holder of a control session too: it is the GUI's editing
policy, and the GUI is never the holder.

**Raises**

Nothing.

<a id="ScopeSession.create"></a>
### `ScopeSession.create`

```python
@classmethod
def create(
    cls,
    settings: dict,
    source_path: FilePath | None = None,
    *,
    simulate: bool = False,
    warn_pre_release: bool = True,
    engineering_mode: bool = False,
    no_engineering: bool = False,
    sim_model: str | None = None,
    sim_model_answer: SimulatedModelAnswer | None = None,
    sim_refused_stream_starts: int = 0,
    sim_camera_stall: SimulatedStall | None = None,
    sim_file_stall: SimulatedStall | None = None,
    sim_focal_spread_um: float = 0.0,
    outcome_listener: Callable[[Notification], None] | None = None,
    settings_file: FilePath | None = None,
) -> ScopeSession
```

Marks: in-process.

Create a session, constructing defaults for any missing components.

This is the one composition path: the GUI, REST and scripts all
build their session here, passing what only a host knows as the
keyword arguments below. The factory builds the scope, so every
session's scope is brought up the one way.

`source_path` is the data folder the factory builds the scope on;
None (default) is the installation's own folder. The session's data
folder and catalogues are always its scope's.

The executor bundle is always built, around the scope's own IO and
CAMERA lanes, via executor_registry.create_default, so every caller
gets the topology the GUI runs: FILE + WORKER_POOL executors plus
protocol_thread (started) and scope_display_thread (constructed,
not started).

The settings' log level and autofocus score log are applied to the
process's logging first, so every host logs as its settings say,
bring-up included.

The scope is brought up before this returns (`configure_scope`,
then the camera start gate released) and is torn down by
`shutdown()`.

Keyword arguments, each with the one consumer it feeds:
    simulate: build a simulated scope (the model from
        `settings['microscope']`, the motor board's tier from
        `settings['simulator_tier']`).
    sim_model: the model the simulated scope is, when it is not the
        one `settings` declares: a stored file that disagrees
        with the scope attached, which bring-up settles by the
        scope's answer. None (the default) is the declared model.
        Refused unless `simulate`.
    sim_model_answer: the model the simulated motor board names as
        its own at bring-up, whatever it was built as
        (`scope.simulation.answer_model`), until the scope's
        faults are cleared. Refused unless `simulate`, and by the
        scope on any board but the fast motor board.
    sim_refused_stream_starts: how many of the camera's stream
        starts it refuses from bring-up on
        (`scope.simulation.refuse_stream_start`), so bring-up's
        own start is refused; 0 (the default) refuses none. Refused
        unless `simulate`, and by the scope on any camera but the
        simulated FX2.
    warn_pre_release: whether this construction fires the
        pre-release FutureWarning -- the factory's own call and the
        scope constructor's. A host that ships with the API passes
        False; a separately shipped caller leaves the default.
    engineering_mode: the session's engineering mode to begin with;
        a plugin may turn it on when the host loads plugins.
    no_engineering: the host's word that the engineering plugin is
        not loaded: `load_plugins` leaves it out before it is
        imported, as if it were not installed.
    sim_camera_stall: a stall for the simulated camera's stream, so a
        simulated scope shows a stream that stops delivering; refused
        by the scope itself unless it is simulated with the
        simulated camera.
    sim_file_stall: a stall for the session's file lane, so a
        simulated scope shows a save drive that stops answering:
        `after_s` seconds after bring-up the lane's worker is held
        for `for_s` seconds. Refused unless `simulate`.
    sim_focal_spread_um: how far the simulated sample's focus may
        lie from where bring-up places the stage, drawn once per
        session, so every session needs a real autofocus; 0 (the
        default) puts it in focus there. Refused as
        `sim_camera_stall` is.
    outcome_listener: heard from before the scope is built, so it
        hears what bring-up reports; the session's
        `add_outcome_listener` says what it receives. A factory
        that raises takes it back; otherwise the session's
        `shutdown()` does.
    settings_file: the file the session saves its settings to,
        one second after the first unsaved write and once at
        `shutdown()`, whole or not at all; the hosts name the
        installation's `user_settings_file(source_path)`. None (the default)
        saves nothing: a script's or a test's handed dict stays in
        memory. Writes are kept unsaved, and the session says so
        once (`SettingsNotSavedNotice`), while the settings are
        provisional or when the scope found no hardware at
        bring-up. Saving starts once the scope is up, so a factory
        that raises writes no file.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`, a `sim_` argument without `simulate`: a real scope's file lane writes to a real drive, and only a simulated scope's boards and camera take a fault. `'needs_simulated_camera'`, `sim_camera_stall` for a model simulated with an FX2. `'not_injectable_on_board'`, `sim_model_answer` or `sim_refused_stream_starts` for a scope whose board or camera cannot show it; `'fault_bound'`, a negative `sim_refused_stream_starts`. `'wrong_kind'`, a block of `settings` that is not the kind the shipped template holds (`argument` names it).
- `SettingRefusedError`: `'wrong_kind'` or `'out_of_range'`, a value in `settings` the writer refuses: a dict handed to a session is a write of every key at once, held to the same ranges.
- `ScopeInUseError`: `'scope_in_use'`, a session with `simulate=False` asked for while another LumaViewPro, REST server or script drives the microscope, or while another real scope in this process is open. Nothing was opened.

**Refusals**

- [`ScopeInUseError`](#ScopeInUseError): `scope_in_use`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `needs_simulated_camera`, `not_injectable_on_board`, `fault_bound`, `wrong_kind`
- [`SettingRefusedError`](#SettingRefusedError): `wrong_kind`, `out_of_range`

<a id="ScopeSession.create_empty_protocol"></a>
### `ScopeSession.create_empty_protocol`

```python
def create_empty_protocol(self) -> Protocol
```

Route: `POST /api/v1/create_empty_protocol`.

A protocol with no steps, on this session's labware and timing.

Needs no objective: it has no step to stamp one into, so it can be
created while the objective in the light path is unknown -- at
startup, before the turret is in a known slot. Steps added later
carry the objective they were taken with.

**Raises**

- `ProtocolScheduleRefusedError`: the stored period or duration is one no protocol can run.

**Refusals by REST status**

- 422: [`ProtocolScheduleRefusedError`](#ProtocolScheduleRefusedError): `schedule_not_runnable`

<a id="ScopeSession.create_protocol_runner"></a>
### `ScopeSession.create_protocol_runner`

```python
def create_protocol_runner(self) -> ProtocolRunner
```

Route: `POST /api/v1/create_protocol_runner`.

The session's one ProtocolRunner (memoized).

Wraps the session-composed sequenced-capture engine; repeated
calls return the same instance. The accessor takes no
arguments: every dependency (engine, executors, autofocus
pair) is session composition, so there is nothing per-call to
configure -- and nothing for a second caller's differing
configuration to silently lose.

**Raises**

Nothing.

<a id="ScopeSession.current_slot_objective_question"></a>
### `ScopeSession.current_slot_objective_question`

```python
def current_slot_objective_question(self) -> ObjectiveQuestion
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/current_slot_objective_question`.

The objective question for the turret slot in the light path, owed or not.

A person asking to say which objective is installed in the current
slot: to change an assignment, or to name a slot nothing is assigned
to, at any time rather than only when `objective_question` owes it.
The question is that read's, proposing the slot's assignment, so an
unchanged answer leaves the slot as it is. The answer goes back
through `confirm_objective` with the slot the question names, which
is the slot written even if the turret has moved since. On a scope
without a turret the question names no slot and proposes the
selected objective once one has been confirmed, the catalogue
default before.

A hold or provisional settings refuse nothing here: the answer's own
write refuses a change under a hold, and a write while settings are
provisional is kept in memory like any other setting's, and not
saved (`settings_are_provisional`).

**Raises**

- `ObjectiveUnknownError`: `'slot_unknown'`, the slot in the light path is unknown -- the turret is turning or has not been homed.

**Refusals by REST status**

- 409: [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`

<a id="ScopeSession.delete_step"></a>
### `ScopeSession.delete_step`

```python
def delete_step(self, protocol: Protocol, step_index: int) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/delete_step`.

Remove step `step_index` from `protocol`, as the Delete button does.

**Raises**

- `StepNotFoundError`: `'no_such_step'`, `step_index` is not a step of `protocol`. Nothing is removed.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`StepNotFoundError`](#StepNotFoundError): `no_such_step`

<a id="ScopeSession.diagnostic_claim"></a>
### `ScopeSession.diagnostic_claim`

```python
def diagnostic_claim(self) -> Iterator[HeldClaim]
```

Marks: in-process.

Hold the scope for a diagnostic for the length of a `with` block.

A diagnostic -- a characterization, the support report's hardware
steps -- drives every axis, the LEDs and the camera directly. While
it holds the claim, a run or a recording start is refused, the
controls lock and the objective cannot change, exactly as during a
run. The claim is released when the block ends, including on a
raise, so no caller owns the release path.

A run the diagnostic lent its claim to (`start_autofocus(claim=...)`)
may still be live when the block ends -- a caller that raised
between starting it and waiting on it. The release waits for that
run to go idle first: its LED lease lives under this claim, and
releasing underneath a live run hands the scope to the next taker
mid-sweep. If the run is still live after the wait, this raises and
the claim is kept with the run, released when the run ends.

**Yields**

The held claim.

**Raises**

- `DiagnosticRefusedError`: `'exclusive_activity_running'`, a run, a recording, a home or another diagnostic holds the scope. Nothing was taken.
- `ControlSessionRefusedError`: `'control_session_held'`, a control session is held by someone else. Nothing was taken.
- `ClaimStillLentError`: `'claim_still_lent'`, at the end of a block that ended cleanly, a run under this claim was still live after the wait; the claim is released when the run ends. When the block raised, its own exception is raised and this is reported once instead. Still a `RuntimeError`.

**Refusals**

- [`DiagnosticRefusedError`](#DiagnosticRefusedError): `exclusive_activity_running`
- [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- [`ClaimStillLentError`](#ClaimStillLentError): `claim_still_lent`

<a id="ScopeSession.disable_metrics"></a>
### `ScopeSession.disable_metrics`

```python
def disable_metrics(self) -> None
```

Marks: in-process.

Stop the session's periodic metrics logging. Idempotent.

Never shuts the scheduler itself down -- a shut scheduler
refuses all future schedules, and the host may start metrics
again with the same instance.

**Raises**

Nothing.

<a id="ScopeSession.discard_close_drain"></a>
### `ScopeSession.discard_close_drain`

```python
def discard_close_drain(self) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/discard_close_drain`.

Drop every video frame still queued in either drain, loudly.

The one escape from waiting on the video frames `live_work` lists
-- a recording's file, a run's video -- for a close or any caller;
each file is finished with what was written, and frames already on
disk stay.

**Raises**

Nothing.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`

<a id="ScopeSession.enable_metrics"></a>
### `ScopeSession.enable_metrics`

```python
def enable_metrics(self, display: DisplayRates | None = None) -> None
```

Marks: in-process.

Start the session's periodic metrics logging.

Uses the session's scheduler and the
`settings.profiling.metrics_interval_s` cadence override when
present. Metrics stay opt-in by the call itself: a host that runs
the scope for others calls it -- the GUI does, and so does a
server host -- and a script that only drives the scope need not.

**Args**

- `display`: The host's live display (the GUI's), whose drawn rate and frame intervals the system tick logs beside the camera's. A host with no display passes none.

**Raises**

- `AlreadyStartedError`: `'metrics_running'`, metrics are already running; `disable_metrics` lets them start again. A second `MetricsLogger.start` would silently overwrite its schedule handles and orphan the first set as untracked, forever-ticking events. Still a `RuntimeError`.

**Refusals**

- [`AlreadyStartedError`](#AlreadyStartedError): `metrics_running`

<a id="ScopeSession.end_control"></a>
### `ScopeSession.end_control`

```python
def end_control(self, session_id: str) -> ControlSession
```

Route: `POST /api/v1/end_control`.

End the control session named by its id: its taker ends it, or the person at the scope.

The session ends at once, and the run or recording taken in it is
stopped: it unwinds under its own claim and holds the scope until it
has, as any stopped run does. A run or recording begun after it is
not stopped. A home or a diagnostic taken in it ends on its own. A
write of the session's already under way finishes whole, and no new
session is taken until it has. A request naming the ended session
is refused from now on.

**Args**

- `session_id`: The `id` the session was taken with.

**Returns**

The session ended.

**Raises**

- `ControlSessionRefusedError`: `'control_session_ended'`, no session is held, or the live one has another id.

**Refusals by REST status**

- 422: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_ended`

<a id="ScopeSession.get_current_plate_position"></a>
### `ScopeSession.get_current_plate_position`

```python
def get_current_plate_position(self) -> PlateLocation
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/get_current_plate_position`.

The stage's position in plate coordinates, on the session's labware.

**Returns**

Where the scope is on the plate: X and Y in mm, Z in um.

**Raises**

- `AxisStateUnknownError`: `'position_unknown'`, an axis the scope has does not know its position (before a home, after a lost reference, while it homes), so there is no position to convert. Raised, not reported.
- `AxisMovingError`: `'axis_moving'`, an axis is moving, so its number is a place it is passing through.
- `HardwareCommandRefusedError`: `'not_connected'`, the model's motor controller is not connected.

**Refusals by REST status**

- 409: [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`AxisMovingError`](#AxisMovingError): `axis_moving`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="ScopeSession.get_setting"></a>
### `ScopeSession.get_setting`

```python
def get_setting(self, path: str) -> SettingValue
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/get_setting`.

A copy of one setting, named by its dotted path: `'stage_offset'`, `'video.max_fps'`.

Taken under the lock, so it is never a value part-way through a
write; a copy, so changing it changes no setting (`update_settings`
and the Session's members do).

A setting with a read of its own is read there, so one setting has
one read: `get_setting('binning')` is refused for
`scope.camera.binning`, `get_setting('objective_id')` for
`objective_id`. The live folder and the protocol file, which only
the host moves, are read here.

**Raises**

- `ArgumentRefusedError`: `'not_a_setting'`, the settings have no value at `path`; `'has_member'`, the setting at or above `path` has its own read, named in `member`.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_setting`, `has_member`

<a id="ScopeSession.get_settings_snapshot"></a>
### `ScopeSession.get_settings_snapshot`

```python
def get_settings_snapshot(self) -> dict
```

Marks: in-process; reads (changes nothing).

A deep copy of the settings dict, taken under the lock.

A worker thread takes one of these at task entry and reads from it
for the rest of the task, rather than reading a dict another
thread may be part-way through rewriting. In-process: a wire client
reads a setting by its path, through `get_setting`.

`objective_id` is the active objective, as `get_setting` answers
it: on a turret scope, the slot's in the light path, or None while
that is unknown.

**Raises**

Nothing.

<a id="ScopeSession.get_zstack_params"></a>
### `ScopeSession.get_zstack_params`

```python
def get_zstack_params(self) -> dict
```

Marks: in-process; reads (changes nothing).

The stored z-stack, what New and Add stack with.

`range` and `step_size` in micrometres, and `z_reference`, where
a step's Z sits in its stack, a `ZStackReference`. The store holds
each to its writer's rule at load and at every write.

**Raises**

Nothing.

<a id="ScopeSession.go_to_bookmark"></a>
### `ScopeSession.go_to_bookmark`

```python
def go_to_bookmark(self, axes: Iterable[StageAxis]) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/go_to_bookmark`.

Move `axes` to the bookmark, and return once they have arrived.

`start_go_to_bookmark` followed by its operation's `result()`.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, as `start_go_to_bookmark`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'lid_open_before_motion'` or `'monitor_ended'`, as `MotionAPI.start_move_to_plate`. Nothing was driven.
- `PositionOutOfRangeError`: The bookmark is outside what this stage can reach. Nothing was driven.
- `AxisStateUnknownError`: An axis given does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `MotionAPI.start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_go_to_bookmark`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="ScopeSession.go_to_step"></a>
### `ScopeSession.go_to_step`

```python
def go_to_step(self, protocol: Protocol, step_index: int) -> None
```

Marks: enters: `lane_call`, `guarded_write`.  
Route: `POST /api/v1/go_to_step`.

Go to step `step_index` of `protocol`; return once the camera holds the step's layer and the stage has arrived.

What `start_go_to_step` does, then the started operation's
`result()`: the step's layer goes on the camera while the stage
travels; see `start_go_to_step` for what going to a step does and
refuses.
Both waits run in this caller's thread, so the IO lane takes other
work meanwhile.

It raises what `start_go_to_step` raises, and the following.

**Raises**

- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `MotionAPI.start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first. The layer and the preview are the step's.
- `CameraSettingRejected`: the camera refused a setting of the step's layer, raised once the stage has arrived; see `apply_layer_camera`.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_go_to_step`. Nothing was started.

**Refusals by REST status**

- 409: [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ScopeSession.held_by_other"></a>
### `ScopeSession.held_by_other`

```python
def held_by_other(self, run: Operation | None = None) -> bool
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/held_by_other`.

Whether the scope is held by anything but *run*, the operation a run member returned.

What a run control greys on while leaving its own run's Stop live;
None, or no run given, asks whether anything holds the scope at all
(`live_work` names what does).

**Raises**

Nothing.

<a id="ScopeSession.host_steady_ns"></a>
### `ScopeSession.host_steady_ns`

```python
def host_steady_ns(self) -> int
```

Marks: in-process.

The host's steady time now, in nanoseconds.

The clock every steady stamp the scope takes is on, for an
in-process caller setting its own "now" against them.

**Raises**

Nothing.

<a id="ScopeSession.labware_id"></a>
### `ScopeSession.labware_id`

```python
labware_id: str  # property
```

Route: `GET /api/v1/labware_id`.

The current plate's catalogue key, the one `select_labware` selected.

Its record is `scope.labware[labware_id]`.

**Raises**

Nothing.

<a id="ScopeSession.last_run"></a>
### `ScopeSession.last_run`

```python
def last_run(self) -> Operation | None
```

Route: `POST /api/v1/last_run`.

The last run's operation, live or ended, whoever started it; None before any run.

Every `ProtocolRunner` start is a run: a scan, a protocol, a
composite, an autofocus, a z-stack. A client that learns of a run
from the `status` edges -- one the GUI, a script or the SDK
started -- reads it here, and on the wire it is that run's one job,
whoever reads it: its progress, its outcome and its `job_ended`,
which a run already ended tells at the read. Live or ended, since a
run goes idle before its operation settles, and one short against
the client's round trip would otherwise end unread. Its per-step
events stay with its starter.

A POST, not a read: it hands out a running call. A finished job a
client deleted answers 404 on its id.

**Raises**

Nothing.

<a id="ScopeSession.light_shown_channel"></a>
### `ScopeSession.light_shown_channel`

```python
def light_shown_channel(self, lit: bool) -> ShownChannel
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/light_shown_channel`.

Light the shown channel's LED at its stored current, or darken it,
then put its stored camera settings on the camera.

Every other LED is dark either way. A layer this unit gives no LED
has nothing to light, and stays dark. The camera is left on the
layer's stored settings (`apply_layer_camera`; nothing is written
when it holds them already), unless no camera is connected.

**Returns**

The shown channel now.

**Raises**

- `ArgumentRefusedError`: `'illumination_out_of_range'`, the layer's stored current is outside the attached board's range.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no shown layer; `'not_connected'`, naming the LED controller the light needs; or `'exclusive_activity_running'`, a run, a diagnostic or a home holds the scope.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved.
- `CameraSettingRejected`: the camera refused a setting of the layer; its LED is already lit or dark.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `illumination_out_of_range`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`, `activity_ended`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`; [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ScopeSession.live_folder_listing"></a>
### `ScopeSession.live_folder_listing`

```python
def live_folder_listing(self, name: str = '.') -> tuple[LiveFolderEntry, ...]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/live_folder_listing`.

What the folder `name` under the live folder holds, one level, sorted by name.

Each entry is named by its full name under the live folder,
`/`-separated on every host, so it is taken as it is by any path
parameter, a download and a further listing. `'.'` lists the live
folder itself. An entry that is gone by the time it is read -- a
file a run renames into place, a link to nothing -- is not listed. A
link that leads outside the live folder is listed as what it leads
to, and refused where its name is used, as `live_folder_path`
refuses it.

**Raises**

- `LiveFolderPathRefusedError`: as `live_folder_path` raises it; `'path_not_found'`, `name` names nothing; `'not_a_folder'`, `name` is a file.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_folder`

<a id="ScopeSession.live_folder_path"></a>
### `ScopeSession.live_folder_path`

```python
def live_folder_path(self, name: str) -> Path
```

Marks: in-process; reads (changes nothing).

The absolute path that `name`, a name under the live folder, names.

The one door from a wire caller's path argument to the file system:
a REST bridge passes every path a caller gives through this, so the
caller reaches the live folder and nothing beside it. A Python or GUI
caller passes any path straight to the member it calls. The live
folder is not created here: one that is missing is an unplugged drive
or a stale setting.

**Raises**

- `LiveFolderPathRefusedError`: `'outside_live_folder'`, `name` is empty, absolute, carries a drive (`C:x`, `C:\x`) or a network share, is no file system's name (a NUL byte), or resolves outside the live folder through `..` or a link; `'capture_location_unusable'`, the live folder is missing or is not a folder.

**Refusals**

- [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `capture_location_unusable`, `outside_live_folder`

<a id="ScopeSession.live_work"></a>
### `ScopeSession.live_work`

```python
live_work: LiveWork  # property
```

Marks: in-process.

Everything the session is still doing, each piece from its owner.

What holds the scope (a run, a recording, a home, a diagnostic and
any run acting under its lent claim -- and a run starting or
unwinding with no holder yet or any more);
then what finishes after it lets go: a recording's file, a run's
video steps, a finished run's images and its post-run builds;
post-processing builds running and queued; a support report or logs
zip; an engineering plugin run, from its press to its end; a still
being saved. Each item says how much it has left, and
a build how far it has got. Read lock-free, owner by owner, from any
thread; `closing` while the close runs, `closed` once the
session has shut down. A wire client reads it in `status`.

**Raises**

Nothing.

<a id="ScopeSession.load_plugins"></a>
### `ScopeSession.load_plugins`

```python
def load_plugins(self) -> None
```

Marks: in-process.

Load the installed plugins and the built-ins into this session.

The host's one call to have plugins; a session it is never made on
has none. Each plugin's `register` is handed this session as its
ctx. A plugin that does not load is reported and the rest load;
nothing here raises for a plugin. Under `no_engineering` the
engineering plugin is left out, unreported, as if not installed.
Call it once, after `create`.

**Raises**

- `AlreadyStartedError`: `'plugins_loaded'`, this session already loaded its plugins. Still a `RuntimeError`.

**Refusals**

- [`AlreadyStartedError`](#AlreadyStartedError): `plugins_loaded`

<a id="ScopeSession.load_protocol"></a>
### `ScopeSession.load_protocol`

```python
def load_protocol(self, file_path: FilePath) -> Protocol
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/load_protocol`.

Load the protocol at `file_path` and put the scope on its plate.

A protocol's positions are stated against the plate it names, so it
is handed back only with the scope on that plate: the plate is
selected through `select_labware`, and a refused selection refuses
the load. On a scope with no XY stage there is no plate to be on; the
protocol and the scope both take "Center Plate".

Its contract is the plate. The protocol's period, duration and
per-layer settings are the protocol's own and are not applied to
anything else.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at `file_path`; `'not_a_file'`, a folder is.
- `ProtocolNotLoadedError`: As `ProtocolsAPI.load_protocol`.
- `ProtocolFormatError`: As `ProtocolsAPI.load_protocol`.
- `ProtocolRunRefusedError`: `'objective_not_mounted'`, `'objectives_require_turret'`, `'turret_objectives_unassigned'` or `'layer_not_on_scope'`: the file holds a step this scope cannot perform; `'channel_unknown'`, a step names a channel no release has; `'values_out_of_range'`, its Layer Settings hold a value no layer setting takes; each as `ProtocolsAPI.load_protocol`.
- `HardwareCommandRefusedError`: `'exclusive_activity_running'`, as `select_labware`; the scope stays on the plate it had.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `objective_not_mounted`, `turret_objectives_unassigned`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_file`; [`ProtocolFormatError`](#ProtocolFormatError): `protocol_invalid`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `objectives_require_turret`, `layer_not_on_scope`, `channel_unknown`, `values_out_of_range`
- 500: [`ProtocolNotLoadedError`](#ProtocolNotLoadedError): `protocol_not_loaded`

<a id="ScopeSession.load_user_settings"></a>
### `ScopeSession.load_user_settings`

```python
@staticmethod
def load_user_settings(source_path: FilePath) -> dict
```

Marks: in-process.

The user's configuration, as the GUI would configure a scope from it.

For a host with no GUI -- a script, a server -- to pass to
`create`: ``create(ScopeSession.load_user_settings(root),
simulate=...)``. Reading is its own call rather than a default of
`create` so a caller that meant to pass settings and forgot is
refused for the missing argument instead of quietly configured
from whatever is on disk.

In a process that has already prepared its settings (the GUI's),
those are answered and `source_path` is not read.

**Raises**

- `ArgumentRefusedError`: `'not_an_installation'`, `source_path` holds neither settings file.
- `SettingsFileError`: the user's `current.json` is unusable.
- `InstallationFileError`: the installation's own `scopes.json`, whose layer vocabulary the settings are checked against, or its `settings.json` template, is missing or unusable.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_an_installation`
- [`SettingsFileError`](#SettingsFileError): `settings_file_unreadable`
- [`InstallationFileError`](#InstallationFileError): `installation_file_unusable`

<a id="ScopeSession.make_logs_zip"></a>
### `ScopeSession.make_logs_zip`

```python
def make_logs_zip(
    self,
    *,
    output_dir: FilePath,
    on_progress: ProgressCallback | None = None,
) -> SupportReportSaved
```

Route: `POST /api/v1/make_logs_zip`.

Zip the logs, the data folder, the recent protocols and the video receipts.

Touches no hardware and does not hold the scope: it is for sending
the record of what already happened.

**Args**

- `output_dir`: The folder the ZIP is written into, made if it is missing. A host that writes where a person looks passes `path_utils.desktop_folder()`; the session chooses none.
- `on_progress`: Called with a percentage and what the zip is doing, from the thread it runs on.

**Returns**

Where the ZIP is, and the words that say where to send it.

**Raises**

- `LiveFolderPathRefusedError`: `'not_a_folder'`, a file is at `output_dir` or above it; refused before anything is made.
- `SupportReportNotSavedError`: no ZIP was saved; chained from the failure, whose words it carries.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `not_a_folder`
- 500: [`SupportReportNotSavedError`](#SupportReportNotSavedError): `support_report_not_saved`

<a id="ScopeSession.make_support_report"></a>
### `ScopeSession.make_support_report`

```python
def make_support_report(
    self,
    *,
    output_dir: FilePath,
    include_bandwidth_test: bool = False,
    on_progress: ProgressCallback | None = None,
) -> SupportReportSaved
```

Route: `POST /api/v1/make_support_report`.

Make the full Tech Support Report: the boards, the motors, the camera and the files.

Blocks for minutes. The hardware steps hold the scope for a
diagnostic, so a run cannot start under a homing or a fan sweep; a
report started while a run holds the scope skips those steps and
says so in the ZIP. A step that fails is written into the ZIP; the
report does not stop for it.

**Args**

- `include_bandwidth_test`: Also time the camera's frame delivery (adds minutes).
- `output_dir`: The folder the ZIP is written into, made if it is missing. A host that writes where a person looks passes `path_utils.desktop_folder()`; the session chooses none.
- `on_progress`: Called with a percentage and what the report is doing, from the thread the report runs on.

**Returns**

Where the ZIP is, and the words that say where to send it.

**Raises**

- `LiveFolderPathRefusedError`: `'not_a_folder'`, a file is at `output_dir` or above it; refused before anything is made.
- `SupportReportNotSavedError`: no ZIP was saved; chained from the failure, whose words it carries.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `not_a_folder`
- 500: [`SupportReportNotSavedError`](#SupportReportNotSavedError): `support_report_not_saved`

<a id="ScopeSession.model"></a>
### `ScopeSession.model`

```python
model: str  # property
```

Route: `GET /api/v1/model`.

The saved scope model: the one `select_model` saved, brought up at the next start.

The running scope's model is `scope.channel_identity.model`; the two
differ until the next bring-up, which `model_at_next_start` says.

**Raises**

Nothing.

<a id="ScopeSession.model_at_next_start"></a>
### `ScopeSession.model_at_next_start`

```python
model_at_next_start: str | None  # property
```

Route: `GET /api/v1/model_at_next_start`.

The saved model when it is not the one running, else None.

What `select_model` saved waits for the next bring-up, so until
then the scope runs as one model and the settings name another; a
caller shows this rather than comparing the two itself.

**Raises**

Nothing.

<a id="ScopeSession.new_protocol"></a>
### `ScopeSession.new_protocol`

```python
def new_protocol(
    self,
    *,
    tiling: str = '1x1',
    use_zstacking: bool = False,
    period_s: float | None = None,
    duration_s: float | None = None,
) -> Protocol
```

Route: `POST /api/v1/new_protocol`.

Build a protocol from this session's settings, as the GUI's New does.

One step per acquiring layer at every well of the session's labware,
at the current objective, tiled and z-stacked as asked. Its period and
duration are the ones given, in seconds; one left out (None) is the
stored default's. One scan is 0. The
protocols API refuses the build when no layer acquires, where an
empty protocol would otherwise come back for a click that meant
steps; a labware with no wells still gives an empty protocol, which
`add_step` fills at the current position.

**Raises**

- `ProtocolRunRefusedError`: `'no_acquiring_layer'`, logged and notified once; `'tiling_unknown'`, a grid this installation does not offer; `'overlap_out_of_range'`, the stored tile overlap outside 0 to 50; `'tiles_outside_travel'`, a tile past the X or Y travel; `'zstack_not_configured'`, a z-stack with no extent; `'positions_unreachable'`, a grid on a scope with no X/Y motor or a z-stack on one with no Z motor; `'zslices_outside_travel'`, a slice past the Z travel.
- `ObjectiveUnknownError`: the objective in the light path is unknown, so the config cannot be assembled: `'slot_unknown'`, `'slot_unassigned'` or `'not_in_catalogue'` on a turret scope; `'none_selected'` with no turret.
- `ProtocolScheduleRefusedError`: `period_s` or `duration_s` is one no protocol can run.
- `AxisStateUnknownError`: an acquiring layer has no saved focus, so its steps would save the current Z, and Z does not know its position. Reported once.
- `AxisMovingError`: `'axis_moving'`, Z is moving, so the current Z is a place it is passing through. Nothing is built.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `no_acquiring_layer`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`, `slot_unassigned`, `not_in_catalogue`, `none_selected`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`AxisMovingError`](#AxisMovingError): `axis_moving`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `tiling_unknown`, `overlap_out_of_range`, `tiles_outside_travel`, `zstack_not_configured`, `positions_unreachable`, `zslices_outside_travel`; [`ProtocolScheduleRefusedError`](#ProtocolScheduleRefusedError): `schedule_not_runnable`

<a id="ScopeSession.objective_id"></a>
### `ScopeSession.objective_id`

```python
objective_id: str | None  # property
```

Route: `GET /api/v1/objective_id`.

The active objective's catalogue key, or None while it is not known.

On a turret scope it is the assignment of the slot in the light path,
None while that slot is unknown or unassigned; on a turretless scope,
the one `select_objective` selected
(`scope.runtime_state.resolve_current_objective` says why it is
unknown).

**Raises**

Nothing.

<a id="ScopeSession.objective_question"></a>
### `ScopeSession.objective_question`

```python
def objective_question(self) -> ObjectiveQuestion | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/objective_question`.

Does the objective need confirming? The question, or None.

A read, for any host: the GUI renders the answer as a popup, a
REST caller reads it as state. Two ways the session cannot know
what is in the light path: no person has confirmed the objective
for this scope's light path (never on this install -- the settings
template ships a default that would otherwise set image scale
silently forever -- or only on a scope that differed in having a
turret), or, on a turret scope (`capabilities.has_turret`, the
declared model's answer when the board did not come up, so a stale
objective does not pass unasked then), the slot in the light path
has no assignment. The slot is the live one
(`motion.turret_status`), so an answer names the glass that is
actually in the light path.

Four conditions withhold an owed question, each leaving one log
line per call so a bundle can say why nothing was asked: with no
hardware there is nothing in the light path and no capture to
stamp; while settings are provisional nothing is saved, so an
answer given now would be lost -- the host re-asks when they
resolve; while an activity holds the scope, or a control session
is held by someone other than the caller, the answer is refused
too (`confirm_objective`), so asking would only ask again after
each refusal -- the host re-asks when the hold or the session
ends. No line is
logged for a returned question: the renderer logs its own show,
and a polled read must not log per poll.

**Raises**

- `ObjectiveUnknownError`: `'slot_unknown'`, the objective is not confirmed for this light path and the turret model's slot is unknown -- there is no slot to answer for until the turret is homed or moved.

**Refusals by REST status**

- 409: [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`

<a id="ScopeSession.plugin_api_level"></a>
### `ScopeSession.plugin_api_level`

```python
plugin_api_level: int  # property
```

Route: `GET /api/v1/plugin_api_level`.

What this LumaViewPro does that a plugin may rely on.

`modules.plugins.PLUGIN_API_LEVEL`; each level's additions are
listed in LumascopeSkills.md, "Plugin API level".

**Raises**

Nothing.

<a id="ScopeSession.plugin_health"></a>
### `ScopeSession.plugin_health`

```python
def plugin_health(self) -> PluginHealth | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/plugin_health`.

The loaded plugins, the ones that did not load, and their runtime errors.

None on a session whose host never asked for plugins (`load_plugins`).

**Raises**

Nothing.

<a id="ScopeSession.plugin_run_in_flight"></a>
### `ScopeSession.plugin_run_in_flight`

```python
def plugin_run_in_flight(self) -> Iterator[None]
```

Marks: in-process.

Count a plugin's run as live work from entry to exit, entered and left on any threads.

For a plugin whose run starts on one thread and ends on a worker it
starts: entered at the press, before the worker, and left as the
worker ends, so `live_work` and `in_flight_named` name the run
across the gap before its own claim and after it -- a confirmation
asked first, the worker's end. A run that holds a claim takes it as
it always did; this holds no claim and refuses nothing, and the
close waits for it as for any live work, after the plugins are told
to stop.

**Raises**

Nothing.

<a id="ScopeSession.point_steps_at_turret_slot"></a>
### `ScopeSession.point_steps_at_turret_slot`

```python
def point_steps_at_turret_slot(
    self,
    protocol: Protocol,
    objective_id: str,
    slot: int,
) -> tuple[int, ...]
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/point_steps_at_turret_slot`.

Answer `objective_id`'s turret slot question: its steps now record `slot`.

Points exactly the steps `turret_slot_questions` names for
`objective_id`; every other step keeps its slot. Returns
the indexes of the steps pointed, which are empty when no question names it.

**Raises**

- `ArgumentRefusedError`: `'slot_not_holding_objective'`, `slot` does not hold `objective_id` now (`offered`: the slots that do). Nothing is changed.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `slot_not_holding_objective`

<a id="ScopeSession.protocol_files_stalled"></a>
### `ScopeSession.protocol_files_stalled`

```python
protocol_files_stalled: bool  # property
```

Route: `GET /api/v1/protocol_files_stalled`.

True while a run's files are draining and the write in flight has
stopped making progress -- by the same stall threshold the run
refusal uses, so what a display says of a stuck writer and what a
new run is refused for cannot disagree.

**Raises**

Nothing.

<a id="ScopeSession.protocol_files_stuck_write"></a>
### `ScopeSession.protocol_files_stuck_write`

```python
protocol_files_stuck_write: str  # property
```

Route: `GET /api/v1/protocol_files_stuck_write`.

The write in flight on the file lane, named for a stall report.

**Raises**

Nothing.

<a id="ScopeSession.protocol_size_advisory"></a>
### `ScopeSession.protocol_size_advisory`

```python
def protocol_size_advisory(self, protocol: Protocol) -> ProtocolSizeAdvisory | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/protocol_size_advisory`.

Ask a protocol whether it is large enough to warn the user about.

The composer a client uses, rather than `Protocol.size_advisory`
itself: the two settings the estimate needs (whether video is saved
as frames, and the global FPS cap) are the session's, resolved the
way the run path resolves them, so the advisory and the run it is
advising about cannot be sized differently.

**Raises**

Nothing.

<a id="ScopeSession.put_live_folder_file"></a>
### `ScopeSession.put_live_folder_file`

```python
def put_live_folder_file(self, name: str, data: bytes) -> Path
```

Marks: in-process.

Write `data` as a new file under the live folder: a wire client's upload.

`name` is a name under the live folder, as `live_folder_path`
takes it, and the folders on its way are made. A file already there
is refused, so an upload never replaces a file a capture wrote. On
the wire it is `PUT /api/v1/files/<name>`, the body the file's
bytes; a Python caller writes its own files.

**Returns**

The file written.

**Raises**

- `LiveFolderPathRefusedError`: `'outside_live_folder'` and `'capture_location_unusable'`, as `live_folder_path` raises them; `'file_exists'`, a file is already at the name; `'not_a_folder'`, a file is where one of its folders would be.
- `LiveFolderFileNotWrittenError`: The disk refused the write; nothing is left at the name.

**Refusals**

- [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `capture_location_unusable`, `outside_live_folder`, `file_exists`, `not_a_folder`
- [`LiveFolderFileNotWrittenError`](#LiveFolderFileNotWrittenError): `live_folder_file_not_written`

<a id="ScopeSession.recover_file_writer"></a>
### `ScopeSession.recover_file_writer`

```python
def recover_file_writer(self) -> int
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/recover_file_writer`.

Give up on a finished run's unwritten images and unlock a stuck writer.

When a protocol run's file writer stops making progress, the run
engine reports the stall once (`FileWriterStalledError`) and
every subsequent run is refused with the `files_writing_stalled`
reason until the writer is recovered or the app restarts; both
carry this method as their remedy, and any caller may call it by
name. The run's outstanding images are given up on and counted,
and the worker stuck mid-write is abandoned and replaced. Nothing
else queued on the lane is discarded.

**Returns**

How many of the run's images were given up on.

**Raises**

- `HardwareCommandRefusedError`: a run, a diagnostic or a home holds the scope: the writer is recovered only while the scope is held by nothing, so no live run's captures are given up on.
- `FileWriterNotStuckError`: no write has stopped making progress; the writer will finish on its own, and recovering would lose images for nothing.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`, `capture_in_flight`, `home_in_flight`, `home_not_stoppable`, `position_unread`, `lid_open_before_motion`, `stage_unpowered_before_motion`, `camera_not_streaming`; [`FileWriterNotStuckError`](#FileWriterNotStuckError): `file_writer_not_stuck`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `not_on_this_scope`, `activity_ended`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`, `monitor_ended`, `not_connected`

<a id="ScopeSession.remove_outcome_listener"></a>
### `ScopeSession.remove_outcome_listener`

```python
def remove_outcome_listener(self, listener: Callable[[Notification], None]) -> None
```

Marks: in-process.

Stop `listener` hearing outcomes; a listener never added is a no-op.

**Raises**

Nothing.

<a id="ScopeSession.remove_run_state_listener"></a>
### `ScopeSession.remove_run_state_listener`

```python
def remove_run_state_listener(self, listener: Callable[[], None]) -> None
```

Marks: in-process.

Stop telling `listener` of run-state transitions; a listener never added is a no-op.

**Raises**

Nothing.

<a id="ScopeSession.remove_settings_listener"></a>
### `ScopeSession.remove_settings_listener`

```python
def remove_settings_listener(self, listener: Callable[[tuple[str, ...]], None]) -> None
```

Marks: in-process.

Stop telling `listener`; a listener never added is a no-op.

**Raises**

Nothing.

<a id="ScopeSession.remove_shown_channel_listener"></a>
### `ScopeSession.remove_shown_channel_listener`

```python
def remove_shown_channel_listener(self, listener: Callable[[str, bool], None]) -> None
```

Marks: in-process.

Stop telling `listener`; a listener never added is a no-op.

**Raises**

Nothing.

<a id="ScopeSession.rename_step"></a>
### `ScopeSession.rename_step`

```python
def rename_step(self, protocol: Protocol, step_index: int, name: str) -> str
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/rename_step`.

Give step `step_index` of `protocol` the label `name`; returns its new name.

**Raises**

- `StepNotFoundError`: `'no_such_step'`, `step_index` is not a step of `protocol`. The step keeps its name.
- `StepEditRefusedError`: `'step_value_refused'`, `name` has no letter, digit, dash or underscore. The step keeps its name.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`StepNotFoundError`](#StepNotFoundError): `no_such_step`; [`StepEditRefusedError`](#StepEditRefusedError): `step_value_refused`

<a id="ScopeSession.repair_camera_driver"></a>
### `ScopeSession.repair_camera_driver`

```python
def repair_camera_driver(self) -> tuple[UsbIdentityBindings, ...]
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/repair_camera_driver`.

Install LumaViewPro's driver for a Lumascope Classic and bind its USB identities to it.

Offered as the remedy of the bring-up's `camera_driver_unusable`
on an installed Windows build that carries the Classic driver, and
callable by name from any client. It runs the installed driver step
as an administrator, on the scope PC's desktop: Windows asks there
for an administrator's approval, then whether to install software
from Etaluma (unless the person chose to always trust it), so someone
at the scope answers whoever asked. The step logs every action to the
install folder's `logs`. A camera that is up is never touched:
Repair runs only while no camera is connected and a present device is
on a driver LumaViewPro cannot use, and the camera connects on its new
driver at the next start, which the notice it reports says.

**Returns**

tuple[UsbIdentityBindings, ...]: The bindings read after the step, as `scope.diagnostics.get_camera_usb_bindings`.

**Raises**

- `CameraDriverRepairRefusedError`: `'repair_needs_windows'`, `'repair_needs_installed_build'`, `'repair_needs_driver_package'` (built without the Classic driver), `'no_classic_bindings'`, `'driver_already_usable'` (a camera is up, or every present device is on a driver LumaViewPro can use), `'repair_declined'` when the administrator's approval was not given, or `'driver_not_accepted'` when Windows did not take the package (declined at its question, or refused).
- `CameraDriverRepairFailedError`: The step could not start, did not finish, ended unsuccessfully, or left a present device on a driver LumaViewPro cannot use; the words name its log.
- `CameraUsbLookupFailedError`: The bindings could not be read.

**Refusals by REST status**

- 409: [`CameraDriverRepairRefusedError`](#CameraDriverRepairRefusedError): `repair_declined`, `driver_not_accepted`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraDriverRepairRefusedError`](#CameraDriverRepairRefusedError): `repair_needs_windows`, `repair_needs_installed_build`, `repair_needs_driver_package`, `no_classic_bindings`, `driver_already_usable`
- 500: [`CameraDriverRepairFailedError`](#CameraDriverRepairFailedError): `camera_driver_repair_failed`; [`CameraUsbLookupFailedError`](#CameraUsbLookupFailedError): `camera_usb_lookup_failed`

<a id="ScopeSession.retire_rejected_settings"></a>
### `ScopeSession.retire_rejected_settings`

```python
def retire_rejected_settings(self) -> Path | None
```

Route: `POST /api/v1/retire_rejected_settings`.

Resolve the provisional-settings state: retire the rejected file.

Moves the unusable current.json aside (renamed, never deleted --
it is the user's only copy) so a fresh one can take its place,
and clears the provisional state, so the session saves again: at
its next accepted write, and at its close. Call only
after a human has chosen to start over. Returns the retired
path, or None when nothing was provisional.

**Raises**

- `SettingsFileNotReplacedError`: the rejected file could not be moved aside. The settings stay provisional, so the question can be answered again.

**Refusals by REST status**

- 500: [`SettingsFileNotReplacedError`](#SettingsFileNotReplacedError): `settings_file_not_replaced`

<a id="ScopeSession.run_application_session"></a>
### `ScopeSession.run_application_session`

```python
def run_application_session(
    self,
    *,
    disable_homing: bool = False,
    home_fn: Callable[[HomeTarget], object] | None = None,
    turret_fn: Callable[[int], object] | None = None,
) -> None
```

Marks: in-process.

Run the standard startup motion, and wait for it.

`start_application_session`, waited for: returns when the startup
motion has ended and the scope is free, and raises what its
operation would. Headless / REST callers use this exact call to apply the
standard startup orchestration without copy-pasting from the App.

**Raises**

- `BlockingCallRefusedError`: `'blocking_on_worker'`, called from an executor's worker. The motion runs on the diagnostics worker and waits on the io lane, so a worker waiting for it can be waiting on itself; `start_application_session` is the form a worker uses. Still a `RuntimeError`. `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_application_session` too. Nothing was started.
- `HardwareCommandRefusedError`: as `start_application_session`.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`, `capture_in_flight`, `home_in_flight`, `home_not_stoppable`, `position_unread`, `lid_open_before_motion`, `stage_unpowered_before_motion`, `camera_not_streaming`, `axis_absent`, `not_on_this_scope`, `activity_ended`, `scope_disconnected`, `monitor_ended`, `not_connected`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `wait_on_ui_thread`

<a id="ScopeSession.save_all_bookmarks"></a>
### `ScopeSession.save_all_bookmarks`

```python
def save_all_bookmarks(self) -> PlateLocation
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/save_all_bookmarks`.

Save the live Z as the Z bookmark and as the focus of every layer this scope has.

What Set All Bookmarks does. Returns the Z bookmark stored, as
`bookmark` reads it; X and Y, which it does not save, are None.

**Raises**

- `ProtocolRunRefusedError`: `'positions_unreachable'` -- this scope has no Z axis. Nothing is written.
- `AxisStateUnknownError`: Z lost its reference. Nothing is written.
- `AxisMovingError`: `'axis_moving'`, Z is moving, so its number is a place it is passing through. Nothing is written.

**Refusals by REST status**

- 409: [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `positions_unreachable`

<a id="ScopeSession.save_bookmark"></a>
### `ScopeSession.save_bookmark`

```python
def save_bookmark(self, axes: Iterable[StageAxis]) -> PlateLocation
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/save_bookmark`.

Save where the stage is on `axes` as the bookmark, as the bookmark buttons do.

X and Y are stored in plate millimetres on the selected plate, Z in
micrometres: the frame the go-to-bookmark moves read them in. The one
writer of `bookmark`.

**Returns**

The values stored, as `bookmark` reads them; an axis not in `axes` is None.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, a name in `axes` that is not X, Y or Z. Nothing is written.
- `AxisStateUnknownError`: `'position_unknown'`, an axis it reads does not know its position, so the number it reports is not one to save: the axes in `axes`, X and Y together. Nothing is written.
- `AxisMovingError`: `'axis_moving'`, an axis it reads is moving, so its number is a place it is passing through: the axes in `axes`, X and Y together. Nothing is written.
- `HardwareCommandRefusedError`: `'not_connected'` -- the motor controller is not connected; `'axis_absent'` -- this scope has no motor for an axis in `axes`, so there is no place on it to save. Nothing is written.

**Refusals by REST status**

- 409: [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="ScopeSession.save_channel_focus"></a>
### `ScopeSession.save_channel_focus`

```python
def save_channel_focus(self, channel: str, z_um: float) -> None
```

Marks: sets [`saved_focus_z_um`](#ScopeSession.saved_focus_z_um); enters: `guarded_write`.  
Route: `PUT /api/v1/saved_focus_z_um`.

Store `z_um` as `channel`'s focus, the Z every new step of the layer is born at.

The door for a caller that holds a number rather than a stage: the
Autofocus button saving the Z its run chose, a script saving one it
measured. It says only that the number is a Z this scope's focus can
reach; it guarantees nothing about where the number came from.
`save_focus`, which saves the live Z, writes through the same path.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no `channel`.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved. Nothing is written.
- `ProtocolRunRefusedError`: `'positions_unreachable'` -- this scope has no Z axis. Nothing is written.
- `PositionOutOfRangeError`: `z_um` is NaN or infinite, or lies outside Z's travel. Nothing is written.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `positions_unreachable`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`

<a id="ScopeSession.save_focus"></a>
### `ScopeSession.save_focus`

```python
def save_focus(
    self,
    protocol: Protocol,
    channel: str,
    *,
    step_index: int | None = None,
) -> SavedFocus
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/save_focus`.

Save the live Z as `channel`'s focus, and as step `step_index`'s Z.

The layer's focus is what every new step of the layer is born at.
The step takes the Z only when it is a step of `channel`; a step of
another channel is left alone. No other step is written: every step
of a layer is born at the same focus, so a step matching the old
focus says nothing about whether its user wants the new one.
`apply_focus_to_channel_steps` writes them all.

**Raises**

- `ProtocolRunRefusedError`: `'positions_unreachable'` -- this scope has no Z axis. Nothing is written.
- `AxisStateUnknownError`: Z lost its reference. Nothing is written.
- `AxisMovingError`: `'axis_moving'`, Z is moving, so its number is a place it is passing through. Nothing is written.
- `StepNotFoundError`: `'no_such_step'`, `step_index` is not a step of `protocol`. Nothing is written.
- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no `channel`.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved. Nothing is written.

**Refusals by REST status**

- 409: [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `positions_unreachable`; [`StepNotFoundError`](#StepNotFoundError): `no_such_step`; [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`

<a id="ScopeSession.save_protocol"></a>
### `ScopeSession.save_protocol`

```python
def save_protocol(self, protocol: Protocol, file_path: FilePath) -> Path
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/save_protocol`.

Write a protocol to a file, with this session's Layer Settings.

The file is written under exactly `file_path`, which ends in
`.tsv` (in any case); nothing is added to it. The block holds
each layer set to acquire an image or a video, with the values its
controls hold now; `apply_channel_settings` puts them back. What the
GUI opens at its next start-up is not changed: a script's scratch
save is not the person's protocol.

**Returns**

The path written; a missing folder is made.

**Raises**

- `LiveFolderPathRefusedError`: `'not_a_file'`, a folder is at `file_path`; `'wrong_extension'`, its name does not end in `.tsv`; `'not_a_folder'`, a file is where its folder would be. Nothing is written.
- `ProtocolNotSavedError`: The file could not be written; a file already at the path is unchanged.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `not_a_file`, `wrong_extension`, `not_a_folder`
- 500: [`ProtocolNotSavedError`](#ProtocolNotSavedError): `protocol_not_saved`

<a id="ScopeSession.saved_focus_z_um"></a>
### `ScopeSession.saved_focus_z_um`

```python
def saved_focus_z_um(self, channel: str) -> float
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/saved_focus_z_um`.

The Z saved as `channel`'s focus.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.
- `FocusNotSavedError`: no focus was ever saved for `channel`; a step built for it takes the current Z instead.

**Refusals by REST status**

- 409: [`FocusNotSavedError`](#FocusNotSavedError): `focus_not_saved`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`

<a id="ScopeSession.scale_bar_enabled"></a>
### `ScopeSession.scale_bar_enabled`

```python
scale_bar_enabled: bool  # property
```

Route: `GET /api/v1/scale_bar_enabled`.

Whether captured images are drawn with the scale bar, as `set_scale_bar` set it.

**Raises**

Nothing.

<a id="ScopeSession.select_labware"></a>
### `ScopeSession.select_labware`

```python
def select_labware(self, labware_id: str) -> bool
```

Marks: sets [`labware_id`](#ScopeSession.labware_id); enters: `guarded_write`.  
Route: `PUT /api/v1/labware_id`.

Make `labware_id` the current plate. Returns whether it changed.

The one writer of the active labware for every host: the plate the
settings name, which the scope reads.

The name is stored in the catalogue's spelling: a plate renamed
since a protocol or settings file named it is accepted under the old
name and written under the key, so the settings store never carries
a spelling the catalogue lacks, and whether the plate changed is
decided on the key rather than on how it was spelled.

On a scope with no XY stage the one plate is Center Plate: another
plate's wells are places it cannot move between, and a protocol made
on one would image one field under every well's name.

**Raises**

- `CatalogueNameRefusedError`: `'labware_unknown'`, the labware catalogue has no plate by that name. Nothing is written.
- `HardwareCommandRefusedError`: `'axis_absent'`, this scope has no XY stage and `labware_id` is not Center Plate (`missing` is `MissingPart.XY_STAGE`). A run, a diagnostic or a recording holds the scope and `labware_id` is not the plate in place (`'exclusive_activity_running'`): each states its positions against the plate it started with. Nothing is written.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`

<a id="ScopeSession.select_model"></a>
### `ScopeSession.select_model`

```python
def select_model(self, model: str) -> None
```

Marks: sets [`model`](#ScopeSession.model); enters: `guarded_write`.  
Route: `PUT /api/v1/model`.

Save the operator's scope model for the next start.

The running scope is not changed: its capabilities are fixed at
construction, so re-resolving its layers for the new model would
leave one scope carrying two models -- the new one in its layers,
the old one in its capabilities and its saved files. The selection
takes effect when the scope is next brought up, and on a scope whose
motor board reports its own model the board's report still wins
there.

A model that is not the one running is reported as a
`ScopeModelDeferredNotice`, so every client hears that it waits
for the next start.

**Args**

- `model`: A model the release's catalogue lists.

**Raises**

- `ScopeModelUnknownError`: the scope's catalogue does not list `model`; nothing is saved.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ScopeModelUnknownError`](#ScopeModelUnknownError): `model_unknown`

<a id="ScopeSession.select_objective"></a>
### `ScopeSession.select_objective`

```python
def select_objective(self, objective_id: str) -> bool
```

Marks: sets [`objective_id`](#ScopeSession.objective_id); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/objective_id`.

Make `objective_id` the active objective. Returns whether it changed.

The one writer of the active objective for every host. With no
turret, the selected objective is the `objective_id` setting, which
the scope reads. On a turret scope the active objective is the
assignment of the slot in the light path, so picking one turns the
turret to the one slot holding it (`move_turret`, Z parked and
restored) and waits for it; which glass is in which slot is
`assign_turret_objective`'s, and a pick writes no slot. The resolved
optics are recorded by the scope's runtime state the next time the
objective is read, before any capture stamps it. Picking the
objective already active is a no-op.

**Raises**

- `CatalogueNameRefusedError`: `'objective_not_in_catalogue'`, `objective_id` is not exactly a catalogue key. The refusal lands before any write or move.
- `ObjectiveUnknownError`: `'none_selected'`, `objective_id` is None.
- `ArgumentRefusedError`: On a turret scope, `'objective_in_several_slots'`, two or more slots hold `objective_id` (`offered`: the slots), or `'objective_not_on_turret'`, none does (`offered`: what the slots hold). Nothing is moved.
- `HardwareCommandRefusedError`: A run, a diagnostic or a recording holds the scope (`'exclusive_activity_running'`), nothing written or moved; or the turret move's own refusal, `'not_connected'` or `'axis_absent'`, nothing moved.
- `AxisStateUnknownError`: The turret has not been homed.
- `MoveFaultError`: The Z park, the turret move or the Z restore did not arrive (see `MotionAPI.start_move_z`); the slot is unknown afterwards.
- `MoveStoppedError`: `'stopped'` or `'superseded'`, the Z park, the turret move or the Z restore was ended; the slot is unknown afterwards.

**Refusals by REST status**

- 409: [`ObjectiveUnknownError`](#ObjectiveUnknownError): `none_selected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `objective_not_in_catalogue`; [`ArgumentRefusedError`](#ArgumentRefusedError): `objective_in_several_slots`, `objective_not_on_turret`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `faulted`, `stalled`, `monitor_ended_mid_move`, `board_lost`, `position_unread_after_move`, `status_unread`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ScopeSession.set_acceleration_limit"></a>
### `ScopeSession.set_acceleration_limit`

```python
def set_acceleration_limit(self, limit_pct: int) -> None
```

Marks: sets [`acceleration_limit_pct`](#ScopeSession.acceleration_limit_pct); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/acceleration_limit_pct`.

Set the motors' acceleration limit, as a percent of the firmware's maximum, and store it.

The one writer of `motion.acceleration_max_pct`, stored once the
motor controller took the value.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller to take it. Nothing is stored.
- `AccelerationLimitRefusedError`: `limit_pct` is outside the range the motion API accepts (a ValueError). Nothing is stored.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`AccelerationLimitRefusedError`](#AccelerationLimitRefusedError): `acceleration_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ScopeSession.set_channel_acquire"></a>
### `ScopeSession.set_channel_acquire`

```python
def set_channel_acquire(self, channel: str, mode: AcquireMode | None) -> None
```

Marks: sets [`channel_acquire`](#ScopeSession.channel_acquire); enters: `guarded_write`.  
Route: `PUT /api/v1/channel_acquire`.

Set what a layer captures: an `AcquireMode`, or None (nothing).

The layers set to acquire are the ones `new_protocol` and
`add_step` build steps for and a composite merges. A layer set to
acquire stops stimulating: one layer does not capture and
stimulate at once.

Nothing is changed on any refusal. Setting a layer to acquire
nothing is always admitted.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel; `'not_offered'`, `mode` is not an `AcquireMode` or None.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, `mode` is not None and this scope does not have `channel`.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`, `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`

<a id="ScopeSession.set_channel_auto_gain"></a>
### `ScopeSession.set_channel_auto_gain`

```python
def set_channel_auto_gain(self, channel: str, enabled: bool) -> AutoGainLock | None
```

Marks: sets [`channel_auto_gain`](#ScopeSession.channel_auto_gain); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/channel_auto_gain`.

Turn a layer's auto-gain on or off, as the GUI's Auto Gain/Exp box does.

For the shown layer the camera follows when this returns: turning it
on arms the camera's auto loop (`apply_layer_camera`); turning it
off locks the standing arm (`scope.camera.lock_auto_gain`) and
stores what the camera reached as the layer's manual setting: the
lock's `gain_db` and its `stored_exposure_ms`, rounded to 0.1 dB
and 0.01 ms, the resolution the stored settings carry, each only when
the camera reported it; with no arm standing, or a lock that found
nothing usable, they are left as they were. Either waits on the
camera lane. For a layer the camera does not hold, only the
preference is stored: the camera's arm is another layer's, and its
values are not this layer's to take. While a run, a diagnostic or a
home holds the scope, turning auto-gain on is stored only, and
turning the shown layer's off is refused.

Nothing is changed on any refusal.

**Returns**

The lock when turning off (its `state` is None when no arm stood), so a caller can read the state the camera reached, and `channel_auto_gain_lock` answers it from then on; None when turning on.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, `enabled` is True and this scope does not have `channel`; or the lock's `'exclusive_activity_running'`, when a run, a diagnostic or a home holds the scope.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`, `activity_ended`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ScopeSession.set_live_folder"></a>
### `ScopeSession.set_live_folder`

```python
def set_live_folder(self, folder: str) -> None
```

Marks: in-process.

Make `folder` the live folder, where captures and runs are saved.

The one writer of `live_folder`, stored as it is at load: a folder
given relative to the installation is made absolute, and created.

**Raises**

- `SettingRefusedError`: `'out_of_range'`, `folder` is a string no file system can name (a NUL byte), or a folder that cannot be created (a file in the way, no permission, a drive that is not there). Nothing is written.

**Refusals**

- [`SettingRefusedError`](#SettingRefusedError): `out_of_range`

<a id="ScopeSession.set_protocol_capture_root"></a>
### `ScopeSession.set_protocol_capture_root`

```python
def set_protocol_capture_root(self, protocol: Protocol, root: str) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/set_protocol_capture_root`.

Give `protocol` the capture root that prefixes its runs' file names.

The protocol keeps `root` as given; the prefix a run's names take
from it is `protocol.capture_prefix()`.

**Raises**

Nothing.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`

<a id="ScopeSession.set_protocol_filepath"></a>
### `ScopeSession.set_protocol_filepath`

```python
def set_protocol_filepath(self, file_path: str) -> None
```

Marks: in-process.

Remember `file_path` as the protocol to open at the next start; `''` forgets it.

The one writer of `protocol.filepath`, which the next start opens.

**Raises**

Nothing.

<a id="ScopeSession.set_protocol_labware"></a>
### `ScopeSession.set_protocol_labware`

```python
def set_protocol_labware(self, protocol: Protocol, labware_id: str) -> str
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/set_protocol_labware`.

Put `protocol` on the plate `labware_id`; returns the key it took.

The protocol's plate only: the scope's is `select_labware`'s. On a
scope with no XY stage the protocol takes Center Plate.

**Raises**

- `CatalogueNameRefusedError`: `'labware_unknown'`, `labware_id` is not a plate the catalogue has. The protocol keeps its plate.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`

<a id="ScopeSession.set_protocol_schedule"></a>
### `ScopeSession.set_protocol_schedule`

```python
def set_protocol_schedule(
    self,
    protocol: Protocol,
    *,
    period_s: float,
    duration_s: float,
) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/set_protocol_schedule`.

Give `protocol` a new period and duration in seconds, as its two fields do.

One scan is 0 for both.

**Raises**

- `ProtocolScheduleRefusedError`: either cannot be run; the protocol keeps the schedule it had.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolScheduleRefusedError`](#ProtocolScheduleRefusedError): `schedule_not_runnable`

<a id="ScopeSession.set_scale_bar"></a>
### `ScopeSession.set_scale_bar`

```python
def set_scale_bar(self, enabled: bool) -> None
```

Marks: sets [`scale_bar_enabled`](#ScopeSession.scale_bar_enabled); enters: `guarded_write`.  
Route: `PUT /api/v1/scale_bar_enabled`.

Draw the scale bar on captured images, or stop, and store it.

The one writer of `scale_bar.enabled`, which the imaging API reads
at each capture.

**Raises**

Nothing.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`

<a id="ScopeSession.set_step_autofocus"></a>
### `ScopeSession.set_step_autofocus`

```python
def set_step_autofocus(
    self,
    protocol: Protocol,
    *,
    enabled: bool,
    step_index: int | None = None,
) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/set_step_autofocus`.

Turn autofocus on or off for step `step_index` of `protocol`, or for every step.

**Raises**

- `StepNotFoundError`: `step_index` is not a step of `protocol`. Nothing changes.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`StepNotFoundError`](#StepNotFoundError): `no_such_step`

<a id="ScopeSession.set_ui_dispatcher"></a>
### `ScopeSession.set_ui_dispatcher`

```python
@staticmethod
def set_ui_dispatcher(dispatcher: UiDispatcher | None) -> None
```

Marks: in-process.

Set how the process hands a callback to its UI thread.

A process has one UI thread, so this is one setting for the process,
not for a session: every lane's completion callback, every run
delivery and every listener the GUI marshals reads it at the moment
it dispatches, whichever session or scope it belongs to. A GUI host
sets it once, before it builds a session; a host with no UI thread
(a script, a REST server) never sets it. None (the default) calls
each callback directly on the thread that dispatches it, and a
callback's raise is reported rather than raised.

`dispatcher.thread` is the thread `schedule` delivers on: a
run's `wait()` on that thread is refused, since what it waits for
is delivered there.

**Raises**

Nothing.

<a id="ScopeSession.settings_are_provisional"></a>
### `ScopeSession.settings_are_provisional`

```python
def settings_are_provisional(self) -> bool
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/settings_are_provisional`.

Is the app running on defaults nobody has agreed to keep?

True while the user's current.json could not be used and no one
has decided its fate. While it holds, accepted writes are kept in
memory and not saved (`SettingsNotSavedNotice`) -- resolve with
retire_rejected_settings() after the user has chosen to start
over.

**Raises**

Nothing.

<a id="ScopeSession.show_channel"></a>
### `ScopeSession.show_channel`

```python
def show_channel(self, channel: str) -> ShownChannel
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/show_channel`.

Show `channel`: every other LED dark, `channel` lit at its stored
current if the shown channel was lit, then its camera settings.

The one way the shown channel changes outside a run; a run's steps
and `go_to_step` move it too. Each refusal is decided before any
LED is turned off, so a refused call leaves the LEDs as they were.
A layer this unit gives no LED is shown with every LED dark.

**Returns**

The shown channel now.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel; `'illumination_out_of_range'`, its stored current is outside the attached board's range.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no `channel`; `'not_connected'`, naming the LED controller the light needs or the camera; or `'exclusive_activity_running'`, a run, a diagnostic or a home holds the scope, or a manual recording runs and `channel` is not the shown one (a recording keeps one layer; nothing is changed).
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved.
- `CameraSettingRejected`: the camera refused a setting of the layer; its LEDs are already shown.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`, `illumination_out_of_range`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`, `activity_ended`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`; [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ScopeSession.shown_channel"></a>
### `ScopeSession.shown_channel`

```python
shown_channel: ShownChannel  # property
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/shown_channel`.

The channel the scope shows: its layer, and whether its LED is lit.

**Raises**

Nothing.

<a id="ScopeSession.shutdown"></a>
### `ScopeSession.shutdown`

```python
def shutdown(self) -> None
```

Marks: in-process.

Finish what the session is doing, then tear down everything it constructed.

First the close: from its first step nothing new takes the scope (a
run, a recording, a home or a diagnostic asked for meanwhile, or
still taking the scope once it has finished, raises
`SessionClosingError`); plugins unload, so a plugin's own work
stops through its `unregister`, and metrics stop; the live run is
stopped (its ending `aborted` / `shutdown`) and a live recording
is stopped. Then it waits, without a time limit, until
`live_work` lists nothing: the run's writes and post-run builds,
the recording's file, post-processing builds running or queued, a
support report, a still. `discard_close_drain` is the escape from
waiting on video frames, and any client may call it meanwhile. A
finished run's file writer judged stuck (`protocol_files_stalled`)
is recovered by the close itself, its unwritten images given up on,
counted and reported. While it waits the close logs what it waits for.

Then the teardown. The bundle the session holds always stops: its long-lived consumer
threads first, then the FILE lane and the worker pool. When
`owns_scope` (a factory BUILT the scope), the hardware half
follows: the LEDs drained through the io lane while its worker is
alive, then the scope disconnected, which shuts its IO and CAMERA
lanes before it turns the LEDs off and stops motion inline -- so a
headless host gets the same teardown the GUI gets. The consumers
stop BEFORE the lanes they consume (a consumer mid-iteration that
finds its lane already shut can hang on a dispatch that never
fires; scope_display_thread consumes camera_executor,
protocol_thread drives io + camera + file). Running metrics stop
first, or their ticks would outlive the executors they snapshot.

A scope handed to the constructor is left connected with its lanes
running: it is the caller's. Callable from any thread, a run's `run_ended`
handler included. One close runs: a call on another thread while it
runs waits for it and returns; a call after it is a logged no-op; a
call from inside the close returns at once; a call that raised
part-way can be called again.

**Raises**

- `ScopeDisconnectError`: a part of an owned scope did not shut down cleanly; every teardown step has still run, and a second call completes.

**Refusals**

- [`ScopeDisconnectError`](#ScopeDisconnectError): `disconnect_failed`

<a id="ScopeSession.source_path"></a>
### `ScopeSession.source_path`

```python
source_path: Path  # property
```

Route: `GET /api/v1/source_path`.

The data folder this session runs on: its scope's, never a copy.

**Raises**

Nothing.

<a id="ScopeSession.start_application_session"></a>
### `ScopeSession.start_application_session`

```python
def start_application_session(
    self,
    *,
    disable_homing: bool = False,
    home_fn: Callable[[HomeTarget], object] | None = None,
    turret_fn: Callable[[int], object] | None = None,
) -> Operation
```

Marks: in-process; stopped by [`Operation.stop`](#Operation.stop).

Start the standard startup motion without waiting for it; returns its `Operation`.

The one implementation of the startup motion for every host:
the App's launch and its reconnect handler once open-coded the
same ALL-axis home + turret-positioning pair, and the two
copies drifted.

The motion, in order:

1. home ALL axes via `move_home`. Firmware homes Z, T, X, Y in
   one routine; on Z-only boards it homes what it has and
   reports the missing axes.

2. (when `self.scope.capabilities.has_turret` is True) move T
   to position 1; the active objective is then slot 1's
   assignment.

3. on the simulator, place the stage at its sample plane.

The scope is the startup's from before this returns until the motion
ends, as it is a home's: every other request, from any client, is
refused naming the home, and the controls lock. So a turret pick made
meanwhile is refused rather than undone by step 2. Asked under a live
taking, the motion is that activity's work and takes nothing. The
claim is released before the operation is done, so a caller told
the motion has ended finds the scope free. Its `stop()` is
`stop_motion` while the motion drives, refused `'home_not_stoppable'`
while the home drives on a board that cannot stop one -- the home
runs to its end and nothing after it moves -- and nothing once the
motion has ended.

A failed home is reported (`report_outcome`) and ends the motion
without the turret move -- an absolute move against a reference the
home did not establish; the operation settles with None. A refusal
that is not for the hardware's state is a defect, and settles the
operation with it.

`disable_homing=True` skips every step: no startup motion on any
axis. The turret is left where it is, like the stage axes, in no
known slot -- positioning it without a home would be an absolute
move against a reference the caller asked us not to establish. The
skip is the requested behaviour, so it is logged, not signalled.

A scope whose model has no motor board (`scope.motion_expected`
False) has nothing to home, so it issues no startup motion either;
nor does one whose motor board bring-up recorded as missing, which
bring-up's own report has already named. A skip takes nothing and
returns an operation already done.

**Args**

- `disable_homing`: If True, issue no startup motion at all.
- `home_fn`: Callable taking a `HomeTarget` that homes it and raises as `MotionAPI.home` does; the startup motion hands it `HomeTarget.ALL`. Defaults to the motion API.
- `turret_fn`: Callable taking a turret slot. Defaults to the motion API.

**Raises**

- `HardwareCommandRefusedError`: `'home_in_flight'` or `'exclusive_activity_running'`: something already holds the scope. Nothing was taken.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `home_in_flight`, `exclusive_activity_running`

<a id="ScopeSession.start_go_to_bookmark"></a>
### `ScopeSession.start_go_to_bookmark`

```python
def start_go_to_bookmark(self, axes: Iterable[StageAxis]) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/start_go_to_bookmark`.

Start `axes` moving to the bookmark, as the go-to-bookmark buttons do; return the started moves.

Reads `bookmark` for the axes given, as `save_bookmark` wrote
them: X and Y in mm on the selected plate (`start_move_to_plate`,
the axes given on one lane task), Z in um (`start_move_z`).
An axis not given is left where it is.

**Returns**

One `Operation` for the moves started, done once every one is (`Operation.all_of`).

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, a name in `axes` that is not X, Y or Z. Nothing was driven.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'lid_open_before_motion'` or `'monitor_ended'`, as `MotionAPI.start_move_to_plate`. Nothing was driven.
- `PositionOutOfRangeError`: The bookmark is outside what this stage can reach. Nothing was driven.
- `AxisStateUnknownError`: An axis given does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="ScopeSession.start_go_to_step"></a>
### `ScopeSession.start_go_to_step`

```python
def start_go_to_step(self, protocol: Protocol, step_index: int) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`, `guarded_write`.  
Route: `POST /api/v1/start_go_to_step`.

Start going to step `step_index` of `protocol`, as a click on a step does.

One task on the scope's IO lane: the axes are asked once whether
they know their position; the turret turns to the step's objective
and X, Y and Z are started towards the step together, on the
protocol's own plate (`ProtocolsAPI.step_targets`, the targets a
run computes); the step's values go into its layer's live settings,
the layer acquiring as the step does and its focus at the step's Z,
so the layer and the step agree; and the step's LED preview is
applied -- its channel at its current when `protocol_led_on` is
set, every channel dark when not. It returns once that task has run,
before the stage arrives, with one `Operation` for the moves it
started, X and Y as one request (`MotionAPI.start_move_xy`) and
then Z: done once every one has ended, its `result()` returns None
when all arrived and raises the first that did not, in X, Y, Z
order, as `MotionAPI.start_move_z` says; its
`stop()` stops the moves still under way. `go_to_step` waits on
it. A person's click is a gesture and does not wait; a fault on the
way is the motion monitor's to report. Only the axes this scope has
are moved: a manual scope moves nothing, does the rest and returns
an operation already done.

A repeat of the step this session last went to (a re-click, a
re-typed number) does everything but the preview: a channel the
person lit or put out in between stays as they left it, and when
another layer was shown meanwhile the step's layer is shown again,
lit or dark as the shown channel was.

Once that task has run, the step's layer goes on the camera
(`apply_layer_camera`) before this returns, so a still asked after
it is taken at the step's settings. It waits on the camera lane, so
it is refused on the UI thread; a click calls it from a worker.

**Raises**

- `StepNotFoundError`: `step_index` is not a step of `protocol`. Nothing changes.
- `ProtocolRunRefusedError`: `'values_out_of_range'`, a value the step would store in its layers is outside its layer setting's range, each named by column and bound. Nothing changes.
- `ArgumentRefusedError`: `'channel_unknown'`, the step's stimulation names a layer this release has not. Nothing changes.
- `HardwareCommandRefusedError`: `'not_connected'`, this scope's motor controller or camera is not connected, or its LED controller is not and the step's preview would light; or a run, a diagnostic or a home holds the scope, or a manual recording runs and the step's layer is not the shown one. Nothing changes.
- `ConfigurationDefectError`: `'layers_unresolved'`, this scope's layers could not be resolved. Nothing changes.
- `AxisStateUnknownError`: an axis the step moves does not know its position. Nothing changes.
- `PositionOutOfRangeError`: the step lies outside an axis's travel; the axes before it have moved, nothing else changes.
- `MoveFaultError`: `'driver_failed'`, the board did not take an axis's command; or the turret's move to the step's slot did not arrive; see `MotionAPI.start_move_z`.
- `MoveStoppedError`: `'stopped'` or `'superseded'`, the turret's move to the step's slot was ended.
- `CameraSettingRejected`: the camera refused a setting of the step's layer; the moves are under way, and the layer and the preview are the step's.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread. Nothing was started.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`StepNotFoundError`](#StepNotFoundError): `no_such_step`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`; [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ScopeSession.status"></a>
### `ScopeSession.status`

```python
status: Status  # property
```

Route: `GET /api/v1/status`.

What the scope is doing: its work, its axes, its parts and its camera, in one read.

The read a client makes first and polls as its health check. Each
field comes from its owner, one after another rather than under one
lock (`Status`); no field does serial I/O.

**Raises**

- `RunCheckFailedError`: `'hardware_state_unknown'`, the connection read behind `run_refused` crashed instead of answering.

**Refusals by REST status**

- 500: [`RunCheckFailedError`](#RunCheckFailedError): `hardware_state_unknown`

<a id="ScopeSession.take_back_control"></a>
### `ScopeSession.take_back_control`

```python
def take_back_control(self) -> ControlSession | None
```

Marks: in-process.

End whatever control session is held, and stop the run or recording taken in it, as `end_control`.

The person at the scope taking it back without naming it -- the
GUI's Take back control press and its close, which hold no id. Kept
off the wire: a client ends only the session whose id it was told.

**Returns**

The session ended, or None when none was held.

**Raises**

Nothing.

<a id="ScopeSession.take_control"></a>
### `ScopeSession.take_control`

```python
def take_control(self, name: str) -> str
```

Route: `POST /api/v1/take_control`.

Take the one control session: from now on only work in it writes to the scope.

While it is held every other caller's write -- a lane command, a
setting, a file, a run, a recording, a home, a diagnostic -- is
refused, the display preferences aside; a stop never is. Work in the
session is a call made inside `acting_in_control` with its id, or
a REST request naming it. Taken only while nothing of anyone else's
is under way, so no one's work is ever put under a client's control:
no other session, no run, recording, home or diagnostic, no setting
or file being written, and no run or recording that has ended still
finishing its files.

**Args**

- `name`: What the person at the scope is told holds it.

**Returns**

The session's id, told to its taker and to no one else: what a call in the session names, and what ends it. `control` reads the session without it.

**Raises**

- `ControlSessionRefusedError`: `'control_session_held'`, another session is held; `'exclusive_activity_running'`, a run, a recording, a home or a diagnostic holds the scope; `'write_in_flight'`, a setting or a file is being written; `'work_finishing'`, a run or recording that has ended is still finishing, or a build is on the post-processing lane, which a capture in the session would overlap. Nothing was taken.
- `SessionClosingError`: `'session_closing'`, the session's close is running or has finished.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`, `exclusive_activity_running`, `write_in_flight`, `work_finishing`
- 503: [`SessionClosingError`](#SessionClosingError): `session_closing`

<a id="ScopeSession.tiling_config"></a>
### `ScopeSession.tiling_config`

```python
def tiling_config(self) -> TilingConfig
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/tiling_config`.

The tiling grids this installation offers: the scope's copy, read at bring-up.

What a caller needs to choose a grid: `available` lists the
labels `new_protocol` and `apply_tiling` accept, and
`default` is the one to preselect; `Protocol.tiling()` names
the grid a protocol's steps already carry. Read once with the
installation's other catalogues, so a tiling.json edited while the
program runs is seen at the next start, and every protocol is built
against the same grids. The record is frozen: a caller cannot change
the grids a session offers.

**Raises**

Nothing.

<a id="ScopeSession.turret_objective"></a>
### `ScopeSession.turret_objective`

```python
def turret_objective(self, slot: int) -> str | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/turret_objective`.

The objective bound to the turret's `slot`, or None while it holds none.

What `assign_turret_objective` bound; the active objective is the
slot's in the light path (`objective_id`).

**Raises**

- `HardwareCommandRefusedError`: `'axis_absent'`, the scope has no turret, said before `slot` is judged.
- `PositionOutOfRangeError`: `slot` is not a slot number 1-4.

**Refusals by REST status**

- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`

<a id="ScopeSession.turret_slot_questions"></a>
### `ScopeSession.turret_slot_questions`

```python
def turret_slot_questions(self, protocol: Protocol) -> list[TurretSlotQuestion]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/turret_slot_questions`.

What to ask about `protocol`'s turret slots: one question per objective, or none.

A step that cannot say which slot it means is one whose objective is
in two or more slots and that recorded none, or whose recorded slot
now holds another objective. Each is refused by a run and by going
to it until `point_steps_at_turret_slot` answers its question.
LumaViewPro asks when a protocol is opened. Empty on a scope with no
turret, and for steps whose objective no slot holds, which a load
and a run refuse as `turret_objectives_unassigned`.

**Raises**

Nothing.

<a id="ScopeSession.unload_plugins"></a>
### `ScopeSession.unload_plugins`

```python
def unload_plugins(self) -> None
```

Marks: in-process.

Call each loaded plugin's `unregister`, last loaded first.

A host whose close has to stop plugin work before anything else --
the GUI's, before it stops runs and saves settings -- calls this
first; `shutdown` calls it as its first step for every other host.
A second call, and a call on a session with no plugins, does
nothing. A plugin's own failure is logged and the rest unload.

**Raises**

Nothing.

<a id="ScopeSession.update_settings"></a>
### `ScopeSession.update_settings`

```python
def update_settings(self, path: str, value: SettingValue) -> None
```

Marks: enters: `lane_call`, `guarded_write`.  
Route: `POST /api/v1/update_settings`.

Write one setting, named by its dotted path: `'video.max_fps'`, `'BF.sum'`.

The one write to the live settings for any caller, on any thread.
Reads take `get_settings_snapshot`; a write that skips this can
tear a snapshot being taken concurrently, and takes none of the
checks below.

A setting that has its own member -- the objective, the plate, the
image mode, a layer's acquire mode or focus, the live folder, ... --
is changed only through that member, which checks it against the
scope or changes another setting with it; the refusal names the
member. A setting that decides how the next start reaches the scope
or the machine -- the REST server and its key, the start mode, the
profilers and debugging switches -- is read only from the
installation's settings file, so no caller can reconfigure the scope
it drives. A block (`'video'`) is not written whole: each of its
settings has a path.

**Raises**

- `SettingRefusedError`: `'has_member'`, `path` is owned by a member (named by its path from the session: `scope.camera.set_frame_size` for the camera's); `'host_only'`, its member is in-process, so only the application hosting the scope changes it (`live_folder`, `protocol.filepath`); `'installation_only'`, it is set only by the installation; `'derived'`, it is derived from the release (`step_channel_order`); `'not_a_setting'`, it is not a setting; `'block'`, it names a block; `'wrong_kind'`, `value` is not the setting's kind; `'out_of_range'`, it is outside its range (a `protocol.period` or `protocol.duration` no protocol can run, and a layer's illumination above the attached LED board's maximum, among them). Nothing is written.
- `ConfigurationDefectError`: `'layers_unresolved'`, the write is to the shown layer's camera settings and this scope's layers could not be resolved; the value is stored, and the camera is not changed.

A numpy scalar in `value`, at any depth, is stored as the Python
value it holds.

A layer's gain or exposure is stored as written, the person's
intent; a camera that cannot reach it is driven at its own limit
when the layer is applied (`CameraAPI.applied_gain_db_for`).
A write to the shown layer's gain or exposure (or, while it is on
auto-gain, to the auto-gain bounds) is on the camera, and to its
illumination on its lit LED, when this returns; a write to a hidden
layer, or one made while a run, a diagnostic or a home holds the
scope, is stored only. Such a write waits on the camera or LED lane,
so a caller on the UI thread hands it to a worker. Each auto-gain
bound is stored alone; a minimum left above the maximum is refused
when auto-gain is armed with them (`ArgumentRefusedError`,
`'gain_range_inverted'`), so this write raises it after the store
while the shown layer is on auto-gain, and the camera keeps its arm.
The log level and the autofocus score log (`'logging.*'`) take
effect as they are written.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`SettingRefusedError`](#SettingRefusedError): `has_member`, `host_only`, `installation_only`, `derived`, `not_a_setting`, `block`, `wrong_kind`, `out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ScopeSession.update_step"></a>
### `ScopeSession.update_step`

```python
def update_step(
    self,
    protocol: Protocol,
    step_index: int,
    *,
    channel: str,
    label: str | None = None,
) -> str
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/update_step`.

Rewrite a step of `protocol` from this session's settings and live position.

The entry point a caller with no GUI uses to do what Update Step
does: step `step_index` takes `channel`'s settings, the live stage
position on the protocol's plate and the current objective. The
protocols API performs the update, reads the objective, and refuses
it when the position or the objective is unknown; this composes its
other inputs from the session the same way `add_step` does.

Returns the step's name after the update.

**Raises**

- `AxisMovingError`: `'axis_moving'`, an axis is moving, so the live position is a place it is passing through. The step is unchanged.
- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, this scope has no `channel`. Nothing is changed on either.
- `ObjectiveUnknownError`: the objective in the light path is unknown, as `add_step` raises it (`'slot_unassigned'`, the turret's current slot has no objective). The step is unchanged.

**Refusals by REST status**

- 409: [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unassigned`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`

<a id="ScopeSession.user_settings_file"></a>
### `ScopeSession.user_settings_file`

```python
@staticmethod
def user_settings_file(source_path: FilePath) -> str
```

Marks: in-process.

The file the installation at `source_path` keeps its user configuration in.

For a host that saves accepted writes, to pass to `create`:
``create(settings, source_path=root,
settings_file=ScopeSession.user_settings_file(root))``, as the GUI
and the REST server do. `load_user_settings` reads this file
first. Nothing is read or created here.

**Raises**

Nothing.

<a id="ManualCaptureController"></a>
## `ManualCaptureController`

Reached through `session.manual_capture`; on the wire, `/api/v1/manual_capture/`.  
Held in: [`ScopeSession.manual_capture`](#ScopeSession.manual_capture).

One still at a time, captured on the camera lane and saved.

**Args**

- `scope`: The Lumascope instance.
- `settings_snapshot`: Returns a private copy of the session's settings; one is taken per capture, when the capture is asked for.
- `shown_layer`: Reads the layer of the session's shown channel, on the camera lane as the frame is taken: it names the channel when no LED is lit.
- `layer_camera_request`: Reads a layer's stored camera settings from the session's store, on the camera lane, for the camera to hold.
- `engineering_mode`: Reads the session's engineering mode, once per capture, when the capture is asked for: a plugin may turn it on after the controller is built.
- `channel_colors`: Reads every layer's colour, from the Session's one read of it, once per capture, when the capture is asked for.

<a id="ManualCaptureController.capture"></a>
### `ManualCaptureController.capture`

```python
def capture(
    self,
    *,
    false_color_on: bool,
    bullseye: bool = False,
    crosshairs: bool = False,
) -> Operation
```

Route: `POST /api/v1/manual_capture/capture`.

Capture one still and save it; returns at once.

**Args**

- `false_color_on`: Whether the shown layer is displayed in false colour -- how the file is rendered, never what it says was imaged.
- `bullseye`: Also save a copy with the bullseye colour map.
- `crosshairs`: Also save a copy with the centre crosshairs. With either overlay, one extra file is written, from the same frame, beside the unmarked one.

The still is named for the lit LED's channel, else the shown
channel's layer (`session.show_channel`), read as its frame is
taken, and the camera holds that layer's stored settings for the
frame; after a raw camera write it keeps the written values, and
the frame's record says no layer was held. In the session's
engineering mode the name carries the turret slot.

**Returns**

An `Operation` whose result is a `SavedStill` per file written, the unmarked file first: its file and the instrument's record of it. Done once the still has let go of the camera lane's guard, so a second still asked then is not refused. The still is on the camera lane when this returns, so a run started after it waits for it when it is running or was put on an idle lane. A still has no stop: `stop()` does nothing. The operation raises what the capture raised: `ObjectiveUnknownError` when the objective in the light path is unknown (nothing captured), `LiveFolderPathRefusedError` (reason `'capture_location_unusable'`) when the live folder cannot take the still, `CaptureFailedError` when no frame passed, with the capture's own reason (`capture_and_wait`); `HardwareCommandRefusedError` (reason `'not_connected'`) when the camera is absent or was removed during the still, `'camera_not_streaming'` when its feed is stopped, and `'exclusive_activity_running'` when a run took the scope while the still was queued behind other camera work, waiting or running when it was put on the lane, or when a recording runs and the camera would have to change layers for the still (nothing is written then, and the recording is unchanged); `CancelledError` when the lane dropped it unrun (the scope disconnected); `LaneCallStillRunningError` when it has not finished within the bound a capture and its write are given: the still may yet be taken and saved, its file logged when it is, and a second still is refused `'capture_in_flight'` until then.

**Raises**

- `HardwareCommandRefusedError`: reason `'capture_in_flight'`, while an earlier still has not finished.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `capture_in_flight`

<a id="ManualCaptureController.in_flight"></a>
### `ManualCaptureController.in_flight`

```python
in_flight: bool  # property
```

Route: `GET /api/v1/manual_capture/in_flight`.

True from a capture call until its camera-lane body has ended.

**Raises**

Nothing.

<a id="ManualRecordingController"></a>
## `ManualRecordingController`

Reached through `session.manual_recording`; on the wire, `/api/v1/manual_recording/`.  
Held in: [`ScopeSession.manual_recording`](#ScopeSession.manual_recording).

One manual-recording flow: snapshot, record, drain, finish.

**Args**

- `scope`: The Lumascope instance (frame listener, camera identity, exposure, stage position).
- `settings`: The live settings dict; read ONLY at `start` -- the recording runs from its immutable snapshot.
- `shown_layer`: Reads the layer of the session's shown channel, once, at `start`: it names the channel when no LED is lit.
- `channel_colors`: Reads every layer's colour, from the Session's one read of it, once, at `start`.
- `activity_claim`: The session's compare-and-claim handle; passed through to the engine so a recording and a protocol run are mutually exclusive.
- `scheduler`: The session's scheduler; the controller arms its own periodic health check on it at `start` and disarms at finish. Required: a controller that can start a recording but has no armed health bound is the trap this parameter exists to make unconstructible.
- `post_processing`: The session's builds; a recording's stack is built through it.
- `clock`: Wall-time source (seconds) for what is measured against frame timestamps: the engine's cadence and the camera tick rebaser. Tests drive it.
- `elapsed_clock`: Elapsed-time source (seconds) for the recording's bounds: its elapsed time, the duration cap, the stall watch and the disk probe's interval. Monotonic, because wall time steps (an NTP correction, a host resume) and a step would end a good recording early or let one run past its cap. Tests drive it.

<a id="ManualRecordingController.discard_pending"></a>
### `ManualRecordingController.discard_pending`

```python
def discard_pending(self) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/manual_recording/discard_pending`.

Drop the unwritten backlog loudly (the app-close discard path).

**Raises**

Nothing.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`

<a id="ManualRecordingController.elapsed_s"></a>
### `ManualRecordingController.elapsed_s`

```python
elapsed_s: float  # property
```

Route: `GET /api/v1/manual_recording/elapsed_s`.

Seconds since the recording started; 0.0 when idle.

**Raises**

Nothing.

<a id="ManualRecordingController.end_reason"></a>
### `ManualRecordingController.end_reason`

```python
end_reason: str | None  # property
```

Route: `GET /api/v1/manual_recording/end_reason`.

Why the most recent recording ended, or None before any has.

Manifest vocabulary ('user_stop', 'duration_elapsed',
'camera_stalled', ...), read from the recording's engine as soon
as its selection closes, so a caller watching is_recording flip
False can tell a camera death from a user stop while the frames
still drain. The first close names it: a later stop does not
rewrite it.

**Raises**

Nothing.

<a id="ManualRecordingController.is_busy"></a>
### `ManualRecordingController.is_busy`

```python
is_busy: bool  # property
```

Route: `GET /api/v1/manual_recording/is_busy`.

True until the recording, its drain, AND the finish complete.

The app-close gate reads this: the recording is not safely over
until the post-drain finish (MP4 close, hyperstack) has run.

**Raises**

Nothing.

<a id="ManualRecordingController.is_draining"></a>
### `ManualRecordingController.is_draining`

```python
is_draining: bool  # property
```

Route: `GET /api/v1/manual_recording/is_draining`.

True while queued frames are still being written.

**Raises**

Nothing.

<a id="ManualRecordingController.is_recording"></a>
### `ManualRecordingController.is_recording`

```python
is_recording: bool  # property
```

Route: `GET /api/v1/manual_recording/is_recording`.

True while cadence selection is open.

**Raises**

Nothing.

<a id="ManualRecordingController.pending_writes"></a>
### `ManualRecordingController.pending_writes`

```python
pending_writes: int  # property
```

Route: `GET /api/v1/manual_recording/pending_writes`.

Frames enqueued but not yet on disk.

**Raises**

Nothing.

<a id="ManualRecordingController.save_folder"></a>
### `ManualRecordingController.save_folder`

```python
save_folder: Path | None  # property
```

Route: `GET /api/v1/manual_recording/save_folder`.

The active (or last) recording's output folder.

**Raises**

Nothing.

<a id="ManualRecordingController.start"></a>
### `ManualRecordingController.start`

```python
def start(self, *, false_color_on: bool = False) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop).  
Route: `POST /api/v1/manual_recording/start`.

Snapshot configuration and open a recording; returns its operation.

Channel IDENTITY (what was imaged) and false-color RENDERING (how
it is displayed) are independent. Identity is recorded on every
frame; only rendering is conditional. Collapsing the two is what
made a brightfield hyperstack impossible to build, since the
toggle being off left the channel unnamed. The channel is the lit
LED's, else the shown channel's layer (`session.show_channel`),
read once here.

**Args**

- `false_color_on`: Whether the shown layer is displayed in false colour. Controls rendering alone; grayscale when False.

**Returns**

The recording's `Operation`: done once its frames, MP4 close and any hyperstack build are over and a new recording may start, with the recording's `end_reason`. Its `stop()` stops this recording as `stop` does, and does nothing once it has stopped, so it never stops a recording started later.

**Raises**

- `RecordingRefusedError`: The start cannot proceed, and nothing is committed: `'recording_active'`, a previous recording is still recording, draining or finishing; `'exclusive_activity_running'`, another exclusive activity is running; `'post_processing_running'`, a build is on the post-processing lane; `'falsifying_change_in_flight'`, a turret or frame change is under way; `'camera_exposure_unknown'`, the camera has no known exposure; `'capture_location_unusable'`, the save location cannot be used; `'names_exhausted'`, the recording's name and every numbered variant are taken there; `'insufficient_disk'`, free disk is below the floor.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`. Nothing is committed.
- `ControlSessionRefusedError`: `'control_session_held'`, someone else holds the control session. Nothing is committed.
- `ObjectiveUnknownError`: No one can say which objective is in the light path, so the recording's frames would carry no scale. Nothing is committed.
- `CaptureFailedError`: `'recording_not_started'`, the camera did not accept the frame listener; the start is unwound.

**Refusals by REST status**

- 409: [`RecordingRefusedError`](#RecordingRefusedError): `recording_active`, `exclusive_activity_running`, `post_processing_running`, `falsifying_change_in_flight`, `camera_exposure_unknown`, `capture_location_unusable`, `names_exhausted`, `insufficient_disk`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`, `slot_unassigned`, `not_in_catalogue`, `none_selected`
- 500: [`CaptureFailedError`](#CaptureFailedError): `recording_not_started`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ManualRecordingController.stop"></a>
### `ManualRecordingController.stop`

```python
def stop(self) -> None
```

Route: `POST /api/v1/manual_recording/stop`.

Close selection; the drain and finish continue on their own.

Every caller of this -- the Record button, app close, an L2 client
-- is the user stopping the recording, so its `end_reason` is
`'user_stop'`. The controller's own enders (the duration cap, the
disk floor, a lost camera) name their own, and no caller can write
one of those into the manifest.

**Raises**

Nothing.

<a id="PostProcessingAPI"></a>
## `PostProcessingAPI`

Reached through `session.post_processing`; on the wire, `/api/v1/post_processing/`.  
Held in: [`ScopeSession.post_processing`](#ScopeSession.post_processing).

The post-processing builds, each run on the post-processing lane.

<a id="PostProcessingAPI.build"></a>
### `PostProcessingAPI.build`

```python
def build(
    self,
    folder: FilePath,
    *,
    zproject: ZProject | None = None,
    stitch: Stitch | None = None,
    composite: bool = False,
    video: Video | None = None,
    on_progress: ProgressCallback | None = None,
) -> BuildsResult
```

Route: `POST /api/v1/post_processing/build`.

Run the chosen builds over *folder* in one call, in the order that
keeps the data honest: z-project, stitch, composite, video.

A projection is made of raw slices, never of mosaics; a stitched
composite is made from the stitched channels, never by stitching
composites. Each build takes only what the build before it in this
call carried forward -- what that build made, and the images it did
not consume (an ordinary Add step's image beside a tiled stack) --
so the call makes the chain's product and its inputs, not every
combination the folder's images allow: z-project, stitch and
composite of a 2x2 tiled, 3-slice, 2-channel scan make 8
projections, 2 stitched projections and 1 composite, where the three
builds run one by one make 36 files. A build is chosen by passing its
settings record (*composite*, which has no settings of its own -- its
format and thresholds are the user's configuration -- by True); one
left None is not run. Every chosen build's
settings are checked before any build starts, and the call runs as
one task on the post-processing lane, so no other build touches the
folder between its steps. Each build alone (`stitch`,
`zproject`, `composite`, `video`) is unchanged.

A build with nothing to do in the folder -- video on a single-scan
run -- is skipped and named in the answer, and the next build takes
what it was given. Any other refusal or failure stops the call and is
raised as that build's own outcome, unchanged; the files the earlier
builds made are in the folder and its record, and the same call made
again rebuilds nothing already made and resumes at the build that
stopped, once its cause is gone. One that does not go -- a raw slice
missing -- stops the call there every time; build that folder with
the members one by one.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at *folder*; `'not_a_folder'`, a file is.
- `PostProcessingRefusedError`: reason `invalid_setting`, no build is chosen, or a chosen build's setting is one its own member refuses, before anything is queued; or the refusal of the build that stopped the call (its `operation` names it).
- `PostProcessingFailedError`: the build that stopped the call did not produce everything; its `operation` names it.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_folder`; [`PostProcessingRefusedError`](#PostProcessingRefusedError): `invalid_setting`
- 500: [`PostProcessingFailedError`](#PostProcessingFailedError): `post_processing_failed`, `post_processing_incomplete`

<a id="PostProcessingAPI.cell_count_size_scale"></a>
### `PostProcessingAPI.cell_count_size_scale`

```python
def cell_count_size_scale(
    self,
    method: CellCountMethod,
    *,
    shape: tuple[int, ...] | None = None,
    pixel_size_um: float | None = None,
) -> CellCountSizeScale
```

Route: `POST /api/v1/post_processing/cell_count_size_scale`.

The scale an image of *shape* is counted at by *method*, and what its size sliders stand for.

The scale is the one `preview_cell_count` and `count_cells`
measure at: the method's *pixels_per_um*, else *pixel_size_um*, the
image's own (microns per pixel; None when it states none), else
pixels. Each slider's top stop is the largest size the image holds
at that scale, and its low end is stretched, so the small sizes most
methods bound get most of the travel. With *shape* None (no image
yet) a position stands for its own size.

**Raises**

- `PostProcessingRefusedError`: reason `invalid_setting`, *shape* is not an image's rows and columns, each at least one, or *pixel_size_um* is not a positive number of microns.

**Refusals by REST status**

- 422: [`PostProcessingRefusedError`](#PostProcessingRefusedError): `invalid_setting`

<a id="PostProcessingAPI.count_cells"></a>
### `PostProcessingAPI.count_cells`

```python
def count_cells(
    self,
    folder: FilePath,
    *,
    method: CellCountMethod,
    on_progress: ProgressCallback | None = None,
) -> CellCountResult
```

Route: `POST /api/v1/post_processing/count_cells`.

Count the cells in every image in *folder* by *method*, writing
`results.csv` into the folder.

**Returns**

The results table, how many images were counted, and the outcome in words.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at *folder*; `'not_a_folder'`, a file is.
- `PostProcessingRefusedError`: reason `method_invalid`, a *method* sent as its fields holds what `CellCountMethod` refuses; or the folder has nothing to count (reason `no_data`).
- `PostProcessingFailedError`: the results could not be written, or some images could not be counted; the file holds every image that was, and the error names the rest.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_folder`; [`PostProcessingRefusedError`](#PostProcessingRefusedError): `method_invalid`, `no_data`
- 500: [`PostProcessingFailedError`](#PostProcessingFailedError): `post_processing_failed`, `post_processing_incomplete`

<a id="PostProcessingAPI.default_cell_count_method"></a>
### `PostProcessingAPI.default_cell_count_method`

```python
def default_cell_count_method(self) -> CellCountMethod
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/post_processing/default_cell_count_method`.

The cell-count method LumaViewPro's panel starts from, for
`count_cells(method=)`: `CellCountMethod()`.

**Raises**

Nothing.

<a id="PostProcessingAPI.draw_graph"></a>
### `PostProcessingAPI.draw_graph`

```python
def draw_graph(
    self,
    results: FilePath | None,
    *,
    x: str | None,
    y: str | None,
    trendline: Trendline | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = '',
) -> Figure
```

Marks: in-process.

The figure `graph` saves for the same arguments, for a GUI in this process to show.

With no *results*, or *x* or *y* None, it is the empty frame a
graph shows before its axes are chosen. Each label defaults to its
column's name.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at *results*; `'not_a_file'`, a folder is.
- `PostProcessingRefusedError`: `'results_unreadable'`, the file cannot be graphed; `'axis_not_plottable'`, *x* or *y* is not a column that axis can take; `'fit_impossible'`, the values cannot give the *trendline*.
- `ArgumentRefusedError`: `'not_offered'`, the *trendline* is not one *x* and *y* take, offering those they take.

**Refusals**

- [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_file`
- [`PostProcessingRefusedError`](#PostProcessingRefusedError): `results_unreadable`, `axis_not_plottable`, `fit_impossible`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`

<a id="PostProcessingAPI.enhance"></a>
### `PostProcessingAPI.enhance`

```python
def enhance(
    self,
    target: FilePath,
    *,
    on_progress: ProgressCallback | None = None,
    on_derived_image: Callable[[np.ndarray, int], None] | None = None,
) -> EnhanceResult
```

Route: `POST /api/v1/post_processing/enhance`.

Quick Enhance *target* -- one image, or every image in a folder --
each to its own derived file beside its source.

**Args**

- `on_derived_image`: Called with each enhanced image and its significant bits as it is saved, for a caller that shows it.

**Returns**

Each derived file, the folder they are in, and the outcome in words.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at *target*.
- `PostProcessingRefusedError`: reason `unreadable`, *target* is a file that is not an image Quick Enhance can read; reason `no_data`, a folder that holds none.
- `PostProcessingFailedError`: a file could not be enhanced.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`; [`PostProcessingRefusedError`](#PostProcessingRefusedError): `unreadable`, `no_data`
- 500: [`PostProcessingFailedError`](#PostProcessingFailedError): `post_processing_failed`, `post_processing_incomplete`

<a id="PostProcessingAPI.graph"></a>
### `PostProcessingAPI.graph`

```python
def graph(
    self,
    results: FilePath | None,
    path: FilePath,
    *,
    x: str | None,
    y: str | None,
    trendline: Trendline | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = '',
) -> GraphResult
```

Route: `POST /api/v1/post_processing/graph`.

Save the graph of column *y* against column *x* of the cell-count
*results* file to *path* as a PNG.

*trendline* is one of the kinds `graph_options` names, or None for
none. Each label defaults to its column's name. A missing folder is
made where *path* sends it.

**Returns**

The PNG written and the outcome in words.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at *results*; `'not_a_file'`, a folder is there or at *path*; `'wrong_extension'`, *path* does not end in `.png` (in any case), which is not added; `'not_a_folder'`, a file is where its folder would be. Nothing is written.
- `PostProcessingRefusedError`: `'graph_incomplete'`, no *results* file or no *x* or *y* is given; `'results_unreadable'`, the file cannot be graphed; `'axis_not_plottable'`, *x* or *y* is not a column that axis can take; `'fit_impossible'`, the values cannot give the *trendline*. Nothing is written.
- `ArgumentRefusedError`: `'not_offered'`, the *trendline* is not one *x* and *y* take, offering those they take. Nothing is written.
- `PostProcessingFailedError`: the folder or the file could not be written; the message names the file and the system's reason.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_file`, `wrong_extension`, `not_a_folder`; [`PostProcessingRefusedError`](#PostProcessingRefusedError): `graph_incomplete`, `results_unreadable`, `axis_not_plottable`, `fit_impossible`; [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`
- 500: [`PostProcessingFailedError`](#PostProcessingFailedError): `post_processing_failed`, `post_processing_incomplete`

<a id="PostProcessingAPI.graph_options"></a>
### `PostProcessingAPI.graph_options`

```python
def graph_options(
    self,
    results: FilePath,
    *,
    x: str | None = None,
    y: str | None = None,
) -> GraphOptions
```

Route: `POST /api/v1/post_processing/graph_options`.

The columns of the cell-count *results* file a graph can take, and
the trendlines the axes given so far take.

Read on the caller's thread, not the post-processing lane, so a form
filling its choices does not wait behind a build.

**Raises**

- `LiveFolderPathRefusedError`: `'path_not_found'`, nothing is at *results*; `'not_a_file'`, a folder is.
- `PostProcessingRefusedError`: `'results_unreadable'`, the file cannot be graphed; `'axis_not_plottable'`, *x* or *y* is not a column that axis can take.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `path_not_found`, `not_a_file`; [`PostProcessingRefusedError`](#PostProcessingRefusedError): `results_unreadable`, `axis_not_plottable`

<a id="PostProcessingAPI.load_cell_count_method"></a>
### `PostProcessingAPI.load_cell_count_method`

```python
def load_cell_count_method(self, path: FilePath) -> CellCountMethod
```

Route: `POST /api/v1/post_processing/load_cell_count_method`.

The cell-count method saved in the file at *path*.

Read on the caller's thread, not the post-processing lane, so a
form taking a method does not wait behind a build. A file saved
before the count read each image's scale carries the old fixed
default of 1.0 pixels per micron, which the load drops, saying so
once.

**Raises**

- `PostProcessingRefusedError`: reason `method_unreadable`, the file cannot be read or is not a method file; reason `method_invalid`, it holds a method the count cannot use. The message names the file.

**Refusals by REST status**

- 422: [`PostProcessingRefusedError`](#PostProcessingRefusedError): `method_unreadable`, `method_invalid`

<a id="PostProcessingAPI.preview_cell_count"></a>
### `PostProcessingAPI.preview_cell_count`

```python
def preview_cell_count(
    self,
    image: np.ndarray,
    method: CellCountMethod,
    *,
    significant_bits: int,
    pixel_size_um: float | None,
) -> CellCountPreview
```

Marks: in-process.

Count the cells in *image* by *method*, as `count_cells` counts
each image of a folder, and answer the outlines and the counts.

Counted on the caller's thread, not the post-processing lane, so a
form previewing a method while a build or a run goes on is not held
behind it. *image* is the pixels the caller holds, so a form that
adjusts the method reads no file. It is measured at the scale
`count_cells` would measure its file at: the method's
*pixels_per_um*, else *pixel_size_um*, the image's own (microns
per pixel), else in pixels.

**Args**

- `significant_bits`: The image's payload depth, by which a 16-bit image is scaled to the 8 bits the count reads.
- `pixel_size_um`: The scale the image states, or None.

**Raises**

- `PostProcessingRefusedError`: `'no_scale'`, the image is counted in pixels and *method* bounds an area or a perimeter, which are in microns; `'invalid_setting'`, *pixel_size_um* is not a positive number of microns.

**Refusals**

- [`PostProcessingRefusedError`](#PostProcessingRefusedError): `no_scale`, `invalid_setting`

<a id="PostProcessingAPI.projection_methods"></a>
### `PostProcessingAPI.projection_methods`

```python
def projection_methods(self) -> tuple[ZProjectMethod, ...]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/post_processing/projection_methods`.

The methods `zproject(method=)` and `ZProject(method)` take.

**Raises**

Nothing.

<a id="PostProcessingAPI.save_cell_count_method"></a>
### `PostProcessingAPI.save_cell_count_method`

```python
def save_cell_count_method(self, method: CellCountMethod, path: FilePath) -> None
```

Route: `POST /api/v1/post_processing/save_cell_count_method`.

Write *method* to *path* as a cell-count method file, which
`load_cell_count_method` reads back.

Written on the caller's thread, as the load is read. A missing
folder is made where *path* sends it.

**Raises**

- `LiveFolderPathRefusedError`: `'not_a_file'`, a folder is at *path*; `'wrong_extension'`, its name does not end in `.json` (in any case), which is not added; `'not_a_folder'`, a file is where its folder would be. Nothing is written.
- `PostProcessingRefusedError`: reason `method_invalid`, a *method* sent as its fields holds what `CellCountMethod` refuses. Nothing is written.

**Refusals by REST status**

- 422: [`LiveFolderPathRefusedError`](#LiveFolderPathRefusedError): `not_a_file`, `wrong_extension`, `not_a_folder`; [`PostProcessingRefusedError`](#PostProcessingRefusedError): `method_invalid`

<a id="PostProcessingAPI.stitch_modes"></a>
### `PostProcessingAPI.stitch_modes`

```python
def stitch_modes(self) -> tuple[StitchMode, ...]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/post_processing/stitch_modes`.

The modes `stitch(mode=)` and `Stitch(mode)` take.

**Raises**

Nothing.

<a id="Lumascope"></a>
## `Lumascope`

Reached through `session.scope`; on the wire, `/api/v1/scope/`.  
Held in: [`ScopeSession.scope`](#ScopeSession.scope).

**Fields**

- <a id="Lumascope.camera"></a>`camera`: `CameraAPI` -- (no description)
- <a id="Lumascope.capabilities"></a>`capabilities`: `ScopeCapabilities` -- (no description)
- <a id="Lumascope.channel_identity"></a>`channel_identity`: `ChannelIdentity` -- (no description)
- <a id="Lumascope.diagnostics"></a>`diagnostics`: `DiagnosticsAPI` -- (no description)
- <a id="Lumascope.illumination"></a>`illumination`: `IlluminationAPI` -- (no description)
- <a id="Lumascope.imaging"></a>`imaging`: `ImagingAPI` -- (no description)
- <a id="Lumascope.motion"></a>`motion`: `MotionAPI` -- (no description)
- <a id="Lumascope.runtime_state"></a>`runtime_state`: `RuntimeState` -- (no description)

<a id="Lumascope.labware"></a>
### `Lumascope.labware`

```python
labware: Mapping[str, Labware]  # property
```

Route: `GET /api/v1/scope/labware`.

The installation's plates, by name, read once at bring-up.

Read-only, as each record is: no caller can change what another
reads. Which one is selected is `ScopeSession.labware_id`.
Slides and Petri dishes are not offered yet.

**Raises**

Nothing.

<a id="Lumascope.motion_expected"></a>
### `Lumascope.motion_expected`

```python
motion_expected: bool  # property
```

Route: `GET /api/v1/scope/motion_expected`.

Whether this scope's model has a motor board at all.

False for a manual scope (an LS620 or LS560): it is complete
without one, so its absence is not a disconnection. Set from the
model's catalogue entry by `initialize()`; True before that.

**Raises**

Nothing.

<a id="Lumascope.no_hardware"></a>
### `Lumascope.no_hardware`

```python
no_hardware: bool  # property
```

Route: `GET /api/v1/scope/no_hardware`.

True if no real hardware was detected (LED, motor, and camera all missing).

**Returns**

bool: True when all three subsystems are absent or stubbed.

**Raises**

Nothing.

<a id="Lumascope.objectives"></a>
### `Lumascope.objectives`

```python
objectives: Mapping[str, Objective]  # property
```

Route: `GET /api/v1/scope/objectives`.

The installation's objectives, by id, read once at bring-up.

Read-only, as each record is: no caller can change what another
reads. Which one is in the light path is
`runtime_state.resolve_current_objective`'s answer.

**Raises**

Nothing.

<a id="Lumascope.scope_models"></a>
### `Lumascope.scope_models`

```python
scope_models: Mapping[str, ScopeModel]  # property
```

Route: `GET /api/v1/scope/scope_models`.

The model catalogue (scopes.json's `Models`): a read-only mapping of model name to its `ScopeModel`.

Read-only, as each record is: no caller can change what another
reads.

**Raises**

Nothing.

<a id="Lumascope.settings_template"></a>
### `Lumascope.settings_template`

```python
settings_template: dict  # property
```

Route: `GET /api/v1/scope/settings_template`.

The shipped settings template: which settings exist, and their shipped values.

The caller's own copy: a caller that changed what it was given
would otherwise change every later reader's template.

**Raises**

Nothing.

<a id="Lumascope.simulation"></a>
### `Lumascope.simulation`

```python
simulation: SimulationAPI  # property
```

The faults a simulated scope presents, switched on by name.

On the wire as in-process, but for two members that take or hand
out an in-process object. Every member refuses on a real scope.

**Raises**

Nothing.

<a id="Lumascope.unconnected_parts"></a>
### `Lumascope.unconnected_parts`

```python
def unconnected_parts(self) -> tuple[Part, ...]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/unconnected_parts`.

The parts this scope needs that are not connected.

Each term is the one the matching single-board question asks, so
a run gate and a per-board gate cannot disagree about the same
hardware. Each term reads its driver's own flags, with no exchange
with the board or the camera (`drivers/protocols.py`,
`Camera.is_connected`).

A scope whose model has no motor board is not asked for one: a
manual scope is complete without it.

**Returns**

Each of `LED`, `MOTOR` and `CAMERA` that is not connected, in that order; empty when all are. A sentence names each by its `noun`.

**Raises**

Nothing.

<a id="CameraAPI"></a>
## `CameraAPI`

Reached through `session.scope.camera`; on the wire, `/api/v1/scope/camera/`.  
Held in: [`Lumascope.camera`](#Lumascope.camera).

Camera sub-API. Owns the camera's setters, reads, holds and applies,
the camera's state (cache, held layer, auto-gain arm, listeners) and the
saving of the stored camera settings.

<a id="CameraAPI.add_exposure_listener"></a>
### `CameraAPI.add_exposure_listener`

```python
def add_exposure_listener(self, listener: Callable[[float], None]) -> None
```

Marks: in-process; event: [`ExposureChanged`](#ExposureChanged).

Register a callback for each change of the camera's exposure.

The listener is called with the exposure now in effect, in ms,
whenever a write changes it. It fires from the thread that caused
the change, so listeners **must** schedule UI work via
`Clock.schedule_once`.

Note: this fires on set_exposure_ms (a user action), NOT on every
camera frame grab -- zero overhead on display framerate.

**Args**

- `listener`: `callable(exposure_ms: float)`

**Raises**

Nothing.

<a id="CameraAPI.add_gain_listener"></a>
### `CameraAPI.add_gain_listener`

```python
def add_gain_listener(self, listener: Callable[[float], None]) -> None
```

Marks: in-process; event: [`GainChanged`](#GainChanged).

Register a callback for each change of the camera's gain.

The listener is called with the gain now in effect, in dB, whenever
a write changes it. It fires from the thread that caused the change,
so listeners **must** schedule UI work via `Clock.schedule_once`.

Note: this fires on set_gain_db (a user action), NOT on every camera
frame grab -- zero overhead on display framerate.

**Args**

- `listener`: `callable(gain_db: float)`

**Raises**

Nothing.

<a id="CameraAPI.applied_auto_gain_for"></a>
### `CameraAPI.applied_auto_gain_for`

```python
def applied_auto_gain_for(self, stored_auto_gain: bool) -> AppliedAutoGain
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/applied_auto_gain_for`.

What a stored auto-gain preference becomes on the attached camera.

See `applied_gain_db_for`; the same contract for the mode. A camera
without hardware auto-gain runs manual whatever the layer stored, and
the stored preference is left alone so a camera that has the mode
takes it again.

**Raises**

Nothing.

<a id="CameraAPI.applied_exposure_ms_for"></a>
### `CameraAPI.applied_exposure_ms_for`

```python
def applied_exposure_ms_for(self, stored_exposure_ms: float) -> AppliedExposure
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/applied_exposure_ms_for`.

What a stored exposure becomes on the attached camera.

See `applied_gain_db_for`; the same contract for the other
quantity, so both travel the same path to hardware and to display.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, the exposure is not a finite number.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`

<a id="CameraAPI.applied_gain_db_for"></a>
### `CameraAPI.applied_gain_db_for`

```python
def applied_gain_db_for(self, stored_gain_db: float) -> AppliedGain
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/applied_gain_db_for`.

What a stored gain becomes on the attached camera.

The one place the gain cap is applied. A caller that narrows a
stored value itself -- against this cap or against a widget's
range -- is a second answerer, and the store it writes back is
how a user's setting gets destroyed by connecting a smaller body.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, the gain is not a finite number.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`

<a id="CameraAPI.apply"></a>
### `CameraAPI.apply`

```python
def apply(
    self,
    gain_db: float,
    exposure_ms: float,
    auto_gain: bool | None = None,
    auto_gain_settings: dict | None = None,
    resume_after_capture: bool = True,
    channel: str = '(unspecified)',
) -> dict | None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/camera/apply`.

Apply a gain, an exposure and an auto-gain mode in one batched call, and wait.

The batch is one command on the camera's lane, so no other camera
command lands between its writes. Turning auto-gain off goes first
and arming it last, so the gain and the exposure are written with
the camera's loop off; an arm may lower the exposure to the
channel's auto-exposure ceiling. Every write is attempted even when
an earlier one is refused, so one refused setting does not leave the
others at a previous channel's values; the refusal is raised once all
have run. The range refusals are `set_gain_db`'s and
`set_exposure_ms`'s.

**Args**

- `gain_db`: The gain in dB.
- `exposure_ms`: The exposure time in milliseconds.
- `auto_gain`: The mode the camera is left in. True arms auto-gain from `auto_gain_settings`; False turns it off; None, the default, leaves the mode as it is and writes none.
- `auto_gain_settings`: The arm's settings, read only when `auto_gain` is True, which needs them.
- `resume_after_capture`: With `auto_gain` True, whether a capture that locks the arm re-arms it afterwards (live view) or leaves the camera at the locked values (a protocol step).
- `channel`: The channel these settings belong to, named in the log line; left out, the line says `(unspecified)`.

**Returns**

dict | None: `{'gain_db': ..., 'exposure_ms': ...}`, each the value now in effect, which differs from the request when the camera snapped it. `None` when no camera is active.

**Raises**

- `ArgumentRefusedError`: `'missing_key'`, `auto_gain` is True and `auto_gain_settings` is None or lacks a key the auto-gain write reads; `'not_a_number'`, one of its values, the gain or the exposure is not a finite number; `'key_unknown'`, it holds a key the write does not take; `'gain_range_inverted'`, its minimum gain is above its maximum. Nothing was written.
- `CameraSettingOutOfRangeError`: `'gain_db_out_of_range'` or `'exposure_ms_out_of_range'`, the gain or the exposure is outside the range the camera declares; nothing was written.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; nothing was written.
- `CameraSettingRejected`: The camera did not take the gain, the exposure or the auto-gain change. When more than one, `setting` names each, comma-separated, `requested` holds their values in that order, the message carries each sentence, and the reason is `'camera_no_answer'` when any of them got no answer.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `missing_key`, `not_a_number`, `key_unknown`, `gain_range_inverted`; [`CameraSettingOutOfRangeError`](#CameraSettingOutOfRangeError): `gain_db_out_of_range`, `exposure_ms_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.auto_exposure"></a>
### `CameraAPI.auto_exposure`

```python
auto_exposure: CameraAutoMode | None  # property
```

Route: `GET /api/v1/scope/camera/auto_exposure`.

The camera's auto-exposure mode, as the camera last reported or took
it; None when unknown. Its writers: `set_auto_exposure`; auto-gain,
whose arm, lock and one-shot switch exposure with the gain; and
`read_auto_exposure`.

**Raises**

Nothing.

<a id="CameraAPI.auto_gain"></a>
### `CameraAPI.auto_gain`

```python
auto_gain: CameraAutoMode | None  # property
```

Route: `GET /api/v1/scope/camera/auto_gain`.

The camera's auto-gain mode, as the camera last reported or took it;
None when unknown. Its writers: an arm, the lock that ends it, a
one-shot (`ONCE` until a read finds it ended), and
`read_auto_gain`.

**Raises**

Nothing.

<a id="CameraAPI.binning"></a>
### `CameraAPI.binning`

```python
binning: int | None  # property
```

Route: `GET /api/v1/scope/camera/binning`.

The binning factor the camera last reported or took; None when unknown.

**Raises**

Nothing.

<a id="CameraAPI.black_level"></a>
### `CameraAPI.black_level`

```python
black_level: float | None  # property
```

Route: `GET /api/v1/scope/camera/black_level`.

The black level the camera last reported or took (see
`read_black_level`); None when unknown or the camera reports none.

**Raises**

Nothing.

<a id="CameraAPI.exposure_ms"></a>
### `CameraAPI.exposure_ms`

```python
exposure_ms: float | None  # property
```

Route: `GET /api/v1/scope/camera/exposure_ms`.

The exposure in ms the camera last reported or took; None when unknown.

Under a continuous auto-gain arm the camera chooses its own exposure,
and this is the one from before the arm; `longest_exposure_ms`
bounds what it may be using.

**Raises**

Nothing.

<a id="CameraAPI.frame_at_binning"></a>
### `CameraAPI.frame_at_binning`

```python
def frame_at_binning(self, size: int) -> dict
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/frame_at_binning`.

The frame `set_binning(size)` will ask the camera for; nothing is applied.

The same arithmetic the apply uses, answered from the saved region
and the camera's limits, so a display can show the new frame beside
the new binning while the apply is still running. The camera's own
answer (its grid, its minimum at the new binning) can differ; the
saved frame after the apply is the truth.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected.
- `CameraSettingUnsupportedError`: `'binning_unsupported'`, this camera does not offer `size`, as `set_binning` refuses it.

**Refusals by REST status**

- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `binning_unsupported`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="CameraAPI.frame_size"></a>
### `CameraAPI.frame_size`

```python
frame_size: FrameSize | None  # property
```

Route: `GET /api/v1/scope/camera/frame_size`.

The size of the frames the camera delivers, as it last reported or
took it, in pixels after binning; None when unknown.

**Raises**

Nothing.

<a id="CameraAPI.gain_db"></a>
### `CameraAPI.gain_db`

```python
gain_db: float | None  # property
```

Route: `GET /api/v1/scope/camera/gain_db`.

The gain in dB the camera last reported or took; None when unknown.

**Raises**

Nothing.

<a id="CameraAPI.get_black_level_range"></a>
### `CameraAPI.get_black_level_range`

```python
def get_black_level_range(self) -> tuple[float, float] | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/get_black_level_range`.

Read the range `set_black_level` accepts, live, in the units of
`read_black_level`.

The camera's own minimum and maximum for its current pixel format;
the range changes with the format. `set_black_level` refuses a value
outside it before anything reaches the camera.

Read live, never held: the range follows the pixel format.

**Returns**

tuple[float, float] | None: `(minimum, maximum)`; None when the camera has no black level setting. A camera holding its black level automatically still answers its range; `set_black_level` is then refused by the camera.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: `'camera_read_failed'`, the camera has a black level setting and the read failed.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.get_pixel_alignment"></a>
### `CameraAPI.get_pixel_alignment`

```python
def get_pixel_alignment(self) -> dict
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/get_pixel_alignment`.

Return the camera's deliverable frame-size granularity.

Every camera delivers the size asked for exactly, by acquiring the next
window up on its grid and cropping back, so the only constraint is even
dimensions (H.264): `{2, 2}`, the camera profile's step.

**Returns**

dict: `{'width': int, 'height': int}`.

**Raises**

Nothing.

<a id="CameraAPI.high_conversion_gain"></a>
### `CameraAPI.high_conversion_gain`

```python
high_conversion_gain: bool | None  # property
```

Route: `GET /api/v1/scope/camera/high_conversion_gain`.

Whether the camera's high conversion gain is on, as the camera last
took it; None when unknown, as after a value the camera refused. No
driver reads it back. A camera without the mode is at its standard
conversion gain: False.

**Raises**

Nothing.

<a id="CameraAPI.image_mode"></a>
### `CameraAPI.image_mode`

```python
image_mode: ImageMode  # property
```

Route: `GET /api/v1/scope/camera/image_mode`.

The image mode captures are taken in: the saved one, which
`set_image_mode` writes. The camera holds no copy of a mode, only
the pixel format the mode chose (`pixel_format`).

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session bound this scope.

**Refusals by REST status**

- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="CameraAPI.line_noise_reduction"></a>
### `CameraAPI.line_noise_reduction`

```python
line_noise_reduction: bool | None  # property
```

Route: `GET /api/v1/scope/camera/line_noise_reduction`.

Whether the camera's line-noise filter is on, as the camera last took
it; None when unknown. No driver reads it back. A camera without the
filter reads unfiltered: False.

**Raises**

Nothing.

<a id="CameraAPI.lock_auto_gain"></a>
### `CameraAPI.lock_auto_gain`

```python
def lock_auto_gain(self) -> AutoGainLock
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/camera/lock_auto_gain`.

Lock a standing continuous auto-gain arm and return the result.

The camera's auto loop chose an exposure and a gain no caller
asked for; the lock writes them back as manual values, so the next
frame is held to them like any manual setting. They are read from
the newest frame taken under the arm when the camera reports them,
else from the camera itself. The lock is one command on the camera's
lane, waited for. A caller leaving
auto-gain stores `stored_exposure_ms` / `gain_db` as the manual
setting; when no arm stands the result's `state` is None and
nothing was written. With no camera connected no arm can stand, and
that is the answer.

**Raises**

Nothing.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="CameraAPI.longest_exposure_ms"></a>
### `CameraAPI.longest_exposure_ms`

```python
longest_exposure_ms: float | None  # property
```

Route: `GET /api/v1/scope/camera/longest_exposure_ms`.

The longest exposure the camera may be using now, in ms.

While continuous auto-gain is armed the camera picks its own
exposure up to the ceiling the arm was given (the sensor maximum
when it was given none), and the held exposure is the value from
before the arm; otherwise the held exposure is the one in force.
With the exposure unknown it is the sensor maximum, the longest a
frame can take. Right after a write made the exposure shorter, a
frame begun before the write is still exposing under the longer one
until it ends, so that one answers until then. A consumer bounding
how long a frame may take -- a feed-death watch, a grab's wait --
needs this, not the held value.

**Returns**

float | None: The bound in ms; None when it is unknown and no camera reports a maximum.

**Raises**

Nothing.

<a id="CameraAPI.pixel_format"></a>
### `CameraAPI.pixel_format`

```python
pixel_format: str | None  # property
```

Route: `GET /api/v1/scope/camera/pixel_format`.

The pixel format the camera last reported or took (e.g. `'Mono12'`),
chosen by `set_image_mode`; None when unknown.

**Raises**

Nothing.

<a id="CameraAPI.read_auto_exposure"></a>
### `CameraAPI.read_auto_exposure`

```python
def read_auto_exposure(self) -> CameraAutoMode | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_auto_exposure`.

Read the camera's auto-exposure mode from the camera.

A camera without hardware auto-exposure answers `OFF` unasked.

**Returns**

CameraAutoMode | None: The mode; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_auto_gain"></a>
### `CameraAPI.read_auto_gain`

```python
def read_auto_gain(self) -> CameraAutoMode | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_auto_gain`.

Read the camera's auto-gain mode from the camera.

A camera without hardware auto-gain answers `OFF` unasked.

**Returns**

CameraAutoMode | None: The mode; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_binning"></a>
### `CameraAPI.read_binning`

```python
def read_binning(self) -> int | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_binning`.

Read the camera's binning factor from the camera.

**Returns**

int | None: The factor, 1 or more; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_black_level"></a>
### `CameraAPI.read_black_level`

```python
def read_black_level(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_black_level`.

Read the camera's black level from the camera.

The value is the camera's own black level parameter, in the camera's
own units, as the vendors' EMVA 1288 sheets state it: on a Basler
body one unit moves the gray value by a model-specific step (0.0625 DN
at 12-bit depth on the daA3840), on an IDS body it is DN of the current
pixel format, on the FX2 it is the sensor's Row Black Target. The
offset it makes in DN is measured, not derived from it.

**Returns**

float | None: The black level; None when the camera reports none.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_exposure_min_ms"></a>
### `CameraAPI.read_exposure_min_ms`

```python
def read_exposure_min_ms(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_exposure_min_ms`.

Read from the camera the shortest exposure it takes now, in milliseconds.

On IDS bodies the floor rises above the value read at connect
(`capabilities.camera_limits.exposure_min_ms`) as the pixel clock,
frame rate and frame size change, and this reads the camera's live
floor; every other driver answers the floor it read at connect.

**Returns**

float | None: The floor; None when the camera declares none.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_exposure_ms"></a>
### `CameraAPI.read_exposure_ms`

```python
def read_exposure_ms(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_exposure_ms`.

Read the camera's exposure time from the camera, in milliseconds.

**Returns**

float | None: The exposure; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_frame_rate_fps"></a>
### `CameraAPI.read_frame_rate_fps`

```python
def read_frame_rate_fps(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_frame_rate_fps`.

Read from the camera the frame rate its current settings allow, in
frames per second.

The camera's own figure: Basler's resulting acquisition frame rate,
IDS's AcquisitionFrameRate maximum, the simulator's pacing rate. The
FX2 reports none. It is what the camera says it can do, not what
reaches the host; what reaches the host is `get_delivered_rate`.

**Returns**

float | None: Frames per second; None when the camera reports none.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_frame_size"></a>
### `CameraAPI.read_frame_size`

```python
def read_frame_size(self) -> FrameSize | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_frame_size`.

Read the size of the frames the camera delivers, from the camera.

**Returns**

FrameSize | None: The delivered size, in pixels after binning; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_gain_db"></a>
### `CameraAPI.read_gain_db`

```python
def read_gain_db(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_gain_db`.

Read the camera's gain from the camera, in dB.

**Returns**

float | None: The gain; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.read_pixel_format"></a>
### `CameraAPI.read_pixel_format`

```python
def read_pixel_format(self) -> str | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/camera/read_pixel_format`.

Read the camera's pixel format from the camera (e.g. `'Mono12'`).

The format follows `set_image_mode`, which chooses it.

**Returns**

str | None: The format; None when unknown.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected; `'scope_disconnected'` after `disconnect()`.
- `CameraReadFailedError`: The camera did not answer the read.

**Refusals by REST status**

- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.remove_exposure_listener"></a>
### `CameraAPI.remove_exposure_listener`

```python
def remove_exposure_listener(self, listener: Callable[[float], None]) -> None
```

Marks: in-process.

Unregister an exposure listener.

**Args**

- `listener`: A callable previously passed to `add_exposure_listener`. Silently ignores listeners that are not currently registered.

**Raises**

Nothing.

<a id="CameraAPI.remove_gain_listener"></a>
### `CameraAPI.remove_gain_listener`

```python
def remove_gain_listener(self, listener: Callable[[float], None]) -> None
```

Marks: in-process.

Unregister a gain listener.

**Args**

- `listener`: A callable previously passed to `add_gain_listener`. Silently ignores listeners that are not currently registered.

**Raises**

Nothing.

<a id="CameraAPI.restore_camera_state"></a>
### `CameraAPI.restore_camera_state`

```python
def restore_camera_state(self, snapshot: dict) -> None
```

Marks: in-process; engineering.

Restore every camera setting a saved state holds, and wait.

The binning, frame size, pixel format and black level go back
first, each only where it differs from the camera's value now, since
a write of an unchanged one restarts grabbing for nothing; then the
gain, the exposure and the auto-gain arm. A setting absent from the
snapshot, or recorded as None, is skipped and named in the log. The
auto-gain loop goes back to the snapshot's state: re-armed when an
arm was recorded, turned off first when none was and one stands now,
untouched when the snapshot does not say. The restore is one command
on the camera's lane, so its writes cannot interleave with another
caller's. A camera already declared removed when the restore
starts has nothing to put back: it returns with nothing written, its
removal already told by whatever noticed it.

**Args**

- `snapshot`: Return value from `save_camera_state`.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera: none is connected and the snapshot holds a setting to restore, or the restore found it removed.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="CameraAPI.save_camera_state"></a>
### `CameraAPI.save_camera_state`

```python
def save_camera_state(self, tag: str) -> dict
```

Marks: in-process; engineering.

Snapshot the camera's settings, for restoration.

The gain and exposure are the values held (the last the camera
reported or took, with no read of the camera now), None for one that
is unknown, which the restore leaves as it is. `auto_gain_arm` is always present: the standing
continuous arm, or None when none stood -- a run that ends with
its step-end disarm otherwise leaves the loop off while the layer
toggle shows on, until a slider write happens to re-arm it.

With a camera active the snapshot also holds its frame size and
pixel format, its binning where it offers more than one size, and its
black level where that can be set (`capabilities`). These are read
live, and a failed read raises: the camera cannot report its own
setting, and a restore that left it out would leave the scope changed
without saying so. With them, `stored` holds the saved image mode,
binning, frame and unbinned region, which the restore saves back: three
image modes share one format, so the mode cannot be read back from the
camera. A scope no session composed has no settings to read them from.

**Args**

- `tag`: Descriptive name for the snapshot (for logging).

**Returns**

dict: Snapshot suitable for passing to `restore_camera_state`.

**Raises**

- `CameraReadFailedError`: `'camera_read_failed'`, a setting the camera offers could not be read.
- `ConfigurationDefectError`: `'scope_not_composed'`, no session bound this scope; read only with a camera active.

**Refusals**

- [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`
- [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="CameraAPI.set_auto_exposure"></a>
### `CameraAPI.set_auto_exposure`

```python
def set_auto_exposure(self, mode: CameraAutoMode) -> bool | None
```

Marks: sets [`auto_exposure`](#CameraAPI.auto_exposure); enters: `lane_call`.  
Route: `PUT /api/v1/scope/camera/auto_exposure`.

Put the camera's auto-exposure in `mode`, and wait for it.

One command on the camera's lane, waited for; a write the camera
does not apply raises. With no camera connected, turning
auto-exposure off is satisfied: True, as a camera already in manual
answers; so is turning off a mode a camera without hardware
auto-exposure never had, with nothing written.

**Args**

- `mode`: `CameraAutoMode.OFF` or `CONTINUOUS` (`'Off'` or `'Continuous'` on the wire).

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, for any other `mode`: a one-shot is `auto_gain_once`'s. Nothing is sent.
- `CameraSettingRejected`: The camera refused the change; nothing is shown here.
- `CameraSettingUnsupportedError`: Turning auto-exposure on for a camera without it.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, for turning auto-exposure on with no camera connected or one removed during the write; turning it off is satisfied either way.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `auto_gain_unsupported`, `auto_exposure_unsupported`, `auto_gain_target_brightness_unsupported`, `binning_unsupported`, `black_level_unsupported`, `conversion_gain_mode_unsupported`, `line_noise_reduction_unsupported`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_auto_gain"></a>
### `CameraAPI.set_auto_gain`

```python
def set_auto_gain(
    self,
    state: bool,
    settings: dict,
    *,
    resume_after_capture: bool = True,
) -> bool | None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/camera/set_auto_gain`.

Enable or disable automatic gain adjustment, and wait for it.

One command on the camera's lane, waited for. A write the camera
does not apply raises and records no arm. With no camera connected,
turning auto-gain off is satisfied: True, as a camera already in
manual answers.

**Args**

- `state`: True to turn auto-gain on, False to turn it off.
- `settings`: The arm's settings: `target_brightness`, `min_gain_db`, `max_gain_db`, and optionally `max_exposure_ms` and `min_exposure_ms`, the bounds the exposure may settle in. Read only to arm; turning auto-gain off reads none.
- `resume_after_capture`: With `state` True, whether a capture that locks this arm re-arms it afterwards (live view) or leaves the camera at the locked values (a protocol step).

**Returns**

bool | None: The camera's answer, True or None; True, with nothing written, for turning off a mode a camera without hardware auto-gain never had.

**Raises**

- `CameraSettingRejected`: The camera refused the change. Nothing was recorded, and nothing is shown here.
- `CameraSettingUnsupportedError`: Turning auto-gain on for a camera without it.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, for turning auto-gain on with no camera connected or one removed during the write; turning it off is satisfied either way.
- `ArgumentRefusedError`: `'missing_key'`, `settings` lacks a key the write reads; `'wrong_kind'` or `'not_a_number'`, it or one of its values is not of the kind read; `'key_unknown'`, it holds a key the write does not take; `'gain_range_inverted'`, its minimum gain is above its maximum.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `auto_gain_unsupported`, `auto_exposure_unsupported`, `auto_gain_target_brightness_unsupported`, `binning_unsupported`, `black_level_unsupported`, `conversion_gain_mode_unsupported`, `line_noise_reduction_unsupported`; [`ArgumentRefusedError`](#ArgumentRefusedError): `missing_key`, `wrong_kind`, `not_a_number`, `key_unknown`, `gain_range_inverted`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_binning"></a>
### `CameraAPI.set_binning`

```python
def set_binning(self, size: int) -> FrameSize | None
```

Marks: sets [`binning`](#CameraAPI.binning); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/scope/camera/binning`.

Bin the camera by `size`, keep the framed region, and save both.

The one writer of the binning for every host. The framed region is
saved unbinned (`frame.native_width` / `frame.native_height`), so
a new binning divides that region by the new factor rather than
rescaling the last displayed size, and cycling the binning
round-trips. The binning goes first and the frame after it, because
the camera's frame limits depend on the binning in force; one caller
running both in order is what keeps a frame edit from being worked
out against a binning the camera has not reached.

**Returns**

The frame the camera delivers.

**Raises**

- `HardwareCommandRefusedError`: A recording holds the scope: its frames and their pixel size follow the binning it started with. `'not_connected'`, naming the camera, with none connected or one removed during the write. Nothing is saved.
- `CameraSettingUnsupportedError`: `'binning_unsupported'`, this camera does not offer `size`. Nothing reaches the camera.
- `CameraSettingRejected`: The camera refused the binning (nothing is saved) or the frame after it (the binning is saved, with the frame the camera reports holding at it, and no frame when it has never reported one).

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `binning_unsupported`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_black_level"></a>
### `CameraAPI.set_black_level`

```python
def set_black_level(self, level: float) -> float | None
```

Marks: sets [`black_level`](#CameraAPI.black_level); enters: `lane_call`.  
Route: `PUT /api/v1/scope/camera/black_level`.

Set the camera's black level, wait for it, and answer with the
value in effect.

See `read_black_level` for what the value means. The value holds
until the next connect, when the camera reloads its default user set.

**Returns**

float | None: The black level now in effect, which differs from the request when the camera snapped it.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one removed during the write.
- `CameraSettingUnsupportedError`: This camera has no black level setting (`capabilities.camera_supports_black_level` is False). Nothing reached the camera.
- `ArgumentRefusedError`: `'not_a_number'`, `level` is not a finite number; nothing was sent to the camera.
- `CameraSettingOutOfRangeError`: `level` is outside the range the camera reports for its current pixel format; nothing was sent.
- `CameraSettingRejected`: The camera refused it, as a camera holding its black level automatically does, or raised from the write (chained).
- `CameraReadFailedError`: `'camera_read_failed'`, the camera's probe of its black level setting or the read of its range failed. Nothing was sent.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `auto_gain_unsupported`, `auto_exposure_unsupported`, `auto_gain_target_brightness_unsupported`, `binning_unsupported`, `black_level_unsupported`, `conversion_gain_mode_unsupported`, `line_noise_reduction_unsupported`; [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`CameraSettingOutOfRangeError`](#CameraSettingOutOfRangeError): `gain_db_out_of_range`, `exposure_ms_out_of_range`, `frame_width_out_of_range`, `frame_height_out_of_range`, `black_level_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_exposure_ms"></a>
### `CameraAPI.set_exposure_ms`

```python
def set_exposure_ms(self, exposure_ms: float) -> float | None
```

Marks: sets [`exposure_ms`](#CameraAPI.exposure_ms); enters: `lane_call`.  
Route: `PUT /api/v1/scope/camera/exposure_ms`.

Set the camera exposure time, wait for it, and answer with the
exposure in effect.

One command on the camera's lane, waited for, as `set_gain_db`
is; a write the camera does not apply raises. The write is a raw
one: afterwards the camera holds no channel's stored settings.

**Args**

- `exposure_ms`: The exposure time in milliseconds, within the camera's declared range (`ScopeCapabilities.camera_limits`).

**Returns**

float | None: The exposure in milliseconds now in effect, which differs from the request when the camera clamped it to its floor or quantized it. `None` when the camera's teardown landed after the lane asked for it.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, the exposure is not a finite number; nothing was sent to the camera.
- `CameraSettingOutOfRangeError`: The exposure is outside the range the camera declares; nothing was sent to it.
- `CameraSettingRejected`: A live driver refused the exposure or raised from the write (chained). It carries the words its reporter shows; nothing is shown here.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one removed during the write.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`CameraSettingOutOfRangeError`](#CameraSettingOutOfRangeError): `gain_db_out_of_range`, `exposure_ms_out_of_range`, `frame_width_out_of_range`, `frame_height_out_of_range`, `black_level_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_frame_size"></a>
### `CameraAPI.set_frame_size`

```python
def set_frame_size(self, width: int, height: int) -> FrameSize | None
```

Marks: sets [`frame_size`](#CameraAPI.frame_size); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/scope/camera/frame_size`.

Frame the camera at `width` x `height` at the binning it holds, and save it.

The one writer of the frame for every host. The size is what the
person sees and captures (post-binning); the unbinned region it
implies is saved beside it. A size the camera already delivers is
not written again.

**Returns**

The frame the camera delivers: the request floored to even sides.

**Raises**

- `HardwareCommandRefusedError`: A recording holds the scope: its frames are fitted to the geometry it started with. `'not_connected'`, naming the camera, with none connected or one removed during the write. Nothing is saved.
- `CameraSettingOutOfRangeError`: `'frame_width_out_of_range'` or `'frame_height_out_of_range'`, the size, floored to even sides, is below the camera's minimum frame or above the scope's maximum at the binning held: the sensor, or the model's smaller maximum (`capabilities.camera_limits`). Nothing reaches the camera or is saved; an undeclared end is not checked.
- `CameraSettingRejected`: The camera refused the frame. Nothing is saved.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingOutOfRangeError`](#CameraSettingOutOfRangeError): `frame_width_out_of_range`, `frame_height_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_gain_db"></a>
### `CameraAPI.set_gain_db`

```python
def set_gain_db(self, gain_db: float) -> float | None
```

Marks: sets [`gain_db`](#CameraAPI.gain_db); enters: `lane_call`.  
Route: `PUT /api/v1/scope/camera/gain_db`.

Set the camera gain, wait for it, and answer with the gain in effect.

The write is one command on the camera's lane: it waits for any
camera command already in flight, and returns once the camera has
taken the gain. A write the camera does not apply raises, so the
returned value is the success and a refusal cannot pass for one. The
write is a raw one: afterwards the camera holds no channel's stored
settings, and a frame taken next records `held_channel` None.

**Args**

- `gain_db`: The gain in dB, within the camera's declared range (`ScopeCapabilities.camera_limits`).

**Returns**

float | None: The gain in dB now in effect, which differs from the request when the camera snapped it. `None` when the camera's teardown landed after the lane asked for it.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, the gain is not a finite number; nothing was sent to the camera.
- `CameraSettingOutOfRangeError`: The gain is outside the range the camera declares; nothing was sent to it.
- `CameraSettingRejected`: A live driver refused the gain or raised from the write (chained). It carries the words its reporter shows; nothing is shown here.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one removed during the write.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`CameraSettingOutOfRangeError`](#CameraSettingOutOfRangeError): `gain_db_out_of_range`, `exposure_ms_out_of_range`, `frame_width_out_of_range`, `frame_height_out_of_range`, `black_level_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_high_conversion_gain"></a>
### `CameraAPI.set_high_conversion_gain`

```python
def set_high_conversion_gain(self, enabled: bool) -> None
```

Marks: sets [`high_conversion_gain`](#CameraAPI.high_conversion_gain); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/scope/camera/high_conversion_gain`.

Turn the camera's high conversion gain on or off, then save it.

High conversion gain lowers the sensor's read-noise floor at the cost
of dynamic range. The one writer of `camera.high_conversion_gain`:
the setting is saved only once the camera took it, so the saved
settings never name a mode the camera is not in. Off on a camera
without the mode is its state, and is saved.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one removed during the write. Nothing is saved.
- `CameraSettingUnsupportedError`: `'conversion_gain_mode_unsupported'`, on, for a camera without the mode. Nothing reaches the camera or is saved.
- `CameraSettingRejected`: The camera refused it or raised from it (chained). Nothing is saved.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `conversion_gain_mode_unsupported`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_image_mode"></a>
### `CameraAPI.set_image_mode`

```python
def set_image_mode(self, mode: ImageMode) -> None
```

Marks: sets [`image_mode`](#CameraAPI.image_mode); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/scope/camera/image_mode`.

Capture in `mode`: apply the camera format it needs, then save it.

The one writer of `image_mode` for every host. The mode names a
capture depth, and the format is chosen from the ones this camera
reports, so a sensor is never asked for a format it lacks; a pixel
format is set only this way, so a capture's depth and its save
policy agree. A camera that reports none (none is connected) has
nothing to apply; the mode is saved and bring-up applies it.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `mode` is not an image mode. Nothing is applied or saved.
- `HardwareCommandRefusedError`: A recording holds the scope: its frames are written at the depth it started with. `'not_connected'`, naming the camera, when it went away between the format query and the apply. Nothing is saved.
- `CameraSettingRejected`: The camera refused the format. Nothing is saved, so captures are never tagged with a depth the camera is not delivering.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.set_line_noise_reduction"></a>
### `CameraAPI.set_line_noise_reduction`

```python
def set_line_noise_reduction(self, enabled: bool) -> None
```

Marks: sets [`line_noise_reduction`](#CameraAPI.line_noise_reduction); enters: `lane_call`, `guarded_write`.  
Route: `PUT /api/v1/scope/camera/line_noise_reduction`.

Turn the camera's line-noise filter on or off, then save it.

The one writer of `camera.line_noise_reduction`, saved only once
the camera took it, as `set_high_conversion_gain` is.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one removed during the write. Nothing is saved.
- `CameraSettingUnsupportedError`: `'line_noise_reduction_unsupported'`, on, for a camera without the filter. Nothing reaches the camera or is saved.
- `CameraSettingRejected`: The camera refused it or raised from it (chained). Nothing is saved.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `line_noise_reduction_unsupported`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="CameraAPI.update_auto_gain_target_brightness"></a>
### `CameraAPI.update_auto_gain_target_brightness`

```python
def update_auto_gain_target_brightness(self, target_brightness: float) -> bool | None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/camera/update_auto_gain_target_brightness`.

Set the auto-gain target brightness, and wait for it.

One command on the camera's lane, waited for; a write the camera
does not apply raises.

**Args**

- `target_brightness`: The brightness auto-gain settles on, as a fraction of full scale, 0.0 to 1.0.

**Returns**

bool | None: The camera's answer. `None` when no camera is active.

**Raises**

- `CameraSettingRejected`: The camera refused the change; nothing is shown here.
- `CameraSettingUnsupportedError`: The camera has no hardware auto-gain, so there is no target to set.
- `ArgumentRefusedError`: `'not_a_number'`, the target is not a finite number.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one removed during the write.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`CameraSettingUnsupportedError`](#CameraSettingUnsupportedError): `auto_gain_unsupported`, `auto_exposure_unsupported`, `auto_gain_target_brightness_unsupported`, `binning_unsupported`, `black_level_unsupported`, `conversion_gain_mode_unsupported`, `line_noise_reduction_unsupported`; [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraSettingRejected`](#CameraSettingRejected): `camera_rejected`, `camera_no_answer`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="DiagnosticsAPI"></a>
## `DiagnosticsAPI`

Reached through `session.scope.diagnostics`; on the wire, `/api/v1/scope/diagnostics/`.  
Held in: [`Lumascope.diagnostics`](#Lumascope.diagnostics).

Diagnostics sub-API. Forwards to Lumascope composition root.

Every member that transmits runs on its device's lane, so while a run, a
diagnostic or a home holds the scope it is refused to anyone not acting under
the holder's taking. The one exception is the camera temperature read,
which the lane admits whatever holds: it changes nothing a holder
depends on, and the temperature log keeps running through a run. The
board commands go on the IO lane through the motion API's dispatcher,
the lane's one blocking dispatch with a wait bound.

<a id="DiagnosticsAPI.enter_led_engineering_mode"></a>
### `DiagnosticsAPI.enter_led_engineering_mode`

```python
def enter_led_engineering_mode(self, timeout_s: float = 5.0) -> None
```

Marks: in-process; engineering.

Enter LED engineering mode via the driver-canonical handshake, and wait for it.

**Args**

- `timeout_s`: Max seconds to wait for the engineering-mode handshake to complete.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'` (the LED controller) or `'not_on_this_scope'` (`MissingPart.LED_ENGINEERING_MODE`, a controller without the handshake). Nothing was sent.
- `BoardNoReplyError`: `'board_no_reply'`, the board did not answer FACTORY.
- `BoardReportedError`: `'board_reported_error'`, the board did not complete the handshake: no Y/N prompt; or no engineering-mode banner after Y, when the driver has already brought it back to safe mode.
- `LedFirmwareWedgedError`: `'led_firmware_wedged'`, the handshake failed after Y and the board's soft reset did not bring its firmware back; it needs a power cycle.
- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was sent.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`, `not_connected`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`
- [`BoardReportedError`](#BoardReportedError): `board_reported_error`
- [`LedFirmwareWedgedError`](#LedFirmwareWedgedError): `led_firmware_wedged`

<a id="DiagnosticsAPI.exit_led_engineering_mode"></a>
### `DiagnosticsAPI.exit_led_engineering_mode`

```python
def exit_led_engineering_mode(self) -> None
```

Marks: in-process; engineering.

Exit LED engineering mode via the driver-canonical handshake, and wait for it.

Driver method drains and sleeps after Q so the LED firmware
actually transitions out of eng mode.

**Raises**

- `HardwareCommandRefusedError`: as `enter_led_engineering_mode`.
- `LedFirmwareWedgedError`: `'led_firmware_wedged'`, Q failed and the board's soft reset did not bring its firmware back; it needs a power cycle.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`, `capture_in_flight`, `home_in_flight`, `home_not_stoppable`, `position_unread`, `lid_open_before_motion`, `stage_unpowered_before_motion`, `camera_not_streaming`, `axis_absent`, `not_on_this_scope`, `activity_ended`, `scope_disconnected`, `monitor_ended`, `not_connected`
- [`LedFirmwareWedgedError`](#LedFirmwareWedgedError): `led_firmware_wedged`

<a id="DiagnosticsAPI.get_camera_link_info"></a>
### `DiagnosticsAPI.get_camera_link_info`

```python
def get_camera_link_info(self) -> CameraLink | None
```

Marks: reads (changes nothing); enters: `lane_call`.  
Route: `GET /api/v1/scope/diagnostics/get_camera_link_info`.

Read the camera's link, live.

Runs on the camera lane, admitted whatever holds the scope, like the
temperature read: the reads are node reads that must not interleave
with a camera write, and a run does not stop them.

**Returns**

CameraLink | None: The link, each field None where the camera does not report it. None when no camera is connected, after `disconnect()` too.

**Raises**

- `CameraReadFailedError`: `'camera_read_failed'`, a field the camera reports could not be read.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="DiagnosticsAPI.get_camera_temperatures_degc"></a>
### `DiagnosticsAPI.get_camera_temperatures_degc`

```python
def get_camera_temperatures_degc(self) -> dict[str, float] | None
```

Marks: reads (changes nothing); enters: `lane_call`.  
Route: `GET /api/v1/scope/diagnostics/get_camera_temperatures_degc`.

Get all camera temperature sensor readings.

Runs on the camera lane, admitted whatever holds the scope: the read
sets the camera's temperature selector, which must not interleave
with another camera write, but a run does not stop it.

**Returns**

dict[str, float] | None: Each sensor's temperature in degC, by the sensor's name, which is the camera's own (a Basler body's `'FpgaCore'`; `'Device'` on an IDS body with one sensor). Empty only for a camera with no temperature sensor (`capabilities.camera_reports_temperature` False). None when no camera is connected, after `disconnect()` too: a read with no camera to describe.

**Raises**

- `CameraReadFailedError`: `'camera_read_failed'`, a camera is active and the read failed.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraReadFailedError`](#CameraReadFailedError): `camera_read_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="DiagnosticsAPI.get_camera_usb_bindings"></a>
### `DiagnosticsAPI.get_camera_usb_bindings`

```python
def get_camera_usb_bindings(self) -> tuple[UsbIdentityBindings, ...] | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/diagnostics/get_camera_usb_bindings`.

Which driver Windows has bound to each USB identity of a Lumascope Classic.

The FX2 is two USB devices in turn, the bootloader and the running
firmware, and Windows binds each on its own. Read without elevation,
the same read the camera's connect checks, so it answers whether or
not the camera came up -- the case it is for. Takes no lane: it asks
the host's device list, not the scope.

**Returns**

tuple[UsbIdentityBindings, ...] | None: The bootloader's, then the running firmware's. None where the host binds no driver to a USB device (macOS, Linux) and on a simulated scope with no FX2.

**Raises**

- `CameraUsbLookupFailedError`: Windows' device list could not be read.

**Refusals by REST status**

- 500: [`CameraUsbLookupFailedError`](#CameraUsbLookupFailedError): `camera_usb_lookup_failed`

<a id="DiagnosticsAPI.get_led_info"></a>
### `DiagnosticsAPI.get_led_info`

```python
def get_led_info(self) -> LedBoardInfo
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/diagnostics/get_led_info`.

The LED board's firmware and command set.

**Returns**

LedBoardInfo: Each field None when unknown, all of them with no board connected.

**Raises**

Nothing.

<a id="DiagnosticsAPI.get_motor_info"></a>
### `DiagnosticsAPI.get_motor_info`

```python
def get_motor_info(self) -> MotorBoardInfo
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/diagnostics/get_motor_info`.

The motor board's identity.

Served from what FULLINFO and INFO cached at connect, never re-queried
over the serial bus: identity is fixed for the life of a connection,
and this is read for every saved frame's metadata.

**Returns**

MotorBoardInfo: Each field None when unknown, all of them with no board connected -- none installed, or one lost since connect, whose cached identity no longer describes a board that is there.

**Raises**

Nothing.

<a id="DiagnosticsAPI.motor_fan_spin_times"></a>
### `DiagnosticsAPI.motor_fan_spin_times`

```python
def motor_fan_spin_times(self) -> dict[str, float] | None
```

Marks: in-process; reads (changes nothing); engineering.

How long the motor-board fan takes to reach a new duty, in seconds.

`{'spin_up_s': ..., 'spin_down_s': ...}`: up to speed from a stop,
and down to a stop from speed, as the board's driver states them. A
`read_motor_fan_rpm` sooner after `set_motor_fan_duty` reads the
old speed. None when no motor board's driver drives a fan; whether
the board's firmware takes a fan write is `set_motor_fan_duty`'s
refusal, asked on the board.

**Raises**

Nothing.

<a id="DiagnosticsAPI.read_led_currents_ma"></a>
### `DiagnosticsAPI.read_led_currents_ma`

```python
def read_led_currents_ma(self) -> dict[int, float | None]
```

Marks: in-process; reads (changes nothing); engineering.

Read the measured current of every LED channel, in mA.

One pass over the board's channels on the IO lane. The v2 firmware
answers these reads only in engineering mode; the caller enters it.

**Returns**

dict: Channel -> mA, or None for a channel whose current could not be read (no reply, an unparseable one, or a board with no current sensing). Empty when no LED board is connected.

**Raises**

Nothing.

<a id="DiagnosticsAPI.read_motor_drv_status"></a>
### `DiagnosticsAPI.read_motor_drv_status`

```python
def read_motor_drv_status(self, axis: Axis) -> int | None
```

Marks: in-process; reads (changes nothing); engineering.

Read TMC5072 DRV_STATUS register for an axis.

Returns the raw register value as int (caller decodes bits),
or None when the firmware does not implement DRVSTAT_&lt;axis>.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `axis` is no axis name; nothing was read.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`

<a id="DiagnosticsAPI.read_motor_fan_rpm"></a>
### `DiagnosticsAPI.read_motor_fan_rpm`

```python
def read_motor_fan_rpm(self) -> int | None
```

Marks: in-process; reads (changes nothing); engineering.

Read motor-board fan tachometer RPM.

Returns RPM as int (0 if no tach wire) or None when firmware
does not implement FANSPEED.

**Raises**

Nothing.

<a id="DiagnosticsAPI.run_camera_bandwidth_test"></a>
### `DiagnosticsAPI.run_camera_bandwidth_test`

```python
def run_camera_bandwidth_test(
    self,
    num_frames: int,
    *,
    timeout_s: float = 60.0,
    progress_cb: ProgressCallback | None = None,
) -> dict
```

Marks: in-process; engineering.

Run an N-frame camera throughput test through the production capture path.

Every frame is grabbed through `ImagingAPI.get_image`, so the
numbers are those of the path a protocol and the live view capture
through.

**Args**

- `num_frames`: Total frames to grab.
- `timeout_s`: Hard wall-clock cutoff in seconds; the test stops early and marks `passed=False` if exceeded.
- `progress_cb`: Optional `callback(percent_int, message_str)` called every 250 frames.

**Returns**

dict: num_frames_requested, num_frames_received, num_frames_failed (the grabs that raised `CaptureFailedError`), failed_by_reason (those, counted by the failure's reason), total_bytes, elapsed_seconds, mb_per_second, fps_actual, frame_sizes, errors (the first 20 failures' words), passed, and resolution (`'WxH'`) and pixel_format, as the camera held them, None when unknown.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not from 0 to 30 days, the longest wait a member takes, or `num_frames` is outside 1 to 100,000. Nothing reached the camera.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, absent at the call or removed during the test. Only a capture failure is counted: any other raise from a grab ends the test as itself.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="DiagnosticsAPI.run_grab_lifecycle_benchmark"></a>
### `DiagnosticsAPI.run_grab_lifecycle_benchmark`

```python
def run_grab_lifecycle_benchmark(
    self,
    num_cycles: int = 100,
    inter_cycle_delay_s: float = 0.0,
    vary_settings: bool = False,
    *,
    slow_threshold_s: float = 3.0,
    progress_cb: ProgressCallback | None = None,
) -> dict
```

Marks: in-process; engineering.

Characterize stop_grabbing/start_grabbing latency, on the camera lane, and wait.

Each cycle stops the camera's grab and starts it again, timing both:
a measured floor for the interval a camera needs between a stop and
the next start. The typical cycle is 130-150 ms; one near 11 s has
been seen when a stop lands within about 275 ms of a start, before
the camera produced a frame. Sweeping `inter_cycle_delay_s` across
runs shows the smallest delay with no slow cycle. The benchmark is
one command on the camera's lane, waited for, bounded per cycle.

**Args**

- `num_cycles`: Stop-and-start cycles to run, 1 to 10,000.
- `inter_cycle_delay_s`: Seconds between a stop and the next start (and any settings change), 0 to 60.
- `vary_settings`: True to alternate the gain (1 and 4 dB) and the exposure (10 and 50 ms) between cycles, as a protocol's steps do.
- `slow_threshold_s`: A cycle's seconds at which it counts as slow, 0 to 600; slow cycles are counted apart.
- `progress_cb`: Called as `callback(percent, message)` every 10 cycles, on the camera lane's worker.

**Returns**

dict: `num_cycles`, `inter_cycle_delay_s`, `vary_settings`, `slow_threshold_s`, `slow_cycle_count`, `slow_cycles` (each `idx`, `cycle_s`, `stop_s`, `start_s`), the 50th, 95th and 99th percentiles of the cycle, stop and start times (`cycle_p50` ...), `total_elapsed_s`, `camera_model`, `pylon_version`, `errors` and `written_to`.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or once a cycle finds it removed.
- `ArgumentRefusedError`: `'not_a_number'`, a time is not a finite number; `'out_of_range'`, `num_cycles` is outside 1 to 10,000, `inter_cycle_delay_s` outside 0 to 60 or `slow_threshold_s` outside 0 to 600. Nothing reached the camera.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="DiagnosticsAPI.run_pylon_diagnostic_probe"></a>
### `DiagnosticsAPI.run_pylon_diagnostic_probe`

```python
def run_pylon_diagnostic_probe(
    self,
    duration_s: float = 3.0,
    *,
    drain_camera_side_errors: bool = True,
    progress_cb: ProgressCallback | None = None,
) -> dict
```

Marks: in-process; engineering.

One-shot Pylon-camera diagnostic probe, on the camera lane, and wait.

Records the camera's identity, its configuration and the change in
its stream statistics over `duration_s`, with the host's OS, name
and camera SDK version, in one JSON file under the log folder's
`camera_probe/`, named for the model, serial, firmware, host and
time so a sweep's files sort together. The grab state is not
changed: a camera not grabbing shows counters near zero, so call it
with the live view running. The probe is one command on the camera's
lane, waited for. It pops the camera's error queue, so it is a write
to the camera and refused while another activity holds the scope.

**Args**

- `duration_s`: The sampling window in seconds, 0 to 3,600.
- `drain_camera_side_errors`: True to drain the camera's own error queue and record its codes, which Basler documents for its support only.
- `progress_cb`: Called as `callback(percent, message)` at the start, mid-window and the end, on the camera lane's worker.

**Returns**

dict: The camera's snapshot with the host, the times and the `output_path`; `{'connected': False, 'errors': [...]}` with no camera active, and `'supported': False` besides for a camera whose driver has no probe.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected.
- `ArgumentRefusedError`: `'not_a_number'`, `duration_s` is not a finite number; `'out_of_range'`, it is outside 0 to 3,600. Nothing reached the camera.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="DiagnosticsAPI.set_motor_fan_duty"></a>
### `DiagnosticsAPI.set_motor_fan_duty`

```python
def set_motor_fan_duty(self, duty_pct: int) -> None
```

Marks: in-process; engineering.

Set motor-board fan PWM duty cycle (0..100), and wait for the board to take it.

The fan reaches the new duty later: `motor_fan_spin_times`.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'` (the motor controller); `'axis_absent'`, `MissingPart.MOTORS` on a manual scope; or `'not_on_this_scope'`, `MissingPart.FAN_CONTROL` for a controller without fan control. Nothing was sent.
- `ArgumentRefusedError`: `'fan_duty_out_of_range'`, `duty_pct` is outside 0..100. Asked after the hardware, before anything is sent.
- `BoardNoReplyError`: `'board_no_reply'`, the board did not answer the write.
- `BoardReportedError`: `'board_reported_error'`, the board answered the write with an error.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `not_on_this_scope`, `not_connected`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `fan_duty_out_of_range`
- [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`
- [`BoardReportedError`](#BoardReportedError): `board_reported_error`

<a id="IlluminationAPI"></a>
## `IlluminationAPI`

Reached through `session.scope.illumination`; on the wire, `/api/v1/scope/illumination/`.  
Held in: [`Lumascope.illumination`](#Lumascope.illumination).

Illumination sub-API. Owns LED state, ownership tracking, and
listener registry. Stateful bodies live here post-Phase 3d.

<a id="IlluminationAPI.add_led_listener"></a>
### `IlluminationAPI.add_led_listener`

```python
def add_led_listener(
    self,
    listener: Callable[[int, str | None, LedState, float | None], None],
) -> None
```

Marks: in-process; event: [`LedReading`](#LedReading).

Register a callback for LED state changes.

The listener is called with the changed address's reading,
`(address, name, state, illumination_ma)` as `LedReading`
holds them, whenever any LED address changes state. It fires from
the thread that caused the change, so listeners **must** schedule
UI work via `Clock.schedule_once`.

**Args**

- `listener`: `callable(address: int, name: str | None, state: LedState, illumination_ma: float | None)`

**Raises**

Nothing.

<a id="IlluminationAPI.get_led_states"></a>
### `IlluminationAPI.get_led_states`

```python
def get_led_states(self) -> tuple[LedReading, ...] | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/illumination/get_led_states`.

Read every LED address of the attached board.

Every address is reported, whether or not the unit's identity
names it, so a lit address is never hidden by its missing name.

**Returns**

One reading per address, in address order, each named through the unit's identity. None when no LED board is installed (none came up, or after `disconnect()`).

**Raises**

Nothing.

<a id="IlluminationAPI.led_off"></a>
### `IlluminationAPI.led_off`

```python
def led_off(self, channel: str) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/illumination/led_off`.

Turn off a channel's LED, and wait for it.

The channel is darkened at its LED's address (`led_off_by_address`
takes the address itself). An off whose end state already holds is
satisfied with no write: a channel the API holds dark, or any off
with no LED controller and nothing believed lit. Only an off of a
channel believed lit asks for the controller.

**Args**

- `channel`: The channel to darken, by its name in this release.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel of this release, refused at the door whatever the scope is doing; nothing was sent.
- `HardwareCommandRefusedError`: `'not_connected'`, the channel is believed lit and no LED controller is connected to darken it; `'exclusive_activity_running'`, another activity holds the scope or the LEDs. Nothing was sent.
- `BoardNoReplyError`: the board did not answer; whether the LED is lit is unknown.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="IlluminationAPI.led_off_by_address"></a>
### `IlluminationAPI.led_off_by_address`

```python
def led_off_by_address(self, address: int) -> None
```

Marks: in-process; engineering.

Darken an LED board address, and wait for it.

The engineering counterpart of `led_off`, as `led_on_by_address`
is of `led_on`; an off whose end state already holds is satisfied
with no write, as `led_off` says.

**Args**

- `address`: The LED board address.

**Raises**

- `ArgumentRefusedError`: `'led_address_unknown'`, `address` is not on the attached board's table, offering its addresses; nothing was sent.
- `HardwareCommandRefusedError`: `'not_connected'`, the address is believed lit and no LED controller is connected to darken it; `'exclusive_activity_running'`, another activity holds the scope or the LEDs. Nothing was sent.
- `BoardNoReplyError`: the board did not answer; whether the LED is lit is unknown.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`, `not_connected`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `led_address_unknown`
- [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`

<a id="IlluminationAPI.led_on"></a>
### `IlluminationAPI.led_on`

```python
def led_on(self, channel: str, illumination_ma: float) -> float
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/illumination/led_on`.

Turn on a channel's LED at the specified current, and wait for it.

The channel is lit at its LED's address (`led_on_by_address`
takes the address itself), as one command on the LED board's lane,
waited for. The board lights at the nearest current its step can
deliver, and a request above zero but below one step at one step;
that commanded current, not the request, is what is stored and
returned. 0 mA leaves the channel on at 0 mA, which the board treats
as off.

**Args**

- `channel`: The channel to light, by its name in this release.
- `illumination_ma`: The current in mA, from 0 to the attached board's maximum (`limits`).

**Returns**

The commanded current in mA.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`, `channel` is not a channel of this release, refused at the door whatever the scope is doing; `'not_a_number'`, `illumination_ma` is not a finite number; `'illumination_out_of_range'`, it is outside 0 to the attached board's maximum. Nothing was sent.
- `HardwareCommandRefusedError`: `'not_connected'` with no LED controller connected; `'not_on_this_scope'` for a channel whose LED this model does not drive; `'exclusive_activity_running'`, another activity holds the scope or the LEDs. Nothing was sent.
- `BoardNoReplyError`: the board did not answer; whether the LED is lit is unknown.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`, `not_a_number`, `illumination_out_of_range`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`, `activity_ended`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="IlluminationAPI.led_on_by_address"></a>
### `IlluminationAPI.led_on_by_address`

```python
def led_on_by_address(self, address: int, illumination_ma: float) -> float
```

Marks: in-process; engineering.

Light an LED board address at the specified current, and wait for it.

For the bench and for a model this release does not know; every
other caller lights a channel by its name (`led_on`). The address
is lit on the lane `led_on` lights a name's address on, judged in
the same order, and read back and reported by its address as a
name's is. The current is commanded on the board's step, as
`led_on` says.

**Args**

- `address`: The LED board address.
- `illumination_ma`: The current in mA, from 0 to the attached board's maximum (`limits`).

**Returns**

The commanded current in mA.

**Raises**

- `ArgumentRefusedError`: `'led_address_unknown'`, `address` is not on the attached board's table, offering its addresses; `'not_a_number'` or `'illumination_out_of_range'`, as `led_on` says. Nothing was sent.
- `HardwareCommandRefusedError`: `'not_connected'` with no LED controller connected; `'not_on_this_scope'` (`MissingPart.led_address`) for an address this model does not drive; `'exclusive_activity_running'`, another activity holds the scope or the LEDs. Nothing was sent.
- `BoardNoReplyError`: the board did not answer; whether the LED is lit is unknown.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`, `not_on_this_scope`, `not_connected`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `led_address_unknown`, `not_a_number`, `illumination_out_of_range`
- [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`

<a id="IlluminationAPI.leds_off"></a>
### `IlluminationAPI.leds_off`

```python
def leds_off(self) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/illumination/leds_off`.

Turn off all LEDs, and wait for it.

Whatever holds the LEDs, every one is turned off: any lease is
ignored. With an LED controller connected the board's all-off is
written whatever the API believes lit; without one, it is satisfied
when nothing is believed lit.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, a channel believed lit and no LED controller connected to darken it.
- `BoardNoReplyError`: the board did not answer; whether each LED is lit is unknown.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="IlluminationAPI.remove_led_listener"></a>
### `IlluminationAPI.remove_led_listener`

```python
def remove_led_listener(
    self,
    listener: Callable[[int, str | None, LedState, float | None], None],
) -> None
```

Marks: in-process.

Unregister an LED listener.

**Args**

- `listener`: A callable previously passed to `add_led_listener`. Silently ignores listeners that are not currently registered.

**Raises**

Nothing.

<a id="IlluminationAPI.restore_led_state"></a>
### `IlluminationAPI.restore_led_state`

```python
def restore_led_state(self, snapshot: LedSnapshot) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/illumination/restore_led_state`.

Restore LEDs to a previously saved state, as one task on the IO lane, and wait.

Turns off the addresses the snapshot does not have lit, then lights
the addresses it has lit at a positive current.

**Args**

- `snapshot`: Return value from `save_led_state`.

**Raises**

- `HardwareCommandRefusedError`: `'exclusive_activity_running'`, another activity holds the scope or the LEDs, raised at the first channel it would change.
- `BoardNoReplyError`: the board did not answer a write; that channel is unknown and the writes after it are not made.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="IlluminationAPI.save_led_state"></a>
### `IlluminationAPI.save_led_state`

```python
def save_led_state(self, tag: str) -> LedSnapshot | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/illumination/save_led_state`.

Snapshot every LED address for later restoration.

**Args**

- `tag`: Descriptive name for the snapshot (for logging).

**Returns**

Snapshot suitable for passing to `restore_led_state`; None when no LED board is installed, so there is no state to keep.

**Raises**

Nothing.

<a id="ImagingAPI"></a>
## `ImagingAPI`

Reached through `session.scope.imaging`; on the wire, `/api/v1/scope/imaging/`.  
Held in: [`Lumascope.imaging`](#Lumascope.imaging).

Imaging sub-API: the capture. Owns the grab, the capture's record,
the frame listeners, the stream check, the scale bar and the
capture/focus events. The camera's settings and state are
`CameraAPI`'s, which this holds and calls.

<a id="ImagingAPI.add_frame_listener"></a>
### `ImagingAPI.add_frame_listener`

```python
def add_frame_listener(
    self,
    cb: Callable[[Frame], None],
    name: str | None = None,
) -> None
```

Marks: in-process; event: [`FrameDelivered`](#FrameDelivered).

Register a per-frame listener fired on every successful grab.

The canonical entry point for live_processing plugins (see
`ctx.plugins.live_processing`) and the manual-record path.
The supplied handler is wrapped in a budget enforcer
(`HANDLER_BUDGET_MS` per call; `HANDLER_DROP_K` consecutive
over-budget or raising invocations remove it, reported once as a
`FrameHandlerRemovedError` warning). Callback
signature is `cb(frame)`, the camera's `Frame` record
(`modules.lumascope_api.Frame`: `image`, `seq`,
`timestamp`, `arrived_ns`, `significant_bits`, `chunks`);
runs on the SDK
callback thread (Pylon `PylonImageGrab` / IDS grab loop /
simulated acquisition thread). Listeners MUST NOT block -- heavy work belongs
on an executor.

**Args**

- `cb`: Per-frame handler. Signature `cb(frame)`.
- `name`: Display name for log + notification messages on over-budget / auto-removal. Defaults to the handler's qualname.

The record is shared across all listeners, the preview and a
capture, and is read-only: `frame.image` refuses a write
(`ValueError`) and `frame.chunks` is a read-only mapping. Never
set the image writeable again: every other holder would see the
change. Draw on your own copy. Keeping the record is safe: no later
frame changes it.

Registration is idempotent for the same callable -- a second
call with the same `cb` is a no-op (the original wrapper +
name are kept).

**Raises**

- `FrameListenerNotRegisteredError`: The camera driver refused the registration, chained from its error. Nothing is left registered, so a later call can retry.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera. Nothing is registered.

**Refusals**

- [`FrameListenerNotRegisteredError`](#FrameListenerNotRegisteredError): `frame_listener_refused`
- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="ImagingAPI.capture_and_wait"></a>
### `ImagingAPI.capture_and_wait`

```python
def capture_and_wait(
    self,
    force_to_8bit: bool = True,
    *,
    accept_dark: bool = False,
    exclude_sources: tuple[FrameValiditySource, ...] = (),
    all_ones_check: bool = False,
    timeout_s: float = 0.0,
    sum_count: int = 1,
    sum_delay_s: float = 0.0,
    sum_iteration_callback: Callable[[], None] | None = None,
) -> np.ndarray
```

Marks: in-process.

Capture a frame-valid image on the camera worker, and wait for it.

A frame guaranteed to reflect the hardware as it is now: the capture
passes over stale frames until every change since the last one (an
LED, the gain, the exposure, a move) is met by its source's rule
(`ScopeCapabilities.frame_rules`) -- a frame count, time on the
camera's clock, an axis settling -- and keeps only a frame that
counts for every change it does not exclude. A change landing while
the capture runs is seen, and the capture starts its wait again,
within a deadline; the deadline stands still while a commanded move
is settling. The capture is one command on the camera's lane, and
this returns once it has run. `capture_still` is the same capture,
returning the frame's record with it.

**Args**

- `force_to_8bit`: True for an 8-bit image; False for the camera's own depth, in a uint16 array for a 12- or 16-bit camera.
- `accept_dark`: While a channel is lit (at a current above 0 mA), a frame with essentially no lit pixel is taken again until `timeout_s` passes, then kept and recorded dark. True skips that judgment, for a caller whose dark frames are legitimate while lit, such as an autofocus sweep.
- `exclude_sources`: Changes not to wait for, e.g. `(FrameValiditySource.Z_MOVE,)`. The frame can then predate such a change, so a frame that is measured or saved never excludes one.
- `all_ones_check`: Measure the frame's saturation: a blown frame is grabbed again once and, blown again, kept.
- `timeout_s`: Seconds the grab, and each summed frame, keeps taking frames for one that passes its checks, once the stale ones are passed over; 0, the default, gives each one try.
- `sum_count`: Frames summed into the image, for noise reduction; a sum's array is uint16.
- `sum_delay_s`: Seconds between summed frames.
- `sum_iteration_callback`: Called after each summed frame, on the camera's lane.

**Returns**

The image: (H, W) for a mono camera, (H, W, 3) for a color one.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, a time is not a wait from 0 to 30 days, or the wait for the capture and every frame of its sum together is longer. Nothing reached the camera.
- `CaptureFailedError`: `'no_frame'`, the camera delivered none within its grab window; `'deadline_expired'`, changes kept landing until the deadline passed; `'no_fresh_frame'`, frames came but none within the grab's bound counted for the `setting` changed; `'chunk_rejected'`, frames counted but none at the `setting` asked; `'record_read_failed'`, a frame came but the camera did not answer a read of the gain, exposure or black level its record needs, so it was not kept, chained from the `CameraReadFailedError`.
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, absent at the call or removed during the capture; `'camera_not_streaming'`, connected but not grabbing.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `camera_not_streaming`, `not_connected`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`CaptureFailedError`](#CaptureFailedError): `no_frame`, `deadline_expired`, `no_fresh_frame`, `chunk_rejected`, `record_read_failed`

<a id="ImagingAPI.disable_streaming"></a>
### `ImagingAPI.disable_streaming`

```python
def disable_streaming(self) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/imaging/disable_streaming`.

Stop camera streaming, and wait.

After this, `capture_and_wait()` fails until streaming resumes;
`get_image()` still returns the last frame the camera delivered.
Satisfied with no camera connected: nothing is streaming. Dispatched
as `enable_streaming` is.

**Raises**

Nothing.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ImagingAPI.enable_streaming"></a>
### `ImagingAPI.enable_streaming`

```python
def enable_streaming(self) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/imaging/enable_streaming`.

Begin camera streaming -- the public way to start the live feed, and wait.

After `connect()` the camera is configured but NOT grabbing (the
camera-lifecycle split); `Lumascope.initialize` makes the one-time
bring-up start. This restarts a feed that was deliberately stopped,
and is idempotent on a feed already running. Refused, naming the camera,
when none is connected, or when the start finds it removed. A start
the FX2 does not take with the device still present raises its fault
(`CameraStreamStartFailedError` for a START the device refused), the
camera left connected and not streaming; a later call tries again.
The start is one command on the camera's lane, waited for up to the
lane's geometry bound, since a start of the grab has been measured
near 11 s on a Pylon body.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, with none connected or one the start finds removed.
- `CameraStreamStartFailedError`: the FX2 did not take the START with the device still present; the camera is left connected and not streaming.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`CameraStreamStartFailedError`](#CameraStreamStartFailedError): `camera_stream_start_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="ImagingAPI.frames_until_valid"></a>
### `ImagingAPI.frames_until_valid`

```python
def frames_until_valid(
    self,
    exclude_sources: tuple[FrameValiditySource, ...] = (),
) -> int
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/imaging/frames_until_valid`.

Number of frames that must be grabbed before the next valid frame.

Delegates to frame_validity.

**Args**

- `exclude_sources`: Sources to exclude from the validity check.

**Returns**

int: Number of additional frames to drain before validity. 0 if already valid.

**Raises**

Nothing.

<a id="ImagingAPI.get_delivered_rate"></a>
### `ImagingAPI.get_delivered_rate`

```python
def get_delivered_rate(self) -> DeliveredRate
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/imaging/get_delivered_rate`.

Read the rate the camera delivered frames, and their bytes on the link.

Measured over the last second from the frames every driver stores,
in every host, so it is the camera's rate whether a display is
drawing, paused or absent. A window older than two checks -- the
camera stopped streaming, was disconnected, or the check stopped --
reads 0, never the last rate seen.

**Returns**

DeliveredRate: Frames and link bytes per second; `NOT_DELIVERING` (both 0) when nothing was delivered in the last window.

**Raises**

Nothing.

<a id="ImagingAPI.get_image"></a>
### `ImagingAPI.get_image`

```python
def get_image(
    self,
    force_to_8bit: bool = True,
    timeout_s: float = 5.0,
    all_ones_check: bool = False,
    sum_count: int = 1,
    sum_delay_s: float = 0.0,
    sum_iteration_callback: Callable[[], None] | None = None,
    force_new_capture: bool = False,
    new_capture_timeout_s: float = 5.0,
    verify_chunk_targets: bool = False,
) -> np.ndarray
```

Marks: in-process.

Grab and return an image from the camera -- the UNGATED primitive.

Never dark-rejects: liveness probes and diagnostics must see what
the camera sees, dark or not. For any frame you will SAVE or
MEASURE as truth, call capture_and_wait, which drains the camera
pipeline until the validity marker is GREEN and derives the
dark-floor expectation from commanded LED state.

Explicit parameters, never a **kwargs forward: the signature is a
pinned contract (introspected by tests), and a silent kwarg
pass-through would re-open the door this split closed.

By default the last buffered frame is returned; `force_new_capture`
waits for a fresh one. The grab is one command on the camera's lane,
waited for. It does not wait for frame validity, so the frame may not
reflect a just-changed gain, exposure, LED or stage position.

**Args**

- `force_to_8bit`: True for an 8-bit image; False for the camera's own depth, in a uint16 array for a 12- or 16-bit camera.
- `timeout_s`: Seconds to wait for a frame.
- `all_ones_check`: Measure each frame's saturation: a blown frame is grabbed again once, then kept.
- `sum_count`: Frames summed into the image, for noise reduction; a sum's array is uint16.
- `sum_delay_s`: Seconds between summed frames.
- `sum_iteration_callback`: Called after each summed frame, on the camera's lane.
- `force_new_capture`: True to wait for a frame the camera captures after the call, rather than take the last buffered one.
- `new_capture_timeout_s`: Seconds to wait for that new frame.
- `verify_chunk_targets`: Pass over frames whose own record of exposure and gain (the camera's chunk data) differs from the last values asked for, until one matches or `timeout_s` passes; no effect on a camera without chunk data, or while hardware auto-gain owns the values.

**Returns**

The image: (H, W) for a mono camera, (H, W, 3) for a color one, with no channel's false colour applied.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, a time is not a wait from 0 to 30 days, or the wait for every frame of the sum together is longer. Nothing reached the camera.
- `CaptureFailedError`: `'no_frame'`, no frame within the timeouts; `'chunk_rejected'`, frames came but none at the `setting` asked (`verify_chunk_targets`).
- `HardwareCommandRefusedError`: `'not_connected'`, naming the camera, absent at the call or removed during the grab.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`CaptureFailedError`](#CaptureFailedError): `no_frame`, `chunk_rejected`
- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="ImagingAPI.get_image_from_buffer"></a>
### `ImagingAPI.get_image_from_buffer`

```python
def get_image_from_buffer(
    self,
    force_to_8bit: bool = True,
    out_8bit: np.ndarray | None = None,
) -> tuple
```

Marks: in-process.

Grab the latest buffered frame from the camera without forcing a new capture.

Copy budget (per frame):
  - grab(): 0 copies (returns the read-only stored frame)
  - add_scale_bar(): 1 copy when the bar is drawn (the stored frame
    is shared with every other consumer and is never written)
  - convert_to_8bit(): 1 copy (LUT indexing creates new array),
    or 0 fresh allocations when the caller supplies out_8bit.
  - Total without the bar: 0 copies (8-bit) or 1 copy (12-bit with
    force_to_8bit); the bar adds 1
  The caller adds 1 more copy via tobytes() for GPU blit.

**Args**

- `force_to_8bit`: Convert 12-bit images to 8-bit output.
- `out_8bit`: Optional caller-owned (H, W) uint8 buffer reused as the 12->8 LUT destination, avoiding a fresh per-frame allocation on the 30 fps preview path. Each caller must supply its OWN buffer (the preview thread and the histogram run on different threads); a None / mismatched buffer falls back to a fresh allocation. The returned array is the buffer itself when used, so the caller must copy (e.g. tobytes()) before the next call overwrites it.

**Returns**

tuple: (image, timestamp) where image is numpy.ndarray and timestamp is from the camera SDK, or (None, None) if unavailable. Per the Sentinel-return contract: `if image is None: ...`.

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope, so it has no scale-bar setting to read.

**Refusals**

- [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="ImagingAPI.held_frame_rules"></a>
### `ImagingAPI.held_frame_rules`

```python
def held_frame_rules(self, record: dict) -> Iterator[None]
```

Marks: in-process.

Judge the changes recorded inside the `with` block by counts not yet on disk.

Use as `with scope.imaging.held_frame_rules(record): ...`. The
rules are built from `record` as bring-up builds them from the
camera's timing record file (`capabilities.frame_rules`), and are
handed to the gate for the block: each change recorded inside it
waits its held count. When the block ends, however it ends, the
installation's rules are back, and each change recorded after it
waits the installation's count. A change keeps the rule it was
recorded under, on either side. For a characterization's verify
phase, which captures under the counts it just measured before they
are saved; `capabilities.frame_rules` still publishes the
installation's.

**Args**

- `record`: A camera timing record, schema 3, checked as a file is (`FrameValidity.check_timing_record`).

**Raises**

- `ArgumentRefusedError`: `'wrong_kind'`, a record this release cannot take as measured; nothing was held.
- `ControlSessionRefusedError`: `'control_session_held'`, a control session is held by someone else; nothing was held.

**Refusals**

- [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `wrong_kind`

<a id="ImagingAPI.is_streaming"></a>
### `ImagingAPI.is_streaming`

```python
def is_streaming(self) -> bool
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/imaging/is_streaming`.

Whether the camera is currently acquiring frames.

Queries the driver directly (unlike `scope.camera.active`, which
reads the held connected-state). False when no camera is attached.

**Raises**

Nothing.

<a id="ImagingAPI.judge_frame"></a>
### `ImagingAPI.judge_frame`

```python
def judge_frame(
    self,
    frame: Frame,
    mapping: ClockMapping | None,
) -> tuple[Judgment, ...]
```

Marks: in-process.

Whether `frame` counts for each source's last change, by the rule a capture waits on.

The one decision (`FrameValidity.judge`), asked of every change
recorded so far (gain, exposure, black level, LED, a move, ...), by
the rule each change's source runs (`capabilities.frame_rules`): a
capture's record carries it for the frame it returns, and a frame
listener can ask it of any frame it receives. Each `Judgment`
carries the change it was judged against: the last frame the driver
had been handed when it was recorded, when it was recorded, its rule,
the frames it needed and who counted them, and for a move when its
axis settled.

**Args**

- `frame`: A frame the camera delivered (a frame listener's `Frame`).
- `mapping`: The camera's clock mapped onto the host's, which the clock rule judges a frame's start by: a capture's `frame_record.clock_mapping`. None for a camera without a latch; a clock-rule change then counts no frame.

**Raises**

Nothing.

<a id="ImagingAPI.last_capture_info"></a>
### `ImagingAPI.last_capture_info`

```python
last_capture_info: dict | None  # property
```

Marks: in-process; engineering.

How the most recent capture on this scope waited, frame or not.

What a capture leaves whether or not it returned a frame: the
protocol file writer's log line, the manual still's retake bound
and the engineering plugin's measure of the hold and drain read it.
What a frame is -- its exposure and gain and where each came from,
its auto-gain outcome, its judgments -- is its `FrameRecord`,
handed back with it (`capture_still`), never this slot.

**Returns**

dict | None: `{'hold_ms', 'drained', 'rechecks', 'deadline_s', 'n_entry', 'active_s'}` for the latest capture past its call, whether it returned a frame or raised why it got none; None before the first. A capture refused at its call (no camera, or one not streaming) records nothing, so read this only after a capture that returned or raised `CaptureFailedError`. Why a capture got no frame is what it raised, never this record. The next capture replaces it.

**Raises**

Nothing.

<a id="ImagingAPI.recorded_changes"></a>
### `ImagingAPI.recorded_changes`

```python
def recorded_changes(self) -> dict[FrameValiditySource, Change]
```

Marks: in-process.

Each source's last recorded change, settled or not, by source.

What a frame is judged against (`judge_frame`): when the change was
recorded, the last frame the driver had been handed then, its rule,
and for a move when its axis settled. A change stays its source's until the
next one replaces it.

**Raises**

Nothing.

<a id="ImagingAPI.remove_frame_listener"></a>
### `ImagingAPI.remove_frame_listener`

```python
def remove_frame_listener(
    self,
    cb: Callable[[np.ndarray, datetime.datetime, dict | None], None],
) -> None
```

Marks: in-process.

Remove a listener registered via `add_frame_listener`.

No new call reaches the handler once this returns (a call already
running completes), whatever the camera driver does: the wrapper is
marked removed before the driver is asked, so a driver that fails to
unregister it goes on calling a wrapper that no longer calls through.
No-op when the listener was never registered.

**Raises**

Nothing.

<a id="ImagingAPI.saturated_fraction"></a>
### `ImagingAPI.saturated_fraction`

```python
@staticmethod
def saturated_fraction(arr: np.ndarray | None, significant_bits: int) -> float
```

Marks: in-process; engineering.

Fraction of pixels at or above the near-full-scale threshold.

Full scale comes from the frame's payload depth, not the container
dtype: a 12-bit frame in a uint16 container tops out at 4095, so
measuring against 65535 would report a fully blown frame as 0%
saturated and let it slip past the evidence check.

Public because clipping is a property of the frame that callers
outside this class have to be able to ask about: a whole-frame mean
cannot distinguish an evenly lit field from one half blown out and
half black, so any caller judging whether an exposure is usable needs
the pixel count, not a summary statistic. `significant_bits` is
required rather than defaulted -- a caller that has to guess the
payload depth is the case this measures wrong.

**Raises**

- `ArgumentRefusedError`: `'out_of_range'`, *significant_bits* is outside 1 to 16, the depths a frame's 16-bit container holds.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `out_of_range`

<a id="ImagingAPI.scale_bar_color"></a>
### `ImagingAPI.scale_bar_color`

```python
scale_bar_color: str | None  # property
```

Route: `GET /api/v1/scope/imaging/scale_bar_color`.

The colour the scale bar is drawn in on captured images, as `set_scale_bar_color` set it; None until it is set.

Whether it is drawn at all is `ScopeSession.scale_bar_enabled`.

**Raises**

Nothing.

<a id="ImagingAPI.set_scale_bar_color"></a>
### `ImagingAPI.set_scale_bar_color`

```python
def set_scale_bar_color(self, color: str) -> None
```

Marks: sets [`scale_bar_color`](#ImagingAPI.scale_bar_color).  
Route: `PUT /api/v1/scope/imaging/scale_bar_color`.

The colour the scale bar is drawn in on captured images.

Whether it is drawn at all is the session's setting
(`ScopeSession.set_scale_bar`), which this never changes.

**Args**

- `color`: Scale bar colour, e.g. a layer name ("BF", "Ex589").

**Raises**

Nothing.

<a id="ImagingAPI.significant_bits"></a>
### `ImagingAPI.significant_bits`

```python
significant_bits: int  # property
```

Route: `GET /api/v1/scope/imaging/significant_bits`.

Meaningful payload bits of frames the current camera delivers.

The depth a single captured frame should be scaled / tagged by (12 for a
Mono12 sensor, 8 for an 8-bit one); it follows `set_image_mode`. A
summed frame carries the bits the sum can reach and is not described
by this -- `CapturedStill.significant_bits` answers for it. The deepest the
camera offers is `capabilities.camera_max_significant_bits`.

Derived from the HELD pixel format (the one the camera last reported
or took)
via the driver's own depth rule (`significant_bits_for_format`, so
a constant-depth driver like FX2 stays authoritative), not a live
driver read: a transient format-read failure once made a 12-bit
camera report depth 16 here, so a fully blown frame passed the
saturation evidence check at 0.0% and files were stamped at the
wrong depth. A frame you are HOLDING carries its own depth
(`Frame.significant_bits`; `CapturedStill.significant_bits` for a capture).

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'`, the scope has no camera, so no frames have a depth.

**Refusals by REST status**

- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="MotionAPI"></a>
## `MotionAPI`

Reached through `session.scope.motion`; on the wire, `/api/v1/scope/motion/`.  
Held in: [`Lumascope.motion`](#Lumascope.motion).

Motion sub-API. Hosts stateless (Phase 2b) and stateful (Phase 2c) bodies.

<a id="MotionAPI.add_position_listener"></a>
### `MotionAPI.add_position_listener`

```python
def add_position_listener(
    self,
    listener: Callable[[StageAxis, AxisPosition], None],
) -> None
```

Marks: in-process; event: [`PositionChanged`](#PositionChanged).

Register a callback for position/state changes on a stage axis: X, Y or Z.

The listener is called with `(axis, AxisPosition)` whenever the
position cache or axis state changes: the axis's state and its
polled position as `axis_positions` answers them then, from the
same snapshot -- not the target, and no position (None) while the
axis is HOMING or UNKNOWN. The turret's changes go to
`add_turret_listener`. It fires from the thread that caused the
change (IO executor, motion monitor, etc.), so listeners **must**
schedule any UI work via `Clock.schedule_once`.

**Args**

- `listener`: `callable(axis: StageAxis, at: AxisPosition)`

**Raises**

Nothing.

<a id="MotionAPI.add_turret_listener"></a>
### `MotionAPI.add_turret_listener`

```python
def add_turret_listener(self, listener: Callable[[TurretStatus], None]) -> None
```

Marks: in-process; event: [`TurretChanged`](#TurretChanged).

Register a callback for each change of the turret's state or slot.

The listener is called with the `TurretStatus` that
`turret_status` answers then: on each change of the turret's state
(homing, moving, idle, unknown), on each read the motion monitor
makes of it while it turns, and once the slot a turret command
reached is recorded, which is after the turret says idle. It fires
from the thread that caused the change, so listeners **must**
schedule any UI work via `Clock.schedule_once`. Never called on a
scope with no turret.

**Args**

- `listener`: `callable(at: TurretStatus)`

**Raises**

Nothing.

<a id="MotionAPI.axes_without_position"></a>
### `MotionAPI.axes_without_position`

```python
def axes_without_position(self) -> dict[Axis, AxisState]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/axes_without_position`.

Which of this scope's axes do not know their position, and why.

The question a run asks before it starts and before each capture:
every run moves every axis the scope has, and an image taken where
an axis is not known is saved with a position that is not true.
Each axis is answered with its own state because the two causes
need different words -- a HOMING axis is about to know, and its
user should wait; an UNKNOWN one has lost its reference, and its
user must home.

Offered as a question rather than only as an exception: a caller
whose move is optional -- one that should be skipped rather than
attempted on an axis whose position was never established -- could
otherwise only discover the answer by provoking the refusal and
catching it, which is indistinguishable from swallowing a real one.
Stricter than a move's own refusal by one state: a HOMING axis is
answered here, because its reference is still being established,
while a move lets it drive so the home can finish.

A scope with no axes answers nothing: it has no position to lose,
so nothing about it is unknown.

**Returns**

dict[Axis, AxisState]: Axis to its state (`AxisState.UNKNOWN` or `AxisState.HOMING`) for every axis whose position is not known, in the scope's axis order. Empty when every axis knows its position. Every axis, UNKNOWN, with the motor board lost.

**Raises**

Nothing.

<a id="MotionAPI.axis_positions"></a>
### `MotionAPI.axis_positions`

```python
def axis_positions(self) -> dict[StageAxis, AxisPosition]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/axis_positions`.

Every stage axis's state and its position, or None where the position is not known.

One snapshot: the state and the cache are read together, so a
caller writing a position into a file cannot pair a number with a
state that changed between two reads. An axis is answered with a
position only while it is IDLE or MOVING; UNKNOWN and HOMING
answer None whatever the cache holds, because the cache keeps the
last number an axis reported after its reference is lost. X, Y and
Z, in um; the turret has a slot, not a position (`turret_status`).
With the motor board lost every axis is UNKNOWN, with no position.
No serial I/O.

The state lock is taken first and the cache lock inside it. No
other path nests the two, so this order cannot meet its reverse.

**Returns**

dict[StageAxis, AxisPosition]: Stage axis to (state, position), in the scope's axis order.

**Raises**

Nothing.

<a id="MotionAPI.get_actual_position"></a>
### `MotionAPI.get_actual_position`

```python
def get_actual_position(self, axis: StageAxis) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/get_actual_position`.

Query a stage axis's actual hardware position via serial (not cached), in um.

Unlike get_current_position(), which serves the in-memory cache,
this asks the motor controller directly. Use during continuous
motion sweeps where the stage is moving and the cache doesn't
reflect the true position. The commanded target is a third thing
again -- get_target_positions() answers that.

Costs one serial round-trip (~5ms).

**Args**

- `axis`: Axis name ("X", "Y", "Z"). The turret has a slot, not a position: `turret_status`.

**Returns**

float | None: Current position in um; None with the null board installed or for an axis this scope does not have.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `axis` is not 'X', 'Y' or 'Z'.
- `HardwareCommandRefusedError`: `'not_connected'`, the installed motor controller is not connected (its cable pulled).
- `BoardNoReplyError`: `'board_no_reply'`, the controller did not answer the read.
- `BoardReplyUnreadableError`: `'board_reply_unreadable'`, the controller's answer is not a position.
- `BoardPortFailedError`: `'board_port_failed'`, the port to the controller failed or is not open.
- `BoardReportedError`: `'board_reported_error'`, the controller refused the read.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`
- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`BoardReplyUnreadableError`](#BoardReplyUnreadableError): `board_reply_unreadable`; [`BoardPortFailedError`](#BoardPortFailedError): `board_port_failed`; [`BoardReportedError`](#BoardReportedError): `board_reported_error`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="MotionAPI.get_axis_limits"></a>
### `MotionAPI.get_axis_limits`

```python
def get_axis_limits(self, axis: Axis) -> AxisLimits | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/get_axis_limits`.

Get the travel limits for an axis, in um.

**Args**

- `axis`: Axis name ("X", "Y", "Z", or "T").

**Returns**

The travel moves are refused against, or `None` if the axis has no configured limits (typical for the turret T axis). Callers must handle the None case.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `axis` is no axis name.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no such axis (see `refuse_absent`); an axis the scope does not have has no travel to answer with.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="MotionAPI.get_limit_switch_status"></a>
### `MotionAPI.get_limit_switch_status`

```python
def get_limit_switch_status(self, axis: Axis) -> tuple[int, int]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/get_limit_switch_status`.

Get the limit switch status for an axis.

Tells a caller WHY a move stopped short: an axis that reached a limit
reports it here rather than raising, so the difference between "the
move finished" and "the stage ran out of travel" is only visible by
asking.

**Args**

- `axis`: Axis name ("X", "Y", "Z", "T").

**Returns**

tuple[int, int]: `(left, right)`, each 1 when that switch is engaged and 0 when clear.

**Raises**

- `BoardNoReplyError`: `'board_no_reply'`, the board did not answer the read.
- `BoardReplyUnreadableError`: `'board_reply_unreadable'`, the board's answer is not a status.
- `BoardPortFailedError`: `'board_port_failed'`, the port to the board failed or is not open.
- `BoardReportedError`: `'board_reported_error'`, the board refused the read.
- `ArgumentRefusedError`: `'not_offered'`, `axis` is no axis name.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no such axis (see `refuse_absent`); a switch the scope does not have is never answered as clear.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`
- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`BoardReplyUnreadableError`](#BoardReplyUnreadableError): `board_reply_unreadable`; [`BoardPortFailedError`](#BoardPortFailedError): `board_port_failed`; [`BoardReportedError`](#BoardReportedError): `board_reported_error`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="MotionAPI.get_target_positions"></a>
### `MotionAPI.get_target_positions`

```python
def get_target_positions(self) -> dict[StageAxis, float]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/get_target_positions`.

The target of every stage axis the scope has a position for (where it is commanded to go), by axis name, in um.

The last target this API wrote to the board for each axis, while
the move runs and after it arrives: the commanded number, not the
polled position, which sits up to a microstep off it. A relative
move's target is the board's own target plus the offset, so it
carries that target's microstep rounding. Where no target was
reached it is the polled position: before the axis's first move,
after a home, after a STOP ended the move, while the axis is
UNKNOWN, and while a move has not yet written its final target (a
Z backlash leg). An axis whose position has not been read since
bring-up -- its bring-up read failed -- is UNKNOWN and answers
0.0, which nothing read. A refused move writes nothing, so the
previous target stands.

An axis with no hardware behind it is left out, and so is the
turret, which has a slot (`turret_status`). No serial I/O.

**Raises**

Nothing.

<a id="MotionAPI.home"></a>
### `MotionAPI.home`

```python
def home(self, axis: HomeTarget = HomeTarget.ALL) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/home`.

Home the given axis set, and wait for it.

Returns only when the home established a reference: every homed
axis has a known position (for `'ALL'`, the axes the board has).
The API sets no time limit on the wait: the home ends when the board
ends it, which bounds each of its phases and halts when one runs out,
or when the driver stops waiting for a board that has stopped
answering; either ends it in `HomingFaultError`.

**Args**

- `axis`: `'Z'` homes the Z axis only. `'T'` homes the turret (parks Z at 0, homes T, restores Z). `'ALL'` (default) homes every axis the board has; the firmware routine homes Z, then T, then X/Y.

**Raises**

- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread; nothing was started (`start_home` is the form for a button).
- `ArgumentRefusedError`: `'not_offered'`, `axis` is not `'Z'`, `'T'` or `'ALL'`; nothing was asked of the scope.
- `HardwareCommandRefusedError`: `'home_in_flight'`, while a home asked earlier, by any caller, has not ended.
- `HomingFaultError`: the home was driven and did not establish a reference: the driver raised, or a homed axis's position could not be read. The axes it names are UNKNOWN.
- `HomingStoppedError`: a Stop ended the home. The axes it names are UNKNOWN; none, when the Stop landed before the home was driven.

While it runs, the home holds the session's activity claim as a run
does: every other request to the scope is refused, naming the home,
and the controls lock until it ends. Asked by an activity that
already holds the scope (the support report's diagnostic), it runs
as that activity's work and takes no claim of its own.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`, `lid_open_before_motion`, `home_in_flight`; [`HomingStoppedError`](#HomingStoppedError): `stopped`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`; [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`
- 500: [`HomingFaultError`](#HomingFaultError): `error`, `unread`, `lid_open`, `stage_unpowered`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="MotionAPI.interlocks"></a>
### `MotionAPI.interlocks`

```python
def interlocks(self) -> frozenset[Interlock]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/interlocks`.

The stage's hardware interlocks open now (`'lid_open'`, `'stage_unpowered'`).

Empty on a board with none. Asked of the driver directly, not
queued on the IO lane: the driver reads it under its own lock,
one exchange serialized with any in flight, so the answer never
waits for a move or a home to finish, as the motion monitor's own
reads do not. It does wait for what holds that lock now: one
exchange or one poll of a home, and at most a Stop's settle (2 s)
or a reopen of the port.

**Returns**

frozenset[Interlock]: The open interlocks.

**Raises**

- `HardwareCommandRefusedError`: `'scope_disconnected'` after `disconnect()`; `'not_connected'`, naming the motor controller. Nothing was asked.
- `BoardNoReplyError`: `'board_no_reply'`, the board did not answer the read; the stage's state is unknown, which is not the same answer as none open.
- `BoardPortFailedError`: `'board_port_failed'`, the port to the board failed or is not open; the state is unknown.
- `BoardReplyUnreadableError`: `'board_reply_unreadable'`, the board's answer could not be read; the state is unknown.
- `BoardReportedError`: `'board_reported_error'`, the board refused the read; the state is unknown.

**Refusals by REST status**

- 500: [`BoardNoReplyError`](#BoardNoReplyError): `board_no_reply`; [`BoardPortFailedError`](#BoardPortFailedError): `board_port_failed`; [`BoardReplyUnreadableError`](#BoardReplyUnreadableError): `board_reply_unreadable`; [`BoardReportedError`](#BoardReportedError): `board_reported_error`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`, `not_connected`

<a id="MotionAPI.jog_step_um"></a>
### `MotionAPI.jog_step_um`

```python
def jog_step_um(self, axis: StageAxis, coarse: bool) -> float
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/jog_step_um`.

The jog step for `axis` under the active objective.

A jog's size scales with the objective, so the active objective's
record answers: `jog_z_coarse_um` / `jog_z_fine_um` for Z,
`jog_xy_coarse_um` / `jog_xy_fine_um` for X and Y, the units
`move_xy_by` and `move_z_by` take.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `axis` is not 'X', 'Y' or 'Z': the turret has no jog.
- `ObjectiveUnknownError`: The objective in the light path is unknown; no step is guessed, so nothing should move.

**Refusals by REST status**

- 409: [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`, `slot_unassigned`, `not_in_catalogue`, `none_selected`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`

<a id="MotionAPI.move_to_plate"></a>
### `MotionAPI.move_to_plate`

```python
def move_to_plate(self, at: PlateLocation) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_to_plate`.

Move X and Y to `at` on the selected plate, and return once they have arrived.

`start_move_to_plate` followed by its operation's `result()`.

**Raises**

- `ArgumentRefusedError`: `'plate_frame_axis'` or `'not_a_number'`, as `start_move_to_plate`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'lid_open_before_motion'` or `'monitor_ended'`, as `start_move_to_plate`.
- `PositionOutOfRangeError`: As `start_move_to_plate`.
- `AxisStateUnknownError`: As `start_move_to_plate`.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_move_to_plate`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `plate_frame_axis`, `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.move_to_well"></a>
### `MotionAPI.move_to_well`

```python
def move_to_well(self, well: str) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_to_well`.

Move X and Y to the centre of `well` on the selected plate, and return once they have arrived.

`start_move_to_well` followed by its operation's `result()`.

**Raises**

- `ArgumentRefusedError`: `'well_unknown'`, as `start_move_to_well`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'lid_open_before_motion'` or `'monitor_ended'`, as `start_move_to_plate`.
- `PositionOutOfRangeError`: As `start_move_to_plate`.
- `AxisStateUnknownError`: As `start_move_to_plate`.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_move_to_well`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `well_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.move_turret"></a>
### `MotionAPI.move_turret`

```python
def move_turret(self, slot: int, restore_z: bool = True) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_turret`.

Move the turret to a slot, and wait for it; a turret already there is not moved.

Three moves -- the Z park, the turret move itself and the Z restore
-- each judged as every move is, so the wait sets no time limit of
its own.

**Args**

- `slot`: The turret slot, 1 to 4.
- `restore_z`: True to put Z back where it was before the park once the turret has arrived; False for a caller that sets Z next anyway, such as a protocol step moving to its own Z.

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no turret, refused before Z is parked, with nothing driven or recorded; or a recording holds the scope: its frames carry the pixel size of the objective it started with.
- `AxisStateUnknownError`: The turret position is unknown.
- `PositionOutOfRangeError`: The slot is not a whole number 1 to 4.
- `MoveFaultError`: The Z park, the turret move or the Z restore did not arrive and its axis is UNKNOWN, as `start_move_z` lists the reasons; or `'driver_failed'`, the board did not take a command. The slot is unknown afterwards, whichever of the three it was.
- `MoveStoppedError`: `'stopped'`, a stop ended one of the three; `'superseded'`, another move or a home took its axis first. The slot is unknown afterwards.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread; nothing was started.

**Refusals by REST status**

- 409: [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="MotionAPI.move_xy"></a>
### `MotionAPI.move_xy`

```python
def move_xy(
    self,
    x_um: float | None = None,
    y_um: float | None = None,
    ignore_limits: bool = False,
) -> None
```

Marks: in-process arguments: `ignore_limits`; enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_xy`.

Move X and Y to a place, in um, each that is given, and return once they have arrived.

`start_move_xy` followed by its operation's `result()`.
`ignore_limits` is in-process only, as `start_move_xy`'s.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, as `start_move_xy`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'monitor_ended'` or `'lid_open_before_motion'`, as `start_move_xy`.
- `PositionOutOfRangeError`: `'position_out_of_range'`, as `start_move_xy`.
- `AxisStateUnknownError`: As `start_move_xy`.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_move_xy`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.move_xy_by"></a>
### `MotionAPI.move_xy_by`

```python
def move_xy_by(self, dx_um: float | None = None, dy_um: float | None = None) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_xy_by`.

Move X and Y by a distance, in um, each that is given, and return once they have arrived.

`start_move_xy_by` followed by its operation's `result()`.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, as `start_move_xy_by`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'position_unread'`, `'monitor_ended'` or `'lid_open_before_motion'`, as `start_move_xy_by`.
- `PositionOutOfRangeError`: `'position_out_of_range'`, as `start_move_xy_by`.
- `AxisStateUnknownError`: As `start_move_xy_by`.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_move_xy_by`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `position_unread`, `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.move_z"></a>
### `MotionAPI.move_z`

```python
def move_z(
    self,
    z_um: float,
    overshoot_enabled: bool | None = None,
    ignore_limits: bool = False,
) -> None
```

Marks: in-process arguments: `ignore_limits`; enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_z`.

Move Z to a place, in um, and return once it has arrived.

`start_move_z` followed by its operation's `result()`.
`ignore_limits` is in-process only, as `start_move_z`'s.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, as `start_move_z`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'position_unread'` or `'monitor_ended'`, as `start_move_z`.
- `PositionOutOfRangeError`: `'position_out_of_range'`, as `start_move_z`.
- `AxisStateUnknownError`: As `start_move_z`.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_move_z`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `position_unread`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.move_z_by"></a>
### `MotionAPI.move_z_by`

```python
def move_z_by(self, dz_um: float, overshoot_enabled: bool | None = None) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/move_z_by`.

Move Z by a distance, in um, and return once it has arrived.

`start_move_z_by` followed by its operation's `result()`.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, as `start_move_z_by`.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'position_unread'` or `'monitor_ended'`, as `start_move_z_by`.
- `PositionOutOfRangeError`: `'position_out_of_range'`, as `start_move_z_by`.
- `AxisStateUnknownError`: As `start_move_z_by`.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command; or an axis did not arrive and is UNKNOWN, the motion monitor's `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`, or `'faulted'`; see `start_move_z`.
- `MoveStoppedError`: `'stopped'`, a stop ended a move; `'superseded'`, another move or a home took an axis first.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_move_z_by`. Nothing was driven.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `position_unread`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`, `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.positions_to_keep"></a>
### `MotionAPI.positions_to_keep`

```python
def positions_to_keep(
    self,
    axes: Iterable[StageAxis],
    *,
    then: str,
) -> dict[StageAxis, float]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/positions_to_keep`.

The positions of `axes`, each one an axis at rest whose reference is known.

The read for a caller that keeps a position: a sweep's centre, a
reference written into a result, the Z a measurement returns to.
An axis's number is one to keep only while it is IDLE: while it
moves, the number is the motion monitor's last poll, a place the
axis was passing through; while it is UNKNOWN or HOMING, it is the
last number the axis reported before its reference was lost. The
states and the numbers come from one snapshot, so a home or a move
that starts between a separate state question and the read cannot
pass the number it left behind. X, Y and Z, in um. No serial I/O.

**Args**

- `axes`: The axes the caller keeps.
- `then`: What the caller does once the scope knows the position, ending the refusal (e.g. `'start the measurement'`).

**Returns**

dict[StageAxis, float]: Axis to its position, in the order asked.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, a name in *axes* is not 'X', 'Y' or 'Z'.
- `HardwareCommandRefusedError`: `'scope_disconnected'` after `disconnect()`; `'not_connected'` or `'axis_absent'`, an asked axis has no motor behind it (see `refuse_absent`).
- `AxisMovingError`: `'axis_moving'`, naming every asked axis that is moving.
- `AxisStateUnknownError`: `'position_unknown'`, naming every asked axis that is UNKNOWN or HOMING.

**Refusals by REST status**

- 409: [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`, `not_connected`

<a id="MotionAPI.remove_position_listener"></a>
### `MotionAPI.remove_position_listener`

```python
def remove_position_listener(
    self,
    listener: Callable[[StageAxis, AxisPosition], None],
) -> None
```

Marks: in-process.

Unregister a position listener.

**Args**

- `listener`: A callable previously passed to `add_position_listener`. Silently ignores listeners that are not currently registered.

**Raises**

Nothing.

<a id="MotionAPI.remove_turret_listener"></a>
### `MotionAPI.remove_turret_listener`

```python
def remove_turret_listener(self, listener: Callable[[TurretStatus], None]) -> None
```

Marks: in-process.

Unregister a turret listener.

**Args**

- `listener`: A callable previously passed to `add_turret_listener`. Silently ignores listeners that are not currently registered.

**Raises**

Nothing.

<a id="MotionAPI.set_precision_mode"></a>
### `MotionAPI.set_precision_mode`

```python
def set_precision_mode(self, axis: Axis, enabled: bool) -> None
```

Marks: enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/set_precision_mode`.

Set motor precision mode for an axis.

Precision mode uses accurate but slightly slower motor stopping.
Use before autofocus fine passes or any measurement requiring
precise Z positioning. Disable for coarse moves where speed
matters more than final position accuracy.

**Args**

- `axis`: Axis name ("X", "Y", "Z", "T").
- `enabled`: True for precise positioning, False for speed.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `axis` is no axis name; nothing was sent.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no such axis (see `refuse_absent`); nothing was sent.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `scope_disconnected`

<a id="MotionAPI.start_home"></a>
### `MotionAPI.start_home`

```python
def start_home(self, axis: HomeTarget = HomeTarget.ALL) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_submit`.  
Route: `POST /api/v1/scope/motion/start_home`.

Start the home `home` runs, without waiting for it; returns its `Operation`.

For a caller that must not block for the length of a home -- a
button. The home is asked, and refused or put on the io lane, before
this returns, so a second home asked while this one is in flight is
refused at once rather than queued behind it. The operation is done
once the home has released the scope, with what `home` would have
returned or raised; its `stop()` is `stop_motion` while the home
drives, refused `'home_not_stoppable'` on a board that cannot stop
a home, and nothing once the home has ended.

**Raises**

- `ArgumentRefusedError`: `'not_offered'`, `axis` is not `'Z'`, `'T'` or `'ALL'`; nothing was asked of the scope.
- `HardwareCommandRefusedError`: `'home_in_flight'`, while a home asked earlier has not ended; or the home was refused as `home` is refused.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `home_in_flight`, `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="MotionAPI.start_move_to_plate"></a>
### `MotionAPI.start_move_to_plate`

```python
def start_move_to_plate(self, at: PlateLocation) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/start_move_to_plate`.

Start X and Y moving to `at` on the selected plate; return the started moves.

`at` is in mm from the plate's top-left, at the live stage
offset, the number a person types into a position box. An axis that
is None is left where it is, a move under way on it included, so a
caller moving X alone leaves Y alone. The axes given are asked once
whether they know their position and are started on one task of the
IO lane (`one_command`), so a capture asked meanwhile never takes
the new X at the old Y. Each is checked against the plate positions
this stage can reach, and refused in mm, before either moves.

**Returns**

One `Operation` for the moves started, done once every one is (`Operation.all_of`); its `result()` raises the first move's raise, as `start_move_xy`'s does. With no axis given nothing moves and it is done at once.

**Raises**

- `ArgumentRefusedError`: `'plate_frame_axis'`, a Z given: a plate position has no Z; `'not_a_number'`, an axis given is not a finite number. Nothing was submitted.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no such axis; `'lid_open_before_motion'`, the stage's lid is open; `'monitor_ended'`, the motion monitor has ended. Nothing was driven.
- `PositionOutOfRangeError`: `'position_out_of_range'`, an axis given is outside the plate positions this stage can reach, in plate mm. Nothing was driven.
- `AxisStateUnknownError`: An axis given does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `plate_frame_axis`, `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.start_move_to_well"></a>
### `MotionAPI.start_move_to_well`

```python
def start_move_to_well(self, well: str) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/start_move_to_well`.

Start X and Y moving to the centre of `well` on the selected plate; return the started moves.

`well` is the label the plate gives it (`'A1'`, `'H12'`),
read from the plate selected when the move is on the lane, so the
plate the well is found on is the plate the move converts on.
Otherwise as `start_move_to_plate` for both axes.

**Raises**

- `ArgumentRefusedError`: `'well_unknown'`, the selected plate has no well of that label (a plate with no wells has none). Nothing was driven.
- `HardwareCommandRefusedError`: `'not_connected'`, `'axis_absent'`, `'lid_open_before_motion'` or `'monitor_ended'`, as `start_move_to_plate`.
- `PositionOutOfRangeError`: As `start_move_to_plate`.
- `AxisStateUnknownError`: As `start_move_to_plate`.
- `MoveFaultError`: As `start_move_to_plate`.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `well_unknown`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`, `faulted`, `stalled`, `monitor_ended_mid_move`, `board_lost`, `position_unread_after_move`, `status_unread`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.start_move_xy"></a>
### `MotionAPI.start_move_xy`

```python
def start_move_xy(
    self,
    x_um: float | None = None,
    y_um: float | None = None,
    ignore_limits: bool = False,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); in-process arguments: `ignore_limits`; enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/start_move_xy`.

Start X and Y moving to a place, in um, each that is given; return the started moves.

One request: every axis given is refused before either drives, so
a refusal of one moves neither. An axis that is None is left as it
is, still, travelling or unknown, and is not read. With neither
given nothing moves and the move is done at once.

`ignore_limits` drives past the configured travel, never past the
motor's safety limit. It is in-process only: a wire request that
sends it is refused as `invalid_request`.

**Returns**

One `Operation` for the axes given, done once every one is (`Operation.all_of`); its `result()` raises the first axis's raise, as `start_move_z`'s does for Z. Its `stop()` is `stop_motion` while a move still has its axis.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, an axis given is not a finite number. Nothing was driven.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no such axis; `'monitor_ended'`, the motion monitor has ended, so nothing would judge the move. Nothing was driven. `'lid_open_before_motion'`, the stage's lid is open: nothing was driven when it was open before the request; when it opened after X was commanded, the stage stopped X and Y was not driven.
- `PositionOutOfRangeError`: `'position_out_of_range'`, an axis given is outside its configured travel and `ignore_limits` is False. Nothing was driven.
- `AxisStateUnknownError`: An axis given does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.start_move_xy_by"></a>
### `MotionAPI.start_move_xy_by`

```python
def start_move_xy_by(
    self,
    dx_um: float | None = None,
    dy_um: float | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/start_move_xy_by`.

Start X and Y moving by a distance, in um, each that is given; return the started moves.

As `start_move_xy`. Each distance is added to the axis's board
target, so a move still under way on it is added to, and every
target is read and refused before either axis drives.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, a distance given is not a finite number. Nothing was driven.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no such axis; `'position_unread'`, the board did not report a target to add to; `'monitor_ended'`, as `start_move_xy`. Nothing was driven. `'lid_open_before_motion'`, as `start_move_xy`.
- `PositionOutOfRangeError`: `'position_out_of_range'`, a distance is beyond the safety limit or its target is outside the axis's travel. Nothing was driven.
- `AxisStateUnknownError`: An axis given does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take a command.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `position_unread`, `lid_open_before_motion`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.start_move_z"></a>
### `MotionAPI.start_move_z`

```python
def start_move_z(
    self,
    z_um: float,
    overshoot_enabled: bool | None = None,
    ignore_limits: bool = False,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); in-process arguments: `ignore_limits`; enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/start_move_z`.

Start Z moving to a place, in um; return the started move.

A move of Z to a place approaches it from below, so the backlash
is always taken the same way. `overshoot_enabled` is for a bench
script that measures the backlash: None is the API's approach,
True or False the script's own.

`ignore_limits` drives past Z's configured travel, never past the
motor's safety limit. It is in-process only: a wire request that
sends it is refused as `invalid_request`.

**Returns**

The move's `Operation`. It is done once Z has been set at rest, or given up, and the position listeners told: its `result()` returns None when Z arrived, and raises `MoveNotCompletedError` otherwise -- the monitor's fault (`'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`) or `'faulted'`, each leaving Z UNKNOWN; `MoveStoppedError` `'stopped'`, a stop the board took while it moved; `MoveStoppedError` `'superseded'`, another move or a home took Z first. The outcome is the one the move ended with, however late it is read. Its `stop()` is `stop_motion` while this move still has Z, and nothing once it has ended; the board's STOP halts every axis.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, the position is not a finite number. Nothing was driven.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no Z; `'position_unread'`, an approach from below whose start the board did not report; `'monitor_ended'`, the motion monitor has ended, so nothing would judge the move. Nothing was driven.
- `PositionOutOfRangeError`: `'position_out_of_range'`, the target is outside Z's configured travel and `ignore_limits` is False. Nothing was driven.
- `AxisStateUnknownError`: Z does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take the command.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `position_unread`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.start_move_z_by"></a>
### `MotionAPI.start_move_z_by`

```python
def start_move_z_by(
    self,
    dz_um: float,
    overshoot_enabled: bool | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/scope/motion/start_move_z_by`.

Start Z moving by a distance, in um; return the started move.

The distance is added to Z's board target, so a move still under
way is added to. A move by a distance takes no approach from below;
`overshoot_enabled` is the bench script's override, as for
`start_move_z`.

**Returns**

The move's `Operation`, as `start_move_z`'s.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, the distance is not a finite number. Nothing was driven.
- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'`, no motor controller or no Z; `'position_unread'`, the board did not report the target to add to (or, approaching from below, the position); `'monitor_ended'`, as `start_move_z`. Nothing was driven.
- `PositionOutOfRangeError`: `'position_out_of_range'`, the distance is beyond the safety limit or its target is outside Z's travel. Nothing was driven.
- `AxisStateUnknownError`: Z does not know its position. Nothing was driven.
- `MoveFaultError`: `'driver_failed'`, the board did not take the command.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `position_unread`, `exclusive_activity_running`; [`AxisStateUnknownError`](#AxisStateUnknownError): `position_unknown`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`, `activity_ended`; [`PositionOutOfRangeError`](#PositionOutOfRangeError): `position_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`
- 500: [`MoveFaultError`](#MoveFaultError): `driver_failed`; [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`, `monitor_ended`, `scope_disconnected`

<a id="MotionAPI.stop_motion"></a>
### `MotionAPI.stop_motion`

```python
def stop_motion(self) -> None
```

Route: `POST /api/v1/scope/motion/stop_motion`.

Stop all in-flight motor moves.

Idempotent. Uses the firmware-side `STOP` command, which the motor
controller implements as `motorstop` (target=actual on all axes).
It does not wait behind the lane, so a stop reaches the board while
a move holds the lane.

A home is stopped only on a board that can take a STOP while it
homes (the LS720's); on any other, a Stop while the home drives is
refused and the home runs to its end.

A press -- taken, failed or refused -- withholds every move asked
before it that has not yet driven, from any client and any queue,
and ends the live run (its press listeners are told first).

**Raises**

- `HardwareCommandRefusedError`: `'not_connected'` or `'axis_absent'` with no motor controller (see `refuse_absent`); `'home_not_stoppable'` while a home drives on a board that cannot stop one. Nothing was sent.
- `MotorStopFailedError`: the board did not take the STOP, so the stage may still be moving. Chained from the driver's error. The stop generation has moved regardless.

**Refusals by REST status**

- 409: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `home_not_stoppable`
- 422: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `axis_absent`
- 500: [`MotorStopFailedError`](#MotorStopFailedError): `motor_stop_failed`
- 503: [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_connected`

<a id="MotionAPI.turret_status"></a>
### `MotionAPI.turret_status`

```python
def turret_status(self) -> TurretStatus | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/motion/turret_status`.

The turret's state and the slot in the light path, read together; None on a scope with no turret.

The slot is the one the last turret command (`move_turret`, a
home) left the turret in, recorded only when that command returned
without error and no stop was issued while it ran. It is None before
the first such command, while one is in flight, after one failed,
and whenever the turret's position is lost. The turret has no
encoder, so nothing else can say which slot is in the light path;
the controller's step count is not a slot. The state is the
turret's `AxisState`, which says what the slot alone cannot:
whether it is homing, moving, or lost.

A scope has a turret as `capabilities.has_turret` answers. With
the motor board lost, or never come up, the turret is UNKNOWN with
no slot. No serial I/O.

**Raises**

Nothing.

<a id="MotionAPI.wait_until_finished_moving"></a>
### `MotionAPI.wait_until_finished_moving`

```python
def wait_until_finished_moving(self, timeout_s: float = 120.0) -> None
```

Route: `POST /api/v1/scope/motion/wait_until_finished_moving`.

Block until every axis moving now has stopped; raise if one did not stop well.

For motion the caller did not start, or started without keeping its
operation. A move the caller started is judged by its operation's
`result()`, which also knows when a stop or a later move ended it;
this wait judges only whether each axis stopped with a known
position. An axis a stop halted is IDLE where it stopped, so after a
stop this returns.

The axes it waits for are those MOVING or HOMING when it is called.
An axis that was not moving is not this wait's business -- one that
was never homed among them -- so a scope with unhomed X and Y can
still wait on its Z. Waits on the per-axis arrival events the motion
monitor sets; no serial I/O from the calling thread.

**Args**

- `timeout_s`: Maximum seconds to wait for all of them (default 120s).

**Raises**

- `MoveFaultError`: An axis it waited for ended UNKNOWN (the monitor's own object when it gave the axis up: `'stalled'`, `'board_lost'`, `'position_unread_after_move'`, `'status_unread'` or `'monitor_ended_mid_move'`; else `'faulted'`).
- `MoveStoppedError`: An axis it waited for ended UNKNOWN after a STOP ended its move, as the close's does: `'stopped'`, or `'superseded'`.
- `MoveWaitExpiredError`: `'still_moving'`, a `TimeoutError`: an axis was still moving when `timeout_s` ran out; its state is left to its own move.
- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days: a NaN deadline would end the wait at once, and one longer than the host can hold would raise out of it.

**Refusals by REST status**

- 409: [`MoveStoppedError`](#MoveStoppedError): `stopped`, `superseded`; [`MoveWaitExpiredError`](#MoveWaitExpiredError): `still_moving`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- 500: [`MoveFaultError`](#MoveFaultError): `stalled`, `board_lost`, `position_unread_after_move`, `status_unread`, `monitor_ended_mid_move`, `faulted`

<a id="RuntimeState"></a>
## `RuntimeState`

Reached through `session.scope.runtime_state`; on the wire, `/api/v1/scope/runtime_state/`.  
Held in: [`Lumascope.runtime_state`](#Lumascope.runtime_state).

Mutable runtime state on a Lumascope, and the answers for the user
configuration the scope acts on (labware / objective / turret / stage),
read from the session's settings.

Fields land per design doc sec 2.5 as the underlying firmware +
driver hooks ship.

<a id="RuntimeState.get_field_of_view"></a>
### `RuntimeState.get_field_of_view`

```python
def get_field_of_view(self) -> FieldOfView | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/get_field_of_view`.

The area the camera's frame covers on the sample, at the current
objective and the camera's binning.

**Returns**

FieldOfView | None: Width and height in micrometres; None when the pixel size cannot be known (`get_pixel_size_um`) or the camera has not reported its frame.

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope.

**Refusals by REST status**

- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.get_pixel_size_um"></a>
### `RuntimeState.get_pixel_size_um`

```python
def get_pixel_size_um(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/get_pixel_size_um`.

The effective pixel size: micrometres per pixel at the current
objective and the camera's binning, the scale a saved image carries.

**Returns**

float | None: None when it cannot be known: no objective is known (`resolve_current_objective` says why), there is no camera, the camera has not reported its binning, or the scope cannot report its optics.

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope.

**Refusals by REST status**

- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.get_stage_offset"></a>
### `RuntimeState.get_stage_offset`

```python
def get_stage_offset(self) -> coord_transformations.StageOffset
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/get_stage_offset`.

The stage offset for coordinate transformations, from the settings.

**Returns**

Where the plate is mounted on the stage, in um.

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope.

**Refusals by REST status**

- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.get_turret_config"></a>
### `RuntimeState.get_turret_config`

```python
def get_turret_config(self) -> dict[int, str | None]
```

Marks: in-process; reads (changes nothing).

The turret's objective assignments, from the settings.

**Returns**

Turret position (1-4) to objective ID, None for an unassigned slot. A copy: changing it changes no assignment (`ScopeSession.assign_turret_objective` does).

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope.

**Refusals**

- [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.get_well_label"></a>
### `RuntimeState.get_well_label`

```python
def get_well_label(self) -> str | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/get_well_label`.

Get the well label for the current stage XY position.

Maps the current target X/Y stage position to a plate-frame
coordinate using the selected labware and stage offset, then
looks up the matching well label.

**Returns**

str | None: Well label (e.g. `"A1"`); `''` when the selected labware has no wells (the Blank plate) or the position is more than half a pitch beyond its outer well centres; None when X or Y does not know its position, or the scope has no X or Y at all. Consumers omit the well from filenames and metadata for both rather than stamping a fabricated one -- an axis that lost its reference keeps answering the last target it had, which names a real well the scope may no longer be over. The two stay distinct so a caller can say which it was.

**Raises**

- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope.
- `CatalogueNameRefusedError`: `'labware_unknown'`, the selected labware is not in the catalogue.

**Refusals by REST status**

- 422: [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.plate_to_stage"></a>
### `RuntimeState.plate_to_stage`

```python
def plate_to_stage(
    self,
    x_mm: float,
    y_mm: float,
    *,
    labware_id: str | None = None,
    stage_offset: coord_transformations.StageOffset | None = None,
) -> coord_transformations.StagePosition
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/plate_to_stage`.

Convert a plate position (mm) to a stage position (um), as `stage_to_plate` inverted.

The plate and offset are `stage_to_plate`'s. Not bounded: a
position off the stage's travel converts, and the move to it is
refused.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, a coordinate or an offset axis is not a finite number.
- `CatalogueNameRefusedError`: `'labware_unknown'`, `labware_id` is not a plate the catalogue has.
- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope, and `labware_id` or `stage_offset` was left to the settings.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.plate_transform"></a>
### `RuntimeState.plate_transform`

```python
def plate_transform(
    self,
    *,
    labware_id: str | None = None,
    stage_offset: coord_transformations.StageOffset | None = None,
) -> Callable[[float, float], coord_transformations.PlatePosition]
```

Marks: in-process; reads (changes nothing).

`stage_to_plate` bound to one plate and offset, taken when this is called.

`stage_to_plate` reads the selected plate and live offset on every
call, so a caller converting many positions over time -- a recording
writing one per frame, a run, a redraw -- would state its later ones
against a plate or offset changed mid-way. The function returned is
bound to the plate and offset of this call (the arguments are
`stage_to_plate`'s), so every position it converts is stated in one
frame of reference and a later change moves none of them.

**Returns**

A function of `(x_um, y_um)` answering their `PlatePosition`; it raises only as the transform refuses a non-finite coordinate.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, an offset axis is not a finite number; asked here, so the function does not.
- `CatalogueNameRefusedError`: `'labware_unknown'`, `labware_id` is not a plate the catalogue has.
- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope, and `labware_id` or `stage_offset` was left to the settings.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`
- [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`
- [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="RuntimeState.resolve_current_objective"></a>
### `RuntimeState.resolve_current_objective`

```python
def resolve_current_objective(self) -> tuple[str, Objective]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/resolve_current_objective`.

The active objective's id and record, or why it is unknown.

On a turreted scope (`capabilities.has_turret`), the objective
assigned to the slot in the light path (`motion.turret_status`),
read fresh every time; with no turret, the selected one.

**Returns**

tuple[str, Objective]: The objective id and its catalogue record.

**Raises**

- `ObjectiveUnknownError`: On a turreted scope, the slot is unknown (`'slot_unknown'`), it has no assignment (`'slot_unassigned'`), or its assignment is not in the catalogue (`'not_in_catalogue'`); with no turret, nothing was selected (`'none_selected'`).

**Refusals by REST status**

- 409: [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`, `slot_unassigned`, `not_in_catalogue`, `none_selected`

<a id="RuntimeState.stage_to_plate"></a>
### `RuntimeState.stage_to_plate`

```python
def stage_to_plate(
    self,
    x_um: float,
    y_um: float,
    *,
    labware_id: str | None = None,
    stage_offset: coord_transformations.StageOffset | None = None,
) -> coord_transformations.PlatePosition
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/runtime_state/stage_to_plate`.

Convert a stage position (um) to a plate position (mm).

`plate_to_stage` inverts it. Stated on the selected plate at the
live stage offset unless `labware_id` or `stage_offset` names
another: a protocol's positions are on the plate it stores, and a
run keeps the offset it started with.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, a coordinate or an offset axis is not a finite number.
- `CatalogueNameRefusedError`: `'labware_unknown'`, `labware_id` is not a plate the catalogue has.
- `ConfigurationDefectError`: `'scope_not_composed'`, no session has bound this scope, and `labware_id` or `stage_offset` was left to the settings.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`; [`CatalogueNameRefusedError`](#CatalogueNameRefusedError): `labware_unknown`
- 500: [`ConfigurationDefectError`](#ConfigurationDefectError): `scope_not_composed`

<a id="SimulationAPI"></a>
## `SimulationAPI`

Reached through `session.scope.simulation`; on the wire, `/api/v1/scope/simulation/`.  
Returned by: [`Lumascope.simulation`](#Lumascope.simulation).

Fault injection on a simulated scope; every member refuses on a real one.

Holds the simulated boards the scope built, which are the drivers
themselves on the fast tier, the boards behind the production drivers
on the firmware tiers, and the devices behind them on a Classic model.

<a id="SimulationAPI.answer_model"></a>
### `SimulationAPI.answer_model`

```python
def answer_model(self, model: str | None) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/answer_model`.

The motor board names `model` as its own, until `clear`; None names none.

Its axes stay the ones it was built with, as a board whose firmware
names a model its motors are not: `diagnostics.get_motor_info`
reports the answer, and bring-up reads it to decide the scope's
model.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'fault_unknown'` for a scope with no motor board; `'not_injectable_on_board'` on every board but the fast motor board: the firmware boards and the TMCM-6110 answer with the model they were built as.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.bind_camera_usb_driver"></a>
### `SimulationAPI.bind_camera_usb_driver`

```python
def bind_camera_usb_driver(self, hardware_id: str, service: str | None) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/bind_camera_usb_driver`.

Windows binds a USB identity of the simulated FX2 to `service`.

The simulated FX2 comes up on WinUSB, as Windows binds it with
LumaViewPro's package installed; this rebinds one identity, as a
Zadig run or LumaView Classic's package would. The camera already up
keeps running: the binding is read by
`diagnostics.get_camera_usb_bindings` and checked at the next
connect.

**Args**

- `hardware_id`: `USB\VID_04B4&PID_8613` (the bootloader) or `USB\VID_04B4&PID_EA17` (the running firmware).
- `service`: The driver's service, as Windows names it (`'libusbK'`); None for no driver.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'fault_unknown'` for any other hardware ID; `'not_injectable_on_board'` on any camera but the FX2.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.clear"></a>
### `SimulationAPI.clear`

```python
def clear(self) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/clear`.

Every injected fault taken back, newest first: drops, holds,
refused writes and stalls lifted, boards plugged back in and
answering, faults out of the hardware. A board that found its cable
pulled is plugged in but stays disconnected until it is reconnected,
as a real one does. A removed camera, an FX2 unplugged or
silenced, and a reboot under way are not undone, as on a real scope.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`

<a id="SimulationAPI.delay_delivery"></a>
### `SimulationAPI.delay_delivery`

```python
def delay_delivery(self, frames: int) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/delay_delivery`.

The camera's frames are stored `frames` behind their hand-over
from now on, as through a driver queue that has fallen behind.

Each frame is numbered and timed when the driver has it, and stored
once `frames` more have been handed over after it, so a frame
handed over before a change can be stored well after it and is still
placed before it. The backlog builds over the next `frames`
frames, in which none is stored; the first frame stored after this
call says it stands. `clear` takes it back and the held frames
are stored at once.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_a_number'` or `'fault_bound'` for fewer than 1 frame; `'not_injectable_on_board'` on the FX2, whose frames come over its own link.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_a_number`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.delay_reply"></a>
### `SimulationAPI.delay_reply`

```python
def delay_reply(self, board: Part, delay_s: float) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/delay_reply`.

The board's next reply comes `delay_s` seconds late.

A reply later than the driver waits for is no reply to it: the
board did what it was told, and the host was not told so.

**Args**

- `board`: `MOTOR` or `LED`, a board this scope has.
- `delay_s`: How late the reply comes, in seconds, above 0.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'` for a board this scope lacks; `'not_a_number'` or `'fault_bound'` for a delay not above 0 or past 30 days, the longest wait every host can hold; `'not_injectable_on_board'` on every board but a firmware board, the only one with a link to delay.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `not_a_number`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.drop_replies"></a>
### `SimulationAPI.drop_replies`

```python
def drop_replies(
    self,
    board: Part,
    count: int | None = 1,
    command: str | None = None,
) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/drop_replies`.

The board answers nothing to its next `count` commands.

**Args**

- `board`: `MOTOR` or `LED`, a board this scope has.
- `count`: How many replies go missing, 1 or more; None for every one until `clear`, as a register the board stops reporting is missing from every poll. Taken by the fast motor board only: a firmware board's link drops a counted few.
- `command`: Only commands with this first word, such as `'ACTUAL_RZ'` or `'STOP'`; any command when None. Taken by the fast motor board only: a firmware board's link loses whichever reply comes next.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'` for a board this scope lacks; `'fault_bound'` for a count below 1; `'argument_not_on_board'` for a count of None on a firmware board; `'not_injectable_on_board'` on the fast LED board, which has no link to drop a reply on, and on a Classic model's boards, which have `silence`; `'argument_not_on_board'` for a command on a firmware board.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `fault_bound`, `argument_not_on_board`, `not_injectable_on_board`

<a id="SimulationAPI.exposed_under"></a>
### `SimulationAPI.exposed_under`

```python
def exposed_under(self, frame: Frame) -> SensorExposure
```

Marks: in-process; reads (changes nothing).

What a frame the simulated camera delivered was exposed under:
the settings its sensor applied when its exposure started, and Z
when it ended.

A frame's chunks name the register when it was tagged, which a
change reaches before the frames already started show it; this is
what the pixels show, so a test can judge a frame by what it was
made under, never by its pixels or by the gate's own answer.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_injectable_on_board'` on the FX2, which keeps no record; `'exposure_not_recorded'` for a frame this camera did not deliver, or one older than the records it keeps (`SimulatedCamera.EXPOSURE_RECORDS_KEPT`).

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_injectable_on_board`, `exposure_not_recorded`

<a id="SimulationAPI.fail_home"></a>
### `SimulationAPI.fail_home`

```python
def fail_home(self, axis: Axis, how: HomeFault) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/fail_home`.

`axis`'s home fails in the simulated hardware, until `clear`.

**Args**

- `axis`: The axis whose home fails.
- `how`: `'switch_never_trips'` (its home switch never closes), `'stall'` (the motor stalls) or `'absent'` (no motor).

**Raises**

- `ArgumentRefusedError`: `'not_offered'` for `axis` or `how`; `'needs_simulated_scope'`; `'fault_unknown'` for a scope with no motor board; `'not_injectable_on_board'` on the fast motor board, which runs no firmware to fail a home; `'argument_not_on_board'` for a `how` the TMCM-6110 cannot show (it has `'switch_never_trips'`).

**Refusals by REST status**

- 409: [`ArgumentRefusedError`](#ArgumentRefusedError): `switch_never_trips`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`, `needs_simulated_scope`, `fault_unknown`, `not_injectable_on_board`, `argument_not_on_board`

<a id="SimulationAPI.hold_axis"></a>
### `SimulationAPI.hold_axis`

```python
def hold_axis(self, axis: Axis, at_fraction: float = 0.5) -> threading.Event
```

Marks: in-process.

`axis`'s next move halts part of the way and never arrives, until `clear`.

A stop, a home or a new target on the axis ends the move it holds,
and with it the hold, as on a real stage.

**Args**

- `axis`: The axis whose next move is held.
- `at_fraction`: How far along its travel it halts, between 0 and 1.

**Returns**

An event set once the move has halted at the hold, for a caller acting on a move while it travels: waited on, it replaces a guess with a clock at how far the move has got. `clear` releases the move, which then travels the rest.

**Raises**

- `ArgumentRefusedError`: `'not_offered'` for `axis`; `'needs_simulated_scope'`; `'not_a_number'` or `'fault_bound'` for a fraction outside 0 to 1; `'fault_unknown'` for a scope with no motor board; `'not_injectable_on_board'` on every board but the fast motor board: the firmware's axis reaches its target whatever is injected, and the TMCM-6110's arrives too.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_offered`, `needs_simulated_scope`, `not_a_number`, `fault_bound`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.injected"></a>
### `SimulationAPI.injected`

```python
def injected(self) -> tuple[InjectedFault, ...]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/scope/simulation/injected`.

The faults injected since the last `clear`, oldest first, as asked.

A one-shot fault stays listed after it has taken its command.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`.

**Refusals by REST status**

- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`

<a id="SimulationAPI.interlock"></a>
### `SimulationAPI.interlock`

```python
def interlock(self, lid_open: bool | None = None, powered: bool | None = None) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/interlock`.

The stage's interlock inputs read as given; None leaves one as it is.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'fault_unknown'` for a scope with no motor board; `'not_injectable_on_board'` on every board but the TMCM-6110, the only one with interlock inputs.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.reboot"></a>
### `SimulationAPI.reboot`

```python
def reboot(self, board: Part, booting_for_s: float = 0.0) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/reboot`.

The board's firmware restarts and forgets everything it knew.

Its port fails at once, as USB re-enumerates, and the board is found
again after `booting_for_s` seconds.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'`; `'not_a_number'` or `'fault_bound'` for a negative boot or one past 30 days; `'not_injectable_on_board'` on every board but a firmware board, the only one with firmware to restart.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `not_a_number`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.refuse_camera_read"></a>
### `SimulationAPI.refuse_camera_read`

```python
def refuse_camera_read(
    self,
    node: CameraReadNode,
    answer: CameraReadAnswer,
    count: int = 1,
) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/refuse_camera_read`.

The camera's next `count` reads of `node` fail as the real
driver that reads the node fails (`SimulatedCamera.FAILED_READS`).

The gain, the exposure and the binning answer -1, the frame size
and the pixel format None; the black level, the frame rate, the
minimum exposure and the auto modes raise `CameraReadFailedError`.
The minimum exposure's is IDS's answer, the one driver that reads it
live, offered here so the live-floor read is proven without an IDS
body. A real driver fails the pixel-format read only before its
first good read after a connect, since it keeps the format it read.

**Args**

- `node`: The setting read: `'gain'`, `'exposure'`, `'frame_size'`, `'binning'`, `'pixel_format'`, `'black_level'`, `'frame_rate'`, `'minimum_exposure'` (`camera.read_exposure_min_ms`), `'auto_gain'` or `'auto_exposure'`.
- `answer`: `'failed'`, the node's answer; or `'removed'`, the same answer with the camera gone from the bus during the read (it stays removed for the session, as after `remove_camera`).
- `count`: How many reads in a row meet it, 1 or more.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'` for a node or an answer; `'fault_bound'` for a count below 1; `'not_injectable_on_board'` on the FX2 camera.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.refuse_camera_write"></a>
### `SimulationAPI.refuse_camera_write`

```python
def refuse_camera_write(
    self,
    node: CameraWriteNode,
    answer: CameraWriteAnswer,
    count: int = 1,
) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/refuse_camera_write`.

The camera's next `count` writes of `node` answer `answer` and move nothing.

**Args**

- `node`: The setting written: `'gain'`, `'exposure'`, `'auto_exposure'`, `'frame_size'`, `'binning'`, `'pixel_format'`, `'black_level'`, `'auto_gain'`, `'auto_gain_once'` or `'auto_gain_target_brightness'`.
- `answer`: What the node write answers: `'false'`, `'none'` (no confirmation), `'true'` (a bare True that confirms no value; not for `'frame_size'`, whose write answers the window it delivers), `'raise'`, or `'removed'` (False, with the camera gone from the bus during the write; it stays removed for the session, as after `remove_camera`), or `'no_answer'` (`CameraNoAnswerError`: the write timed out, and the camera stays on the bus).
- `count`: How many writes in a row meet it, 1 or more: the API sends a write the camera did not answer once more, so a second unanswered write takes 2.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'` for a node, or an answer that node's write does not give, offering those it gives; `'fault_bound'` for a count below 1; `'not_injectable_on_board'` on the FX2 camera.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.refuse_led_write"></a>
### `SimulationAPI.refuse_led_write`

```python
def refuse_led_write(self, count: int = 1) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/refuse_led_write`.

The LED board refuses its next `count` writes and stays on the bus.

An LED command whose write is refused raises the LED board's
`BoardPortFailedError` to its caller, and the LED stays connected
for the next command. On the FX2 each LED command is three writes.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'fault_unknown'` on a scope with no LED board; `'not_a_number'` or `'fault_bound'` for fewer than 1 write; `'not_injectable_on_board'` on any LED board but the FX2.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `fault_unknown`, `not_a_number`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.refuse_stream_start"></a>
### `SimulationAPI.refuse_stream_start`

```python
def refuse_stream_start(self, count: int = 1) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/refuse_stream_start`.

The camera refuses its next `count` stream starts and stays on the bus.

A start the camera refuses raises its fault to whoever started the
stream, and the camera is left connected and not streaming.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_a_number'` or `'fault_bound'` for fewer than 1 start; `'not_injectable_on_board'` on the simulated camera, made like a Pylon body, whose driver logs a start that fails.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_a_number`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.remove_camera"></a>
### `SimulationAPI.remove_camera`

```python
def remove_camera(self) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/remove_camera`.

The camera leaves the bus, as a real one does when removed.

It stays removed for the session: `clear` does not bring it back,
as nothing brings back a removed real camera but a new session.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'fault_unknown'` on a scope whose camera did not come up.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `fault_unknown`

<a id="SimulationAPI.replug"></a>
### `SimulationAPI.replug`

```python
def replug(self, board: Part) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/replug`.

The board's cable goes back in: it is found again.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'`; `'not_injectable_on_board'` where `unplug` is, and on the FX2, which re-enumerates on its own.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.silence"></a>
### `SimulationAPI.silence`

```python
def silence(self, board: Part) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/silence`.

The board answers no command from now on, until `unsilence`.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'`; `'not_injectable_on_board'` on the fast LED board and a firmware board, whose silence is `unplug`.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.stall_stream"></a>
### `SimulationAPI.stall_stream`

```python
def stall_stream(self, after_s: float, for_s: float) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/stall_stream`.

The camera's stream delivers no frame for `for_s` seconds,
starting `after_s` seconds from now; it stays connected and grabbing.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_a_number'` or `'fault_bound'` for a negative start or a stall of no length; `'not_injectable_on_board'` on the FX2, whose stream does not resume (`silence('led')` stops it).

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_a_number`, `fault_bound`, `not_injectable_on_board`

<a id="SimulationAPI.unplug"></a>
### `SimulationAPI.unplug`

```python
def unplug(self, board: Part) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/unplug`.

The board's cable is pulled: its port fails from the next command.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'`; `'not_injectable_on_board'` on the fast LED board, which has no link to lose.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `not_injectable_on_board`

<a id="SimulationAPI.unsilence"></a>
### `SimulationAPI.unsilence`

```python
def unsilence(self, board: Part) -> None
```

Marks: enters: `guarded_write`.  
Route: `POST /api/v1/scope/simulation/unsilence`.

The board answers again after `silence`.

**Raises**

- `ArgumentRefusedError`: `'needs_simulated_scope'`; `'not_offered'`, `board` is not a `Part`; `'fault_unknown'`; `'not_injectable_on_board'` where `silence` is, and on the FX2, whose stream does not resume.

**Refusals by REST status**

- 409: [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `needs_simulated_scope`, `not_offered`, `fault_unknown`, `not_injectable_on_board`

<a id="AppliedAutoGain"></a>
## `AppliedAutoGain`

A record: it crosses the wire as its data.  
Returned by: [`CameraAPI.applied_auto_gain_for`](#CameraAPI.applied_auto_gain_for).

What a stored auto-gain preference becomes on the attached camera (`applied_auto_gain_for`).

**Fields**

- <a id="AppliedAutoGain.applied"></a>`applied`: `bool` -- (no description)
- <a id="AppliedAutoGain.capped"></a>`capped`: `bool` -- (no description)
- <a id="AppliedAutoGain.stored"></a>`stored`: `bool` -- (no description)

<a id="AppliedExposure"></a>
## `AppliedExposure`

A record: it crosses the wire as its data.  
Returned by: [`CameraAPI.applied_exposure_ms_for`](#CameraAPI.applied_exposure_ms_for).

What a stored exposure becomes on the attached camera, in ms (`applied_exposure_ms_for`).

**Fields**

- <a id="AppliedExposure.applied_ms"></a>`applied_ms`: `float` -- (no description)
- <a id="AppliedExposure.capped"></a>`capped`: `bool` -- (no description)
- <a id="AppliedExposure.stored_ms"></a>`stored_ms`: `float` -- (no description)

<a id="AppliedGain"></a>
## `AppliedGain`

A record: it crosses the wire as its data.  
Returned by: [`CameraAPI.applied_gain_db_for`](#CameraAPI.applied_gain_db_for).

What a stored gain becomes on the attached camera, in dB (`applied_gain_db_for`).

**Fields**

- <a id="AppliedGain.applied_db"></a>`applied_db`: `float` -- (no description)
- <a id="AppliedGain.capped"></a>`capped`: `bool` -- (no description)
- <a id="AppliedGain.stored_db"></a>`stored_db`: `float` -- (no description)

<a id="AutoGainLock"></a>
## `AutoGainLock`

A record: it crosses the wire as its data.  
Returned by: [`CameraAPI.lock_auto_gain`](#CameraAPI.lock_auto_gain), [`ScopeSession.channel_auto_gain_lock`](#ScopeSession.channel_auto_gain_lock), [`ScopeSession.set_channel_auto_gain`](#ScopeSession.set_channel_auto_gain).

The result of locking an auto-gain arm; `state` is None when no
arm was recorded (nothing was locked, no driver traffic happened).

**Fields**

- <a id="AutoGainLock.ceiling_ms"></a>`ceiling_ms`: `float | None` -- (no description)
- <a id="AutoGainLock.exposure_ms"></a>`exposure_ms`: `float | None` -- (no description)
- <a id="AutoGainLock.floor_ms"></a>`floor_ms`: `float | None` -- (no description)
- <a id="AutoGainLock.gain_db"></a>`gain_db`: `float | None` -- (no description)
- <a id="AutoGainLock.resume_after_capture"></a>`resume_after_capture`: `bool` -- (no description)
- <a id="AutoGainLock.settings"></a>`settings`: `dict | None` -- (no description)
- <a id="AutoGainLock.state"></a>`state`: `AutoGainConvergence | None` -- (no description)
- <a id="AutoGainLock.stored_exposure_ms"></a>`stored_exposure_ms`: `float | None` -- (no description)

<a id="AxisLimits"></a>
## `AxisLimits`

A record: it crosses the wire as its data.  
Returned by: [`MotionAPI.get_axis_limits`](#MotionAPI.get_axis_limits).

An axis's travel, in um: a move outside it is refused.

**Fields**

- <a id="AxisLimits.max_um"></a>`max_um`: `float` -- (no description)
- <a id="AxisLimits.min_um"></a>`min_um`: `float` -- (no description)

<a id="AxisPosition"></a>
## `AxisPosition`

A record: it crosses the wire as its data.  
Returned by: [`MotionAPI.axis_positions`](#MotionAPI.axis_positions).  
Held in: [`PositionChanged.at`](#PositionChanged.at), [`Status.axes`](#Status.axes).  
Taken by: [`MotionAPI.add_position_listener`](#MotionAPI.add_position_listener), [`MotionAPI.remove_position_listener`](#MotionAPI.remove_position_listener).

One stage axis's state and its position, read together: X, Y or Z.

`position_um` is None unless the axis is IDLE or MOVING: an axis whose
reference is lost or still being established keeps answering the last
number it reported, and a caller writing a position into a file must
not be handed it. The turret has a slot, not a position: `TurretStatus`.

**Fields**

- <a id="AxisPosition.position_um"></a>`position_um`: `float | None` -- (no description)
- <a id="AxisPosition.state"></a>`state`: `AxisState` -- (no description)

<a id="BringUpRecord"></a>
## `BringUpRecord`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.bring_up_record`](#ScopeSession.bring_up_record).

What bring-up found, substituted and set aside.

**Fields**

- <a id="BringUpRecord.parts"></a>`parts`: `tuple[PartStatus, ...]` -- (no description)
- <a id="BringUpRecord.settings_set_aside"></a>`settings_set_aside`: `SettingsSetAside | None` -- (no description)
- <a id="BringUpRecord.substitutions"></a>`substitutions`: `tuple[Substitution, ...]` -- (no description)

<a id="BringUpRecord.missing"></a>
### `BringUpRecord.missing`

```python
missing: tuple[PartStatus, ...]  # property
```

The parts this scope should have and lacks.

**Raises**

Nothing.

<a id="BuildResult"></a>
## `BuildResult`

A record: it crosses the wire as its data.  
Held in: [`BuildStep.result`](#BuildStep.result).

What a stitch, z-projection, composite or video build made.

**Fields**

- <a id="BuildResult.artifact_paths"></a>`artifact_paths`: `tuple[pathlib.Path, ...]` -- Each file it made, where it landed.
- <a id="BuildResult.degraded_outputs"></a>`degraded_outputs`: `tuple[DegradedOutput, ...]` -- The files a fallback algorithm made; empty when none.
- <a id="BuildResult.excluded_inputs"></a>`excluded_inputs`: `tuple[ExcludedInput, ...]` -- The files of each kind in the folder the build did not take (a stitch's per-tile composites), not counting its own earlier outputs; empty when none. The message names them, with the route to their output where there is one.
- <a id="BuildResult.message"></a>`message`: `str` -- The outcome in words, with what was skipped and why.
- <a id="BuildResult.new_count"></a>`new_count`: `int` -- How many files it made.
- <a id="BuildResult.output_root"></a>`output_root`: `pathlib.Path` -- The folder it read and wrote under.

<a id="BuildStep"></a>
## `BuildStep`

A record: it crosses the wire as its data.  
Held in: [`BuildsResult.built`](#BuildsResult.built).

One build `build` ran, and what it made.

**Fields**

- <a id="BuildStep.operation"></a>`operation`: `PostProcessingOperation` -- The build.
- <a id="BuildStep.result"></a>`result`: `BuildResult` -- What it made.

<a id="BuildsResult"></a>
## `BuildsResult`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.build`](#PostProcessingAPI.build).

What `build` made.

**Fields**

- <a id="BuildsResult.built"></a>`built`: `tuple[BuildStep, ...]` -- Each build run and what it made, in the order run.
- <a id="BuildsResult.message"></a>`message`: `str` -- Each build's outcome in words, in the order run.
- <a id="BuildsResult.skipped"></a>`skipped`: `tuple[SkippedBuild, ...]` -- Each build that had nothing to do, and why.

<a id="CameraLimits"></a>
## `CameraLimits`

A record: it crosses the wire as its data.  
Held in: [`ScopeCapabilities.camera_limits`](#ScopeCapabilities.camera_limits).

The ranges the attached camera takes, read once at connect.

`set_gain_db`, `set_exposure_ms` and `set_frame_size` refuse a
value outside them before it reaches the camera. A limit the camera
does not declare is None, and that end is not checked.

`gain_max_db` is the most gain the camera applies, digital stage
included; `analog_gain_max_db` the most before a digital stage, past
which the camera multiplies digitised values and the noise with the
signal: the analog maximum the camera's profile documents, or, for a
camera whose profile states no digital stage, its total maximum (all
its gain is analog), None for a camera with a digital stage and no
documented split. `frame_max` is the largest frame the scope
delivers, unbinned: the smallest of the sensor size the camera's
profile documents, the maximum the camera reports at connect
multiplied back by the binning it reported then, and the model's
catalogue `MaxFrame` (a lens that images less than the sensor: the
LS560's 1700), each where it is known. The profile stays in that
comparison because a camera can report a few rows and columns more
than its documented sensor (the LS850T's daA3840 reports 3860 x 2178
against 3840 x 2160). `set_frame_size` divides it by the
binning in force. `frame_min` is the smallest frame the camera
reported at connect.

**Fields**

- <a id="CameraLimits.analog_gain_max_db"></a>`analog_gain_max_db`: `float | None` -- (no description)
- <a id="CameraLimits.exposure_max_ms"></a>`exposure_max_ms`: `float | None` -- (no description)
- <a id="CameraLimits.exposure_min_ms"></a>`exposure_min_ms`: `float | None` -- (no description)
- <a id="CameraLimits.frame_max"></a>`frame_max`: `FrameSize | None` -- (no description)
- <a id="CameraLimits.frame_min"></a>`frame_min`: `FrameSize | None` -- (no description)
- <a id="CameraLimits.gain_max_db"></a>`gain_max_db`: `float | None` -- (no description)
- <a id="CameraLimits.gain_min_db"></a>`gain_min_db`: `float | None` -- (no description)

<a id="CameraLink"></a>
## `CameraLink`

A record: it crosses the wire as its data.  
Returned by: [`DiagnosticsAPI.get_camera_link_info`](#DiagnosticsAPI.get_camera_link_info).

The camera's link to the host, read live; each field None where the
camera does not report it.

**Fields**

- <a id="CameraLink.inter_packet_delay"></a>`inter_packet_delay`: `int | None` -- GigE's inter-packet delay, in the camera's ticks.
- <a id="CameraLink.link_speed"></a>`link_speed`: `float | None` -- The negotiated speed, in `link_speed_unit`, never converted: GenICam's DeviceLinkSpeed has a unit that varies by Basler model, and libusb reports the FX2's in Mbps. A speed whose camera declares no unit is None with its unit, a number no caller could read.
- <a id="CameraLink.link_speed_unit"></a>`link_speed_unit`: `str | None` -- The unit the camera declares for `link_speed`.
- <a id="CameraLink.packet_size_bytes"></a>`packet_size_bytes`: `int | None` -- GigE's stream packet size.
- <a id="CameraLink.transport"></a>`transport`: `str | None` -- `'USB3'`, `'GigE'` or `'USB2'`; the SDK's own name when it is none of these.

<a id="CaptureTally"></a>
## `CaptureTally`

A record: it crosses the wire as its data.  
Held in: [`RunOutcome.captures`](#RunOutcome.captures).

The captures a run was asked for, and what became of each.

Counted once per (scan, step): a scan that is run again after a
transient failure re-captures its earlier steps into the same slots,
so a retry never counts a capture twice. A blown or a dark frame is
captured -- the camera delivered it -- and is counted again among
`saturated` or `dark`.

**Fields**

- <a id="CaptureTally.asked"></a>`asked`: `int` -- Captures the run was asked for: scans times steps for a run that saves images, 0 for one that saves none (an autofocus scan, a standalone autofocus).
- <a id="CaptureTally.captured"></a>`captured`: `int` -- Captures that produced an image.
- <a id="CaptureTally.dark"></a>`dark`: `tuple[TakenCapture, ...]` -- The captures whose frame was dark, in the same order.
- <a id="CaptureTally.failed"></a>`failed`: `tuple[FailedCapture, ...]` -- Captures that produced none, in the order they failed. `asked - captured - len(failed)` were never reached.
- <a id="CaptureTally.saturated"></a>`saturated`: `tuple[TakenCapture, ...]` -- The captures whose frame was blown, in the order their (scan, step) was first taken.

<a id="CaptureTally.missing"></a>
### `CaptureTally.missing`

```python
missing: int  # property
```

Captures asked for that produced no image, failed or never reached.

**Raises**

Nothing.

<a id="CapturedStill"></a>
## `CapturedStill`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.capture_still`](#ScopeSession.capture_still).

One captured still: its pixels and the instrument's record of them, together.

Handed back by the grab that took them, so no caller reads a frame's
record from a slot the next capture replaces. Equality is identity: a
frozen dataclass's own would compare the arrays elementwise and raise.

**Fields**

- <a id="CapturedStill.image"></a>`image`: `ndarray` -- The captured array, read-only: the frame, a sum, or its 8-bit rendering.
- <a id="CapturedStill.record"></a>`record`: `FrameRecord` -- The instrument's account of the frame (`FrameRecord`).

<a id="CellCountFilters"></a>
## `CellCountFilters`

A record: it crosses the wire as its data.  
Held in: [`CellCountMethod.filters`](#CellCountMethod.filters).

Which regions a count keeps, each bound checked as the filters are built.

**Raises**

- `PostProcessingRefusedError`: reason `method_invalid`, a filter that is not a `FilterBounds`, a bound that is not a number or None, or a min above its max; the message names the filter.

**Fields**

- <a id="CellCountFilters.area_um2"></a>`area_um2`: `FilterBounds` -- A region's area, in square microns.
- <a id="CellCountFilters.intensity_max_pct"></a>`intensity_max_pct`: `FilterBounds` -- A region's brightest pixel, in percent of full scale.
- <a id="CellCountFilters.intensity_mean_pct"></a>`intensity_mean_pct`: `FilterBounds` -- A region's mean pixel, in percent of full scale.
- <a id="CellCountFilters.intensity_min_pct"></a>`intensity_min_pct`: `FilterBounds` -- A region's darkest pixel, in percent of full scale.
- <a id="CellCountFilters.perimeter_um"></a>`perimeter_um`: `FilterBounds` -- A region's perimeter, in microns.
- <a id="CellCountFilters.sphericity"></a>`sphericity`: `FilterBounds` -- From 0 to 1.

<a id="CellCountMethod"></a>
## `CellCountMethod`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.default_cell_count_method`](#PostProcessingAPI.default_cell_count_method), [`PostProcessingAPI.load_cell_count_method`](#PostProcessingAPI.load_cell_count_method).  
Taken by: [`PostProcessingAPI.cell_count_size_scale`](#PostProcessingAPI.cell_count_size_scale), [`PostProcessingAPI.count_cells`](#PostProcessingAPI.count_cells), [`PostProcessingAPI.preview_cell_count`](#PostProcessingAPI.preview_cell_count), [`PostProcessingAPI.save_cell_count_method`](#PostProcessingAPI.save_cell_count_method).

How a count finds and keeps cells, checked as it is built.

`CellCountMethod()` is the method LumaViewPro's panel starts from. It
sets no scale, so each image is counted at the scale it states, and no
size filter, so an image that states none is counted in pixels.

**Raises**

- `PostProcessingRefusedError`: reason `method_invalid`, a *pixels_per_um* that is not a positive number or None (0 or below counts nothing, NaN writes NaN areas), a *fluorescent_mode* that is not True or False, a *threshold_pct* that is not a number, or *filters* that is not a `CellCountFilters`; the message names the field.

**Fields**

- <a id="CellCountMethod.filters"></a>`filters`: `CellCountFilters` -- Which regions are kept.
- <a id="CellCountMethod.fluorescent_mode"></a>`fluorescent_mode`: `bool` -- Bright cells on a dark field; False inverts the image first.
- <a id="CellCountMethod.pixels_per_um"></a>`pixels_per_um`: `float | None` -- The scale every image is counted at, in camera pixels per micron; None counts each image at its own scale.
- <a id="CellCountMethod.threshold_pct"></a>`threshold_pct`: `float` -- The brightness that separates a cell from the field, in percent of full scale.

<a id="CellCountPreview"></a>
## `CellCountPreview`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.preview_cell_count`](#PostProcessingAPI.preview_cell_count).

One image counted by a method, as a form shows it.

**Fields**

- <a id="CellCountPreview.counts"></a>`counts`: `CellCounts` -- What the count measured.
- <a id="CellCountPreview.image"></a>`image`: `np.ndarray` -- The image with the kept regions' outlines drawn on it.

<a id="CellCountResult"></a>
## `CellCountResult`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.count_cells`](#PostProcessingAPI.count_cells).

What a cell count wrote.

**Fields**

- <a id="CellCountResult.counted"></a>`counted`: `int` -- How many images it counted.
- <a id="CellCountResult.message"></a>`message`: `str` -- The outcome in words.
- <a id="CellCountResult.results_path"></a>`results_path`: `pathlib.Path` -- The results table it wrote.

<a id="CellCountSizeScale"></a>
## `CellCountSizeScale`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.cell_count_size_scale`](#PostProcessingAPI.cell_count_size_scale).

The scale an image is counted at, and what its size sliders stand for.

**Fields**

- <a id="CellCountSizeScale.area"></a>`area`: `SizeSliderStops` -- The area slider's stops.
- <a id="CellCountSizeScale.perimeter"></a>`perimeter`: `SizeSliderStops` -- The perimeter slider's stops.
- <a id="CellCountSizeScale.pixels_per_um"></a>`pixels_per_um`: `float | None` -- The pixels per micron the count measures at, or None when it counts in pixels.

<a id="CellCounts"></a>
## `CellCounts`

A record: it crosses the wire as its data.  
Held in: [`CellCountPreview.counts`](#CellCountPreview.counts).

What a cell count measured in one image: the values of its row in `results.csv`.

**Fields**

- <a id="CellCounts.num_cells"></a>`num_cells`: `int` -- The regions the method kept.
- <a id="CellCounts.total_object_area_px2"></a>`total_object_area_px2`: `float` -- Their area, in pixels.
- <a id="CellCounts.total_object_area_um2"></a>`total_object_area_um2`: `float | None` -- Their area, in square microns; None when the image was counted in pixels, with no scale.
- <a id="CellCounts.total_object_intensity"></a>`total_object_intensity`: `float` -- Their area in pixels times their mean intensity in percent of full scale, so it does not change with the scale.

<a id="Change"></a>
## `Change`

A record: it crosses the wire as its data.  
Returned by: [`ImagingAPI.recorded_changes`](#ImagingAPI.recorded_changes).  
Held in: [`Judgment.change`](#Judgment.change).

The last recorded change of one source, and the facts a frame is judged against.

Frames do not consume it: it stays the source's change until the next
`invalidate` replaces it, so a capture long after can still say how
long after the change its frame came.

**Fields**

- <a id="Change.at_seq"></a>`at_seq`: `int` -- The last frame the camera's driver had been handed when the change was recorded, a frame still in a host queue included; for a move, once in force, when its stop was stamped (`FrameValidity.motion_settled`). A frame already in the camera's pipeline when the register was written cannot show the new state, so crediting it would retire the skip count against frames that predate the change -- and the capture then returns a frame from before the write. An ordinal rather than a clock reading, because the comparison has to survive the clock moving: wall time runs backwards across a DST fall-back, an NTP correction or a host resume, and a settle count measured against a timestamp that jumped forward never completes. Frame numbers only go up.
- <a id="Change.credits"></a>`credits`: `tuple[Credit, ...]` -- The frames counted toward it, in the order counted, until `skip` was met.
- <a id="Change.earliest_ns"></a>`earliest_ns`: `int` -- The scope's steady clock (`host_clock.steady_ns`) of the earliest moment the change could have reached the hardware: a write's start, as its writer stamped it, since the register may change anywhere inside the write; for a move, once in force, the start of the last read that saw its axis still moving, else `written_ns`. A delay measured from here is never short; the rule judges from `written_ns` or `in_force_ns`, the latest, so the pair never accepts a frame early.
- <a id="Change.in_force_ns"></a>`in_force_ns`: `int | None` -- The scope's steady clock (`host_clock.steady_ns`) when the axis state the settle check reads became settled, stamped by the motion API (`FrameValidity.motion_settled`); None until then, and for a change that does not wait on an axis.
- <a id="Change.needs_in_force"></a>`needs_in_force`: `bool` -- A motion source the axis must also settle for: the frame count alone does not end its wait.
- <a id="Change.rule"></a>`rule`: `FrameRule` -- The rule the change is judged by, as its source's stood when it was recorded.
- <a id="Change.skip"></a>`skip`: `int` -- The frames the change needed, its rule's count when it was recorded.
- <a id="Change.source"></a>`source`: `FrameValiditySource` -- What changed.
- <a id="Change.written_at"></a>`written_at`: `datetime.datetime` -- Host wall-clock time of the same moment, for a reader.
- <a id="Change.written_ns"></a>`written_ns`: `int` -- The scope's steady clock (`host_clock.steady_ns`) when the change was recorded. Every recorder calls `invalidate` after a camera register or LED write returns, and before a move starts.

<a id="ChannelIdentity"></a>
## `ChannelIdentity`

A record: it crosses the wire as its data.  
Held in: [`Lumascope.channel_identity`](#Lumascope.channel_identity).

The resolved per-unit identity snapshot.

`source` records which rung answered: `MOTORCONFIG` (the unit's own
block -- authoritative and complete when present), `SCOPES` (the
model's rows -- what a pre-block unit of this model has), or
`UNRESOLVED` (no block and no resolvable model; empty, and loud at
the point of LED use rather than silently wrong here).

`model` is the scope model the resolver settled on, whichever rung
answered: the override, else the model the board reports, else the
operator's selection -- the only identity an FX2 scope has, since it
cannot report its own. A model the catalogue lacks is still carried,
with no channels; `None` means there was no model at all. Every reader
of "which model is this scope" takes it from here, so the channels, the
capabilities and the saved files name one model.

**Fields**

- <a id="ChannelIdentity.channels"></a>`channels`: `tuple[ChannelRecord, ...]` -- (no description)
- <a id="ChannelIdentity.filterset"></a>`filterset`: `str` -- (no description)
- <a id="ChannelIdentity.model"></a>`model`: `str | None` -- (no description)
- <a id="ChannelIdentity.source"></a>`source`: `IdentitySource` -- (no description)

<a id="ChannelIdentity.find"></a>
### `ChannelIdentity.find`

```python
def find(self, key_name: str) -> ChannelRecord | None
```

Marks: in-process; reads (changes nothing).

Return the layer whose stable key name matches, else None.

The deserialisation boundary: a name arriving from disk or from
an API caller is mapped to its record exactly once, here.

**Raises**

Nothing.

<a id="ChannelIdentity.refuse_unless_on_scope"></a>
### `ChannelIdentity.refuse_unless_on_scope`

```python
def refuse_unless_on_scope(self, channel: str, member: str, *, then: str) -> None
```

Marks: in-process.

Refuse `channel` unless it is a layer of this release and this unit has it.

The one question for a command that takes a layer: its settings,
its focus, a still or a recording named for it. The name first
(`refuse_unknown_layer`), so a name that is no layer is never
told it is missing hardware.

**Raises**

- `ArgumentRefusedError`: `'channel_unknown'`.
- `HardwareCommandRefusedError`: `'not_on_this_scope'`, naming the layer (`MissingPart.layer`).
- `ConfigurationDefectError`: `'layers_unresolved'`, this unit's layers could not be resolved, so `channel` cannot `then`.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`
- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `not_on_this_scope`
- [`ConfigurationDefectError`](#ConfigurationDefectError): `layers_unresolved`

<a id="ChannelRecord"></a>
## `ChannelRecord`

A record: it crosses the wire as its data.  
Returned by: [`ChannelIdentity.find`](#ChannelIdentity.find).  
Held in: [`ChannelIdentity.channels`](#ChannelIdentity.channels), [`ScopeModel.channels`](#ScopeModel.channels).

Identity of one layer on one unit.

`id` is the internal key and the display position -- assigned from
the release catalogue's order, never persisted to any per-unit or
user file, so it can renumber freely between releases.

`key_name` is the stable serialisation name (settings block keys,
protocol TSV `Color`, filename tokens, metadata `channel`). It never
changes when a layer is renamed.

`display_name` is the unit's name for the layer. Freely changeable: a
rename touches this field and nothing persisted. None for a
fluorescence channel a board's Channels block names by its excitation
alone.

`led_addresses` is the LED board address(es) this layer drives, empty
for a layer with no LED (luminescence). A tuple rather than a single
value because a future channel (quantitative phase) drives several
switches as one layer; today every shipped row carries zero or one.
A row in a model's file names them `led_channel`, read into this
field by the one row reader.

`excitation_nm` is the excitation wavelength -- named for what it
holds, since an ambiguous "wavelength" invites silently wrong
metadata -- and the number a fluorescence key names (`Ex<nm>`).
None is the truth for broadband transmitted light and for LED-less
layers.

`modality` and `background` are the layer's catalogue record's
(`CatalogueLayer`). `display_color` is the unit's colour for the layer:
its board block row's, else the catalogue record's.

`label` is what the layer is called wherever it is shown or written:
a fluorescence layer's excitation (`'405 nm'`), any other layer's
`display_name`.

**Fields**

- <a id="ChannelRecord.background"></a>`background`: `Background` -- (no description)
- <a id="ChannelRecord.display_color"></a>`display_color`: `str` -- (no description)
- <a id="ChannelRecord.display_name"></a>`display_name`: `str | None` -- (no description)
- <a id="ChannelRecord.excitation_nm"></a>`excitation_nm`: `float | None` -- (no description)
- <a id="ChannelRecord.id"></a>`id`: `int` -- (no description)
- <a id="ChannelRecord.key_name"></a>`key_name`: `str` -- (no description)
- <a id="ChannelRecord.label"></a>`label`: `str` -- (no description)
- <a id="ChannelRecord.led_addresses"></a>`led_addresses`: `tuple[int, ...]` -- (no description)
- <a id="ChannelRecord.modality"></a>`modality`: `Modality` -- (no description)

<a id="ChannelSettings"></a>
## `ChannelSettings`

A record: it crosses the wire as its data.  
Returned by: [`Protocol.channel_settings`](#Protocol.channel_settings).

One channel's row of a protocol's Layer Settings, each cell typed.

Every field but `acquire` is None where the cell is blank, or where a
row inferred from the channel's first step cannot supply it: applied,
the channel's control keeps what it holds.

**Fields**

- <a id="ChannelSettings.acquire"></a>`acquire`: `AcquireMode` -- What the channel captures.
- <a id="ChannelSettings.auto_gain"></a>`auto_gain`: `bool | None` -- The camera's auto-gain is on.
- <a id="ChannelSettings.exposure_ms"></a>`exposure_ms`: `float | None` -- The camera exposure, in ms.
- <a id="ChannelSettings.false_color"></a>`false_color`: `bool | None` -- The channel's image is false-coloured.
- <a id="ChannelSettings.frames_summed"></a>`frames_summed`: `int | None` -- How many frames a capture sums.
- <a id="ChannelSettings.gain_db"></a>`gain_db`: `float | None` -- The camera gain, in dB.
- <a id="ChannelSettings.illumination_ma"></a>`illumination_ma`: `float | None` -- The LED current, in mA.
- <a id="ChannelSettings.stim_enabled"></a>`stim_enabled`: `bool | None` -- The channel's stimulation is on; never inferred from the steps.

<a id="ClockMapping"></a>
## `ClockMapping`

A record: it crosses the wire as its data.  
Held in: [`FrameRecord.clock_mapping`](#FrameRecord.clock_mapping).  
Taken by: [`ImagingAPI.judge_frame`](#ImagingAPI.judge_frame).

The camera's timestamp clock against the host's steady clock, measured once.

The host latches the camera's clock between two reads of the scope's
steady clock (`host_clock.steady_ns`); the latch fell somewhere inside
that bracket, so its middle is the host time the latched value belongs to
and half its width is how far off that can be. The read of the latched
value comes after the bracket and is no part of it.

**Fields**

- <a id="ClockMapping.camera_ticks"></a>`camera_ticks`: `int` -- The camera's clock the latch latched.
- <a id="ClockMapping.host_ns"></a>`host_ns`: `int` -- The scope's steady clock (`host_clock.steady_ns`) at the middle of the bracket.
- <a id="ClockMapping.tick_hz"></a>`tick_hz`: `int` -- Ticks per second of the camera's clock.
- <a id="ClockMapping.uncertainty_ns"></a>`uncertainty_ns`: `int` -- Half the bracket.

<a id="ControlSession"></a>
## `ControlSession`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.acting_in_control`](#ScopeSession.acting_in_control), [`ScopeSession.control`](#ScopeSession.control), [`ScopeSession.end_control`](#ScopeSession.end_control), [`ScopeSession.take_back_control`](#ScopeSession.take_back_control).  
Held in: [`Status.control`](#Status.control).

The one client controlling the scope: what it was taken as, and by whom.

Carries no id: the id is the only proof of holding, and only the taker's
`ControllerTaking` holds it, so no reader of this record -- a wire
client, a plugin, the GUI -- can act as the holder.

**Fields**

- <a id="ControlSession.name"></a>`name`: `str` -- The name the client took it with, for the person at the scope.
- <a id="ControlSession.taken_at_epoch_s"></a>`taken_at_epoch_s`: `float` -- When it was taken, as `time.time()`.
- <a id="ControlSession.taken_by"></a>`taken_by`: `Caller` -- The host the take came through (`modules.caller`).

<a id="ControlsLock"></a>
## `ControlsLock`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.controls_lock`](#ScopeSession.controls_lock).  
Held in: [`Status.lock`](#Status.lock).

Which of a person's controls the Session locks now, and why.

The Session's editing policy for the controls a person works, not an
API refusal: a REST client may still make any call the API accepts,
and what the API refuses it refuses whatever this says.

**Fields**

- <a id="ControlsLock.control_named"></a>`control_named`: `str | None` -- The control session held, as a sentence a person reads; None while none is. It holds the run surface too, the holder's own run included.
- <a id="ControlsLock.recording"></a>`recording`: `bool` -- Whether a manual recording is live. Its drain is not: a draining recording frees the controls while its claim still refuses new runs.
- <a id="ControlsLock.run_named"></a>`run_named`: `str | None` -- What holds the run surface -- a run, a diagnostic, a home, or a finished run's files still writing -- as a sentence a person reads; None while nothing does. A display that pauses for that work reads this, not `run_lockout`, so it keeps showing the scope under an idle control session.

<a id="ControlsLock.controls_locked"></a>
### `ControlsLock.controls_locked`

```python
controls_locked: bool  # property
```

True while the whole control surface locks: the run surface is held, or a recording is live.

**Raises**

Nothing.

<a id="ControlsLock.named"></a>
### `ControlsLock.named`

```python
named: str | None  # property
```

The sentence for what locks the controls now, None while nothing does.

The control session's while one is held, the holder's own run
included; else what holds the run surface; else a live recording.

**Raises**

Nothing.

<a id="ControlsLock.run_lockout"></a>
### `ControlsLock.run_lockout`

```python
run_lockout: bool  # property
```

True while something holds the run surface: `run_named` or `control_named`.

A finished run's files hold it until the last one is written,
though the run has freed its claim: the controls stay locked until
the queue empties.

**Raises**

Nothing.

<a id="Credit"></a>
## `Credit`

A record: it crosses the wire as its data.  
Held in: [`Change.credits`](#Change.credits).

One frame counted toward a change, and who counted it.

**Fields**

- <a id="Credit.by"></a>`by`: `Counter` -- (no description)
- <a id="Credit.seq"></a>`seq`: `int` -- (no description)

<a id="DegradedOutput"></a>
## `DegradedOutput`

A record: it crosses the wire as its data.  
Held in: [`BuildResult.degraded_outputs`](#BuildResult.degraded_outputs).

A file a build made by its fallback algorithm, and why.

**Fields**

- <a id="DegradedOutput.algorithm"></a>`algorithm`: `StitchAlgorithm` -- The algorithm that made it.
- <a id="DegradedOutput.fallback_from"></a>`fallback_from`: `StitchAlgorithm` -- The algorithm it fell back from.
- <a id="DegradedOutput.fallback_reason"></a>`fallback_reason`: `str` -- Why the first could not make it.
- <a id="DegradedOutput.path"></a>`path`: `pathlib.Path` -- The file.

<a id="DeliveredRate"></a>
## `DeliveredRate`

A record: it crosses the wire as its data.  
Returned by: [`ImagingAPI.get_delivered_rate`](#ImagingAPI.get_delivered_rate).

What the camera delivered to the host over the last second.

`frame_rate_fps` counts the frames stored, whatever any display does with
them. `bytes_per_s` is the bytes those same frames took on the link --
the payload the camera sent, not the transport's protocol overhead -- so
`bytes_per_s / frame_rate_fps` is the bytes one frame takes on the link.
Both are 0 when nothing was delivered in the last window: no camera, a
camera not streaming, or no recent window measured.

**Fields**

- <a id="DeliveredRate.bytes_per_s"></a>`bytes_per_s`: `float` -- (no description)
- <a id="DeliveredRate.frame_rate_fps"></a>`frame_rate_fps`: `float` -- (no description)

<a id="DisplayRates"></a>
## `DisplayRates`

Taken by: [`ScopeSession.enable_metrics`](#ScopeSession.enable_metrics).

A live display's own rates, which only a host with a display has.

The GUI hands its display down through `ScopeSession.enable_metrics`;
a module never reaches up to it.

<a id="DisplayRates.display_fps"></a>
### `DisplayRates.display_fps`

```python
def display_fps(self) -> float
```

Marks: in-process.

The frames drawn in the last second.

**Raises**

Nothing.

<a id="DisplayRates.frame_interval_percentiles_ms"></a>
### `DisplayRates.frame_interval_percentiles_ms`

```python
def frame_interval_percentiles_ms(self) -> dict[str, float]
```

Marks: in-process.

`p50`, `p95`, `p99` and `max` intervals between drawn
frames, in ms, and `n` the samples; empty before the first.

**Raises**

Nothing.

<a id="EnhanceResult"></a>
## `EnhanceResult`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.enhance`](#PostProcessingAPI.enhance).

What a Quick Enhance made.

**Fields**

- <a id="EnhanceResult.created"></a>`created`: `tuple[EnhancedFile, ...]` -- Each derived file, with its source and recipe.
- <a id="EnhanceResult.message"></a>`message`: `str` -- The outcome in words.
- <a id="EnhanceResult.output_folder"></a>`output_folder`: `pathlib.Path` -- The folder the derived files are in.

<a id="EnhancedFile"></a>
## `EnhancedFile`

A record: it crosses the wire as its data.  
Held in: [`EnhanceResult.created`](#EnhanceResult.created).

One file Quick Enhance derived: its source, the derived image, and its recipe.

**Fields**

- <a id="EnhancedFile.output_path"></a>`output_path`: `pathlib.Path` -- (no description)
- <a id="EnhancedFile.recipe_path"></a>`recipe_path`: `pathlib.Path` -- (no description)
- <a id="EnhancedFile.source_path"></a>`source_path`: `pathlib.Path` -- (no description)

<a id="ExcludedInput"></a>
## `ExcludedInput`

A record: it crosses the wire as its data.  
Held in: [`BuildResult.excluded_inputs`](#BuildResult.excluded_inputs), [`PostProcessingFailedError.excluded_inputs`](#PostProcessingFailedError.excluded_inputs).

Files of one kind a post-processing build found in the folder and did not take.

A build's own earlier outputs are not among them: those it has already
made.

**Fields**

- <a id="ExcludedInput.count"></a>`count`: `int` -- How many.
- <a id="ExcludedInput.kind"></a>`kind`: `PostProcessingOperation` -- The build that made the files: `COMPOSITE`, `STITCH`, `Z_PROJECTION`, `VIDEO` or `HYPERSTACK`. A recording's frames, which only a video or a hyperstack takes, are never among them.

<a id="ExposureChanged"></a>
## `ExposureChanged`

A record: it crosses the wire as its data.  
Delivered by: [`CameraAPI.add_exposure_listener`](#CameraAPI.add_exposure_listener).

An exposure event: the exposure now in effect, in ms.

**Fields**

- <a id="ExposureChanged.exposure_ms"></a>`exposure_ms`: `float` -- (no description)

<a id="FailedCapture"></a>
## `FailedCapture`

A record: it crosses the wire as its data.  
Held in: [`CaptureTally.failed`](#CaptureTally.failed).

One capture a run was asked for that produced no image.

**Fields**

- <a id="FailedCapture.cause"></a>`cause`: `str` -- Why no image came, in the words the log records.
- <a id="FailedCapture.scan_index"></a>`scan_index`: `int` -- The scan it belonged to, counting from 0.
- <a id="FailedCapture.step_index"></a>`step_index`: `int` -- The step's row in the protocol, counting from 0.
- <a id="FailedCapture.step_name"></a>`step_name`: `str` -- The step's name, as the protocol has it.

<a id="FieldOfView"></a>
## `FieldOfView`

A record: it crosses the wire as its data.  
Returned by: [`RuntimeState.get_field_of_view`](#RuntimeState.get_field_of_view).

The area a frame covers on the sample, in micrometres.

**Fields**

- <a id="FieldOfView.height_um"></a>`height_um`: `float` -- (no description)
- <a id="FieldOfView.width_um"></a>`width_um`: `float` -- (no description)

<a id="FilesWritten"></a>
## `FilesWritten`

A record: it crosses the wire as its data.  
Delivered by: [`RunEvents.files_written`](#RunEvents.files_written).

A `files_written` event: the run's folder, and what became of its images.

**Fields**

- <a id="FilesWritten.files"></a>`files`: `FilesOutcome` -- (no description)
- <a id="FilesWritten.run_dir"></a>`run_dir`: `pathlib.Path | None` -- (no description)

<a id="FilterBounds"></a>
## `FilterBounds`

A record: it crosses the wire as its data.  
Held in: [`CellCountFilters.area_um2`](#CellCountFilters.area_um2), [`CellCountFilters.intensity_max_pct`](#CellCountFilters.intensity_max_pct), [`CellCountFilters.intensity_mean_pct`](#CellCountFilters.intensity_mean_pct), [`CellCountFilters.intensity_min_pct`](#CellCountFilters.intensity_min_pct), [`CellCountFilters.perimeter_um`](#CellCountFilters.perimeter_um), [`CellCountFilters.sphericity`](#CellCountFilters.sphericity).

One filter's bounds: a region outside them is not counted.

**Fields**

- <a id="FilterBounds.max"></a>`max`: `float | None` -- The highest value kept; None is no upper bound.
- <a id="FilterBounds.min"></a>`min`: `float | None` -- The lowest value kept; None is no lower bound.

<a id="FocusFailure"></a>
## `FocusFailure`

A record: it crosses the wire as its data.  
Held in: [`RunOutcome.focus_failures`](#RunOutcome.focus_failures).

One step whose autofocus did not focus; the run went on at its fallback Z.

**Fields**

- <a id="FocusFailure.cause"></a>`cause`: `str` -- Why it did not focus, in the autofocus failure's words.
- <a id="FocusFailure.scan_index"></a>`scan_index`: `int` -- The scan it belonged to, counting from 0.
- <a id="FocusFailure.step_index"></a>`step_index`: `int` -- The step's row in the protocol, counting from 0.
- <a id="FocusFailure.step_name"></a>`step_name`: `str` -- The step's name, as the protocol has it.

<a id="Frame"></a>
## `Frame`

A record: it crosses the wire as its data.  
Held in: [`FrameDelivered.frame`](#FrameDelivered.frame).  
Taken by: [`ImagingAPI.add_frame_listener`](#ImagingAPI.add_frame_listener), [`ImagingAPI.judge_frame`](#ImagingAPI.judge_frame), [`SimulationAPI.exposed_under`](#SimulationAPI.exposed_under).

One delivered frame and the facts known when it was stored.

The camera's frame store builds it once, as the frame arrives, and every
reader -- a grab, a capture, the preview, each frame listener --
holds that one record, so the pixels, the arrival number, the depth and
the chunks it carries are always the same frame's. Read-only throughout:
the pixels refuse a write and the chunks are a read-only mapping. Two
records are equal only when they are the same delivery (identity).

**Fields**

- <a id="Frame.arrived_ns"></a>`arrived_ns`: `int` -- The scope's steady clock (`host_clock.steady_ns`) at the hand-over, where the driver first had the frame: a steady time to set against when a change was made.
- <a id="Frame.chunks"></a>`chunks`: `Mapping[str, int | float] | None` -- The camera's per-frame metadata (`ExposureTime`, `Gain`, `FrameID`, `Timestamp` on a Basler), or None for a camera that stamps none (IDS, FX2).
- <a id="Frame.image"></a>`image`: `ndarray` -- The pixels, cropped to the camera's window, read-only.
- <a id="Frame.seq"></a>`seq`: `int` -- Hand-over ordinal: how many frames this camera's handler had been handed when the driver first had this one (`HandOver`). Only goes up, whatever the clocks do, and may skip a number for a frame handed over and never stored.
- <a id="Frame.significant_bits"></a>`significant_bits`: `int` -- Payload depth of these pixels (12 for Mono12 in a uint16 container), from the frame's own format.
- <a id="Frame.timestamp"></a>`timestamp`: `datetime` -- Host wall-clock time at arrival, for what a person reads (file names, logs). It can step backwards; order by `seq`.

<a id="FrameCaptured"></a>
## `FrameCaptured`

A record: it crosses the wire as its data.  
Delivered by: [`RunEvents.frame_captured`](#RunEvents.frame_captured).

A `frame_captured` event: the frame as its file gets it, how many summed, its bits.

**Fields**

- <a id="FrameCaptured.frame_significant_bits"></a>`frame_significant_bits`: `int` -- (no description)
- <a id="FrameCaptured.frames_summed"></a>`frames_summed`: `int` -- (no description)
- <a id="FrameCaptured.image"></a>`image`: `np.ndarray` -- (no description)

<a id="FrameDelivered"></a>
## `FrameDelivered`

A record: it crosses the wire as its data.  
Delivered by: [`ImagingAPI.add_frame_listener`](#ImagingAPI.add_frame_listener).

A frame the camera delivered: its `Frame` record -- the read-only
image, its arrival number and times, its depth, and its chunk data when
the camera sends any.

**Fields**

- <a id="FrameDelivered.frame"></a>`frame`: `Frame` -- (no description)

<a id="FrameRecord"></a>
## `FrameRecord`

A record: it crosses the wire as its data.  
Held in: [`CapturedStill.record`](#CapturedStill.record), [`SavedStill.frame`](#SavedStill.frame).

The instrument's own account of one captured frame.

A field the instrument did not report is None, never a stand-in: a
stand-in written into a file reads downstream as a measurement.

Built only by this module's two entries: `grab_record` for a still,
`delivered_frame_record` for a recorded frame.

**Fields**

- <a id="FrameRecord.auto_gain"></a>`auto_gain`: `AutoGainConvergence | None` -- Where the standing auto-gain arm landed when the capture locked it (`AutoGainConvergence`); None when no arm stood. The lock's exposure and gain are `exposure_ms` and `gain_db`.
- <a id="FrameRecord.binning_size"></a>`binning_size`: `int` -- Binning the frame was delivered at.
- <a id="FrameRecord.black_level"></a>`black_level`: `float | None` -- The camera's black level parameter the frame was taken at, in the camera's own units (`CameraAPI.get_black_level`). None for a recorded frame: no frame reports it.
- <a id="FrameRecord.camera_model"></a>`camera_model`: `str | None` -- The camera that took it.
- <a id="FrameRecord.camera_tick_hz"></a>`camera_tick_hz`: `int | None` -- Ticks per second of that clock.
- <a id="FrameRecord.camera_timestamp_ticks"></a>`camera_timestamp_ticks`: `int | None` -- The camera's own clock for the frame (for a summed capture, the last frame's).
- <a id="FrameRecord.captured_at"></a>`captured_at`: `FrameTime` -- When the frame was taken, with what that time is (`FrameTime`).
- <a id="FrameRecord.clock_mapping"></a>`clock_mapping`: `ClockMapping | None` -- The camera's clock mapped onto the host's, latched by the capture at the top of the cycle that grabbed the frame, before its drain (`ClockMapping`); None on a camera without a latch (`capabilities.camera_has_clock_latch`).
- <a id="FrameRecord.dark"></a>`dark`: `bool | None` -- True when the frame was kept with no pixel above the dark floor while illumination was commanded on, after the capture waited out its budget for a lit one; for a summed capture, when any frame was. None when the capture did not judge it (no LED on, or `accept_dark`).
- <a id="FrameRecord.exposure_ms"></a>`exposure_ms`: `float | None` -- Per-frame exposure. For a summed capture this is each frame's exposure, not the total; `frames_summed` says how many. None for a recorded frame that carries no exposure of its own.
- <a id="FrameRecord.exposure_source"></a>`exposure_source`: `SettingSource | None` -- Whether `exposure_ms` is the frame's own or the setting read beside the grab (`SettingSource`); None exactly when `exposure_ms` is.
- <a id="FrameRecord.frame_id"></a>`frame_id`: `int | None` -- The camera's frame counter for the frame.
- <a id="FrameRecord.frame_significant_bits"></a>`frame_significant_bits`: `int` -- The depth each summed frame was delivered at (the driver's per-frame stamp). With `frames_summed` it is what the sum's tag, its rendering and its saturation are worked out from (`image_utils.summed_significant_bits`, `summed_full_scale`, `convert_sum_to_8bit`).
- <a id="FrameRecord.frames_summed"></a>`frames_summed`: `int` -- Number of frames summed into the image.
- <a id="FrameRecord.gain_db"></a>`gain_db`: `float | None` -- Gain the frame was taken at; None as `exposure_ms` is.
- <a id="FrameRecord.gain_source"></a>`gain_source`: `SettingSource | None` -- Whether `gain_db` is the frame's own or the setting read beside the grab; None exactly when `gain_db` is.
- <a id="FrameRecord.held_channel"></a>`held_channel`: `str | None` -- The channel whose stored settings the camera held for the grab (`CameraAPI.get_held_layer`); None when it held no layer's, after a raw camera write or with the settings a run's step wrote. The frame's own exposure and gain are the fields above either way.
- <a id="FrameRecord.illumination_ma"></a>`illumination_ma`: `Mapping[int, float]` -- Drive current of each LED address lit for the frame, by address. An address that was off is absent.
- <a id="FrameRecord.illumination_timing"></a>`illumination_timing`: `LightTiming` -- When those currents were read (`LightTiming`).
- <a id="FrameRecord.judgments"></a>`judgments`: `tuple[Judgment, ...] | None` -- The frame judged against each source's last change (`FrameValidity.judgments`): for a summed capture, the first frame summed. Empty when no change has been recorded; None for a recorded frame, which is never judged.
- <a id="FrameRecord.saturated_fraction"></a>`saturated_fraction`: `float | None` -- The fraction of the frame's pixels at or near full scale (`ImagingAPI.saturated_fraction`), measured when the capture asked for the check (`all_ones_check`); for a summed capture, its most saturated frame. None when it did not ask.

<a id="FrameRule"></a>
## `FrameRule`

A record: it crosses the wire as its data.  
Held in: [`Change.rule`](#Change.rule), [`ScopeCapabilities.frame_rules`](#ScopeCapabilities.frame_rules).

The rule a source's changes are judged by, and where it came from.

**Raises**

- `ValueError`: a count that is not a whole number, 0 or more; a clock rule without its delay or provenance; a credit rule with a delay.

**Fields**

- <a id="FrameRule.count"></a>`count`: `int` -- The frames a change of the source needs counted after it, which under the credit rule come before the first that counts: the largest `delivered_frames.max` the camera's timing record holds for the source, over its directions and the microscopes it was measured on, a move taking the largest over every move; else the shipped count (`SKIP_FRAMES`).
- <a id="FrameRule.delay_us"></a>`delay_us`: `float | None` -- Under the clock rule, the source's measured delay: the larger of its directions' `apply_delay_us`, each the least start over its trials of the first frame showing the change. A frame starting later than the change plus this shows it. 0 for a move with none measured. None under the credit rule.
- <a id="FrameRule.provenance"></a>`provenance`: `Mapping[str, str] | None` -- Under the clock rule, the run that measured the delay (`TIMING_PROVENANCE`), or that none did (`at_the_stop`). Under the credit rule, the run that measured the count, naming the microscope it was measured on (`TIMING_MICROSCOPE`); None for a shipped count.
- <a id="FrameRule.rule"></a>`rule`: `Rule` -- `Rule.CLOCK` when the camera has a clock latch and its timing record holds a measured delay for the source, or the source is a move with no measured count; `Rule.CREDIT` otherwise.

<a id="FrameSize"></a>
## `FrameSize`

A record: it crosses the wire as its data.  
Returned by: [`CameraAPI.frame_size`](#CameraAPI.frame_size), [`CameraAPI.read_frame_size`](#CameraAPI.read_frame_size), [`CameraAPI.set_binning`](#CameraAPI.set_binning), [`CameraAPI.set_frame_size`](#CameraAPI.set_frame_size).  
Held in: [`CameraLimits.frame_max`](#CameraLimits.frame_max), [`CameraLimits.frame_min`](#CameraLimits.frame_min), [`ScopeModel.max_frame`](#ScopeModel.max_frame).

A frame's size, in pixels.

**Fields**

- <a id="FrameSize.height"></a>`height`: `int` -- (no description)
- <a id="FrameSize.width"></a>`width`: `int` -- (no description)

<a id="FrameTime"></a>
## `FrameTime`

A record: it crosses the wire as its data.  
Held in: [`FrameRecord.captured_at`](#FrameRecord.captured_at).

When a frame was taken, with what that time is.

One value, so a time is never stated without its source: two clocks
give a frame its time, and a reader comparing frames must know which.

**Fields**

- <a id="FrameTime.at"></a>`at`: `datetime` -- The frame's time, host local.
- <a id="FrameTime.source"></a>`source`: `TimestampSource` -- What `at` is (`TimestampSource`).

<a id="GainChanged"></a>
## `GainChanged`

A record: it crosses the wire as its data.  
Delivered by: [`CameraAPI.add_gain_listener`](#CameraAPI.add_gain_listener).

A gain event: the gain now in effect, in dB.

**Fields**

- <a id="GainChanged.gain_db"></a>`gain_db`: `float` -- (no description)

<a id="GraphOptions"></a>
## `GraphOptions`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.graph_options`](#PostProcessingAPI.graph_options).

What a graph of a results table can be drawn with.

**Fields**

- <a id="GraphOptions.trendlines"></a>`trendlines`: `tuple[Trendline, ...]` -- The trendline kinds the axes given take: every kind until an axis rules some out (a time X takes no Power or Logarithmic).
- <a id="GraphOptions.x_axes"></a>`x_axes`: `tuple[str, ...]` -- The columns that can be the X axis: numbers and times.
- <a id="GraphOptions.y_axes"></a>`y_axes`: `tuple[str, ...]` -- The columns that can be the Y axis: numbers.

<a id="GraphResult"></a>
## `GraphResult`

A record: it crosses the wire as its data.  
Returned by: [`PostProcessingAPI.graph`](#PostProcessingAPI.graph).

What a graph save wrote.

**Fields**

- <a id="GraphResult.message"></a>`message`: `str` -- The outcome in words.
- <a id="GraphResult.path"></a>`path`: `pathlib.Path` -- The PNG it wrote.

<a id="HeldClaim"></a>
## `HeldClaim`

Returned by: [`ScopeSession.diagnostic_claim`](#ScopeSession.diagnostic_claim).  
Taken by: [`ProtocolRunner.run_autofocus`](#ProtocolRunner.run_autofocus), [`ProtocolRunner.start_autofocus`](#ProtocolRunner.start_autofocus).

The claim as held by the one caller that took it.

A fresh object per taking, so it identifies that taking and no
other: the next run's HeldClaim is a different object even when its
holder description is equal. Only it releases the claim. Nothing
else hands it out -- `ActivityClaim.holder` describes the holder,
it does not return this.

<a id="HeldClaim.holds"></a>
### `HeldClaim.holds`

```python
holds: bool  # property
```

Whether this taking still holds the claim.

False once it is released, and stays False: a later taking of the
same claim is a different object.

**Raises**

Nothing.

<a id="ImageStatistics"></a>
## `ImageStatistics`

A record: it crosses the wire as its data.  
Held in: [`SavedStill.statistics`](#SavedStill.statistics).

What an image's pixel values are, in its own counts.

Over every sample of the array, each colour channel's included, at the
depth it holds: a 12-bit frame's counts run to 4095, a sum's to what it
reached. Saturation and darkness are the frame record's, measured
against the frame's full scale by the capture.

**Fields**

- <a id="ImageStatistics.maximum"></a>`maximum`: `int` -- The highest value.
- <a id="ImageStatistics.mean"></a>`mean`: `float` -- The mean value.
- <a id="ImageStatistics.median"></a>`median`: `int` -- The lowest value at or above half of the samples.
- <a id="ImageStatistics.minimum"></a>`minimum`: `int` -- The lowest value.
- <a id="ImageStatistics.percentile_1"></a>`percentile_1`: `int` -- The lowest value at or above 1% of the samples.
- <a id="ImageStatistics.percentile_99"></a>`percentile_99`: `int` -- The lowest value at or above 99% of the samples.
- <a id="ImageStatistics.standard_deviation"></a>`standard_deviation`: `float` -- The population standard deviation.

<a id="InjectedFault"></a>
## `InjectedFault`

A record: it crosses the wire as its data.  
Returned by: [`SimulationAPI.injected`](#SimulationAPI.injected).

One fault injected on a simulated scope, as it was asked for.

**Fields**

- <a id="InjectedFault.arguments"></a>`arguments`: `dict[str, str | int | float | bool | None]` -- The arguments it was called with, by name.
- <a id="InjectedFault.member"></a>`member`: `str` -- The `scope.simulation` member that injected it.

<a id="Judgment"></a>
## `Judgment`

A record: it crosses the wire as its data.  
Returned by: [`ImagingAPI.judge_frame`](#ImagingAPI.judge_frame).  
Held in: [`FrameRecord.judgments`](#FrameRecord.judgments).

Whether one frame counts for one change, and the facts that decided it.

**Fields**

- <a id="Judgment.after_in_force_ms"></a>`after_in_force_ms`: `float | None` -- Milliseconds from the change's in-force time to the frame's arrival; None when it has none.
- <a id="Judgment.after_write_ms"></a>`after_write_ms`: `float` -- Milliseconds from the change's recording to the frame's arrival.
- <a id="Judgment.change"></a>`change`: `Change` -- The change judged against, as it stood when judged; its `rule` is the rule it was judged by.
- <a id="Judgment.counts"></a>`counts`: `bool` -- Under the credit rule, the frame arrived after the change, the change's count was met by frames before it, and an axis the change waits on had settled before it arrived. Under the clock rule, the frame started after `qualifies_after_ticks`; never when either is unknown.
- <a id="Judgment.frame_seq"></a>`frame_seq`: `int` -- The frame's hand-over ordinal.
- <a id="Judgment.frame_start_ticks"></a>`frame_start_ticks`: `int | None` -- Under the clock rule, when the frame started exposing on the camera's clock (its `Timestamp` chunk); None under the credit rule, and for a frame without the chunk.
- <a id="Judgment.frames_after"></a>`frames_after`: `int` -- Frames handed over after the change was recorded, up to and including this one: an upper bound on the frames stored after it, since a frame handed over may be dropped before the store.
- <a id="Judgment.qualifies_after_ticks"></a>`qualifies_after_ticks`: `int | None` -- Under the clock rule, the camera-clock tick a frame must start after to count: the change, its source's delay and the clock mapping's uncertainty. None under the credit rule, and when it cannot be known (no mapping, or a move not yet in force).

<a id="Labware"></a>
## `Labware`

A record: it crosses the wire as its data.  
Returned by: [`Lumascope.labware`](#Lumascope.labware).

One multi-well plate of the installation's catalogue, read once at bring-up.

Plate positions are in mm. `offset_*` is the first well's centre and
`spacing_*` the pitch between well centres; `width_mm` and
`height_mm` the plate's outline. `well_diameter_mm` is None where
the catalogue states none. A plate with no wells (the Blank labware)
has `rows` and `columns` 0.

**Fields**

- <a id="Labware.columns"></a>`columns`: `int` -- (no description)
- <a id="Labware.height_mm"></a>`height_mm`: `float` -- (no description)
- <a id="Labware.name"></a>`name`: `str` -- (no description)
- <a id="Labware.offset_x_mm"></a>`offset_x_mm`: `float` -- (no description)
- <a id="Labware.offset_y_mm"></a>`offset_y_mm`: `float` -- (no description)
- <a id="Labware.rows"></a>`rows`: `int` -- (no description)
- <a id="Labware.spacing_x_mm"></a>`spacing_x_mm`: `float` -- (no description)
- <a id="Labware.spacing_y_mm"></a>`spacing_y_mm`: `float` -- (no description)
- <a id="Labware.standard"></a>`standard`: `str` -- (no description)
- <a id="Labware.well_diameter_mm"></a>`well_diameter_mm`: `float | None` -- (no description)
- <a id="Labware.width_mm"></a>`width_mm`: `float` -- (no description)

<a id="Labware.get_well_index"></a>
### `Labware.get_well_index`

```python
def get_well_index(self, x_mm: float, y_mm: float) -> tuple[int, int] | None
```

Marks: in-process; reads (changes nothing).

The (column, row) of the well centre nearest plate position (x_mm, y_mm).

None when that centre is outside the grid: the position is more
than half a pitch beyond the outer well centres, off the plate's
wells, or the plate has no wells at all (the Blank labware). No
shipped plate states a well size, so within the grid a position
names its nearest well whether or not it is inside it.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'`, `x_mm` or `y_mm` is not a finite number, which names no well.

**Refusals**

- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`

<a id="Labware.get_well_position"></a>
### `Labware.get_well_position`

```python
def get_well_position(self, i: int, j: int) -> tuple[float, float]
```

Marks: in-process; reads (changes nothing).

The centre of the well in column `i` and row `j`, in mm from the plate's top-left.

**Raises**

Nothing.

<a id="LedBoardInfo"></a>
## `LedBoardInfo`

A record: it crosses the wire as its data.  
Returned by: [`DiagnosticsAPI.get_led_info`](#DiagnosticsAPI.get_led_info).

The LED board's firmware, as it reported it at connect.

Each field is None when it is unknown: no board, a board that did not
answer, or one with no firmware of its own (an FX2 scope's LED
peripheral). Whether a board is connected is `led_connected`.

**Fields**

- <a id="LedBoardInfo.command_set"></a>`command_set`: `LedCommandSet | None` -- `LedCommandSet.V2` (INFO, SELFTEST, I2CSCAN, LEDREAD), `LedCommandSet.LEGACY` (INFO only: firmware older than v2), or None: no text command channel, as on an FX2 scope, or no board.
- <a id="LedBoardInfo.firmware_date"></a>`firmware_date`: `str | None` -- As `MotorBoardInfo.firmware_date`.
- <a id="LedBoardInfo.firmware_version"></a>`firmware_version`: `str | None` -- As `MotorBoardInfo.firmware_version`.

<a id="LedReading"></a>
## `LedReading`

A record: it crosses the wire as its data.  
Returned by: [`IlluminationAPI.get_led_states`](#IlluminationAPI.get_led_states).  
Held in: [`LedSnapshot.readings`](#LedSnapshot.readings).  
Delivered by: [`IlluminationAPI.add_led_listener`](#IlluminationAPI.add_led_listener).

One LED address: the layer it lights, what it is doing, and its current.

The answer of every LED read and the `led` event.

**Fields**

- <a id="LedReading.address"></a>`address`: `int` -- The LED board address.
- <a id="LedReading.illumination_ma"></a>`illumination_ma`: `float | None` -- The commanded current in mA when lit; None otherwise.
- <a id="LedReading.name"></a>`name`: `str | None` -- The layer the unit's identity names for the address, read at the moment of the reading; None when the identity names none, which never hides the address.
- <a id="LedReading.state"></a>`state`: `LedState` -- Lit, dark, or unknown.

<a id="LedSnapshot"></a>
## `LedSnapshot`

A record: it crosses the wire as its data.  
Returned by: [`IlluminationAPI.save_led_state`](#IlluminationAPI.save_led_state).  
Taken by: [`IlluminationAPI.restore_led_state`](#IlluminationAPI.restore_led_state).

Every LED address as it read at one moment, for `restore_led_state`.

**Fields**

- <a id="LedSnapshot.readings"></a>`readings`: `tuple[LedReading, ...]` -- One reading per address of the board.
- <a id="LedSnapshot.tag"></a>`tag`: `str` -- A label for logs.

<a id="LiveFolderEntry"></a>
## `LiveFolderEntry`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.live_folder_listing`](#ScopeSession.live_folder_listing).

One file or folder in a live-folder listing.

**Fields**

- <a id="LiveFolderEntry.kind"></a>`kind`: `EntryKind` -- Whether it is a file or a folder.
- <a id="LiveFolderEntry.modified"></a>`modified`: `datetime` -- When it was last modified, in local time with its offset.
- <a id="LiveFolderEntry.name"></a>`name`: `str` -- Its full name under the live folder, `/`-separated on every host: what a path parameter, a download and a further listing take.
- <a id="LiveFolderEntry.size_bytes"></a>`size_bytes`: `int` -- Its size in bytes, as the file system reports it.

<a id="LiveWork"></a>
## `LiveWork`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.live_work`](#ScopeSession.live_work).  
Held in: [`Status.live_work`](#Status.live_work).

Everything a session is still doing, read once.

Each item is read from its owner as that owner answers, one after
another rather than under one lock, so two items can describe
slightly different moments; the close waits until a read finds none.

**Fields**

- <a id="LiveWork.closed"></a>`closed`: `bool` -- True once the session has shut down.
- <a id="LiveWork.closing"></a>`closing`: `bool` -- True while the session's close runs: nothing new takes the scope, and the close waits for this work.
- <a id="LiveWork.work"></a>`work`: `tuple[WorkItem, ...]` -- The work still under way, in the order the session asked: what holds the scope first, then what finishes after.

<a id="MotorBoardInfo"></a>
## `MotorBoardInfo`

A record: it crosses the wire as its data.  
Returned by: [`DiagnosticsAPI.get_motor_info`](#DiagnosticsAPI.get_motor_info).

The motor board's identity, as it reported it at connect.

Each field is None when it is unknown: no board, a board that did not
answer, or a reply that could not be read.

**Fields**

- <a id="MotorBoardInfo.command_set"></a>`command_set`: `MotorCommandSet | None` -- `MotorCommandSet.TEXT` for a board with a text command channel; None for one without (the TMCM-6110, which speaks binary datagrams) and for no board.
- <a id="MotorBoardInfo.firmware_date"></a>`firmware_date`: `str | None` -- The date INFO carried.
- <a id="MotorBoardInfo.firmware_version"></a>`firmware_version`: `str | None` -- The parsed version, or `ORIGINAL_FIRMWARE` (`'original'`) for a board that answered INFO with no version string.
- <a id="MotorBoardInfo.model"></a>`model`: `str | None` -- The scope model the board reports (`'LS850T'`), the board's own report and not `channel_identity.model`, which falls back to the operator's selection.
- <a id="MotorBoardInfo.serial_number"></a>`serial_number`: `str | None` -- The board's serial number.

<a id="NamespaceHealth"></a>
## `NamespaceHealth`

A record: it crosses the wire as its data.  
Held in: [`PluginHealth.namespaces`](#PluginHealth.namespaces).

Per-namespace snapshot for tech-support + diagnostic probes.

**Fields**

- <a id="NamespaceHealth.last_runtime_errors"></a>`last_runtime_errors`: `tuple[PluginRuntimeError, ...]` -- (no description)
- <a id="NamespaceHealth.loaded"></a>`loaded`: `tuple[PluginStatus, ...]` -- (no description)
- <a id="NamespaceHealth.namespace"></a>`namespace`: `str` -- (no description)

<a id="Notification"></a>
## `Notification`

A record: it crosses the wire as its data.  
Taken by: [`ScopeSession.add_outcome_listener`](#ScopeSession.add_outcome_listener), [`ScopeSession.create`](#ScopeSession.create), [`ScopeSession.remove_outcome_listener`](#ScopeSession.remove_outcome_listener).  
Delivered by: [`ScopeSession.add_outcome_listener`](#ScopeSession.add_outcome_listener).

Immutable notification payload delivered to listeners.

Every listener receives every post, shown or not: `shown` is the
centre's decision whether this one is for display now (shutdown, an
unattended run's mute, an attended run that already showed the same post,
and the dedup window say no), so a listener that
displays opens only the shown ones and a listener that records keeps
them all.

**Fields**

- <a id="Notification.category"></a>`category`: `str` -- (no description)
- <a id="Notification.fatal"></a>`fatal`: `bool` -- (no description)
- <a id="Notification.kind"></a>`kind`: `OutcomeKind` -- (no description)
- <a id="Notification.message"></a>`message`: `str` -- (no description)
- <a id="Notification.monotonic_s"></a>`monotonic_s`: `float` -- (no description)
- <a id="Notification.operation_key"></a>`operation_key`: `str` -- (no description)
- <a id="Notification.outcome_id"></a>`outcome_id`: `int` -- (no description)
- <a id="Notification.reason"></a>`reason`: `str` -- (no description)
- <a id="Notification.remedy"></a>`remedy`: `Remedy | None` -- (no description)
- <a id="Notification.severity"></a>`severity`: `Severity` -- (no description)
- <a id="Notification.shown"></a>`shown`: `bool` -- (no description)
- <a id="Notification.solicited"></a>`solicited`: `bool` -- (no description)
- <a id="Notification.source"></a>`source`: `str` -- (no description)
- <a id="Notification.title"></a>`title`: `str` -- (no description)
- <a id="Notification.wall_time_epoch_s"></a>`wall_time_epoch_s`: `float` -- (no description)

<a id="Objective"></a>
## `Objective`

A record: it crosses the wire as its data.  
Returned by: [`Lumascope.objectives`](#Lumascope.objectives), [`RuntimeState.resolve_current_objective`](#RuntimeState.resolve_current_objective).  
Taken by: [`Protocol.validate_steps`](#Protocol.validate_steps).

One objective of the installation's catalogue, read once at bring-up.

`id` is the catalogue key, the objective's one identity everywhere in
the program; `short_name` the token step and file names carry, derived
from the id alone (`objective_short_name`). Autofocus starts at
`autofocus_step_max_um` and refines down to `autofocus_step_min_um`
across `autofocus_half_range_um` either side of where it starts. A
jog moves the `jog_*` distance on its axis. `focus_peak_fwhm_dense_um`
is measured by the engineering plugin's characterization, None until an
objective has been.

**Fields**

- <a id="Objective.autofocus_half_range_um"></a>`autofocus_half_range_um`: `float` -- (no description)
- <a id="Objective.autofocus_step_max_um"></a>`autofocus_step_max_um`: `float` -- (no description)
- <a id="Objective.autofocus_step_min_um"></a>`autofocus_step_min_um`: `float` -- (no description)
- <a id="Objective.depth_of_field_um"></a>`depth_of_field_um`: `float` -- (no description)
- <a id="Objective.description"></a>`description`: `str` -- (no description)
- <a id="Objective.focal_length_mm"></a>`focal_length_mm`: `float` -- (no description)
- <a id="Objective.focus_peak_fwhm_dense_um"></a>`focus_peak_fwhm_dense_um`: `float | None` -- (no description)
- <a id="Objective.id"></a>`id`: `str` -- (no description)
- <a id="Objective.jog_xy_coarse_um"></a>`jog_xy_coarse_um`: `float` -- (no description)
- <a id="Objective.jog_xy_fine_um"></a>`jog_xy_fine_um`: `float` -- (no description)
- <a id="Objective.jog_z_coarse_um"></a>`jog_z_coarse_um`: `float` -- (no description)
- <a id="Objective.jog_z_fine_um"></a>`jog_z_fine_um`: `float` -- (no description)
- <a id="Objective.magnification"></a>`magnification`: `float` -- (no description)
- <a id="Objective.numerical_aperture"></a>`numerical_aperture`: `float` -- (no description)
- <a id="Objective.short_name"></a>`short_name`: `str` -- (no description)
- <a id="Objective.working_distance_mm"></a>`working_distance_mm`: `float` -- (no description)
- <a id="Objective.z_settle_timeout_ms"></a>`z_settle_timeout_ms`: `float` -- (no description)
- <a id="Objective.z_settle_tolerance_um"></a>`z_settle_tolerance_um`: `float` -- (no description)

<a id="ObjectiveQuestion"></a>
## `ObjectiveQuestion`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.current_slot_objective_question`](#ScopeSession.current_slot_objective_question), [`ScopeSession.objective_question`](#ScopeSession.objective_question).

Which objective is in the light path; what to ask the user.

Returned by `ScopeSession.objective_question` when no one has ever
confirmed the objective on this install, or the turret sits on a slot
with no assignment; and by `current_slot_objective_question` whenever
a person asks to say what the current turret slot holds, or on a scope
without a turret which objective is installed. The pixel size
derived from the objective is stamped into the scale bar and every
saved image's metadata, and a wrong scale cannot be told from a
measured one afterwards -- so the session exposes the question instead
of assuming silently, and a host renders it however it likes. The answer goes back through
`ScopeSession.confirm_objective`.

**Fields**

- <a id="ObjectiveQuestion.choices"></a>`choices`: `tuple[str, ...]` -- The catalogue, in its shipped order.
- <a id="ObjectiveQuestion.proposed"></a>`proposed`: `str` -- The catalogue id to offer as the default.
- <a id="ObjectiveQuestion.turret_slot"></a>`turret_slot`: `int | None` -- The slot the answer binds to, or None on a non-turret model.

<a id="Operation"></a>
## `Operation`

Returned by: [`ManualCaptureController.capture`](#ManualCaptureController.capture), [`ManualRecordingController.start`](#ManualRecordingController.start), [`MotionAPI.start_home`](#MotionAPI.start_home), [`MotionAPI.start_move_to_plate`](#MotionAPI.start_move_to_plate), [`MotionAPI.start_move_to_well`](#MotionAPI.start_move_to_well), [`MotionAPI.start_move_xy`](#MotionAPI.start_move_xy), [`MotionAPI.start_move_xy_by`](#MotionAPI.start_move_xy_by), [`MotionAPI.start_move_z`](#MotionAPI.start_move_z), [`MotionAPI.start_move_z_by`](#MotionAPI.start_move_z_by), [`ProtocolRunner.start_autofocus`](#ProtocolRunner.start_autofocus), [`ProtocolRunner.start_autofocus_all_steps`](#ProtocolRunner.start_autofocus_all_steps), [`ProtocolRunner.start_composite`](#ProtocolRunner.start_composite), [`ProtocolRunner.start_protocol`](#ProtocolRunner.start_protocol), [`ProtocolRunner.start_single_scan`](#ProtocolRunner.start_single_scan), [`ProtocolRunner.start_zstack`](#ProtocolRunner.start_zstack), [`ScopeSession.last_run`](#ScopeSession.last_run), [`ScopeSession.start_application_session`](#ScopeSession.start_application_session), [`ScopeSession.start_go_to_bookmark`](#ScopeSession.start_go_to_bookmark), [`ScopeSession.start_go_to_step`](#ScopeSession.start_go_to_step).  
Taken by: [`ScopeSession.held_by_other`](#ScopeSession.held_by_other).

One operation in flight: its outcome, its done-callbacks, its stop and its progress.

**Fields**

- <a id="Operation.files"></a>`files`: `Operation | None` -- A run's files, as their own operation: done once the run's images are on disk or given up on, its hyperstack build (when it has one) has ended and its `files_written` has been delivered, with the run's `RunFiles`. None on every other operation.

<a id="Operation.progress"></a>
### `Operation.progress`

```python
def progress(self) -> RunProgress | None
```

Marks: reads (changes nothing).

Where this operation has got, read now; None for one that keeps no progress record.

**Raises**

Nothing.

<a id="Operation.stop"></a>
### `Operation.stop`

```python
def stop(self) -> None
```

Stop this operation's work; only asks, and the operation ends on its own.

Does nothing once the operation is done, or once its work has ended
and it is still telling its listeners, or when it has no stop (a
still): a stop that arrives late never touches another operation.

**Raises**

- `HardwareCommandRefusedError`: a move's or a home's stop, which is `stop_motion`'s: `'not_connected'` or `'axis_absent'`, no motor controller; `'home_not_stoppable'`, a home drives on a board that cannot stop one. A run's and a recording's stop raise nothing.
- `MotorStopFailedError`: a move's or a home's STOP was not taken by the board, so the stage may still be moving.

**Refusals**

- [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `home_not_stoppable`, `axis_absent`, `not_connected`
- [`MotorStopFailedError`](#MotorStopFailedError): `motor_stop_failed`

<a id="Outcome"></a>
## `Outcome`

A record: it crosses the wire as its data.  
Returned by: [`modules.notification_center.outcome_of`](#modules.notification_center.outcome_of).

What an exception's type says it is, read once for every host that answers it.

**Fields**

- <a id="Outcome.cause"></a>`cause`: `RefusalCause | None` -- Whether the same request could succeed later, as a refusal's, a quiet outcome's or a raised notice's type declares (`RefusalCause`); None for a fault, a notice that is only reported, and a by-contract cancel, which declares none.
- <a id="Outcome.fatal"></a>`fatal`: `bool` -- A fault that ends what was running.
- <a id="Outcome.for_person"></a>`for_person`: `bool` -- Whether `words` are written for the person: a refusal's, a notice's, a quiet outcome's, and a fault's whose type writes them so. Any other fault's are a developer's.
- <a id="Outcome.kind"></a>`kind`: `OutcomeKind` -- A refusal, a notice, a quiet outcome or a fault.
- <a id="Outcome.reason"></a>`reason`: `str` -- Its machine-readable code; empty when its type declares none.
- <a id="Outcome.remedy"></a>`remedy`: `Remedy | None` -- The one action that answers it, when it has one.
- <a id="Outcome.title"></a>`title`: `str | None` -- The heading its type declares; None for a fault whose type declares none, which each host names in its own way.
- <a id="Outcome.words"></a>`words`: `str` -- Its own words, or a sentence saying they could not be written.

<a id="PartStatus"></a>
## `PartStatus`

A record: it crosses the wire as its data.  
Returned by: [`BringUpRecord.missing`](#BringUpRecord.missing).  
Held in: [`BringUpRecord.parts`](#BringUpRecord.parts), [`Status.parts`](#Status.parts).

One part of the scope at bring-up.

**Fields**

- <a id="PartStatus.cause"></a>`cause`: `str | None` -- Why it is not up, as a code from `CAUSE_PHRASES`; or, on a part that is up, a problem it met coming up (the LED board's safety LEDs-off not confirmed; the camera's stream that did not start, which a later start that succeeds does not clear). None when nothing is wrong.
- <a id="PartStatus.detail"></a>`detail`: `str` -- The driver's or the error's own words, for the log and the record; empty when there are none.
- <a id="PartStatus.expected"></a>`expected`: `bool` -- Whether this scope's model has one. A manual scope has no motor board, so its absence is not a failure; until the model is known every part is expected.
- <a id="PartStatus.part"></a>`part`: `Part` -- Which part.
- <a id="PartStatus.up"></a>`up`: `bool` -- Whether a real driver came up for it.

<a id="PartStatus.describe"></a>
### `PartStatus.describe`

```python
def describe(self) -> str
```

Marks: in-process; reads (changes nothing).

The part's label with its cause: `Motor Controller (port in use)`.

**Raises**

Nothing.

<a id="PartStatus.missing"></a>
### `PartStatus.missing`

```python
missing: bool  # property
```

Expected and not up: the part this scope should have and lacks.

**Raises**

Nothing.

<a id="PlateLocation"></a>
## `PlateLocation`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.bookmark`](#ScopeSession.bookmark), [`ScopeSession.get_current_plate_position`](#ScopeSession.get_current_plate_position), [`ScopeSession.save_all_bookmarks`](#ScopeSession.save_all_bookmarks), [`ScopeSession.save_bookmark`](#ScopeSession.save_bookmark).  
Taken by: [`MotionAPI.move_to_plate`](#MotionAPI.move_to_plate), [`MotionAPI.start_move_to_plate`](#MotionAPI.start_move_to_plate), [`ProtocolRunner.run_single_scan`](#ProtocolRunner.run_single_scan), [`ProtocolRunner.start_single_scan`](#ProtocolRunner.start_single_scan), [`ScopeSession.add_step`](#ScopeSession.add_step).

Where the scope is on a plate, focus included: X and Y in mm on the plate, Z in um.

None on an axis is no position there: a move to it leaves that axis where it is.

**Fields**

- <a id="PlateLocation.x_mm"></a>`x_mm`: `float | None` -- (no description)
- <a id="PlateLocation.y_mm"></a>`y_mm`: `float | None` -- (no description)
- <a id="PlateLocation.z_um"></a>`z_um`: `float | None` -- (no description)

<a id="PlatePosition"></a>
## `PlatePosition`

A record: it crosses the wire as its data.  
Returned by: [`RuntimeState.plate_transform`](#RuntimeState.plate_transform), [`RuntimeState.stage_to_plate`](#RuntimeState.stage_to_plate).

A position on a plate, in mm from the plate's top-left corner.

**Fields**

- <a id="PlatePosition.x_mm"></a>`x_mm`: `float` -- (no description)
- <a id="PlatePosition.y_mm"></a>`y_mm`: `float` -- (no description)

<a id="PluginHealth"></a>
## `PluginHealth`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.plugin_health`](#ScopeSession.plugin_health).

Every namespace's health and the plugins that did not load, for tech-support reports.

**Fields**

- <a id="PluginHealth.namespaces"></a>`namespaces`: `tuple[NamespaceHealth, ...]` -- (no description)
- <a id="PluginHealth.not_loaded"></a>`not_loaded`: `tuple[PluginNotLoaded, ...]` -- (no description)

<a id="PluginNotLoaded"></a>
## `PluginNotLoaded`

A record: it crosses the wire as its data.  
Held in: [`PluginHealth.not_loaded`](#PluginHealth.not_loaded).

A plugin that did not load, and why.

It has no namespace: a plugin is filed under one only by registering,
which a plugin that never loaded did not do.

**Fields**

- <a id="PluginNotLoaded.name"></a>`name`: `str` -- (no description)
- <a id="PluginNotLoaded.reason"></a>`reason`: `str` -- (no description)
- <a id="PluginNotLoaded.version"></a>`version`: `str` -- (no description)

<a id="PluginRuntimeError"></a>
## `PluginRuntimeError`

A record: it crosses the wire as its data.  
Held in: [`NamespaceHealth.last_runtime_errors`](#NamespaceHealth.last_runtime_errors).

A runtime error caught from a plugin handler.

Distinct from a load failure: the plugin loaded fine, but failed
while servicing a callback (e.g. auto_run_on_protocol_complete). Reported by
PluginRegistry.record_runtime_error and kept in
NamespaceHealth.last_runtime_errors so diagnostic probes can
attribute fault to the right plugin.

**Fields**

- <a id="PluginRuntimeError.exc_type"></a>`exc_type`: `str` -- (no description)
- <a id="PluginRuntimeError.hook"></a>`hook`: `str` -- (no description)
- <a id="PluginRuntimeError.message"></a>`message`: `str` -- (no description)
- <a id="PluginRuntimeError.namespace"></a>`namespace`: `str` -- (no description)
- <a id="PluginRuntimeError.plugin_name"></a>`plugin_name`: `str` -- (no description)

<a id="PluginStatus"></a>
## `PluginStatus`

A record: it crosses the wire as its data.  
Held in: [`NamespaceHealth.loaded`](#NamespaceHealth.loaded).

A loaded plugin, for health reports.

**Fields**

- <a id="PluginStatus.name"></a>`name`: `str` -- (no description)
- <a id="PluginStatus.namespace"></a>`namespace`: `str` -- (no description)
- <a id="PluginStatus.version"></a>`version`: `str` -- (no description)

<a id="PositionChanged"></a>
## `PositionChanged`

A record: it crosses the wire as its data.  
Delivered by: [`MotionAPI.add_position_listener`](#MotionAPI.add_position_listener).

A position event: a stage axis, and its state and position as `axis_positions` answers.

**Fields**

- <a id="PositionChanged.at"></a>`at`: `AxisPosition` -- (no description)
- <a id="PositionChanged.axis"></a>`axis`: `StageAxis` -- (no description)

<a id="Protocol"></a>
## `Protocol`

A wire client holds one as a handle: `/api/v1/handles/Protocol/{handle_id}/`.  
Returned by: [`ScopeSession.create_empty_protocol`](#ScopeSession.create_empty_protocol), [`ScopeSession.load_protocol`](#ScopeSession.load_protocol), [`ScopeSession.new_protocol`](#ScopeSession.new_protocol).  
Held in: [`RunEnded.protocol`](#RunEnded.protocol), [`RunEvents.run_ended`](#RunEvents.run_ended).  
Taken by: [`ProtocolRunner.run_autofocus_all_steps`](#ProtocolRunner.run_autofocus_all_steps), [`ProtocolRunner.run_protocol`](#ProtocolRunner.run_protocol), [`ProtocolRunner.run_single_scan`](#ProtocolRunner.run_single_scan), [`ProtocolRunner.start_autofocus_all_steps`](#ProtocolRunner.start_autofocus_all_steps), [`ProtocolRunner.start_protocol`](#ProtocolRunner.start_protocol), [`ProtocolRunner.start_single_scan`](#ProtocolRunner.start_single_scan), [`ScopeSession.add_step`](#ScopeSession.add_step), [`ScopeSession.apply_channel_settings`](#ScopeSession.apply_channel_settings), [`ScopeSession.apply_focus_to_channel_steps`](#ScopeSession.apply_focus_to_channel_steps), [`ScopeSession.apply_tiling`](#ScopeSession.apply_tiling), [`ScopeSession.apply_zstacking`](#ScopeSession.apply_zstacking), [`ScopeSession.delete_step`](#ScopeSession.delete_step), [`ScopeSession.go_to_step`](#ScopeSession.go_to_step), [`ScopeSession.point_steps_at_turret_slot`](#ScopeSession.point_steps_at_turret_slot), [`ScopeSession.protocol_size_advisory`](#ScopeSession.protocol_size_advisory), [`ScopeSession.rename_step`](#ScopeSession.rename_step), [`ScopeSession.save_focus`](#ScopeSession.save_focus), [`ScopeSession.save_protocol`](#ScopeSession.save_protocol), [`ScopeSession.set_protocol_capture_root`](#ScopeSession.set_protocol_capture_root), [`ScopeSession.set_protocol_labware`](#ScopeSession.set_protocol_labware), [`ScopeSession.set_protocol_schedule`](#ScopeSession.set_protocol_schedule), [`ScopeSession.set_step_autofocus`](#ScopeSession.set_step_autofocus), [`ScopeSession.start_go_to_step`](#ScopeSession.start_go_to_step), [`ScopeSession.turret_slot_questions`](#ScopeSession.turret_slot_questions), [`ScopeSession.update_step`](#ScopeSession.update_step).

**Fields**

- <a id="Protocol.CURRENT_VERSION"></a>`CURRENT_VERSION`: `ClassVar[int]` -- (no description)
- <a id="Protocol.PROTOCOL_FILE_HEADER"></a>`PROTOCOL_FILE_HEADER`: `ClassVar[str]` -- (no description)

<a id="Protocol.capture_prefix"></a>
### `Protocol.capture_prefix`

```python
def capture_prefix(self) -> str
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/capture_prefix`.

The capture root as the prefix of a saved file's name.

The one derivation, read by the image writer and by post-processing:
two derivations named the images `exp202610_A1_BF` and looked for
`exp/2026:10_A1_BF`. Empty when there is no root.

**Raises**

Nothing.

<a id="Protocol.capture_root"></a>
### `Protocol.capture_root`

```python
def capture_root(self) -> str
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/capture_root`.

The capture root as it was typed or loaded.

**Raises**

Nothing.

<a id="Protocol.channel_settings"></a>
### `Protocol.channel_settings`

```python
def channel_settings(self) -> dict[str, ChannelSettings]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/channel_settings`.

Each channel's Layer Settings row as its record, keyed by channel.

A protocol loaded from a file with a 'Layer Settings' block returns
that block. One without (a v5 file, or a protocol built in memory)
has it inferred from its steps: each channel with steps yields one
record seeded from its first step, `acquire` 'video' when any of
its steps records video, else 'image'. `stim_enabled` is not
inferred: a v5 step carries its stimulation per step.

Every cell is read as its column's type (`ChannelSettings`); a
blank cell but Acquire is None, and so is an inferred one its step
cannot supply (see _channel_settings_inferred_from_steps).

**Raises**

- `ProtocolFormatError`: A block cell is not of its column's type. A file is refused at load for it, so a loaded protocol does not.

**Refusals by REST status**

- 422: [`ProtocolFormatError`](#ProtocolFormatError): `protocol_invalid`

<a id="Protocol.duration_s"></a>
### `Protocol.duration_s`

```python
def duration_s(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/duration_s`.

How long the protocol runs its scans, in seconds; 0 or None for one scan.

**Raises**

Nothing.

<a id="Protocol.estimate_write_mb"></a>
### `Protocol.estimate_write_mb`

```python
def estimate_write_mb(
    self,
    *,
    video_as_frames: bool = False,
    global_max_fps: float,
) -> float
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/estimate_write_mb`.

Estimate the disk this whole protocol will write, in MB.

One owner for "how big is this protocol", so the pre-run free-space
guard and anything that wants to tell the user up front cannot drift
apart. Lives on Protocol because the estimate has to know the steps
frame schema, and Protocol is what guarantees that schema at any row
count.

Stateless by design -- no cached total. A cache here has to be
invalidated by every mutator, and the in-place ones (modify_step
flipping Acquire, delete_step dropping rows) do not go through the
one place that would clear it; a stale total silently under-reserves
disk for a run. Recomputing costs single-digit milliseconds on a
protocol large enough to care about.

Returns the estimate only, NOT floored at the run's minimum free-disk
figure. That floor is the disk guard's refusal policy, not a property
of the protocol -- folding it in would project gigabytes for a
three-step protocol.

Total by construction, like the per-step estimator it sums: it must
not raise on any protocol, because the run loop calls it inside a
broad except where a raise would silently disable the disk guard.

**Args**

- `video_as_frames`: Run-level flag -- video saved as individual frames rather than a compressed MP4.
- `global_max_fps`: The run's snapshot of the global video FPS cap (0 = uncapped), so a video step is never sized at a rate the recording will not run at.

**Returns**

Estimated megabytes the whole protocol will write.

**Raises**

Nothing.

<a id="Protocol.labware_id"></a>
### `Protocol.labware_id`

```python
def labware_id(self) -> str
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/labware_id`.

The catalogue key of the plate the protocol's steps are placed on.

**Raises**

Nothing.

<a id="Protocol.num_scans"></a>
### `Protocol.num_scans`

```python
def num_scans(self) -> int
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/num_scans`.

How many scans a full run of the protocol takes: its duration over its period, whole.

A period of 0 or None is one scan, and so is a duration shorter than
one period: a run captures its first scan before any period elapses.
Divided as timedeltas, which is exact where float seconds drop a scan
on some ratios.

**Raises**

Nothing.

<a id="Protocol.num_steps"></a>
### `Protocol.num_steps`

```python
def num_steps(self) -> int
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/num_steps`.

How many steps the protocol holds.

**Raises**

Nothing.

<a id="Protocol.period_s"></a>
### `Protocol.period_s`

```python
def period_s(self) -> float | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/period_s`.

How often the protocol scans, in seconds; 0 or None for one scan.

**Raises**

Nothing.

<a id="Protocol.size_advisory"></a>
### `Protocol.size_advisory`

```python
def size_advisory(
    self,
    *,
    video_as_frames: bool = False,
    global_max_fps: float,
) -> ProtocolSizeAdvisory | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/size_advisory`.

Describe this protocol if it is large enough to warn about, else None.

The threshold and the sentence live here rather than in the GUI: a
REST caller asking how big a protocol is should get the same answer,
and the GUI's job is to render what comes back.

Advisory only. Nothing here refuses anything -- a legitimate large
protocol runs untouched, and the run-start disk guard is unchanged.

**Raises**

Nothing.

<a id="Protocol.step_list_revision"></a>
### `Protocol.step_list_revision`

```python
step_list_revision: int  # property
```

Route: `GET /api/v1/handles/Protocol/{handle_id}/step_list_revision`.

A count that changes whenever the step list does, for a reader to tell it changed.

**Raises**

Nothing.

<a id="Protocol.step_records"></a>
### `Protocol.step_records`

```python
def step_records(self) -> tuple[StepRecord, ...]
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/step_records`.

Every step, in order, as its record: the step table in one read.

**Raises**

- `ProtocolFormatError`: a stimulation entry of a step is not a config.

**Refusals by REST status**

- 422: [`ProtocolFormatError`](#ProtocolFormatError): `protocol_invalid`

<a id="Protocol.tiling"></a>
### `Protocol.tiling`

```python
def tiling(self) -> str | None
```

Marks: reads (changes nothing).  
Route: `GET /api/v1/handles/Protocol/{handle_id}/tiling`.

The tile grid this protocol's steps carry.

The no-tiling label when no step carries a tile; the grid's label
when the steps' tiles form a grid this installation offers; None
when they are tiled in a layout it does not offer, which only a
hand-edited file produces, since every offered grid is square.

**Raises**

Nothing.

<a id="Protocol.validate_steps"></a>
### `Protocol.validate_steps`

```python
def validate_steps(
    self,
    objectives: Mapping[str, Objective],
    *,
    led_max_ma: int | None,
) -> list
```

Marks: in-process; reads (changes nothing).

Validate all step fields and return a list of error strings.

In-process: a wire client needs no call of its own. It is told of a
loaded protocol's invalid steps when the protocol loads, and a run
start refuses them.

**Args**

- `objectives`: The scope's objectives (`scope.objectives`); a step naming an objective they lack is an error.
- `led_max_ma`: The connected scope's LED current cap (`capabilities.led_max_ma`). Asked of the caller, not carried on the protocol: a copy made for a run would not carry it, and the gate would admit a step the LED refuses. None on a scope that came up without its LED board: there is no cap to judge against, and a run on it is refused for the missing LED controller.

Returns an empty list if all steps are valid.

**Raises**

Nothing.

<a id="ProtocolRunner"></a>
## `ProtocolRunner`

A wire client holds one as a handle: `/api/v1/handles/ProtocolRunner/{handle_id}/`.  
Returned by: [`ScopeSession.create_protocol_runner`](#ScopeSession.create_protocol_runner).

GUI-independent protocol runner wrapping the session's
SequencedCaptureRunner.

One engine per session: the runner wraps the session-composed
instance rather than constructing a second one, so the GUI, L2, and
REST all drive (and observe) the same run state behind the same
claim.

<a id="ProtocolRunner.run_autofocus"></a>
### `ProtocolRunner.run_autofocus`

```python
def run_autofocus(
    self,
    channel: str | None,
    *,
    save_characterization_data: bool = False,
    sequence_name: str = 'autofocus',
    parent_dir: FilePath | None = None,
    events: RunEvents | None = None,
    claim: HeldClaim | None = None,
    timeout_s: float | None = None,
) -> RunOutcome
```

Marks: in-process.

`start_autofocus`, waited for: returns the run's outcome once it has let go of the scope.

Takes `start_autofocus`'s arguments, and `timeout_s`: how long to wait
for the run to end; None waits for its end. A run that ends any way
returns its outcome, a stopped one included.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was started.
- `ProtocolRunRefusedError`: As `start_autofocus`; nothing was started.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_autofocus`. Nothing was started.
- `RunWaitExpiredError`: `'still_running'`, a `TimeoutError`: `timeout_s` ran out; its `operation` holds the run, still going.

**Refusals**

- [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`, `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`, `not_connected`
- [`RunWaitExpiredError`](#RunWaitExpiredError): `still_running`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`

<a id="ProtocolRunner.run_autofocus_all_steps"></a>
### `ProtocolRunner.run_autofocus_all_steps`

```python
def run_autofocus_all_steps(
    self,
    protocol: Protocol,
    *,
    events: RunEvents | None = None,
    timeout_s: float | None = None,
) -> RunOutcome
```

Marks: in-process.

`start_autofocus_all_steps`, waited for: returns the run's outcome once it has let go of the scope.

Takes `start_autofocus_all_steps`'s arguments, and `timeout_s`: how long to wait
for the run to end; None waits for its end. A run that ends any way
returns its outcome, a stopped one included.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was started.
- `ProtocolRunRefusedError`: As `start_autofocus_all_steps`; nothing was started.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_autofocus_all_steps`. Nothing was started.
- `RunWaitExpiredError`: `'still_running'`, a `TimeoutError`: `timeout_s` ran out; its `operation` holds the run, still going.

**Refusals**

- [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`, `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`, `not_connected`
- [`RunWaitExpiredError`](#RunWaitExpiredError): `still_running`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`

<a id="ProtocolRunner.run_composite"></a>
### `ProtocolRunner.run_composite`

```python
def run_composite(
    self,
    *,
    sequence_name: str = 'composite',
    parent_dir: FilePath | None = None,
    events: RunEvents | None = None,
    timeout_s: float | None = None,
) -> RunOutcome
```

Marks: in-process.

Capture one frame per acquiring channel and merge them.

A composite is a single-position run through the same engine as
every other run kind: one step per channel at the current stage
position, each at its own stored focus, followed by the merge that
combines them into one image.

The channel set, the capture format and the depth all come from the
settings snapshot rather than from any widget, so a headless caller
gets the same run a GUI click does.

The merged artifact is the run's real product, so this BLOCKS until
the merge settles and returns the run's outcome, the artifact's path
on it. A run that reported 'completed' while the merged file was
missing would be indistinguishable from a successful one to every
headless caller, which is the boundary this run kind exists to fix.
A composite is merged from every channel asked for or not at all: a
channel that captured nothing leaves no composite, the run's status
'incomplete' with the failed channels in `captures`.

**Args**

- `sequence_name`: The name of the protocol file the run saves in its folder ('.tsv' added): a name, not a path.
- `parent_dir`: Parent directory for output. Defaults to 'Manual/Composites' under the live folder, as start_composite.
- `events`: The run's event handlers (`RunEvents`); None for none.
- `timeout_s`: How long to wait for the run and its merge; None waits for them to end.

**Returns**

The run's RunOutcome: `merged` True, `artifact_path` the merged composite, `status` 'completed' or 'incomplete', and `captures` what each channel produced.

**Raises**

- `ProtocolRunRefusedError`: Fewer than two channels are set to capture an image, so nothing could be merged; or the runner refused the run itself (already running, files still writing, hardware not connected).
- `CompositeFailedError`: The run finished but produced no composite. The error names the machine-readable reason, so a caller can tell an aborted run from a failed merge.
- `RunWaitExpiredError`: `'still_running'`, a `TimeoutError`: `timeout_s` ran out before the run or its merge ended; its `operation` holds the run, still going.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_composite`. Nothing was started.
- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was started.

**Refusals**

- [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`, `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`, `not_connected`
- [`RunWaitExpiredError`](#RunWaitExpiredError): `still_running`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`CompositeFailedError`](#CompositeFailedError): `run_not_completed`, `no_run_dir`, `no_composite_config`, `merge_failed`, `merge_error`

<a id="ProtocolRunner.run_protocol"></a>
### `ProtocolRunner.run_protocol`

```python
def run_protocol(
    self,
    protocol: Protocol,
    *,
    sequence_name: str = 'protocol',
    parent_dir: FilePath | None = None,
    enable_image_saving: bool = True,
    events: RunEvents | None = None,
    timeout_s: float | None = None,
) -> RunOutcome
```

Marks: in-process.

`start_protocol`, waited for: returns the run's outcome once it has let go of the scope.

Takes `start_protocol`'s arguments, and `timeout_s`: how long to wait
for the run to end; None waits for its end. A run that ends any way
returns its outcome, a stopped one included.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was started.
- `ProtocolRunRefusedError`: As `start_protocol`; nothing was started.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_protocol`. Nothing was started.
- `RunWaitExpiredError`: `'still_running'`, a `TimeoutError`: `timeout_s` ran out; its `operation` holds the run, still going.

**Refusals**

- [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`, `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`, `not_connected`
- [`RunWaitExpiredError`](#RunWaitExpiredError): `still_running`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`

<a id="ProtocolRunner.run_single_scan"></a>
### `ProtocolRunner.run_single_scan`

```python
def run_single_scan(
    self,
    protocol: Protocol,
    *,
    sequence_name: str = 'scan',
    parent_dir: FilePath | None = None,
    enable_image_saving: bool = True,
    events: RunEvents | None = None,
    return_to_position: PlateLocation | None = None,
    timeout_s: float | None = None,
) -> RunOutcome
```

Marks: in-process.

`start_single_scan`, waited for: returns the run's outcome once it has let go of the scope.

Takes `start_single_scan`'s arguments, and `timeout_s`: how long to wait
for the run to end; None waits for its end. A run that ends any way
returns its outcome, a stopped one included.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was started.
- `ProtocolRunRefusedError`: As `start_single_scan`; nothing was started.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_single_scan`. Nothing was started.
- `RunWaitExpiredError`: `'still_running'`, a `TimeoutError`: `timeout_s` ran out; its `operation` holds the run, still going.

**Refusals**

- [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`, `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`, `not_connected`
- [`RunWaitExpiredError`](#RunWaitExpiredError): `still_running`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`

<a id="ProtocolRunner.run_zstack"></a>
### `ProtocolRunner.run_zstack`

```python
def run_zstack(
    self,
    channel: str | None,
    *,
    sequence_name: str = 'zstack',
    parent_dir: FilePath | None = None,
    events: RunEvents | None = None,
    return_to_start: bool = True,
    enable_image_saving: bool = True,
    timeout_s: float | None = None,
) -> RunOutcome
```

Marks: in-process.

`start_zstack`, waited for: returns the run's outcome once it has let go of the scope.

Takes `start_zstack`'s arguments, and `timeout_s`: how long to wait
for the run to end; None waits for its end. A run that ends any way
returns its outcome, a stopped one included.

**Raises**

- `ArgumentRefusedError`: `'not_a_number'` or `'out_of_range'`, `timeout_s` is not a wait from 0 to 30 days. Nothing was started.
- `ProtocolRunRefusedError`: As `start_zstack`; nothing was started.
- `BlockingCallRefusedError`: `'wait_on_ui_thread'`, called on the UI thread, whose form is `start_zstack`. Nothing was started.
- `RunWaitExpiredError`: `'still_running'`, a `TimeoutError`: `timeout_s` ran out; its `operation` holds the run, still going.

**Refusals**

- [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`, `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`, `not_connected`
- [`RunWaitExpiredError`](#RunWaitExpiredError): `still_running`
- [`ArgumentRefusedError`](#ArgumentRefusedError): `not_a_number`, `out_of_range`
- [`BlockingCallRefusedError`](#BlockingCallRefusedError): `wait_on_ui_thread`

<a id="ProtocolRunner.start_autofocus"></a>
### `ProtocolRunner.start_autofocus`

```python
def start_autofocus(
    self,
    channel: str | None,
    *,
    save_characterization_data: bool = False,
    sequence_name: str = 'autofocus',
    parent_dir: FilePath | None = None,
    events: RunEvents | None = None,
    claim: HeldClaim | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/handles/ProtocolRunner/{handle_id}/start_autofocus`.

Autofocus once on *channel*, at the current stage position.

The headless twin of the standalone autofocus button: a
one-position, one-layer run with autofocus on, saving no images and
no run artifacts, that leaves the stage at the focus it found.

The layer is named rather than discovered. A GUI reads it from
whichever drawer is open, which is a fact about a running GUI and
means nothing to a caller that has none, so this asks for it and
has no default -- an autofocus on a layer nobody chose is not a
useful answer.

Everything else comes from the settings store, so this run and a
click on the button resolve the same way.

The stage is deliberately left where the focus was found, and there
is no return-to-position input: ending at the focus is the point.
A caller sweeping the same field repeatedly sets its own Z between
runs, which it must do anyway for the measurements to be comparable.

**Args**

- `channel`: Which channel to focus on ('BF', 'Ex488', ...).
- `save_characterization_data`: Whether to write the per-position focus scores this sweep measured. Off by default: a caller that does not ask for data gets no folder. When on, the run's outcome reports whether the file was written and names it (af_data_saved / af_data_path), which is the only way a headless caller can tell a delivered file from a requested one.
- `sequence_name`: The name of the protocol file the run saves in its folder ('.tsv' added): a name, not a path.
- `parent_dir`: Where characterization data goes. Defaults to 'Autofocus Characterization' under the live folder, where the button already puts it. Unused when no data is saved: that run writes nowhere, so where it would have written cannot refuse it.
- `events`: The run's event handlers (`RunEvents`); None for none.
- `claim`: A claim the caller holds -- the one `session.diagnostic_claim()` yields -- to run under instead of taking the scope. The run acts inside the caller's activity and cannot release its claim; it is refused if that claim no longer holds. None takes the scope for this run alone. The run is attended -- its failures are shown -- unless it runs under a claim.

**Returns**

The run's operation, to wait on or to ignore; its result() is the run's outcome.

**Raises**

- `AxisMovingError`: `'axis_moving'`, an axis is moving, so the position the run is built around is a place it is passing through. Nothing is started.
- `ArgumentRefusedError`: `'channel_unknown'`, or `'no_channel_selected'` for None; asked before anything is read from the scope.
- `ProtocolRunRefusedError`: The runner refused the request (already running, files still writing, hardware not connected); no state was committed.

**Refusals by REST status**

- 409: [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`, `no_channel_selected`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ProtocolRunner.start_autofocus_all_steps"></a>
### `ProtocolRunner.start_autofocus_all_steps`

```python
def start_autofocus_all_steps(
    self,
    protocol: Protocol,
    *,
    events: RunEvents | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/handles/ProtocolRunner/{handle_id}/start_autofocus_all_steps`.

Autofocus at every step of *protocol*, and write the focus into it.

One scan that visits each step with autofocus on, whatever each
step's own autofocus setting, and captures nothing. When the scan
completes, each step's Z becomes the focus found for it, written
before the run lets go of the scope, so it is in the protocol when
run_ended is sent.
A scan that does not complete writes nothing, because it focused
only some of the steps.

The focus belongs to the steps it was found at, so a protocol whose
steps changed during the scan -- a different number of them, or a
step at a different position, channel or objective -- is left
unchanged, and the person is told once ('Focus Not Saved'). The
outcome's focus_written says which happened.

**Args**

- `protocol`: The protocol to focus. Its autofocus settings are not changed; only its Z values are written.
- `events`: The run's event handlers (`RunEvents`); None for none.

**Returns**

The run's operation, to wait on or to ignore; its result() is the run's outcome.

**Raises**

- `ProtocolRunRefusedError`: The runner refused the request (already running, files still writing, hardware not connected, a step it cannot run); no state was committed.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ProtocolRunner.start_composite"></a>
### `ProtocolRunner.start_composite`

```python
def start_composite(
    self,
    *,
    sequence_name: str = 'composite',
    parent_dir: FilePath | None = None,
    events: RunEvents | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/handles/ProtocolRunner/{handle_id}/start_composite`.

Assemble a composite run and launch it, returning once committed.

Split out of run_composite so a caller that must not block -- a GUI
click on the thread that draws the button -- gets the same assembly
without the wait. The alternative was for such a caller to build the
run config itself, which is the duplicate composite implementation
this run kind exists to retire.

**Args**

- `sequence_name`: The name of the protocol file the run saves in its folder ('.tsv' added): a name, not a path.
- `parent_dir`: Parent directory for output. Defaults to 'Manual/Composites' under the live folder, where the button already puts it, so a script's composite and a click's land in the same place.
- `events`: The run's event handlers (`RunEvents`); None for none.

**Returns**

The run's operation, to wait on or to ignore; its result() is the run's outcome.

**Raises**

- `AxisMovingError`: `'axis_moving'`, an axis is moving, so the position the run is built around is a place it is passing through. Nothing is started.
- `ProtocolRunRefusedError`: Fewer than two channels are set to capture an image, so nothing could be merged; or the runner refused the run itself (already running, files still writing, hardware not connected).

**Refusals by REST status**

- 409: [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ProtocolRunner.start_protocol"></a>
### `ProtocolRunner.start_protocol`

```python
def start_protocol(
    self,
    protocol: Protocol,
    *,
    sequence_name: str = 'protocol',
    parent_dir: FilePath | None = None,
    enable_image_saving: bool = True,
    events: RunEvents | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/handles/ProtocolRunner/{handle_id}/start_protocol`.

Run a full protocol (multiple scans over time).

**Args**

- `protocol`: Protocol defining the steps, period, and duration
- `sequence_name`: The name of the protocol file the run saves in its folder ('.tsv' added): a name, not a path
- `parent_dir`: Parent directory for output
- `enable_image_saving`: Whether to save captured images
- `events`: The run's event handlers (`RunEvents`); None for none.

**Returns**

The run's operation. Its result() is the run's outcome -- the status, reason, title and message the run ended with -- once the run has let go of the scope and its run_ended has been delivered.

**Raises**

- `ProtocolRunRefusedError`: The run was refused before any state was committed; no operation is returned and no run is started.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ProtocolRunner.start_single_scan"></a>
### `ProtocolRunner.start_single_scan`

```python
def start_single_scan(
    self,
    protocol: Protocol,
    *,
    sequence_name: str = 'scan',
    parent_dir: FilePath | None = None,
    enable_image_saving: bool = True,
    events: RunEvents | None = None,
    return_to_position: PlateLocation | None = None,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/handles/ProtocolRunner/{handle_id}/start_single_scan`.

Run a single scan through the protocol steps.

**Args**

- `protocol`: Protocol defining the steps to execute
- `sequence_name`: The name of the protocol file the run saves in its folder ('.tsv' added): a name, not a path
- `parent_dir`: Parent directory for output (defaults to settings['live_folder']/ProtocolData)
- `enable_image_saving`: Whether to save captured images
- `events`: The run's event handlers (`RunEvents`); None for none.
- `return_to_position`: Where to return once the scan ends, on the protocol's plate, since the run moves in its protocol's frame; an axis that is None is left where the scan left it. None returns nowhere.

**Returns**

The run's operation. Its result() is the run's outcome -- the status, reason, title and message the run ended with -- once the run has let go of the scope and its run_ended has been delivered.

**Raises**

- `ProtocolRunRefusedError`: The run was refused before any state was committed; no operation is returned and no run is started.

**Refusals by REST status**

- 409: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ProtocolRunner.start_zstack"></a>
### `ProtocolRunner.start_zstack`

```python
def start_zstack(
    self,
    channel: str | None,
    *,
    sequence_name: str = 'zstack',
    parent_dir: FilePath | None = None,
    events: RunEvents | None = None,
    return_to_start: bool = True,
    enable_image_saving: bool = True,
) -> Operation
```

Marks: stopped by [`Operation.stop`](#Operation.stop); enters: `lane_call`.  
Route: `POST /api/v1/handles/ProtocolRunner/{handle_id}/start_zstack`.

Capture a z-stack on *channel*, around the current stage position.

The one implementation of a z-stack run, for a script and for the
Acquire button on the z-stack panel alike: a one-position,
one-layer run that expands into one step per slice. The slices are
its product, so unlike an autofocus this one saves its images
unless the caller says otherwise.

The stack's range, step size and reference -- whether the current
position is the top, centre or bottom of the sweep -- come from the
settings store, as does everything else about the capture, so this
run and a click on the button resolve the same way. The layer is named by the
caller for the same reason start_autofocus asks for it: an open
drawer is a fact about a running GUI.

Autofocus is forced OFF for the step, matching the button. A
z-stack with autofocus enabled refocuses at each slice and
flattens the stack it was asked to capture.

Stimulation configs are carried onto the step as stored, enabled or
not. Stated rather than
inherited silently: if the API should instead carry only the
enabled ones, this is the line that changes.

**Args**

- `channel`: Which channel to capture ('BF', 'Ex488', ...).
- `sequence_name`: The name of the protocol file the run saves in its folder ('.tsv' added): a name, not a path.
- `parent_dir`: Parent directory for output. Defaults to 'Manual/Z-Stacks' under the live folder, where the button already puts it.
- `events`: The run's event handlers (`RunEvents`); None for none.
- `return_to_start`: Whether to put the stage back where the stack was centred when the run ends. On by default, because a stack leaves Z at whichever end it finished on, which is not where the operator was looking. A bool rather than a position, so a caller cannot hand back coordinates in a frame this run never used.
- `enable_image_saving`: Whether the slices are written. On by default; the engineering panel's "disable image saving" switch is the one caller that turns it off, to exercise the stage without filling the disk.

**Returns**

The run's operation, to wait on or to ignore; its result() is the run's outcome.

**Raises**

- `AxisMovingError`: `'axis_moving'`, an axis is moving, so the position the run is built around is a place it is passing through. Nothing is started.
- `ArgumentRefusedError`: `'channel_unknown'`, or `'no_channel_selected'` for None; asked before anything is read from the scope.
- `ObjectiveUnknownError`: The objective in the light path is unknown, so no slice could say what it was taken with.
- `ProtocolRunRefusedError`: The runner refused the request (already running, files still writing, hardware not connected); no state was committed.

**Refusals by REST status**

- 409: [`AxisMovingError`](#AxisMovingError): `axis_moving`; [`ObjectiveUnknownError`](#ObjectiveUnknownError): `slot_unknown`, `slot_unassigned`, `not_in_catalogue`, `none_selected`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `already_running`, `exclusive_activity_running`, `recording_active`, `files_writing`, `files_writing_stalled`, `post_processing_running`, `autofocus_running`, `camera_not_streaming`, `position_unknown`, `lid_open_before_motion`, `capture_location_unusable`, `no_acquiring_layer`, `turret_slot_ambiguous`, `turret_slot_reassigned`, `step_position_unknown`, `turret_objectives_unassigned`, `objective_not_mounted`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `exclusive_activity_running`; [`ControlSessionRefusedError`](#ControlSessionRefusedError): `control_session_held`
- 422: [`ArgumentRefusedError`](#ArgumentRefusedError): `channel_unknown`, `no_channel_selected`; [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `channel_already_in_add`, `composite_two_transmitted_layers`, `sequence_name_invalid`, `empty_protocol`, `validation_failed`, `composite_needs_two_channels`, `tiling_unknown`, `tiling_scale_unknown`, `zstack_not_configured`, `already_tiled`, `objective_not_in_catalogue`, `tiles_outside_travel`, `zslices_outside_travel`, `positions_outside_travel`, `camera_setting_out_of_range`, `positions_unreachable`, `step_position_missing`, `layer_not_on_scope`, `channel_unknown`, `objectives_require_turret`, `objective_not_given`, `focus_not_given`, `overlap_out_of_range`, `values_out_of_range`; [`BlockingCallRefusedError`](#BlockingCallRefusedError): `blocking_on_worker`, `blocking_inline`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `activity_ended`
- 500: [`LaneCallStillRunningError`](#LaneCallStillRunningError): `call_overran`
- 503: [`ProtocolRunRefusedError`](#ProtocolRunRefusedError): `not_connected`; [`HardwareCommandRefusedError`](#HardwareCommandRefusedError): `scope_disconnected`

<a id="ProtocolSizeAdvisory"></a>
## `ProtocolSizeAdvisory`

A record: it crosses the wire as its data.  
Returned by: [`Protocol.size_advisory`](#Protocol.size_advisory), [`ScopeSession.protocol_size_advisory`](#ScopeSession.protocol_size_advisory).

A protocol big enough that the user should be told before running it.

Carries the numbers alongside the sentence so a caller that wants to
render something other than the sentence -- REST, a log line -- does not
have to parse it back out.

**Fields**

- <a id="ProtocolSizeAdvisory.message"></a>`message`: `str` -- (no description)
- <a id="ProtocolSizeAdvisory.num_steps"></a>`num_steps`: `int` -- (no description)
- <a id="ProtocolSizeAdvisory.projected_mb"></a>`projected_mb`: `float` -- (no description)

<a id="Remedy"></a>
## `Remedy`

A record: it crosses the wire as its data.  
Held in: [`Notification.remedy`](#Notification.remedy), [`Outcome.remedy`](#Outcome.remedy).  
Taken by: [`ScopeSession.apply_remedy`](#ScopeSession.apply_remedy).

The one action that answers a refusal, offered to whoever was refused.

Data, not a callable: the layer that refuses cannot reach the member that
answers (a run's engine refuses; the Session recovers), and a REST caller
receives the remedy as a name it can send back. The Session is the one
place a name becomes an action (`ScopeSession.apply_remedy`). The offer's
prompt is the refusal's own title and message, so its words live in one
place.

**Fields**

- <a id="Remedy.cancel_text"></a>`cancel_text`: `str` -- The words that decline it and leave things as they are.
- <a id="Remedy.confirm_text"></a>`confirm_text`: `str` -- The words that take the remedy, naming what it costs.
- <a id="Remedy.member"></a>`member`: `RemedyName` -- The Session member that answers the refusal.

<a id="RunEnded"></a>
## `RunEnded`

A record: it crosses the wire as its data.  
Delivered by: [`RunEvents.run_ended`](#RunEvents.run_ended).

A `run_ended` event: the outcome, the run's folder (None when it saved nothing), its protocol.

**Fields**

- <a id="RunEnded.outcome"></a>`outcome`: `RunOutcome` -- (no description)
- <a id="RunEnded.protocol"></a>`protocol`: `Protocol` -- (no description)
- <a id="RunEnded.run_dir"></a>`run_dir`: `pathlib.Path | None` -- (no description)

<a id="RunEvents"></a>
## `RunEvents`

A record: it crosses the wire as its data.  
Taken by: [`ProtocolRunner.run_autofocus`](#ProtocolRunner.run_autofocus), [`ProtocolRunner.run_autofocus_all_steps`](#ProtocolRunner.run_autofocus_all_steps), [`ProtocolRunner.run_composite`](#ProtocolRunner.run_composite), [`ProtocolRunner.run_protocol`](#ProtocolRunner.run_protocol), [`ProtocolRunner.run_single_scan`](#ProtocolRunner.run_single_scan), [`ProtocolRunner.run_zstack`](#ProtocolRunner.run_zstack), [`ProtocolRunner.start_autofocus`](#ProtocolRunner.start_autofocus), [`ProtocolRunner.start_autofocus_all_steps`](#ProtocolRunner.start_autofocus_all_steps), [`ProtocolRunner.start_composite`](#ProtocolRunner.start_composite), [`ProtocolRunner.start_protocol`](#ProtocolRunner.start_protocol), [`ProtocolRunner.start_single_scan`](#ProtocolRunner.start_single_scan), [`ProtocolRunner.start_zstack`](#ProtocolRunner.start_zstack).

The handlers a run calls as it goes, each optional; a misspelt one fails here.

Every event but `frame_captured` arrives on the thread that sends it,
unless `on_ui_thread` is set: then through the process's UI dispatcher
(`ScopeSession.set_ui_dispatcher`), on a GUI host on its UI thread, and
with none set on the thread that sends it. The run carries the choice,
so a GUI's handlers run on its UI thread while a REST client's or a
script's run in the same process never waits on that thread. A handler's own
exception is reported once, as the exception it is, under the event's
name: a refusal stays a refusal and a fault a fault. During an unattended
run a `step_started` or `frame_captured` failure is logged, muted by
the run's scope, and a `run_ended` or `files_written` failure is
shown. No handler needs to guard itself.

Each handler field declares its event's record (`ScanStarted`, ...), whose
fields name the handler's arguments in order; a host that carries the
event elsewhere builds the record from them.

**Fields**

- <a id="RunEvents.files_written"></a>`files_written`: `Callable[[pathlib.Path | None, FilesOutcome], object] | None` -- `(run_dir, files)` once the run's last image write has landed: `files` is `FilesOutcome.WRITTEN`, or `INCOMPLETE` when some of its images are not on disk.
- <a id="RunEvents.frame_captured"></a>`frame_captured`: `Callable[[np.ndarray, int, int], object] | None` -- `(image, frames_summed, frame_significant_bits)` for each captured frame whose write was handed to the run's writer, once it was: the frame as the file gets it (a sum when summing), read-only. Delivered on the run's own thread, which waits for it, so it must not block; a GUI renders and shows it.
- <a id="RunEvents.on_ui_thread"></a>`on_ui_thread`: `bool` -- Whether the handlers above but `frame_captured` run on the UI dispatcher's thread. A GUI sets it for handlers that draw; the default runs each on the sending thread.
- <a id="RunEvents.run_ended"></a>`run_ended`: `Callable[[RunOutcome, pathlib.Path | None, Protocol], object] | None` -- `(outcome, run_dir, protocol)` once the run has let go of the scope and its outcome has settled: for every run at the release, for a composite once its merge settles, which may be before or after its `files_written`. `run_dir` is None for a run that saved nothing.
- <a id="RunEvents.scan_ended"></a>`scan_ended`: `Callable[[int, int, float | None], object] | None` -- `(scan_index, scans_remaining, period_s)` as a scan ends, a stopped one included: how many remain after it.
- <a id="RunEvents.scan_started"></a>`scan_started`: `Callable[[int, int, float | None], object] | None` -- `(scan_index, scans_remaining, period_s)` as a scan begins: the scan's index from 0, how many scans remain with this one, and the protocol's period in seconds.
- <a id="RunEvents.step_started"></a>`step_started`: `Callable[[int], object] | None` -- `(step_index)` as each step begins, step 0 included, the index read at the step.
- <a id="RunEvents.video_progress"></a>`video_progress`: `Callable[[VideoProgress], object] | None` -- `(progress)`, a `VideoProgress`, as a video step records and writes.

<a id="RunFiles"></a>
## `RunFiles`

A record: it crosses the wire as its data.

What became of one run's images, once its writes are done.

The run's batch's own account, read when it completed: its outcome,
and when it is `INCOMPLETE`, how many images are not on disk and why
(the batch's `not_written_reason`).

**Fields**

- <a id="RunFiles.not_written"></a>`not_written`: `int` -- (no description)
- <a id="RunFiles.not_written_reason"></a>`not_written_reason`: `str | None` -- (no description)
- <a id="RunFiles.outcome"></a>`outcome`: `FilesOutcome` -- (no description)
- <a id="RunFiles.written"></a>`written`: `int` -- (no description)

<a id="RunOutcome"></a>
## `RunOutcome`

A record: it crosses the wire as its data.  
Returned by: [`ProtocolRunner.run_autofocus`](#ProtocolRunner.run_autofocus), [`ProtocolRunner.run_autofocus_all_steps`](#ProtocolRunner.run_autofocus_all_steps), [`ProtocolRunner.run_composite`](#ProtocolRunner.run_composite), [`ProtocolRunner.run_protocol`](#ProtocolRunner.run_protocol), [`ProtocolRunner.run_single_scan`](#ProtocolRunner.run_single_scan), [`ProtocolRunner.run_zstack`](#ProtocolRunner.run_zstack).  
Held in: [`RunEnded.outcome`](#RunEnded.outcome), [`RunEvents.run_ended`](#RunEvents.run_ended).

How a run ended, and what its merge produced.

The first four fields are the run's ENDING, copied from the
RunEnding whatever ended the run recorded; the next three describe
the merge, the next two the autofocus characterization data, then
what the run captured, which steps putting the scope back after it
did not finish, whether a scan's focus was written into the caller's
protocol, and last which steps' autofocus did not focus. A
non-composite run carries merged=False and an empty merge_reason
under whatever status it ended in, and a run that saved
no autofocus data carries af_data_saved=False, so a caller reads one
shape for every run kind.

**Fields**

- <a id="RunOutcome.af_data_path"></a>`af_data_path`: `pathlib.Path | None` -- The characterization file that was written; None whenever af_data_saved is False. Never the folder: that is allocated before the sweep runs and exists even when nothing was written, which is the answer a caller must not be given.
- <a id="RunOutcome.af_data_saved"></a>`af_data_saved`: `bool` -- True only when autofocus characterization data was WRITTEN. False covers every run that asked for none, and every run that asked and got none -- a sweep that collected nothing, or one whose queued save an abort discarded.
- <a id="RunOutcome.af_focus_z_um"></a>`af_focus_z_um`: `float | None` -- The Z a standalone autofocus run chose as focus, where it left the stage; None when it chose none (a degenerate curve, an abort, an error), in which case the stage is back where the run found it. A standalone autofocus run that chose none ends 'failed' ('autofocus_failed', or 'autofocus_not_started' when its first window reached past Z's travel), unless its restore lost Z ('position_lost'); a Stop is a stop. Set only by a standalone autofocus run: a run that autofocuses at several steps has no one focus to report, and reads None.
- <a id="RunOutcome.artifact_path"></a>`artifact_path`: `pathlib.Path | None` -- Where it landed; None whenever merged is False.
- <a id="RunOutcome.captures"></a>`captures`: `CaptureTally | None` -- What the run captured of what it was asked for. None only for a run settled before its cleanup counted -- a run torn down by a shutdown before it ended.
- <a id="RunOutcome.cleanup_failures"></a>`cleanup_failures`: `tuple[str, ...] | None` -- The steps putting the scope back after the run that did not finish ('Restore LED states', 'Return to position', ...); empty when every one did, so the LEDs, camera settings and stage are as the run found them. None only for a run settled before its cleanup ran. A host's own display work at the run's end is its `run_ended` handler's, never a step here.
- <a id="RunOutcome.focus_failures"></a>`focus_failures`: `tuple[FocusFailure, ...]` -- Each step whose autofocus did not focus, a flat curve included, in the order they failed: a capture protocol captured it at its fallback focus, an autofocus scan left its focus as it was. Counted once per scan and step, so a scan run again that focuses the step clears it. Empty when every autofocus focused, and for a run with none.
- <a id="RunOutcome.focus_written"></a>`focus_written`: `bool | None` -- Whether the focus a scan found was written into the protocol the caller asked it to write to. None for a run asked to write no focus; False when it was asked and wrote none -- the scan did not complete, or the protocol's steps changed during it (reported once, as 'Focus Not Saved') -- and the protocol is unchanged.
- <a id="RunOutcome.merge_reason"></a>`merge_reason`: `str` -- Why the MERGE produced no file ('write_batch_timeout', 'write_batch_abandoned', 'write_batch_not_taken', 'write_batch_save_failed', 'write_batch_disk_full', 'write_batch_video_unfinished', 'no_run_dir', 'shutdown', ...). Empty on success and on every run that has no merge, so a caller can branch on merged and still log one field.
- <a id="RunOutcome.merged"></a>`merged`: `bool` -- True only when an artifact was produced.
- <a id="RunOutcome.message"></a>`message`: `str` -- The sentence a user reads.
- <a id="RunOutcome.reason"></a>`reason`: `str` -- Why the RUN ended ('stopped', 'position_lost', 'captures_failed', ...).
- <a id="RunOutcome.status"></a>`status`: `RunStatus` -- How the run ended.
- <a id="RunOutcome.title"></a>`title`: `str` -- Short human sentence, suitable as a popup heading.

<a id="RunProgress"></a>
## `RunProgress`

A record: it crosses the wire as its data.  
Returned by: [`Operation.progress`](#Operation.progress).

Where one run has got, read in one hold of the run lock: every field is that run's.

**Fields**

- <a id="RunProgress.is_last_run"></a>`is_last_run`: `bool` -- The run is the engine's most recent, live or finished.
- <a id="RunProgress.is_live"></a>`is_live`: `bool` -- The run holds the scope: started and not yet torn down. False once its teardown has finished, while its `run_ended` may still be on its way, so a run can be not live and its operation not yet done.
- <a id="RunProgress.is_stopping"></a>`is_stopping`: `bool` -- Live, and a Stop of it accepted: True until its teardown finishes.
- <a id="RunProgress.num_steps"></a>`num_steps`: `int | None` -- The run's step count; None once it is not live.
- <a id="RunProgress.period_s"></a>`period_s`: `float | None` -- The run's scan period in seconds; None once it is not live.
- <a id="RunProgress.run_dir"></a>`run_dir`: `Path | None` -- The run's folder; None for a run that saves nothing or never started.
- <a id="RunProgress.scans_remaining"></a>`scans_remaining`: `int | None` -- The scans still to run; None once it is not live.
- <a id="RunProgress.step_index"></a>`step_index`: `int | None` -- The step executing now, its row from 0; None once the run is not live.

<a id="SavedFocus"></a>
## `SavedFocus`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.save_focus`](#ScopeSession.save_focus).

What `ScopeSession.save_focus` wrote.

**Fields**

- <a id="SavedFocus.step_index"></a>`step_index`: `int | None` -- The step that took it as its Z, or None when no step did.
- <a id="SavedFocus.z_um"></a>`z_um`: `float` -- The Z saved as the layer's focus.

<a id="SavedStill"></a>
## `SavedStill`

A record: it crosses the wire as its data.

One file a still wrote, with the instrument's record of it.

Built at the capture from what the still read beside its grab, for
every format alike: a JPG embeds none of it, so the record is the one
place a JPG's channel, position and frame are stated.

**Fields**

- <a id="SavedStill.channel"></a>`channel`: `str` -- The layer it is named and set for.
- <a id="SavedStill.file"></a>`file`: `Path` -- The file written.
- <a id="SavedStill.frame"></a>`frame`: `FrameRecord` -- The instrument's account of the frame (`FrameRecord`); an overlay copy carries its frame's.
- <a id="SavedStill.objective_id"></a>`objective_id`: `str` -- The objective in the light path when it was taken.
- <a id="SavedStill.output_format"></a>`output_format`: `OutputFormat` -- The format it was saved in; never the hyperstack, which a single capture has no meaning for.
- <a id="SavedStill.plate_x_mm"></a>`plate_x_mm`: `float | None` -- Plate X where it was taken; None when the axis had no reference.
- <a id="SavedStill.plate_y_mm"></a>`plate_y_mm`: `float | None` -- Plate Y, likewise.
- <a id="SavedStill.significant_bits"></a>`significant_bits`: `int` -- The depth of the array saved: the frame's as captured, or 8 for an overlay copy, the frame's 8-bit rendering. A scaled encoding stores the frame left-justified in its file.
- <a id="SavedStill.stage_z_um"></a>`stage_z_um`: `float | None` -- Stage Z, likewise.
- <a id="SavedStill.statistics"></a>`statistics`: `ImageStatistics` -- The values of the array saved, in its counts at `significant_bits`.
- <a id="SavedStill.well_label"></a>`well_label`: `str | None` -- The well at that position; None off the plate's wells, on a plate with none, or at an unknown position.

<a id="ScanEnded"></a>
## `ScanEnded`

A record: it crosses the wire as its data.  
Delivered by: [`RunEvents.scan_ended`](#RunEvents.scan_ended).

A `scan_ended` event: the scan's index from 0, the scans left after it, the period in seconds.

**Fields**

- <a id="ScanEnded.period_s"></a>`period_s`: `float | None` -- (no description)
- <a id="ScanEnded.scan_index"></a>`scan_index`: `int` -- (no description)
- <a id="ScanEnded.scans_remaining"></a>`scans_remaining`: `int` -- (no description)

<a id="ScanStarted"></a>
## `ScanStarted`

A record: it crosses the wire as its data.  
Delivered by: [`RunEvents.scan_started`](#RunEvents.scan_started).

A `scan_started` event: the scan's index from 0, the scans left with it, the period in seconds.

**Fields**

- <a id="ScanStarted.period_s"></a>`period_s`: `float | None` -- (no description)
- <a id="ScanStarted.scan_index"></a>`scan_index`: `int` -- (no description)
- <a id="ScanStarted.scans_remaining"></a>`scans_remaining`: `int` -- (no description)

<a id="ScopeCapabilities"></a>
## `ScopeCapabilities`

A record: it crosses the wire as its data.  
Held in: [`Lumascope.capabilities`](#Lumascope.capabilities).

Immutable snapshot of what a scope has.

Built once at `Lumascope.__init__` from the three drivers. Fields
are tuples (not lists) to reinforce immutability -- a caller that
wants to mutate would have to shallow-copy into their own list.

**Fields**

- <a id="ScopeCapabilities.axes"></a>`axes`: `tuple[Axis, ...]` -- (no description)
- <a id="ScopeCapabilities.buffer_bit_depth"></a>`buffer_bit_depth`: `int` -- (no description)
- <a id="ScopeCapabilities.camera_binning_sizes"></a>`camera_binning_sizes`: `tuple[int, ...]` -- (no description)
- <a id="ScopeCapabilities.camera_firmware_version"></a>`camera_firmware_version`: `str | None` -- (no description)
- <a id="ScopeCapabilities.camera_has_clock_latch"></a>`camera_has_clock_latch`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_limits"></a>`camera_limits`: `CameraLimits | None` -- (no description)
- <a id="ScopeCapabilities.camera_max_significant_bits"></a>`camera_max_significant_bits`: `int | None` -- (no description)
- <a id="ScopeCapabilities.camera_model"></a>`camera_model`: `str | None` -- (no description)
- <a id="ScopeCapabilities.camera_pixel_formats"></a>`camera_pixel_formats`: `tuple[str, ...]` -- (no description)
- <a id="ScopeCapabilities.camera_reports_temperature"></a>`camera_reports_temperature`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_sdk_version"></a>`camera_sdk_version`: `str | None` -- (no description)
- <a id="ScopeCapabilities.camera_sensor"></a>`camera_sensor`: `str | None` -- (no description)
- <a id="ScopeCapabilities.camera_serial_number"></a>`camera_serial_number`: `str | None` -- (no description)
- <a id="ScopeCapabilities.camera_shutter"></a>`camera_shutter`: `CameraShutter | None` -- (no description)
- <a id="ScopeCapabilities.camera_supports_auto_exposure"></a>`camera_supports_auto_exposure`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_supports_auto_gain"></a>`camera_supports_auto_gain`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_supports_black_level"></a>`camera_supports_black_level`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_supports_conversion_gain_mode"></a>`camera_supports_conversion_gain_mode`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_supports_line_noise_reduction"></a>`camera_supports_line_noise_reduction`: `bool` -- (no description)
- <a id="ScopeCapabilities.camera_timestamp_tick_hz"></a>`camera_timestamp_tick_hz`: `int | None` -- (no description)
- <a id="ScopeCapabilities.frame_rules"></a>`frame_rules`: `Mapping[FrameValiditySource, FrameRule]` -- (no description)
- <a id="ScopeCapabilities.has_firmware_stim"></a>`has_firmware_stim`: `bool` -- (no description)
- <a id="ScopeCapabilities.has_focus"></a>`has_focus`: `bool` -- (no description)
- <a id="ScopeCapabilities.has_turret"></a>`has_turret`: `bool` -- (no description)
- <a id="ScopeCapabilities.has_xy_stage"></a>`has_xy_stage`: `bool` -- (no description)
- <a id="ScopeCapabilities.is_color_native"></a>`is_color_native`: `bool` -- (no description)
- <a id="ScopeCapabilities.led_addresses"></a>`led_addresses`: `tuple[int, ...] | None` -- (no description)
- <a id="ScopeCapabilities.led_colors"></a>`led_colors`: `tuple[str, ...]` -- (no description)
- <a id="ScopeCapabilities.led_max_ma"></a>`led_max_ma`: `int | None` -- (no description)
- <a id="ScopeCapabilities.lens_focal_length_mm"></a>`lens_focal_length_mm`: `float | None` -- (no description)
- <a id="ScopeCapabilities.pixel_size_um"></a>`pixel_size_um`: `float | None` -- (no description)

<a id="ScopeModel"></a>
## `ScopeModel`

A record: it crosses the wire as its data.  
Returned by: [`Lumascope.scope_models`](#Lumascope.scope_models).

One model of the installation's catalogue (`scopes.json`'s `Models`), read once at bring-up.

`id` is the model's name, the key the operator selects and a board
reports. `has_focus`, `has_xy_stage` and `has_turret` are the motor
axes the model is built with: Z, X and Y, T. `channels` are the model's
channel rows, resolved against the release's layer vocabulary: what a unit
of the model has when its board carries no Channels block of its own.
`led_board` and `motor_board` name the model's boards, `motor_board`
None for a model with no motor axes; production finds the boards it has
and reads neither, and the simulator builds from them. `filterset` is
None where the row names none. `pixel_size_um` (the sensor's pixel
pitch) and `lens_focal_length_mm` (the tube lens) are a Classic
model's optics, which no motorconfig carries; `max_frame` is the
largest frame, unbinned, of a model whose lens images less than its
sensor. Each is None where the row declares none.

**Fields**

- <a id="ScopeModel.channels"></a>`channels`: `tuple[ChannelRecord, ...]` -- (no description)
- <a id="ScopeModel.filterset"></a>`filterset`: `str | None` -- (no description)
- <a id="ScopeModel.has_focus"></a>`has_focus`: `bool` -- (no description)
- <a id="ScopeModel.has_turret"></a>`has_turret`: `bool` -- (no description)
- <a id="ScopeModel.has_xy_stage"></a>`has_xy_stage`: `bool` -- (no description)
- <a id="ScopeModel.id"></a>`id`: `str` -- (no description)
- <a id="ScopeModel.led_board"></a>`led_board`: `LedBoardFamily` -- (no description)
- <a id="ScopeModel.lens_focal_length_mm"></a>`lens_focal_length_mm`: `float | None` -- (no description)
- <a id="ScopeModel.max_frame"></a>`max_frame`: `FrameSize | None` -- (no description)
- <a id="ScopeModel.motor_board"></a>`motor_board`: `MotorBoardFamily | None` -- (no description)
- <a id="ScopeModel.pixel_size_um"></a>`pixel_size_um`: `float | None` -- (no description)

<a id="SensorExposure"></a>
## `SensorExposure`

A record: it crosses the wire as its data.  
Returned by: [`SimulationAPI.exposed_under`](#SimulationAPI.exposed_under).

What one simulated frame was exposed under: the settings the sensor
applied when its exposure started, which is what its pixels show.

A write reaches the camera's register at once -- the frame's chunk
reports the register -- but a frame already started comes out under the
settings it started with. This record is that difference made
readable: a frame's chunk can name the new gain while this names the
gain its pixels were made at.

**Fields**

- <a id="SensorExposure.black_level"></a>`black_level`: `float` -- The black level applied.
- <a id="SensorExposure.depth_of_field_um"></a>`depth_of_field_um`: `float | None` -- The objective's depth of field the frame was focused through; None when no objective is known.
- <a id="SensorExposure.exposure_us"></a>`exposure_us`: `float` -- The exposure applied.
- <a id="SensorExposure.frame_id"></a>`frame_id`: `int` -- The camera's frame counter, as the frame's `FrameID` chunk carries it.
- <a id="SensorExposure.gain_db"></a>`gain_db`: `float` -- The gain applied.
- <a id="SensorExposure.illumination_ma"></a>`illumination_ma`: `float` -- The LED current lighting the sample, summed over the lit addresses; 0 for an unlit field.
- <a id="SensorExposure.light"></a>`light`: `float` -- The light on the sample per second of exposure at unity gain: each lit address's current times its layer's light per mA-second, summed; 0 for an unlit field.
- <a id="SensorExposure.pixel_size_um"></a>`pixel_size_um`: `float | None` -- The objective's image scale the frame was drawn at, micrometres of stage per sensor pixel at binning 1; None when no objective or no optics are known, and the frame is drawn at the scene's nominal scale.
- <a id="SensorExposure.started_ticks"></a>`started_ticks`: `int` -- The camera's clock when the exposure started, as the frame's `Timestamp` chunk carries it.
- <a id="SensorExposure.x_um"></a>`x_um`: `float | None` -- The stage's X when the exposure started; None on a scope with no X axis.
- <a id="SensorExposure.y_um"></a>`y_um`: `float | None` -- The stage's Y, likewise.
- <a id="SensorExposure.z_end_um"></a>`z_end_um`: `float | None` -- Z when the exposure ended (the next frame's start); None until then, and on a scope with no focus axis.
- <a id="SensorExposure.z_um"></a>`z_um`: `float | None` -- Z when the exposure started; None on a scope with no focus axis.

<a id="SettingsChanged"></a>
## `SettingsChanged`

A record: it crosses the wire as its data.  
Delivered by: [`ScopeSession.add_settings_listener`](#ScopeSession.add_settings_listener).

Settings one write stored, told once the write has let go of the store.

**Fields**

- <a id="SettingsChanged.paths"></a>`paths`: `tuple[str, ...]` -- The dotted paths stored, each once, in the order stored.

<a id="SettingsSetAside"></a>
## `SettingsSetAside`

A record: it crosses the wire as its data.  
Held in: [`BringUpRecord.settings_set_aside`](#BringUpRecord.settings_set_aside).

The user's settings file that could not be used, and why.

The app runs on the shipped template while this stands, and refuses to
save over the file until the person decides what happens to it.

**Fields**

- <a id="SettingsSetAside.path"></a>`path`: `pathlib.Path` -- (no description)
- <a id="SettingsSetAside.reason"></a>`reason`: `str` -- (no description)

<a id="ShownChannel"></a>
## `ShownChannel`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.light_shown_channel`](#ScopeSession.light_shown_channel), [`ScopeSession.show_channel`](#ScopeSession.show_channel), [`ScopeSession.shown_channel`](#ScopeSession.shown_channel).  
Delivered by: [`ScopeSession.add_shown_channel_listener`](#ScopeSession.add_shown_channel_listener).

The channel the scope shows: the channel whose camera settings are on
the camera, and whether its LED is lit with every other one dark.

**Fields**

- <a id="ShownChannel.channel"></a>`channel`: `str` -- The channel's key name.
- <a id="ShownChannel.lit"></a>`lit`: `bool` -- Whether its LED is lit.

<a id="SimulatedModelAnswer"></a>
## `SimulatedModelAnswer`

A record: it crosses the wire as its data.  
Taken by: [`ScopeSession.create`](#ScopeSession.create).

The model a simulated motor board names as its own (`answer_model`).

A record rather than the bare model, so that naming no model -- None --
is an answer a caller can ask for.

**Fields**

- <a id="SimulatedModelAnswer.model"></a>`model`: `str | None` -- The model the board names, or None for none.

<a id="SimulatedStall"></a>
## `SimulatedStall`

A record: it crosses the wire as its data.  
Taken by: [`ScopeSession.create`](#ScopeSession.create).

When a simulated camera's stream stalls, and for how long.

A stall is a stretch in which the camera stays connected and grabbing
and no frame arrives -- what a real camera's link or grab loop can fall
into without the device being removed. Built where it is asked for (a
launch argument, a test), so a stall that cannot happen is refused there.

**Raises**

- `ValueError`: `after_s` is negative or `for_s` is not positive.

**Fields**

- <a id="SimulatedStall.after_s"></a>`after_s`: `float` -- Seconds from the moment it is applied until the stall begins.
- <a id="SimulatedStall.for_s"></a>`for_s`: `float` -- How long the stall lasts, in seconds.

<a id="SizeSliderStops"></a>
## `SizeSliderStops`

A record: it crosses the wire as its data.  
Held in: [`CellCountSizeScale.area`](#CellCountSizeScale.area), [`CellCountSizeScale.perimeter`](#CellCountSizeScale.perimeter).

What a size slider's positions stand for: between two stops, a size in proportion.

**Fields**

- <a id="SizeSliderStops.positions"></a>`positions`: `tuple[float, ...]` -- The slider positions of the stops, ascending.
- <a id="SizeSliderStops.reach"></a>`reach`: `float` -- The largest size the image can hold; the slider goes no further than its position.
- <a id="SizeSliderStops.sizes"></a>`sizes`: `tuple[float, ...]` -- The size each stop stands for, in microns (square microns for an area), or in pixels when the image has no scale.

<a id="SizeSliderStops.positions_of"></a>
### `SizeSliderStops.positions_of`

```python
def positions_of(self, sizes: tuple[float, float]) -> tuple[float, float]
```

Marks: in-process; reads (changes nothing).

The slider positions that show two sizes.

**Raises**

Nothing.

<a id="SizeSliderStops.size_at"></a>
### `SizeSliderStops.size_at`

```python
def size_at(self, positions: tuple[float, float]) -> tuple[float, float]
```

Marks: in-process; reads (changes nothing).

The sizes a slider's two positions stand for.

**Raises**

Nothing.

<a id="SkippedBuild"></a>
## `SkippedBuild`

A record: it crosses the wire as its data.  
Held in: [`BuildsResult.skipped`](#BuildsResult.skipped).

A build `build` was asked for that had nothing to do in the folder.

**Fields**

- <a id="SkippedBuild.message"></a>`message`: `str` -- Why, in the build's own words.
- <a id="SkippedBuild.operation"></a>`operation`: `PostProcessingOperation` -- The build.

<a id="StageOffset"></a>
## `StageOffset`

A record: it crosses the wire as its data.  
Returned by: [`RuntimeState.get_stage_offset`](#RuntimeState.get_stage_offset).  
Taken by: [`RuntimeState.plate_to_stage`](#RuntimeState.plate_to_stage), [`RuntimeState.plate_transform`](#RuntimeState.plate_transform), [`RuntimeState.stage_to_plate`](#RuntimeState.stage_to_plate).

Where the plate is mounted on the stage, in um: the offset the two conversions apply.

**Fields**

- <a id="StageOffset.x_um"></a>`x_um`: `float` -- (no description)
- <a id="StageOffset.y_um"></a>`y_um`: `float` -- (no description)

<a id="StagePosition"></a>
## `StagePosition`

A record: it crosses the wire as its data.  
Returned by: [`RuntimeState.plate_to_stage`](#RuntimeState.plate_to_stage).

A stage X/Y position, in um.

**Fields**

- <a id="StagePosition.x_um"></a>`x_um`: `float` -- (no description)
- <a id="StagePosition.y_um"></a>`y_um`: `float` -- (no description)

<a id="Status"></a>
## `Status`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.status`](#ScopeSession.status).  
Delivered by: [`ScopeSession.add_run_state_listener`](#ScopeSession.add_run_state_listener).

What the scope is doing, in one read, for a client that has just connected.

Each field is read from its owner, one after another rather than under
one lock, so two fields can describe slightly different moments: an
axis can read HOMING in `axes` a moment before the home appears in
`live_work`.

**Fields**

- <a id="Status.axes"></a>`axes`: `dict[StageAxis, AxisPosition]` -- Each stage axis's state and position, None where the position is not known (`MotionAPI.axis_positions`): X, Y and Z, in um.
- <a id="Status.camera_streaming"></a>`camera_streaming`: `bool` -- Whether the camera is grabbing frames, asked of the camera itself (`ImagingAPI.is_streaming`).
- <a id="Status.control"></a>`control`: `ControlSession | None` -- The control session held, None when none is (`ScopeSession.control`).
- <a id="Status.live_work"></a>`live_work`: `LiveWork` -- Everything the session is still doing (`ScopeSession.live_work`).
- <a id="Status.lock"></a>`lock`: `ControlsLock` -- Which of a person's controls the Session locks, and why (`ScopeSession.controls_lock`): its editing policy, not a refusal.
- <a id="Status.parts"></a>`parts`: `tuple[PartStatus, ...]` -- Which parts came up at bring-up, and why each that did not (`ScopeSession.bring_up_record`).
- <a id="Status.run_refused"></a>`run_refused`: `ProtocolRunRefusedError | None` -- The refusal a run would get now for the scope's state alone, None when the state refuses none (`SequencedCaptureRunner.state_refusal`). None does not promise a run starts: the protocol, an axis's position, the camera feed and the interlocks are judged when the run is asked.
- <a id="Status.turret"></a>`turret`: `TurretStatus | None` -- The turret's state and slot (`MotionAPI.turret_status`), None on a scope with no turret.

<a id="StepRecord"></a>
## `StepRecord`

A record: it crosses the wire as its data.  
Returned by: [`Protocol.step_records`](#Protocol.step_records).

One step of a protocol, a field per column of its steps table.

**Fields**

- <a id="StepRecord.acquire"></a>`acquire`: `AcquireMode` -- What the step captures.
- <a id="StepRecord.auto_focus"></a>`auto_focus`: `bool` -- The step is autofocused before its capture.
- <a id="StepRecord.auto_gain"></a>`auto_gain`: `bool` -- The camera's auto-gain is on for the step.
- <a id="StepRecord.auto_named"></a>`auto_named`: `bool` -- The name is rendered from the step's fields.
- <a id="StepRecord.channel"></a>`channel`: `str` -- The channel the step captures.
- <a id="StepRecord.channel_group_id"></a>`channel_group_id`: `int | None` -- The steps captured together as one composite share it; None for a step recorded before groups were.
- <a id="StepRecord.custom_step"></a>`custom_step`: `bool` -- The step was made at a position of the caller's, not at a well of the plate.
- <a id="StepRecord.exposure_ms"></a>`exposure_ms`: `float` -- The camera exposure, in ms.
- <a id="StepRecord.false_color"></a>`false_color`: `bool` -- The channel's image is false-coloured.
- <a id="StepRecord.frames_summed"></a>`frames_summed`: `int` -- How many frames the capture sums.
- <a id="StepRecord.gain_db"></a>`gain_db`: `float` -- The camera gain, in dB.
- <a id="StepRecord.illumination_ma"></a>`illumination_ma`: `float` -- The LED current, in mA.
- <a id="StepRecord.label"></a>`label`: `str` -- The caller's label for the step; '' for none.
- <a id="StepRecord.name"></a>`name`: `str` -- The step's name, rendered from its other fields.
- <a id="StepRecord.objective"></a>`objective`: `str` -- The objective's catalogue key.
- <a id="StepRecord.stim_config"></a>`stim_config`: `dict[str, Stimulation] | None` -- Each channel's stimulation, by channel; None where the step carries no stimulation config.
- <a id="StepRecord.tile"></a>`tile`: `str` -- The step's tile in its grid; '' for no tile.
- <a id="StepRecord.tile_group_id"></a>`tile_group_id`: `int | None` -- The tile grid the step is one tile of; None for none.
- <a id="StepRecord.turret_slot"></a>`turret_slot`: `int | None` -- The turret slot the step uses; None for none.
- <a id="StepRecord.video_config"></a>`video_config`: `VideoConfig | None` -- What the step records as video; None where the step carries no video config.
- <a id="StepRecord.well"></a>`well`: `str` -- The plate's label for the well; '' for a step made at a position of the caller's.
- <a id="StepRecord.x_mm"></a>`x_mm`: `float | None` -- X in mm on the protocol's plate; None where the step has no position on X.
- <a id="StepRecord.y_mm"></a>`y_mm`: `float | None` -- Y in mm on the protocol's plate; None as for X.
- <a id="StepRecord.z_slice"></a>`z_slice`: `int | None` -- The step's slice in its z-stack; None for none.
- <a id="StepRecord.z_stack_group_id"></a>`z_stack_group_id`: `int | None` -- The z-stack the step is one slice of; None for none.
- <a id="StepRecord.z_um"></a>`z_um`: `float | None` -- Z in um; None as for X.

<a id="StepStarted"></a>
## `StepStarted`

A record: it crosses the wire as its data.  
Delivered by: [`RunEvents.step_started`](#RunEvents.step_started).

A `step_started` event: the index of the step beginning.

**Fields**

- <a id="StepStarted.step_index"></a>`step_index`: `int` -- (no description)

<a id="Stimulation"></a>
## `Stimulation`

A record: it crosses the wire as its data.  
Held in: [`StepRecord.stim_config`](#StepRecord.stim_config).

One channel's stimulation for a step: on or off, its current, and its pulses.

None where the step's config does not say, as in a file from before a
value was kept.

**Fields**

- <a id="Stimulation.enabled"></a>`enabled`: `bool | None` -- (no description)
- <a id="Stimulation.frequency_hz"></a>`frequency_hz`: `float | None` -- (no description)
- <a id="Stimulation.illumination_ma"></a>`illumination_ma`: `float | None` -- (no description)
- <a id="Stimulation.pulse_count"></a>`pulse_count`: `int | None` -- (no description)
- <a id="Stimulation.pulse_width_ms"></a>`pulse_width_ms`: `float | None` -- (no description)

<a id="Stitch"></a>
## `Stitch`

A record: it crosses the wire as its data.  
Taken by: [`PostProcessingAPI.build`](#PostProcessingAPI.build).

A stitch, as `build` takes it: each tiled scan to one mosaic.

**Fields**

- <a id="Stitch.mode"></a>`mode`: `StitchMode` -- How the tiles are registered.

<a id="Substitution"></a>
## `Substitution`

A record: it crosses the wire as its data.  
Held in: [`BringUpRecord.substitutions`](#BringUpRecord.substitutions).

A saved setting the camera could not take, and what bring-up used instead.

What becomes of the saved value is the setting's own rule, not this
record's: the record says what was saved and what ran. The binning the
camera delivered at is stored in the saved binning's place, with the
frame it belongs to, and so is a frame refitted to the scope's maximum;
the saved image mode stays saved.

**Fields**

- <a id="Substitution.saved"></a>`saved`: `SettingValue` -- (no description)
- <a id="Substitution.setting"></a>`setting`: `str` -- (no description)
- <a id="Substitution.used"></a>`used`: `SettingValue` -- (no description)

<a id="SupportReportSaved"></a>
## `SupportReportSaved`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.make_logs_zip`](#ScopeSession.make_logs_zip), [`ScopeSession.make_support_report`](#ScopeSession.make_support_report).

A support report or logs zip that was saved, and the words that say where.

**Fields**

- <a id="SupportReportSaved.path"></a>`path`: `Path` -- The ZIP.
- <a id="SupportReportSaved.report"></a>`report`: `SupportReportKind` -- Which ZIP it is.

<a id="SupportReportSaved.message"></a>
### `SupportReportSaved.message`

```python
message: str  # property
```

Where the ZIP is and where to send it; the folder is the one it was written to.

**Raises**

Nothing.

<a id="SupportReportSaved.title"></a>
### `SupportReportSaved.title`

```python
title: str  # property
```

The title the saved report's notice is shown under.

**Raises**

Nothing.

<a id="TakenCapture"></a>
## `TakenCapture`

A record: it crosses the wire as its data.  
Held in: [`CaptureTally.dark`](#CaptureTally.dark), [`CaptureTally.saturated`](#CaptureTally.saturated).

One capture a run was asked for that produced an image, and what its frame was.

**Fields**

- <a id="TakenCapture.blown"></a>`blown`: `bool | None` -- Whether the frame's record found it blown (`FrameRecord.blown`); None when nothing judged it: a video step, which has no one frame.
- <a id="TakenCapture.dark"></a>`dark`: `bool | None` -- Whether the frame's record found it dark (`FrameRecord.dark`); None when nothing judged it: no LED on, or a video step.
- <a id="TakenCapture.scan_index"></a>`scan_index`: `int` -- The scan it belonged to, counting from 0.
- <a id="TakenCapture.step_index"></a>`step_index`: `int` -- The step's row in the protocol, counting from 0.
- <a id="TakenCapture.step_name"></a>`step_name`: `str` -- The step's name, as the protocol has it.

<a id="TilingConfig"></a>
## `TilingConfig`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.tiling_config`](#ScopeSession.tiling_config).

The tiling grids an installation offers.

**Fields**

- <a id="TilingConfig.available"></a>`available`: `tuple[str, ...]` -- Each grid's label, as `create_protocol` and `Protocol.apply_tiling` accept it.
- <a id="TilingConfig.default"></a>`default`: `str` -- The grid to preselect, one of `available`.

<a id="TurretChanged"></a>
## `TurretChanged`

A record: it crosses the wire as its data.  
Delivered by: [`MotionAPI.add_turret_listener`](#MotionAPI.add_turret_listener).

A turret event: its state and slot as `turret_status` answers.

**Fields**

- <a id="TurretChanged.at"></a>`at`: `TurretStatus` -- (no description)

<a id="TurretSlotQuestion"></a>
## `TurretSlotQuestion`

A record: it crosses the wire as its data.  
Returned by: [`ScopeSession.turret_slot_questions`](#ScopeSession.turret_slot_questions).

Which turret slot some steps of a protocol mean: what to ask the person.

Returned by `ScopeSession.turret_slot_questions`, one per objective
whose steps cannot say which slot they mean: the objective is in two
or more slots and the steps recorded none (a protocol from before steps
recorded their slot), or the slot a step recorded now holds another
objective. The answer goes back through
`ScopeSession.point_steps_at_turret_slot`; until then a run and going
to one of these steps are refused.

**Fields**

- <a id="TurretSlotQuestion.choices"></a>`choices`: `tuple[int, ...]` -- The slots that hold `objective_id` now, ascending.
- <a id="TurretSlotQuestion.objective_id"></a>`objective_id`: `str` -- The objective the steps name.
- <a id="TurretSlotQuestion.reassigned_step_indexes"></a>`reassigned_step_indexes`: `tuple[int, ...]` -- Those of `step_indexes` whose recorded slot now holds another objective; the others recorded no slot.
- <a id="TurretSlotQuestion.step_indexes"></a>`step_indexes`: `tuple[int, ...]` -- The indexes of the steps the answer points, ascending.

<a id="TurretStatus"></a>
## `TurretStatus`

A record: it crosses the wire as its data.  
Returned by: [`MotionAPI.turret_status`](#MotionAPI.turret_status).  
Held in: [`Status.turret`](#Status.turret), [`TurretChanged.at`](#TurretChanged.at).  
Taken by: [`MotionAPI.add_turret_listener`](#MotionAPI.add_turret_listener), [`MotionAPI.remove_turret_listener`](#MotionAPI.remove_turret_listener).

The turret's state and the slot in the light path, read together.

`slot` is the slot the last turret command left the turret in, None
before the first, while one is in flight, after one failed, and
whenever the turret's position is lost (`MotionAPI.turret_status`). `state` is the
turret's `AxisState`, which the slot alone cannot say: homing, moving
or unknown.

**Fields**

- <a id="TurretStatus.slot"></a>`slot`: `int | None` -- (no description)
- <a id="TurretStatus.state"></a>`state`: `AxisState` -- (no description)

<a id="UiDispatcher"></a>
## `UiDispatcher`

A record: it crosses the wire as its data.  
Taken by: [`ScopeSession.set_ui_dispatcher`](#ScopeSession.set_ui_dispatcher).

How the GUI delivers a callback, and the thread it delivers on.

One value, so the thread can never be stale for the dispatcher beside
it: a wait on that thread for something only a later delivery there can
do would never return, and a run's waits refuse it.

**Fields**

- <a id="UiDispatcher.schedule"></a>`schedule`: `Callable[[Callable[[float], object], float], object]` -- A function with signature (func, timeout) that schedules func on the GUI thread. Typically Clock.schedule_once.
- <a id="UiDispatcher.thread"></a>`thread`: `Thread | None` -- The thread `schedule` delivers on; None for a dispatcher that calls inline on the caller's thread.

<a id="UnfitArgument"></a>
## `UnfitArgument`

A record: it crosses the wire as its data.  
Held in: [`ServerRefusedError.errors`](#ServerRefusedError.errors).

One argument of a request that does not fit its member, as the request reader found it.

**Fields**

- <a id="UnfitArgument.loc"></a>`loc`: `tuple[str | int, ...]` -- (no description)
- <a id="UnfitArgument.msg"></a>`msg`: `str` -- (no description)
- <a id="UnfitArgument.type"></a>`type`: `str` -- (no description)

<a id="UsbDeviceBinding"></a>
## `UsbDeviceBinding`

A record: it crosses the wire as its data.  
Held in: [`UsbIdentityBindings.devices`](#UsbIdentityBindings.devices).

One device of a USB identity and the driver Windows has bound to it.

**Fields**

- <a id="UsbDeviceBinding.date"></a>`date`: `str | None` -- The bound package's date.
- <a id="UsbDeviceBinding.inf"></a>`inf`: `str | None` -- The bound package's published INF (`oem12.inf`).
- <a id="UsbDeviceBinding.instance_id"></a>`instance_id`: `str` -- Windows' instance ID: the identity, a backslash, then the device's serial or port.
- <a id="UsbDeviceBinding.present"></a>`present`: `bool` -- False for a device seen before on another port or at another time, which keeps its binding until it is matched afresh.
- <a id="UsbDeviceBinding.problem_code"></a>`problem_code`: `int | None` -- Device Manager's problem code; None for none.
- <a id="UsbDeviceBinding.provider"></a>`provider`: `str | None` -- The bound package's provider.
- <a id="UsbDeviceBinding.service"></a>`service`: `str | None` -- The bound driver's service (`WinUSB`, `libusbK`, `libusb0`); None when no driver is bound, and the package fields with it.
- <a id="UsbDeviceBinding.version"></a>`version`: `str | None` -- The bound package's version.

<a id="UsbIdentityBindings"></a>
## `UsbIdentityBindings`

A record: it crosses the wire as its data.  
Returned by: [`DiagnosticsAPI.get_camera_usb_bindings`](#DiagnosticsAPI.get_camera_usb_bindings), [`ScopeSession.apply_remedy`](#ScopeSession.apply_remedy), [`ScopeSession.repair_camera_driver`](#ScopeSession.repair_camera_driver).

A USB identity of a Lumascope Classic and each of its devices' drivers.

**Fields**

- <a id="UsbIdentityBindings.devices"></a>`devices`: `tuple[UsbDeviceBinding, ...]` -- Every device of it, present or seen before.
- <a id="UsbIdentityBindings.hardware_id"></a>`hardware_id`: `str` -- `USB\VID_04B4&PID_8613` (the bootloader) or `USB\VID_04B4&PID_EA17` (the running firmware).
- <a id="UsbIdentityBindings.name"></a>`name`: `str` -- Its name in Device Manager.
- <a id="UsbIdentityBindings.needs"></a>`needs`: `tuple[str, ...]` -- The drivers LumaViewPro can use it on, lower case.
- <a id="UsbIdentityBindings.usable"></a>`usable`: `bool` -- False when a present device is on none of them; a device seen before does not count, since it is matched afresh when plugged.

<a id="Video"></a>
## `Video`

A record: it crosses the wire as its data.  
Taken by: [`PostProcessingAPI.build`](#PostProcessingAPI.build).

A video, as `build` takes it: each position's time series to one video.

**Fields**

- <a id="Video.frame_rate_fps"></a>`frame_rate_fps`: `float | str | None` -- The playback rate, a number of at least 1 or its text; None plays a recording at its own measured rate.
- <a id="Video.timestamp_overlay"></a>`timestamp_overlay`: `bool` -- Whether each frame shows its time.

<a id="VideoConfig"></a>
## `VideoConfig`

A record: it crosses the wire as its data.  
Held in: [`StepRecord.video_config`](#StepRecord.video_config).

What a video step records: its length in seconds and its frame rate.

None where the step's config does not say; a step that records no video
may carry neither.

**Fields**

- <a id="VideoConfig.duration_s"></a>`duration_s`: `float | None` -- (no description)
- <a id="VideoConfig.frame_rate_fps"></a>`frame_rate_fps`: `float | None` -- (no description)

<a id="VideoProgress"></a>
## `VideoProgress`

A record: it crosses the wire as its data.  
Held in: [`RunEvents.video_progress`](#RunEvents.video_progress).  
Delivered by: [`RunEvents.video_progress`](#RunEvents.video_progress).

Where a video step is, as its `video_progress` event says.

`phase` is `RECORDING` (`elapsed_s` of `total_s`), `WRITING`
(`progress_pct` of the recorded frames written) or `ENDED`, which follows
once any recording or writing phase on every path that had one; a step
that ends before it records sends no phase and no `ENDED`.

**Fields**

- <a id="VideoProgress.elapsed_s"></a>`elapsed_s`: `float | None` -- (no description)
- <a id="VideoProgress.phase"></a>`phase`: `VideoPhase` -- (no description)
- <a id="VideoProgress.progress_pct"></a>`progress_pct`: `float | None` -- (no description)
- <a id="VideoProgress.total_s"></a>`total_s`: `float | None` -- (no description)

<a id="WorkItem"></a>
## `WorkItem`

A record: it crosses the wire as its data.  
Held in: [`LiveWork.work`](#LiveWork.work).

One piece of work the session is still doing.

**Fields**

- <a id="WorkItem.discardable"></a>`discardable`: `bool` -- True when `ScopeSession.discard_close_drain` drops what this work has left; a host offers that discard only while some item says so.
- <a id="WorkItem.kind"></a>`kind`: `WorkKind` -- Where the work runs.
- <a id="WorkItem.left"></a>`left`: `int | None` -- How many frames, images or builds it still has to finish, when it counts them; None when it does not.
- <a id="WorkItem.name"></a>`name`: `str` -- What it is, in words a person reads.
- <a id="WorkItem.progress_pct"></a>`progress_pct`: `float | None` -- How far a build has got, as its own progress last said; None before it has said, or for work that reports none.

<a id="ZProject"></a>
## `ZProject`

A record: it crosses the wire as its data.  
Taken by: [`PostProcessingAPI.build`](#PostProcessingAPI.build).

A z-projection, as `build` takes it: each Z-stack to one image.

**Fields**

- <a id="ZProject.method"></a>`method`: `ZProjectMethod` -- How each stack's slices are reduced.

<a id="modules.notification_center"></a>
## `modules.notification_center`

Functions of the module; each is in-process, since the wire routes only a class's members.

<a id="modules.notification_center.outcome_of"></a>
### `modules.notification_center.outcome_of`

```python
def outcome_of(exception: BaseException) -> Outcome
```

Marks: in-process.

What *exception* is, as its type says: the one reading of an outcome's kind and words.

**Raises**

Nothing.

<a id="rest.app"></a>
## `rest.app`

Functions of the module; each is in-process, since the wire routes only a class's members.

<a id="rest.app.build_app"></a>
### `rest.app.build_app`

```python
def build_app(session: ScopeSession) -> fastapi.FastAPI
```

Marks: in-process.

The application serving *session*'s wire members under `/api/v1/`.

The OpenAPI description is at `/api/v1/openapi.json` and the
interactive reference at `/docs`. The event stream hears the scope
while the application is served, from its startup to its shutdown.
Its close is `app.state.closing`; its live view `app.state.live`.

**Raises**

- `ConfigurationDefectError`: `'route_shadowed'`, a Session member is named as one of the server's own routes, so a client could not reach it.

**Refusals**

- [`ConfigurationDefectError`](#ConfigurationDefectError): `route_shadowed`

## Closed sets

A closed set crosses the wire as its value, and in Python equals its value's string.

<a id="AcquireMode"></a>
### `AcquireMode`

Returned by: [`ScopeSession.channel_acquire`](#ScopeSession.channel_acquire).  
Held in: [`ChannelSettings.acquire`](#ChannelSettings.acquire), [`StepRecord.acquire`](#StepRecord.acquire).  
Taken by: [`ScopeSession.set_channel_acquire`](#ScopeSession.set_channel_acquire).

What a layer captures: an image or a video.

A layer that captures nothing holds None, not a member. The values are
the ones the settings and a protocol file store, so neither changes.

- `image`
- `video`

<a id="AppRuntime"></a>
### `AppRuntime`

Returned by: [`ScopeSession.app_runtime`](#ScopeSession.app_runtime).

How this process was launched: the one answer every reader takes.

`SOURCE`: a Python interpreter running a checkout or pip install.
`BUNDLE`: a PyInstaller build the installer did not install (a build
run from `dist`). `INSTALLED`: a PyInstaller build in the folder the
installer wrote, its marker beside the exe.

- `source`
- `bundle`
- `installed`

<a id="AutoGainConvergence"></a>
### `AutoGainConvergence`

Held in: [`AutoGainLock.state`](#AutoGainLock.state), [`FrameRecord.auto_gain`](#FrameRecord.auto_gain).

Where a continuous auto-gain arm landed when the capture locked it.

CONVERGED: exposure inside the layer class's usable range.
MAXED: exposure pinned at the class ceiling -- the scene is too dark
    for the range, so target brightness was not reached.
AT_MINIMUM: exposure at or below the class usable floor -- the scene
    is too bright for the range.
FAILED: the camera reported no usable achieved value, so the capture
    ran without an exposure/gain evidence gate.
The limit states are outcomes, not failures: the frame is saved with
the achieved values and a run continues.

- `CONVERGED` (`CONVERGED`)
- `MAXED` (`MAXED`)
- `AT_MINIMUM` (`AT_MINIMUM`)
- `FAILED` (`FAILED`)

<a id="Axis"></a>
### `Axis`

Returned by: [`MotionAPI.axes_without_position`](#MotionAPI.axes_without_position).  
Held in: [`PositionOutOfRangeError.axis`](#PositionOutOfRangeError.axis), [`ScopeCapabilities.axes`](#ScopeCapabilities.axes).  
Taken by: [`DiagnosticsAPI.read_motor_drv_status`](#DiagnosticsAPI.read_motor_drv_status), [`MotionAPI.get_axis_limits`](#MotionAPI.get_axis_limits), [`MotionAPI.get_limit_switch_status`](#MotionAPI.get_limit_switch_status), [`MotionAPI.set_precision_mode`](#MotionAPI.set_precision_mode), [`SimulationAPI.fail_home`](#SimulationAPI.fail_home), [`SimulationAPI.hold_axis`](#SimulationAPI.hold_axis).

An axis a motion member names: a stage axis, or the turret, T.

The turret has a slot, not a position (`TurretStatus`), so a member
that answers a position takes a `StageAxis`.

- `X` (`X`)
- `Y` (`Y`)
- `Z` (`Z`)
- `T` (`T`)

<a id="AxisState"></a>
### `AxisState`

Returned by: [`MotionAPI.axes_without_position`](#MotionAPI.axes_without_position).  
Held in: [`AxisPosition.state`](#AxisPosition.state), [`TurretStatus.state`](#TurretStatus.state).

Possible states for a motion axis.

- `unknown`
- `idle`
- `moving`
- `homing`

<a id="Background"></a>
### `Background`

Held in: [`ChannelRecord.background`](#ChannelRecord.background).

What a layer's field of view looks like, which is not its modality's:
darkfield is transmitted light on a dark field.

- `bright`
- `dark`

<a id="Caller"></a>
### `Caller`

Held in: [`ControlSession.taken_by`](#ControlSession.taken_by), [`ControlSessionRefusedError.holder_trigger`](#ControlSessionRefusedError.holder_trigger), [`DiagnosticRefusedError.holder_trigger`](#DiagnosticRefusedError.holder_trigger), [`HardwareCommandRefusedError.holder_trigger`](#HardwareCommandRefusedError.holder_trigger), [`ProtocolRunRefusedError.holder_trigger`](#ProtocolRunRefusedError.holder_trigger), [`RecordingRefusedError.holder_trigger`](#RecordingRefusedError.holder_trigger).

Which host a call came through.

- `gui`
- `rest`
- `api`

<a id="CameraAutoMode"></a>
### `CameraAutoMode`

Returned by: [`CameraAPI.auto_exposure`](#CameraAPI.auto_exposure), [`CameraAPI.auto_gain`](#CameraAPI.auto_gain), [`CameraAPI.read_auto_exposure`](#CameraAPI.read_auto_exposure), [`CameraAPI.read_auto_gain`](#CameraAPI.read_auto_gain).  
Taken by: [`CameraAPI.set_auto_exposure`](#CameraAPI.set_auto_exposure).

What one of the camera's auto loops is doing, as the camera reports it.

ONCE: a one-shot the camera is still running; it turns itself Off when
    done, so it is the one mode that changes without a write.

- `Off` (`OFF`)
- `Once` (`ONCE`)
- `Continuous` (`CONTINUOUS`)

<a id="CameraReadAnswer"></a>
### `CameraReadAnswer`

Taken by: [`SimulationAPI.refuse_camera_read`](#SimulationAPI.refuse_camera_read).

How a failed camera read fails.

- `failed`
- `removed`

<a id="CameraReadNode"></a>
### `CameraReadNode`

Taken by: [`SimulationAPI.refuse_camera_read`](#SimulationAPI.refuse_camera_read).

A camera setting whose read a simulated fault can fail.

- `gain`
- `exposure`
- `frame_size`
- `binning`
- `pixel_format`
- `black_level`
- `frame_rate`
- `minimum_exposure`
- `auto_gain`
- `auto_exposure`

<a id="CameraShutter"></a>
### `CameraShutter`

Held in: [`ScopeCapabilities.camera_shutter`](#ScopeCapabilities.camera_shutter).

How a camera's sensor exposes a frame, as its datasheet entry says.

- `rolling`
- `global`

<a id="CameraWriteAnswer"></a>
### `CameraWriteAnswer`

Taken by: [`SimulationAPI.refuse_camera_write`](#SimulationAPI.refuse_camera_write).

What a refused camera write answers.

- `false`
- `none`
- `true`
- `raise`
- `removed`
- `no_answer`

<a id="CameraWriteNode"></a>
### `CameraWriteNode`

Taken by: [`SimulationAPI.refuse_camera_write`](#SimulationAPI.refuse_camera_write).

A camera setting whose write a simulated fault can refuse.

- `gain`
- `exposure`
- `auto_exposure`
- `frame_size`
- `binning`
- `pixel_format`
- `black_level`
- `auto_gain`
- `auto_gain_once`
- `auto_gain_target_brightness`

<a id="Counter"></a>
### `Counter`

Held in: [`Credit.by`](#Credit.by).

Who counted a frame toward a change: the capture's drain, a grab
(a capture's own, or an ungated `get_image`), or the live preview.

A capture's wait can be met by frames another consumer counted, so the
record of a wait names who counted each frame.

- `drain`
- `grab`
- `preview`

<a id="DisconnectPart"></a>
### `DisconnectPart`

Held in: [`ScopeDisconnectError.part_reasons`](#ScopeDisconnectError.part_reasons), [`ScopeDisconnectError.parts`](#ScopeDisconnectError.parts).

A part of the microscope `Lumascope.disconnect` shuts down, in teardown order.

- `motor stop` (`MOTOR_STOP`)
- `LED board` (`LED_BOARD`)
- `motor board` (`MOTOR_BOARD`)
- `camera`

<a id="EntryKind"></a>
### `EntryKind`

Held in: [`LiveFolderEntry.kind`](#LiveFolderEntry.kind).

What a live-folder listing's entry is.

- `file`
- `folder`

<a id="FilesOutcome"></a>
### `FilesOutcome`

Held in: [`FilesWritten.files`](#FilesWritten.files), [`RunEvents.files_written`](#RunEvents.files_written), [`RunFiles.outcome`](#RunFiles.outcome).

What became of a run's images once its writes are done.

`WRITTEN` when every image the run captured is on disk;
`INCOMPLETE` when some are not -- a write that failed, or writes
given up on by a writer recovery or a shutdown.

- `written`
- `incomplete`

<a id="FrameValiditySource"></a>
### `FrameValiditySource`

Returned by: [`ImagingAPI.recorded_changes`](#ImagingAPI.recorded_changes).  
Held in: [`CaptureFailedError.source`](#CaptureFailedError.source), [`Change.source`](#Change.source), [`ScopeCapabilities.frame_rules`](#ScopeCapabilities.frame_rules).  
Taken by: [`ImagingAPI.capture_and_wait`](#ImagingAPI.capture_and_wait), [`ImagingAPI.frames_until_valid`](#ImagingAPI.frames_until_valid), [`ScopeSession.capture_still`](#ScopeSession.capture_still).

What changed, for a capture to wait out: each a key of `FrameValidity.SKIP_FRAMES`.

- `led`
- `gain`
- `exposure`
- `xy_move`
- `z_move`
- `turret`
- `auto_gain`
- `pixel_format`
- `frame_size`
- `binning`
- `black_level`
- `conversion_gain_mode`
- `line_noise_reduction`

<a id="Holder"></a>
### `Holder`

Held in: [`ControlSessionRefusedError.holder`](#ControlSessionRefusedError.holder), [`DiagnosticRefusedError.holder`](#DiagnosticRefusedError.holder), [`HardwareCommandRefusedError.holder`](#HardwareCommandRefusedError.holder), [`ProtocolRunRefusedError.holder`](#ProtocolRunRefusedError.holder), [`RecordingRefusedError.holder`](#RecordingRefusedError.holder).

What holds the microscope when a refusal names its holder.

- `diagnostic`
- `home`
- `protocol`
- `recording`
- `autofocus`

<a id="HomeFault"></a>
### `HomeFault`

Taken by: [`SimulationAPI.fail_home`](#SimulationAPI.fail_home).

How a home fails, as the EL-0940 firmware's simulated hardware takes it.

- `switch_never_trips`
- `stall`
- `absent`

<a id="HomeTarget"></a>
### `HomeTarget`

Taken by: [`MotionAPI.home`](#MotionAPI.home), [`MotionAPI.start_home`](#MotionAPI.start_home), [`ScopeSession.run_application_session`](#ScopeSession.run_application_session), [`ScopeSession.start_application_session`](#ScopeSession.start_application_session).

What a home takes: Z, the turret, or every axis the board has.

X and Y home only with the rest.

- `Z` (`Z`)
- `T` (`T`)
- `ALL` (`ALL`)

<a id="IdentitySource"></a>
### `IdentitySource`

Held in: [`ChannelIdentity.source`](#ChannelIdentity.source).

Which rung of the identity resolver answered.

- `motorconfig`
- `scopes`
- `unresolved`

<a id="ImageMode"></a>
### `ImageMode`

Returned by: [`CameraAPI.image_mode`](#CameraAPI.image_mode).  
Taken by: [`CameraAPI.set_image_mode`](#CameraAPI.set_image_mode).

The image mode captures are taken and saved in (the module docstring).

- `8bit` (`EIGHT_BIT`)
- `12bit_scientific` (`SCIENTIFIC`)
- `12bit_scaled` (`SCALED`)
- `12bit_false_color_rgb` (`FALSE_COLOR_RGB`)

<a id="Interlock"></a>
### `Interlock`

Returned by: [`MotionAPI.interlocks`](#MotionAPI.interlocks).

A stage hardware interlock that, open, stops the stage.

Its value is also the reason the driver's fault carries when the
interlock stops a command, which stays a string, as every reason does;
a command it stopped before anything moved is refused with its
`REFUSED_BEFORE_MOTION` reason.

- `lid_open`
- `stage_unpowered`

<a id="LedBoardFamily"></a>
### `LedBoardFamily`

Held in: [`ScopeModel.led_board`](#ScopeModel.led_board).

The kind of board a model's LEDs are on, as its catalogue row names it.

An EL-0940 is an LED board of its own; an FX2 drives its scope's camera
and LEDs both.

- `EL-0940` (`EL_0940`)
- `FX2` (`FX2`)

<a id="LedCommandSet"></a>
### `LedCommandSet`

Held in: [`LedBoardInfo.command_set`](#LedBoardInfo.command_set).

The LED board's text command set, as `get_led_info().command_set` names it.

INFO predates the v2 firmware; SELFTEST, I2CSCAN and the LEDREAD
current reads came with it. A board with no text command channel at
all (the FX2's LED peripheral) answers None.

- `legacy`
- `v2`

<a id="LedState"></a>
### `LedState`

Held in: [`LedReading.state`](#LedReading.state).  
Taken by: [`IlluminationAPI.add_led_listener`](#IlluminationAPI.add_led_listener), [`IlluminationAPI.remove_led_listener`](#IlluminationAPI.remove_led_listener).

What an LED address is doing, as far as the API can vouch for it.

- `lit`
- `dark`
- `unknown`

<a id="LightTiming"></a>
### `LightTiming`

Held in: [`FrameRecord.illumination_timing`](#FrameRecord.illumination_timing).

When a record's LED currents were read.

BEFORE_GRAB: before the grab that took the frame (a still): the LEDs
    lit for it.
ON_ARRIVAL: as the frame arrived from the camera (a recorded frame):
    the LEDs lit then, which is after the frame's exposure.

- `before_grab`
- `on_arrival`

<a id="Modality"></a>
### `Modality`

Held in: [`ChannelRecord.modality`](#ChannelRecord.modality).

The kind of light a layer images.

- `transmitted`
- `fluorescence`
- `luminescence`

<a id="MotorBoardFamily"></a>
### `MotorBoardFamily`

Held in: [`ScopeModel.motor_board`](#ScopeModel.motor_board).

The kind of board a model's motor axes are on, as its catalogue row names it.

- `EL-0940` (`EL_0940`)
- `TMCM-6110` (`TMCM_6110`)

<a id="MotorCommandSet"></a>
### `MotorCommandSet`

Held in: [`MotorBoardInfo.command_set`](#MotorBoardInfo.command_set).

The motor board's text command set, as `get_motor_info().command_set` names it.

The EL-0940's (INFO, FULLINFO, ACTUAL_R, SPI and the rest). A board
with no text command channel (the TMCM-6110, which speaks binary
datagrams), or no board connected, answers None.

- `text`

<a id="OutcomeKind"></a>
### `OutcomeKind`

Held in: [`Notification.kind`](#Notification.kind), [`Outcome.kind`](#Outcome.kind).

What an outcome is, read from its type by the reporter.

A string enum, so the value crosses a wire as itself. `UNCLASSIFIED`
is a post made straight to the centre rather than through the
reporter: its kind was never declared, so the record says so rather
than guessing from its severity. `QUIET` is never delivered to a
listener, since a quiet outcome is never shown; it is the kind a caller
that receives one as its answer reads.

- `refusal`
- `fault`
- `notice`
- `quiet`
- `unclassified`

<a id="OutputFormat"></a>
### `OutputFormat`

Held in: [`SavedStill.output_format`](#SavedStill.output_format).

The formats a capture is saved in.

The values are spelled as the settings files hold them verbatim -- none
can be renamed without a settings migration; the selectors show their
own labels. `HYPERSTACK` swaps per-step TIFF writes for a post-run
per-well stack build.

- `TIFF` (`TIFF`)
- `OME-TIFF` (`OME_TIFF`)
- `JPG` (`JPG`)
- `OME-TIFF Hyperstack` (`HYPERSTACK`)

<a id="Part"></a>
### `Part`

Returned by: [`Lumascope.unconnected_parts`](#Lumascope.unconnected_parts).  
Held in: [`PartStatus.part`](#PartStatus.part).  
Taken by: [`SimulationAPI.delay_reply`](#SimulationAPI.delay_reply), [`SimulationAPI.drop_replies`](#SimulationAPI.drop_replies), [`SimulationAPI.reboot`](#SimulationAPI.reboot), [`SimulationAPI.replug`](#SimulationAPI.replug), [`SimulationAPI.silence`](#SimulationAPI.silence), [`SimulationAPI.unplug`](#SimulationAPI.unplug), [`SimulationAPI.unsilence`](#SimulationAPI.unsilence).

The parts a scope is made of, in the order the record lists them.

- `motor`
- `led`
- `camera`

<a id="PostProcessingOperation"></a>
### `PostProcessingOperation`

Held in: [`BuildStep.operation`](#BuildStep.operation), [`ExcludedInput.kind`](#ExcludedInput.kind), [`SkippedBuild.operation`](#SkippedBuild.operation).

Every post-processing operation, by the name a person reads.

A post-processing outcome's title is keyed by the operation and its
reason, so the operations are a closed set: an operation outside it fails
where its outcome is built.

- `Composite` (`COMPOSITE`)
- `Stitch` (`STITCH`)
- `Z-Projection` (`Z_PROJECTION`)
- `Video` (`VIDEO`)
- `Hyperstack` (`HYPERSTACK`)
- `Cell Count` (`CELL_COUNT`)
- `Graphing` (`GRAPHING`)
- `Trendline` (`TRENDLINE`)
- `Quick Enhance` (`QUICK_ENHANCE`)
- `Post-Processing` (`POST_PROCESSING`)

<a id="RefusalCause"></a>
### `RefusalCause`

Held in: [`Outcome.cause`](#Outcome.cause).

Whether a refused request could succeed later, and what must change for it to.

Declared on the refusal's type, per reason where one type mixes them, so
a server answers from the declaration (REST: 409 for `STATE`, 422 for
`REQUEST`, 503 for `RESTART`) and never from the exception's base: a
`ValueError` can be refused for the scope's state, as a live folder on
an unplugged drive is. A string enum, so the value crosses a wire as
itself.

**Attributes**

- `STATE`: The scope's state refused it -- a run is live, an axis is not homed, a drive is unplugged -- and the same request may succeed later.
- `REQUEST`: As sent, the request cannot succeed on this scope: a bad argument, a value outside a fixed rule or this hardware's range, hardware this scope does not have, a handle or a control session that no longer names anything live, a file already at the name. The caller changes the request.
- `RESTART`: Nothing the caller changes makes it succeed while this process runs -- the scope is disconnected, the motion monitor has ended, the session or the server is closing. Do not retry: the application must be restarted.

- `state`
- `request`
- `restart`

<a id="RemedyName"></a>
### `RemedyName`

Held in: [`Remedy.member`](#Remedy.member).

The Session members a refusal can name as its remedy (`ScopeSession.apply_remedy`).

- `recover_file_writer`
- `repair_camera_driver`

<a id="Rule"></a>
### `Rule`

Held in: [`FrameRule.rule`](#FrameRule.rule).

How a frame is judged against a change of one source.

- `clock`
- `credit`

<a id="RunStatus"></a>
### `RunStatus`

Held in: [`RunOutcome.status`](#RunOutcome.status).

How a run ended, in five words.

`INCOMPLETE` is a run that reached its end without every capture it
was asked for; `ABORTED` is an ending someone asked for (a Stop, or
the application tearing the run down); `FAILED` is one the
instrument imposed, including the strike ceiling for scans that kept
failing; `FAILED_AT_START` is a run that could not begin after it
committed.

- `completed`
- `incomplete`
- `aborted`
- `failed`
- `failed_at_start`

<a id="SettingSource"></a>
### `SettingSource`

Held in: [`FrameRecord.exposure_source`](#FrameRecord.exposure_source), [`FrameRecord.gain_source`](#FrameRecord.gain_source).

Where a record's exposure or gain came from.

FRAME: the frame's own chunk value, stamped by the camera on the frame
    (Pylon ace 2 / dart, the simulated camera).
READ: a read of the camera's setting beside the grab, for a frame that
    carries no chunk value (IDS, FX2, a Basler frame whose chunk was not
    delivered): the setting in force, not a measurement of the frame.

- `frame`
- `read`

<a id="Severity"></a>
### `Severity`

Held in: [`Notification.severity`](#Notification.severity).

Notification severity, by its name; its logging level is `level`.

NOTICE sits between INFO and WARNING: user-facing status (start/done
of a long unattended operation) that says so without misdeclaring
itself as a fault. WARNING stays "something didn't work". The level
says how a post is logged, not whether it is shown: `shown` on the
notification says that.

Two severities are ordered by their levels, never by their names:
`<`, `<=`, `>` and `>=` raise, so an order read from the
strings cannot be written.

- `DEBUG` (`DEBUG`)
- `INFO` (`INFO`)
- `NOTICE` (`NOTICE`)
- `WARNING` (`WARNING`)
- `ERROR` (`ERROR`)
- `CRITICAL` (`CRITICAL`)

<a id="StageAxis"></a>
### `StageAxis`

Returned by: [`MotionAPI.axis_positions`](#MotionAPI.axis_positions), [`MotionAPI.get_target_positions`](#MotionAPI.get_target_positions), [`MotionAPI.positions_to_keep`](#MotionAPI.positions_to_keep).  
Held in: [`PositionChanged.axis`](#PositionChanged.axis), [`Status.axes`](#Status.axes).  
Taken by: [`MotionAPI.add_position_listener`](#MotionAPI.add_position_listener), [`MotionAPI.get_actual_position`](#MotionAPI.get_actual_position), [`MotionAPI.jog_step_um`](#MotionAPI.jog_step_um), [`MotionAPI.positions_to_keep`](#MotionAPI.positions_to_keep), [`MotionAPI.remove_position_listener`](#MotionAPI.remove_position_listener), [`ScopeSession.go_to_bookmark`](#ScopeSession.go_to_bookmark), [`ScopeSession.save_bookmark`](#ScopeSession.save_bookmark), [`ScopeSession.start_go_to_bookmark`](#ScopeSession.start_go_to_bookmark).

A stage axis, X, Y or Z: an axis with a position in um.

The axes a position, a jog, a bookmark and a position event name. A
name's spelling is exact: `'x'` is not `'X'`, so each axis has one
spelling in every log and on every wire. Whether this scope has the
axis is `ScopeCapabilities.axes`.

- `X` (`X`)
- `Y` (`Y`)
- `Z` (`Z`)

<a id="StitchAlgorithm"></a>
### `StitchAlgorithm`

Held in: [`DegradedOutput.algorithm`](#DegradedOutput.algorithm), [`DegradedOutput.fallback_from`](#DegradedOutput.fallback_from).

The algorithm that made a mosaic, from the stitch's fallback chain.

A mode tries its registration first (`FAST_FFT_PHASE` for a fast
preview, `QUALITY_LOCAL_NCC` for quality) where the tiles overlap,
then places them by their recorded stage positions
(`STAGE_POSITION`), then by their rank in X and Y
(`SIMPLE_POSITION`). The values are the ones a mosaic's file records.

- `fast_fft_phase`
- `quality_local_ncc`
- `stage_position_stitcher` (`STAGE_POSITION`)
- `simple_position_stitcher` (`SIMPLE_POSITION`)

<a id="StitchMode"></a>
### `StitchMode`

Returned by: [`PostProcessingAPI.stitch_modes`](#PostProcessingAPI.stitch_modes).  
Held in: [`Stitch.mode`](#Stitch.mode).

How a stitch registers its tiles.

`QUALITY` registers by bounded local NCC; `FAST_PREVIEW` by bounded
FFT phase correlation. Both place the tiles from the recorded stage
geometry and keep the source pixels. A fast preview's file name carries
the `FastPreview` token; a quality stitch's carries none.

- `quality`
- `fast_preview`

<a id="SupportReportKind"></a>
### `SupportReportKind`

Held in: [`SupportReportSaved.report`](#SupportReportSaved.report).

Which ZIP a support request saved: the full support report, or the logs alone.

- `support report` (`SUPPORT_REPORT`)
- `logs zip` (`LOGS_ZIP`)

<a id="TimestampSource"></a>
### `TimestampSource`

Held in: [`FrameTime.source`](#FrameTime.source).

What a record's time is.

HOST_WALLCLOCK: the host's wall clock read after the grab returned the
    frame (a still).
HOST_ARRIVAL: the host's wall clock when the frame arrived from the
    camera (a recorded frame from a camera that stamps no ticks).
CAMERA_REBASED: the camera's own clock for the frame, offset onto the
    host's wall clock at the recording's first frame that carried it.

- `host_wallclock`
- `host_arrival`
- `camera_rebased`

<a id="Trendline"></a>
### `Trendline`

Held in: [`GraphOptions.trendlines`](#GraphOptions.trendlines).  
Taken by: [`PostProcessingAPI.draw_graph`](#PostProcessingAPI.draw_graph), [`PostProcessingAPI.graph`](#PostProcessingAPI.graph).

The kind of curve a graph's trendline is fitted as.

- `Linear` (`LINEAR`)
- `Quadratic` (`QUADRATIC`)
- `Exponential` (`EXPONENTIAL`)
- `Power` (`POWER`)
- `Logarithmic` (`LOGARITHMIC`)

<a id="VideoPhase"></a>
### `VideoPhase`

Held in: [`VideoProgress.phase`](#VideoProgress.phase).

Which part of a video step a `VideoProgress` reports.

- `recording`
- `writing`
- `ended`

<a id="WorkKind"></a>
### `WorkKind`

Held in: [`WorkItem.kind`](#WorkItem.kind).

The kinds of work a session reports, one per place work runs.

A run, a recording, a home and a diagnostic are the claim's own kinds,
so a holder's kind is its item's kind.

- `protocol`
- `recording`
- `home`
- `diagnostic`
- `recording_finish`
- `run_video_finish`
- `run_files`
- `post_processing`
- `post_processing_queued`
- `support_report`
- `logs_zip`
- `plugin_run`
- `still`

<a id="ZProjectMethod"></a>
### `ZProjectMethod`

Returned by: [`PostProcessingAPI.projection_methods`](#PostProcessingAPI.projection_methods).  
Held in: [`ZProject.method`](#ZProject.method).

How a z-stack's slices are reduced to one image, pixel by pixel.

Each value is its name, the word a caller sends. A projection's file
name and derivation record carry the name lower-cased
(`zproj_max`), so neither depends on these values.

- `Min` (`Min`)
- `Max` (`Max`)
- `Average` (`Average`)
- `Median` (`Median`)
- `Sum` (`Sum`)
- `StdDev` (`StdDev`)

<a id="ZStackReference"></a>
### `ZStackReference`

Taken by: [`ScopeSession.apply_zstacking`](#ScopeSession.apply_zstacking).

Where a stack sits against the Z it is built around: its top, its
centre or its bottom at that Z. The stored `zstack.position` holds the
value; the GUI shows its own label for each.

- `top`
- `center`
- `bottom`

## Outcomes

Every outcome class: its kind, its reasons, whether the same request could succeed later (each reason's cause), and the HTTP status REST answers it with.

<a id="AccelerationLimitRefusedError"></a>
### `AccelerationLimitRefusedError`

A refusal; bases `Refusal`, `ValueError`.

An acceleration limit no board may be given was refused; nothing was commanded or stored.

A refusal, not a fault: the person typed or a caller passed a number
outside the range, and the words tell them the range. Raised as a bare
ValueError it was reported as an operation that failed, with a
traceback, over "Check the main log for details". Subclasses ValueError
so callers that catch a bad argument keep working.

**Attributes**

- `reason`: `'acceleration_out_of_range'`.
- `value`: What was given, as given.
- `limits`: The lowest and highest limit a board takes, in percent.

`limits` is published, so a client corrects the value without reading
the words.

**Fields**

- <a id="AccelerationLimitRefusedError.limits"></a>`limits`: `tuple[int, int]` -- The lowest and highest limit a board takes, in percent.

| Reason | Cause | REST status |
|---|---|---|
| `acceleration_out_of_range` | request | 422 |

<a id="AlreadyStartedError"></a>
### `AlreadyStartedError`

A refusal; bases `Refusal`, `RuntimeError`.

A start a session makes once was asked of a session that has made it.

A host's lifecycle call, refused before anything is started twice: a
second plugin load would register every plugin again, and a second
metrics start would orphan the first schedule as untracked,
forever-ticking events. Still a `RuntimeError`, as it was documented.

**Attributes**

- `reason`: `'plugins_loaded'` -- this session already loaded its plugins; it never loads them again. `'metrics_running'` -- metrics are already running; `disable_metrics` lets them start again.

| Reason | Cause | REST status |
|---|---|---|
| `metrics_running` | state | 409 |
| `plugins_loaded` | request | 422 |

<a id="ArgumentRefusedError"></a>
### `ArgumentRefusedError`

A refusal; bases `Refusal`, `ValueError`.

An argument the scope cannot act on: it breaks a fixed rule, whatever the scope's state.

Raised by the member that owns the rule, before anything is commanded
or stored, so every caller -- REST, the SDK, the GUI, a run's own
engine -- is refused alike. A `ValueError`, as the bare raises it
replaces were, so a caller catching a bad argument keeps working.

**Attributes**

- `reason`: One of: - `'not_offered'` -- a value of a closed set (a `StrEnum`) that is not one of `offered`: not a member, nor a member's plain string, or a member the taker does not take. Values are exact: `'off'` is not `'Off'`. - `'not_a_number'` -- a number was needed and the value is not a finite one: NaN, an infinity, a `bool` or not a number at all. Every comparison with NaN is False, so a range check alone passes it. - `'plate_frame_axis'` -- a plate coordinate for an axis the plate frame does not define (`offered`: X and Y). - `'well_unknown'` -- a well label the selected plate (`plate`) does not have. - `'out_of_range'` -- a number outside the `limits` its member takes. - `'fan_duty_out_of_range'` -- a fan duty outside 0 to 100 percent. - `'channel_unknown'` -- the name is not one of this release's channels (`offered`, the catalogue). Names are exact: `'bf'` is not `'BF'`. A channel this model lacks is a name, and is refused as absent hardware, not here. - `'no_channel_selected'` -- the member needs a channel and was given None (`offered`, the catalogue). - `'led_address_unknown'` -- an LED address that is not one of the attached LED board's addresses (`offered`). A `bool` is not an address. - `'illumination_out_of_range'` -- an LED current outside `limits`, 0 to the attached LED board's maximum, in mA. - `'missing_key'` -- a dictionary argument lacks a key its member reads; `argument` names the key's path. - `'wrong_kind'` -- an argument, or a value in a dictionary argument, is not of the kind its member reads (`takes`, in words). - `'protocol_source_ambiguous'` -- not exactly one of the protocol sources `offered` was given. - `'not_a_setting'` -- a dotted path that names no setting. The same fact a write is refused for as `SettingRefusedError('not_a_setting')`; this type says a read asked it. - `'has_member'` -- a dotted path to a setting that has a read of its own, named in `member` by its path from the session, so one setting has one read. - `'needs_simulated_scope'` -- an argument that exists only on a simulated scope, given for a real one. - `'needs_simulated_camera'` -- an argument that exists only on the simulated camera, given for a model simulated with an FX2. - `'exposure_not_recorded'` -- a frame the simulated camera holds no exposure record for: one it did not deliver, or one older than the records it keeps. - `'not_an_installation'` -- a folder that holds neither settings file, so it is not an LVP installation root. - `'binning_size_invalid'` -- a binning label that is not a positive square `NxN`. - `'fault_unknown'` -- a simulated fault's board, or a device's hardware ID, that is not one of `offered`: sets this scope's boards and this machine's devices decide. - `'fault_bound'` -- a simulated fault's count, fraction or time outside what it takes (`takes`, in words). - `'not_injectable_on_board'` -- a simulated fault the board that would present it cannot produce; `argument` names the `scope.simulation` member, `board` the board. - `'argument_not_on_board'` -- an argument of a simulated fault the board cannot take; `board` names the board, `takes` what it takes for that argument. - `'slot_not_holding_objective'` -- a turret slot given for steps naming an objective the slot does not hold now (`offered`: the slots that hold it). - `'objective_in_several_slots'` -- an objective to turn the turret to that two or more slots hold (`offered`: the slots), so no one slot is meant. - `'objective_not_on_turret'` -- an objective to turn the turret to that no slot holds (`offered`: the objectives the slots hold, in slot order). - `'slot_required'` -- an objective confirmed on a turret scope with no slot named; the confirmation records the glass in a slot. - `'gain_range_inverted'` -- an auto-gain settings dictionary whose `min_gain_db` is above its `max_gain_db`; `value` holds the two. - `'key_unknown'` -- a key a dictionary argument does not take (`offered`: the keys it takes).
- `argument`: The name of the argument refused.
- `value`: What was given, as given.
- `offered`: The values the argument takes, where it takes a list; None otherwise.
- `limits`: The lowest and highest value the argument takes, where it takes a range that depends on the scope; None otherwise.
- `takes`: The value the argument takes, in words, for `'wrong_kind'`, `'fault_bound'` and `'argument_not_on_board'`; None otherwise.
- `board`: The simulated board, in words, for `'not_injectable_on_board'` and `'argument_not_on_board'`; None otherwise.
- `plate`: The name of the plate looked on, for `'well_unknown'`; None otherwise.
- `member`: The member to read instead, for `'has_member'`; None otherwise.

`argument`, `offered`, `limits`, `takes`, `board`, `plate`
and `member` are published, so a REST problem carries them beside the
words; the client already holds the value it sent.

**Fields**

- <a id="ArgumentRefusedError.argument"></a>`argument`: `str` -- The name of the argument refused.
- <a id="ArgumentRefusedError.board"></a>`board`: `str | None` -- The simulated board, in words, for `'not_injectable_on_board'` and `'argument_not_on_board'`; None otherwise.
- <a id="ArgumentRefusedError.limits"></a>`limits`: `tuple[float, float] | None` -- The lowest and highest value the argument takes, where it takes a range that depends on the scope; None otherwise.
- <a id="ArgumentRefusedError.member"></a>`member`: `str | None` -- The member to read instead, for `'has_member'`; None otherwise.
- <a id="ArgumentRefusedError.offered"></a>`offered`: `tuple[str | int, ...] | None` -- The values the argument takes, where it takes a list; None otherwise.
- <a id="ArgumentRefusedError.plate"></a>`plate`: `str | None` -- The name of the plate looked on, for `'well_unknown'`; None otherwise.
- <a id="ArgumentRefusedError.takes"></a>`takes`: `str | None` -- The value the argument takes, in words, for `'wrong_kind'`, `'fault_bound'` and `'argument_not_on_board'`; None otherwise.

| Reason | Cause | REST status |
|---|---|---|
| `argument_not_on_board` | request | 422 |
| `binning_size_invalid` | request | 422 |
| `channel_unknown` | request | 422 |
| `exposure_not_recorded` | request | 422 |
| `fan_duty_out_of_range` | request | 422 |
| `fault_bound` | request | 422 |
| `fault_unknown` | request | 422 |
| `gain_range_inverted` | request | 422 |
| `has_member` | request | 422 |
| `illumination_out_of_range` | request | 422 |
| `key_unknown` | request | 422 |
| `led_address_unknown` | request | 422 |
| `missing_key` | request | 422 |
| `needs_simulated_camera` | request | 422 |
| `needs_simulated_scope` | request | 422 |
| `no_channel_selected` | request | 422 |
| `not_a_number` | request | 422 |
| `not_a_setting` | request | 422 |
| `not_an_installation` | request | 422 |
| `not_injectable_on_board` | request | 422 |
| `not_offered` | request | 422 |
| `objective_in_several_slots` | request | 422 |
| `objective_not_on_turret` | request | 422 |
| `out_of_range` | request | 422 |
| `plate_frame_axis` | request | 422 |
| `protocol_source_ambiguous` | request | 422 |
| `slot_not_holding_objective` | request | 422 |
| `slot_required` | request | 422 |
| `well_unknown` | request | 422 |
| `wrong_kind` | request | 422 |

<a id="ArgumentTypeRefusedError"></a>
### `ArgumentTypeRefusedError`

A refusal; bases `Refusal`, `TypeError`.

An argument whose type its member does not take, refused before the member runs.

Raised by the `@api` door for every caller alike -- the GUI, a
script, the engineering plugin, a member calling another -- from the
member's own annotation, so the body never meets a value of another
type. A `TypeError`, as Python raises for a call it cannot take. The
door judges the type only: a number's range and finiteness, a name's
membership and a dictionary's contents stay their member's.

**Attributes**

- `reason`: `'wrong_argument_type'`.
- `member`: The member called, as `Class.member`.
- `argument`: The name of the argument refused.
- `declared`: The type the member takes, as its annotation reads.
- `value`: What was given, as given.

**Fields**

- <a id="ArgumentTypeRefusedError.argument"></a>`argument`: `str` -- The name of the argument refused.
- <a id="ArgumentTypeRefusedError.declared"></a>`declared`: `str` -- The type the member takes, as its annotation reads.
- <a id="ArgumentTypeRefusedError.member"></a>`member`: `str` -- The member called, as `Class.member`.

| Reason | Cause | REST status |
|---|---|---|
| `wrong_argument_type` | request | 422 |

<a id="AutoGainNotSettledError"></a>
### `AutoGainNotSettledError`

A fault; bases `Fault`, `CaptureError`.

Auto-gain locked with no usable exposure or gain from the camera.

The previous settings were kept, so a capture taken with it had no
exposure check.

| Reason | Cause | REST status |
|---|---|---|
| `auto_gain_not_settled` | -- | 500 |

<a id="AutofocusAborted"></a>
### `AutofocusAborted`

A subject: never raised, its children are; bases `Exception`.

An autofocus sweep cut short before it chose a focus: the subject, never raised itself.

`except AutofocusAborted` catches every way a sweep is cut short; the
child raised says which, and its kind with it.

<a id="AutofocusFailedError"></a>
### `AutofocusFailedError`

A fault; bases `Fault`, `CaptureError`.

An autofocus sweep chose no focus; the stage goes back to where it started.

**Attributes**

- `reason`: `'flat_focus_curve'` -- the final pass's curve has no peak: every score zero or invalid, a fit that cannot be made, or a curve with no contrast (every point above half its maximum) whose fit is not concave, peaks outside the measured range, or shifts past twice the step spacing. `'travel_exhausted'` -- a window the sweep reached after it moved the stage reaches past Z's travel; refused before the stage moves there. `'unexpected_error'` -- the sweep raised; its cause is chained.

| Reason | Cause | REST status |
|---|---|---|
| `flat_focus_curve` | -- | 500 |
| `travel_exhausted` | -- | 500 |
| `unexpected_error` | -- | 500 |

<a id="AutofocusLedsHeldError"></a>
### `AutofocusLedsHeldError`

A refusal; bases `Refusal`, `AutofocusAborted`.

The sweep's run is live, but another lease on top of the run's holds the LEDs.

Without them the sweep would scan an unlit field and commit a wrong Z,
so it does not start. The run goes on and captures at its fallback Z.

**Attributes**

- `reason`: `'leds_held'`.
- `holder`: The purpose of the lease that holds the LEDs.

| Reason | Cause | REST status |
|---|---|---|
| `leds_held` | state | 409 |

<a id="AutofocusOutOfTravelError"></a>
### `AutofocusOutOfTravelError`

A refusal; bases `Refusal`, `CaptureError`.

An autofocus whose first window reaches past Z's travel, refused before the stage moves.

The window is centred on where Z stands, so the same autofocus started
further from the travel limit runs. A window the sweep reaches later,
after the stage has moved, is `AutofocusFailedError('travel_exhausted')`.

**Attributes**

- `reason`: `'out_of_travel'`.

| Reason | Cause | REST status |
|---|---|---|
| `out_of_travel` | state | 409 |

<a id="AutofocusRunEnded"></a>
### `AutofocusRunEnded`

A quiet outcome; bases `Quiet`, `AutofocusAborted`.

The run a sweep belongs to had ended when the sweep went to take its LEDs.

Recorded, never shown: whatever holds the LEDs now is not that run, and
the run's own end is the report.

**Attributes**

- `reason`: `'sweep_run_ended'`.

| Reason | Cause | REST status |
|---|---|---|
| `sweep_run_ended` | state | 409 |

<a id="AutofocusStopped"></a>
### `AutofocusStopped`

A quiet outcome; bases `Quiet`, `AutofocusAborted`.

The sweep was stopped by its caller: a Stop, the run's cleanup, the app closing.

Recorded, never shown: the run's own end is the report.

**Attributes**

- `reason`: `'sweep_stopped'`.

| Reason | Cause | REST status |
|---|---|---|
| `sweep_stopped` | state | 409 |

<a id="AutofocusZNotRestoredError"></a>
### `AutofocusZNotRestoredError`

A fault; bases `Fault`, `CaptureError`.

Autofocus stopped without a result and could not put a known Z back.

The restore's target was refused, so Z is known, where the sweep parked
it. A restore on a lost Z is not this: the fault that lost Z is told
where it was found, and the run ends on the lost position.

**Attributes**

- `reason`: `'z_left_at_search_position'`.

| Reason | Cause | REST status |
|---|---|---|
| `z_left_at_search_position` | -- | 500 |

<a id="AxisMovingError"></a>
### `AxisMovingError`

A refusal; bases `Refusal`, `Exception`.

A position was asked for to be kept while an axis it records is moving.

While an axis moves, the number it answers is the motion monitor's
last poll, behind the board by up to a poll, so a focus, a bookmark or
a step saved then records a place the axis was passing through, real-
looking and never where it stopped. The state and the number are read
in one snapshot, so a move that ends between the two cannot pass a
number taken while it ran.

Distinct from `AxisStateUnknownError`: a moving axis knows its frame
and needs only to stop; the user waits, never homes.

**Attributes**

- `reason`: `'axis_moving'`.
- `axes`: Every moving axis asked for, in the order asked.
- `axis`: The first of them, for callers that map a refusal to a response by a single axis (REST status codes, SDK branches).

| Reason | Cause | REST status |
|---|---|---|
| `axis_moving` | state | 409 |

<a id="AxisStateUnknownError"></a>
### `AxisStateUnknownError`

A refusal; bases `Refusal`, `Exception`.

An axis whose position is not known was asked to move, or to be recorded.

Raised by the motion pre-drive gate when the target axis is UNKNOWN:
a home failed, the board vanished mid-move, or a move stalled out.
An absolute move against an unknown reference frame is never a valid
request -- there is no frame for it to be absolute in -- so refusing
it discards nothing legitimate and makes the failure loud instead of
letting the stage travel somewhere nobody asked for. Also raised when
a position is about to be saved (a step, a focus, a bookmark) while
an axis does not know where it is: the cached number is the last one
the axis reported, real-looking and no longer true.

The recovery paths that must move a still-unknown axis (lowering Z
for turret safety, a deliberate re-home jog) pass `force=True`
rather than pre-checking state, so the gate can never deadlock the
operation that would clear the state it guards.

The message is written for the person at the scope, because the
GUI's background lane shows a typed error's message as the popup
body; it names every axis in one sentence so one refusal of a
several-axis gesture reads as one.

**Attributes**

- `reason`: `'position_unknown'`.
- `axes`: Every refused axis, mapped to its state, in the scope's axis order.
- `axis`: The first of them, for callers that map a refusal to a response by a single axis (REST status codes, SDK branches).

| Reason | Cause | REST status |
|---|---|---|
| `position_unknown` | state | 409 |

<a id="BinningSubstitutedNotice"></a>
### `BinningSubstitutedNotice`

A notice; bases `Notice`, `Exception`.

The saved binning is one this camera does not offer; bring-up ran at the camera's.

The camera's binning is stored in the saved value's place: the binning
is half of the frame's geometry, and the store holds the pair the
camera delivered.

| Reason | Cause | REST status |
|---|---|---|
| `binning_substituted` | -- | 409 |

<a id="BlockingCallRefusedError"></a>
### `BlockingCallRefusedError`

A refusal; bases `Refusal`, `RuntimeError`.

A blocking call was made on a thread it could end up waiting on.

Refused before any wait is made, by where the call was made: its
non-blocking form is the one that thread uses. Still a `RuntimeError`,
as it was documented.

**Attributes**

- `reason`: `'blocking_on_worker'` -- made on an executor's worker, whose own work it may queue behind. `'wait_on_ui_thread'` -- a wait on the UI thread for an operation not yet done, which would hold the window until the operation ends; a done-callback is the UI thread's form. `'blocking_inline'` -- made inside the GUI's inline boundary, whose thread draws the window; `submit_reported` is that call's form.
- `member`: The member that was called.

| Reason | Cause | REST status |
|---|---|---|
| `blocking_inline` | request | 422 |
| `blocking_on_worker` | request | 422 |
| `wait_on_ui_thread` | request | 422 |

<a id="BoardNoReplyError"></a>
### `BoardNoReplyError`

A fault; bases `Fault`, `HardwareError`.

A controller board did not answer a command within its timeout.

The board may be busy, unpowered or unplugged; the same command may be
answered when asked again. A motion command that got no answer may or
may not have been taken, and the raise site's words say which.

| Reason | Cause | REST status |
|---|---|---|
| `board_no_reply` | -- | 500 |

<a id="BoardPortFailedError"></a>
### `BoardPortFailedError`

A fault; bases `Fault`, `HardwareError`.

The serial port to a controller board failed, or is not open.

A cable or power state: nothing reached the board.

| Reason | Cause | REST status |
|---|---|---|
| `board_port_failed` | -- | 500 |

<a id="BoardReplyUnreadableError"></a>
### `BoardReplyUnreadableError`

A fault; bases `Fault`, `HardwareError`.

A controller board answered, and the answer could not be read.

A reply that is not the number or record asked for, or that answers
another command: the board is talking, and what it said cannot be used.

| Reason | Cause | REST status |
|---|---|---|
| `board_reply_unreadable` | -- | 500 |

<a id="BoardReportedError"></a>
### `BoardReportedError`

A fault; bases `Fault`, `HardwareError`.

A controller board answered that it could not do what it was asked.

The firmware's own error, a refused status, or a handshake it did not
complete: the board heard the command and said no.

| Reason | Cause | REST status |
|---|---|---|
| `board_reported_error` | -- | 500 |

<a id="BringUpError"></a>
### `BringUpError`

A subject: never raised, its children are; bases `Exception`.

A part of the scope did not come up as it should have; the rest of the scope runs.

Reported, never raised: bring-up goes on without the part, and the
person is told once what is missing and what to do about it. The record
of the bring-up (`ScopeSession.bring_up_record`) keeps the fact for a
client that asks later. The subject its children share, each a fault
naming what did not come up; never built itself.

**Attributes**

- `reason`: The machine-readable code a client branches on, the child's.

<a id="CameraConfigNotAppliedNotice"></a>
### `CameraConfigNotAppliedNotice`

A notice; bases `Notice`, `Exception`.

The camera came up without some of its connect-time configuration.

Each step of the configuration is tried whatever an earlier one did, so
the camera runs with the rest applied; the ones it did not take are
named, the image orientation among them when it is one.

**Attributes**

- `not_applied`: The configuration steps the camera did not take, by name.

| Reason | Cause | REST status |
|---|---|---|
| `camera_config_not_applied` | -- | 409 |

<a id="CameraDriverRepairFailedError"></a>
### `CameraDriverRepairFailedError`

A fault; bases `Fault`, `Exception`.

The driver step Repair ran as an administrator did not bind the camera.

The step logs every action and every Windows answer, so the words name
its log folder.

| Reason | Cause | REST status |
|---|---|---|
| `camera_driver_repair_failed` | -- | 500 |

<a id="CameraDriverRepairRefusedError"></a>
### `CameraDriverRepairRefusedError`

A refusal; bases `Refusal`, `Exception`.

Repair of the Lumascope Classic's driver binding was asked for and cannot run.

Repair runs the installed driver step as an administrator, so it needs
Windows, an installed build that carries the signed package, an
administrator who approves the prompt, the person's yes to Windows'
question whether to install Etaluma's driver, and a binding that needs it.

**Attributes**

- `reason`: `'repair_needs_windows'` -- the computer is not Windows, which alone binds a USB device to a driver. `'repair_needs_installed_build'` -- LumaViewPro is running from source or from a build in `dist`, not from an installation. `'repair_needs_driver_package'` -- the installation was built without the Classic driver package. `'no_classic_bindings'` -- the scope reads no Classic USB device, such as a simulated one without an FX2. `'repair_declined'` -- the administrator's prompt was not approved. `'driver_not_accepted'` -- Windows did not take the package: installing it was declined at its question, or Windows refused it. `'driver_already_usable'` -- a camera is up, or every present device is on a driver LumaViewPro can use.

| Reason | Cause | REST status |
|---|---|---|
| `driver_already_usable` | request | 422 |
| `driver_not_accepted` | state | 409 |
| `no_classic_bindings` | request | 422 |
| `repair_declined` | state | 409 |
| `repair_needs_driver_package` | request | 422 |
| `repair_needs_installed_build` | request | 422 |
| `repair_needs_windows` | request | 422 |

<a id="CameraDriverRepairedNotice"></a>
### `CameraDriverRepairedNotice`

A notice; bases `Notice`, `Exception`.

Repair bound the Lumascope Classic to LumaViewPro's driver; the camera connects at the next start.

| Reason | Cause | REST status |
|---|---|---|
| `camera_driver_repaired` | -- | 409 |

<a id="CameraFeatureAbsentNotice"></a>
### `CameraFeatureAbsentNotice`

A notice; bases `Notice`, `Exception`.

A camera feature saved on that this camera does not have; it is now saved off.

A camera without high conversion gain or line-noise reduction is at its
standard setting, which is known, so that setting is stored in the saved
value's place, as a substituted binning is.

**Attributes**

- `feature`: The feature, in words (`'high conversion gain'`).

| Reason | Cause | REST status |
|---|---|---|
| `camera_feature_absent` | -- | 409 |

<a id="CameraFirmwareMissingError"></a>
### `CameraFirmwareMissingError`

A fault; bases `Fault`, `Exception`.

The firmware file the camera's driver uploads is not in the installation.

The installation is at fault, not the hardware or its cable, so this is
not a `HardwareError`. The words name each place the driver looked.

| Reason | Cause | REST status |
|---|---|---|
| `camera_firmware_missing` | -- | 500 |

<a id="CameraNoAnswerError"></a>
### `CameraNoAnswerError`

A fault; bases `Fault`, `HardwareError`.

The camera did not answer a write of one of its settings in time.

An SDK or USB timeout on the access that decides whether the write
happened. The camera is still on the bus and may answer the same write
sent again; whether this one was applied is not known.

| Reason | Cause | REST status |
|---|---|---|
| `camera_no_answer` | -- | 500 |

<a id="CameraNotAvailableError"></a>
### `CameraNotAvailableError`

A fault; bases `Fault`, `BringUpError`.

The camera did not come up; the heading and the advice follow the cause.

**Attributes**

- `reason`: `'port_in_use'` -- the part was found and its port could not be opened, most often because another program holds it. `'not_detected'` -- no driver found the part on USB. These two are the hardware registry's own causes, the LED board's too. `'camera_driver_unavailable'` -- no camera was found, and a camera driver did not load on this installation. `'camera_firmware_missing'` -- the firmware file a Classic's driver uploads to the camera is not installed. `'camera_driver_unusable'` -- Windows has a Classic on no driver, or on one LumaViewPro cannot use. `'camera_not_initialized'` -- the camera did not come up for any other cause, among them a camera that was found and could not be opened, one another application holds included.

| Reason | Cause | REST status |
|---|---|---|
| `camera_driver_unavailable` | -- | 500 |
| `camera_driver_unusable` | -- | 500 |
| `camera_firmware_missing` | -- | 500 |
| `camera_not_initialized` | -- | 500 |
| `not_detected` | -- | 500 |
| `port_in_use` | -- | 500 |

<a id="CameraReadFailedError"></a>
### `CameraReadFailedError`

A fault; bases `Fault`, `HardwareError`.

The camera did not answer a read of one of its values, or a probe of one.

The words carry the SDK's own text, so the person reads the title.

| Reason | Cause | REST status |
|---|---|---|
| `camera_read_failed` | -- | 500 |

<a id="CameraSettingOutOfRangeError"></a>
### `CameraSettingOutOfRangeError`

A refusal; bases `Refusal`, `ValueError`.

A camera setting outside the range this camera declares was asked for.

Declined before anything reaches the camera, the same on every camera: a
body that would refuse the value and a body that would silently clamp it
both answer this way, so a caller that does not read the return cannot
record a value the camera never took. Nothing broke, and the caller can
ask again inside the range, so a refusal rather than a fault. A
`ValueError` too, for a caller that treats it as the bad argument it is.

**Attributes**

- `reason`: Machine-readable refusal code, `'<setting>_out_of_range'`. `'gain_db_out_of_range'` -- the gain is outside the camera's declared range. `'exposure_ms_out_of_range'` -- the exposure is outside the camera's declared range. `'frame_width_out_of_range'` -- the frame width is outside the widths the camera declares. `'frame_height_out_of_range'` -- the frame height is outside the heights the camera declares. `'black_level_out_of_range'` -- the black level is outside the camera's declared range.
- `setting`: Machine-readable setting name ('gain_db', 'exposure_ms', 'frame_width', 'frame_height', 'black_level').
- `requested`: The value asked for.
- `limits`: The camera's declared floor and ceiling, in the setting's unit; either is None when the camera declares none, and is not checked.

`limits` is published, so a client corrects the value without reading
the words.

**Fields**

- <a id="CameraSettingOutOfRangeError.limits"></a>`limits`: `tuple[float | None, float | None]` -- The camera's declared floor and ceiling, in the setting's unit; either is None when the camera declares none, and is not checked.

| Reason | Cause | REST status |
|---|---|---|
| `black_level_out_of_range` | request | 422 |
| `exposure_ms_out_of_range` | request | 422 |
| `frame_height_out_of_range` | request | 422 |
| `frame_width_out_of_range` | request | 422 |
| `gain_db_out_of_range` | request | 422 |

<a id="CameraSettingRejected"></a>
### `CameraSettingRejected`

A fault; bases `Fault`, `Exception`.

The camera driver rejected a state-changing setting apply.

Raised by CameraAPI setters (frame size, binning, image mode, gain,
exposure) when a LIVE driver refuses the apply -- distinct from a
camera that is not there, before or during the write, which is refused
as `HardwareCommandRefusedError('not_connected')`. Success is observed by receiving the
applied/delivered value, failure by this raise, so a caller cannot
record a rejected apply as applied by forgetting to check a return code.

A fault, not a refusal: the camera did not take a value it should
have, and the words say so. They are written for the person, so the
one reporter shows them where the flight stops; no setter logs or
notifies it.

**Attributes**

- `reason`: `'camera_rejected'` -- the camera refused the write, or its driver raised from it. `'camera_no_answer'` -- the camera did not answer the write, sent twice, so whether it took it is not known. The setting is in `setting` and in the words.
- `setting`: Machine-readable setting name (e.g. 'frame_size').
- `requested`: The value the caller asked for.

| Reason | Cause | REST status |
|---|---|---|
| `camera_no_answer` | -- | 500 |
| `camera_rejected` | -- | 500 |

<a id="CameraSettingUnsupportedError"></a>
### `CameraSettingUnsupportedError`

A refusal; bases `Refusal`, `ValueError`.

A camera setting this camera does not offer was asked for.

Declined before anything reaches the camera: the value is outside the
list the camera itself reports (its binning factors), so nothing
broke and the person can pick one it offers. A `ValueError` too,
for a caller that treats it as the bad argument it is.

**Attributes**

- `reason`: Machine-readable refusal code, `'<setting>_unsupported'`. `'auto_gain_unsupported'` -- this camera has no automatic gain to turn on. `'auto_exposure_unsupported'` -- this camera has no automatic exposure to turn on. `'auto_gain_target_brightness_unsupported'` -- this camera has no automatic gain, so it takes no target brightness. `'binning_unsupported'` -- the binning size is not one this camera reports. `'black_level_unsupported'` -- this camera has no black level setting. `'conversion_gain_mode_unsupported'` -- this camera has no high conversion gain, or the mode asked for is neither High nor Low. `'line_noise_reduction_unsupported'` -- this camera has no line noise reduction to turn on.
- `setting`: Machine-readable setting name (e.g. 'binning').
- `requested`: The value asked for.
- `offered`: The values the camera reports it supports.

| Reason | Cause | REST status |
|---|---|---|
| `auto_exposure_unsupported` | request | 422 |
| `auto_gain_target_brightness_unsupported` | request | 422 |
| `auto_gain_unsupported` | request | 422 |
| `binning_unsupported` | request | 422 |
| `black_level_unsupported` | request | 422 |
| `conversion_gain_mode_unsupported` | request | 422 |
| `line_noise_reduction_unsupported` | request | 422 |

<a id="CameraStreamStalledError"></a>
### `CameraStreamStalledError`

A fault; bases `Fault`, `CaptureError`.

The camera stopped delivering frames while it was not known to be removed.

A camera's link or grab loop can stall without the device being removed:
it reports itself connected and grabbing, and no frame arrives. Nothing
is waiting on the stream when that happens, so the imaging API reports it
when its stream check sees the frame count stand still for ten times the
longest exposure the camera may be using (the sensor's maximum while the
exposure is unknown), and never less than 5 s. A pulled cable meets the same
check first when its driver reports the removal later than that (the IDS
SDK reports it seconds after the pull), so the words never say the camera
is still connected.

**Attributes**

- `seconds`: How long no new frame had arrived when the stall was seen.

| Reason | Cause | REST status |
|---|---|---|
| `camera_stream_stalled` | -- | 500 |

<a id="CameraStreamStartFailedError"></a>
### `CameraStreamStartFailedError`

A fault; bases `Fault`, `HardwareError`.

The stream's own set-up on an opened camera failed. The words carry
the library's text.

| Reason | Cause | REST status |
|---|---|---|
| `camera_stream_start_failed` | -- | 500 |

<a id="CameraUsbDriverUnusableError"></a>
### `CameraUsbDriverUnusableError`

A fault; bases `Fault`, `HardwareError`.

The camera's USB device is bound to a driver LumaViewPro cannot use.

Windows binds each of the Lumascope Classic's two USB identities to a
driver on its own; a device with none, or one the firmware upload or
the stream cannot drive, cannot come up however it is cabled. The words
name the device, the driver it is bound to and what it needs.

| Reason | Cause | REST status |
|---|---|---|
| `camera_usb_driver_unusable` | -- | 500 |

<a id="CameraUsbLookupFailedError"></a>
### `CameraUsbLookupFailedError`

A fault; bases `Fault`, `HardwareError`.

The system's lookup of the camera's USB device itself failed.

Not an absence: the lookup did not complete, so whether the device is
there is not known. The words name the call and its error code.

| Reason | Cause | REST status |
|---|---|---|
| `camera_usb_lookup_failed` | -- | 500 |

<a id="CameraUsbNotFoundError"></a>
### `CameraUsbNotFoundError`

A fault; bases `Fault`, `HardwareError`.

No USB device the camera driver can stream from answers to its identity.

On Windows: none bound to WinUSB, which is what the stream opens.
The words carry the identity and the system's own text.

| Reason | Cause | REST status |
|---|---|---|
| `camera_usb_not_found` | -- | 500 |

<a id="CameraUsbOpenFailedError"></a>
### `CameraUsbOpenFailedError`

A fault; bases `Fault`, `HardwareError`.

The camera's USB device was found and could not be opened or set up
for streaming. The words name the call and its error code.

| Reason | Cause | REST status |
|---|---|---|
| `camera_usb_open_failed` | -- | 500 |

<a id="CameraWriteFailedError"></a>
### `CameraWriteFailedError`

A fault; bases `Fault`, `HardwareError`.

The camera refused or did not take a write of one of its settings.

The words carry the SDK's own text, so the person reads the title.

| Reason | Cause | REST status |
|---|---|---|
| `camera_write_failed` | -- | 500 |

<a id="CaptureError"></a>
### `CaptureError`

A subject: never raised, its children are; bases `Exception`.

Image capture, save or processing: the subject, never raised itself.

**Attributes**

- `reason`: Machine-readable failure code for callers that map failures to responses (REST status codes, UI branches). Without it a caller has to regex the prose to tell a timeout from a failed merge from a camera that returned nothing.

`reason` is required rather than defaulted. A default would let a
new raise site stay untyped while every caller still had to guess
which raises carry a usable code and which carry a placeholder.

`title` is the heading the reporter shows above the message, which is
written for the person. It is each child's, read from its own table; the
subject has none, as it has no kind.

<a id="CaptureFailedError"></a>
### `CaptureFailedError`

A fault; bases `Fault`, `CaptureError`.

A capture or a recording was attempted and produced nothing.

Raised by `capture_and_wait` and `get_image` for a capture that got
no frame, and by `capture_and_wait` for one whose frame's record could
not be read; a camera that is not there is refused instead
(`HardwareCommandRefusedError`, `'not_connected'`).

**Attributes**

- `reason`: One of: - `'no_frame'` -- the camera delivered no new frame within its grab window, whether the capture was draining, grabbing or waiting for a sum's next frame. - `'deadline_expired'` -- the scope's state kept changing for longer than the capture's budget. - `'chunk_rejected'` -- frames came, but none was exposed at the `source` setting asked. - `'no_fresh_frame'` -- frames came, but none within the capture's bound counted for the `source` change: each was exposed, by its source's rule, before that change reached it. - `'record_read_failed'` -- a frame came, but the camera did not answer a read its record needs (the gain, exposure or black level), so it was not kept; chained from the `CameraReadFailedError`. - `'recording_not_started'` -- the camera did not take the recording's frame listener. - `'unknown_save_encoding'` -- a frame was handed an encoding no writer knows, which the program, not the person, got wrong.
- `source`: The source the capture failed for: for `'chunk_rejected'` the setting no frame matched, `EXPOSURE` or `GAIN`; for `'no_fresh_frame'` the change no frame came after (`Z_MOVE`, `LED`); None otherwise.

**Fields**

- <a id="CaptureFailedError.source"></a>`source`: `FrameValiditySource | None` -- The source the capture failed for: for `'chunk_rejected'` the setting no frame matched, `EXPOSURE` or `GAIN`; for `'no_fresh_frame'` the change no frame came after (`Z_MOVE`, `LED`); None otherwise.

| Reason | Cause | REST status |
|---|---|---|
| `chunk_rejected` | -- | 500 |
| `deadline_expired` | -- | 500 |
| `no_frame` | -- | 500 |
| `no_fresh_frame` | -- | 500 |
| `record_read_failed` | -- | 500 |
| `recording_not_started` | -- | 500 |
| `unknown_save_encoding` | -- | 500 |

<a id="CaptureLocationError"></a>
### `CaptureLocationError`

A refusal; bases `Refusal`, `Exception`.

The capture location cannot hold a new output directory.

The location's state, not the request: a drive is plugged back in, a
folder made writable, old captures moved away, and the same capture
may then succeed.

**Attributes**

- `reason`: `'capture_location_unusable'` -- the location is missing, is not a folder, or refused the new directory; `'names_exhausted'` -- the name and every numbered variant are taken.

| Reason | Cause | REST status |
|---|---|---|
| `capture_location_unusable` | state | 409 |
| `names_exhausted` | state | 409 |

<a id="CapturePositionNotRecordedNotice"></a>
### `CapturePositionNotRecordedNotice`

A notice; bases `Notice`, `Exception`.

A still was saved, without its well or position, because X or Y is unknown.

| Reason | Cause | REST status |
|---|---|---|
| `position_not_recorded` | -- | 409 |

<a id="CatalogueNameRefusedError"></a>
### `CatalogueNameRefusedError`

A refusal; bases `Refusal`, `ConfigError`.

A labware or objective name the installation's catalogue does not hold.

Raised by the catalogue's owner -- `WellPlateLoader.resolve_plate_key`
for a plate, `ObjectiveLoader.get_objective_info` for an objective --
before anything is stored, so every door that names one is refused
alike. A `ConfigError`, because the same check reads a stored setting
at bring-up and the callers that recover from a bad stored name catch
that type.

The catalogue is the installation's, fixed for the process, so the same
name is refused every time it is sent.

**Attributes**

- `reason`: `'labware_unknown'` -- the labware name is no plate in the labware catalogue. `'objective_not_in_catalogue'` -- the objective name is no key of the objective catalogue.
- `argument`: The owner's parameter: `labware_id` or `objective_id`.
- `value`: What was given, as given.
- `offered`: The catalogue's keys.

`argument` and `offered` are published, so a REST problem carries
them beside the words. The run's `ProtocolRunRefusedError` names the
same fact for a protocol's steps; `argument` is on this type only.

**Fields**

- <a id="CatalogueNameRefusedError.argument"></a>`argument`: `str` -- The owner's parameter: `labware_id` or `objective_id`.
- <a id="CatalogueNameRefusedError.offered"></a>`offered`: `tuple[str, ...]` -- The catalogue's keys.

| Reason | Cause | REST status |
|---|---|---|
| `labware_unknown` | request | 422 |
| `objective_not_in_catalogue` | request | 422 |

<a id="CellCountScaleDroppedNotice"></a>
### `CellCountScaleDroppedNotice`

A notice; bases `Notice`, `Exception`.

A method file saved before the count read each image's own scale.

Every such file carries the old default of 1.0 pixels per micron, which
described no image, so the load drops it and the count measures each
image at its own scale. A person who meant 1.0 types it again.

| Reason | Cause | REST status |
|---|---|---|
| `cell_count_scale_dropped` | -- | 409 |

<a id="ChannelBlockSupersededNotice"></a>
### `ChannelBlockSupersededNotice`

A notice; bases `Notice`, `Exception`.

The motor board carries the old Layers block, which is no longer read.

The unit's channels and filterset are its model's. A board's own channels
are its `Channels` block, beside its `Filterset`.

| Reason | Cause | REST status |
|---|---|---|
| `channel_block_superseded` | -- | 409 |

<a id="ChannelBlockUnusableNotice"></a>
### `ChannelBlockUnusableNotice`

A notice; bases `Notice`, `Exception`.

The motor board's Channels block cannot be used, so the scope has no channels.

The block describes one filter assembly, so a block that contradicts
itself is not read in part: no row can be trusted over another.

| Reason | Cause | REST status |
|---|---|---|
| `channel_block_unusable` | -- | 409 |

<a id="ChannelRowsSkippedNotice"></a>
### `ChannelRowsSkippedNotice`

A notice; bases `Notice`, `Exception`.

Rows of the motor board's Channels block that could not be read; the rest are the scope's.

One notice names every skipped row, as one load's replaced settings are
one notice.

**Attributes**

- `skipped`: Each skipped row's key and why it was not read.

| Reason | Cause | REST status |
|---|---|---|
| `channel_rows_skipped` | -- | 409 |

<a id="ClaimStillLentError"></a>
### `ClaimStillLentError`

A fault; bases `Fault`, `RuntimeError`.

A diagnostic's block ended with a run it lent its claim to still live.

The diagnostic's end waited for the run and the run outlived the wait,
so the claim stays with the run and is released when the run ends: the
scope is not free yet. Raised at the block's end when the block ended
cleanly; reported once when the block raised, so the block's own
exception is the one its caller receives. Still a `RuntimeError`, as
it was documented.

**Attributes**

- `timeout_s`: How long the end waited for the run.

| Reason | Cause | REST status |
|---|---|---|
| `claim_still_lent` | -- | 500 |

<a id="CompositeFailedError"></a>
### `CompositeFailedError`

A fault; bases `Fault`, `CaptureError`.

A run's composite was not merged; the run's own images stand.

Its reasons are its own: what the run ended with, or what the merge
raised, is a field, since each is another vocabulary's string.

**Attributes**

- `reason`: `'run_not_completed'` -- the run ended before a composite was made. `'no_run_dir'` -- the run has no folder to merge from. `'no_composite_config'` -- the run carries no composite thresholds. `'merge_failed'` -- the merge finished without writing a file. `'merge_error'` -- the merge failed.
- `run_ending`: The run's ending reason, for `'run_not_completed'`; otherwise None.
- `merge_failure`: The reason of the outcome the merge failed with, for `'merge_error'` when it had one; otherwise None.

**Fields**

- <a id="CompositeFailedError.merge_failure"></a>`merge_failure`: `str | None` -- The reason of the outcome the merge failed with, for `'merge_error'` when it had one; otherwise None.
- <a id="CompositeFailedError.run_ending"></a>`run_ending`: `str | None` -- The run's ending reason, for `'run_not_completed'`; otherwise None.

| Reason | Cause | REST status |
|---|---|---|
| `merge_error` | -- | 500 |
| `merge_failed` | -- | 500 |
| `no_composite_config` | -- | 500 |
| `no_run_dir` | -- | 500 |
| `run_not_completed` | -- | 500 |

<a id="ConfigError"></a>
### `ConfigError`

A subject: never raised, its children are; bases `Exception`.

The application's configuration or settings: the subject, never raised itself.

Its children's words are written for the person unless a child says
otherwise.

<a id="ConfigReadError"></a>
### `ConfigReadError`

A fault; bases `Fault`, `HardwareError`.

The board's per-unit config could not be READ (no answer, or an
unparseable payload), or a value in it is not one the field can hold
(an optics value that is not a number).

Distinct from a board that answers with an empty or minimal config:
after a failed read the per-unit values may exist on the board but
are unavailable, so consumers that would trust "no per-unit value
present" need to know the difference or they silently serve another
source's answer for a unit that has its own.

| Reason | Cause | REST status |
|---|---|---|
| `config_unread` | -- | 500 |

<a id="ConfigurationDefectError"></a>
### `ConfigurationDefectError`

A fault; bases `Fault`, `ConfigError`.

The program, the installation or the scope's composition broke a configuration contract.

Not the request's and not the scope's state: no caller can change what it
sent to make it succeed, so it is a fault, never a refusal. Raised where
a value every door has already judged arrives unjudged, where a scope is
used before it is composed, or where the installation lacks what the
release ships. Its words are a developer's, kept in the log, except where
the reason's are written for the person.

**Attributes**

- `reason`: `'settings_not_prepared'` -- a capture configuration names an output format this build does not write, so the stored image output format was never judged by the settings' preparation. `'composite_thresholds_missing'` -- a composite was asked for without a brightness threshold for every fluorescence layer it blends. `'composite_threshold_out_of_range'` -- a composite's brightness threshold is not a number from 0 to 100 percent. `'capture_config_inconsistent'` -- a capture configuration's capture depth and save encoding do not match its image mode. `'tail_id_mode_unknown'` -- a file name was asked for with a tail id mode the namer does not implement. `'labware_not_given'` -- an image's record was written with no labware to state. `'layers_unresolved'` -- this scope's layers could not be resolved from its model, so no layer can be acted on. `'scope_not_composed'` -- a scope that no session has bound was used for work that reads or saves settings. `'zstack_reference_unchecked'` -- a Z-stack's position reference is none of top, center and bottom. `'route_shadowed'` -- a Session member has the name of one of the REST server's own routes, so a client could not reach it.

| Reason | Cause | REST status |
|---|---|---|
| `capture_config_inconsistent` | -- | 500 |
| `composite_threshold_out_of_range` | -- | 500 |
| `composite_thresholds_missing` | -- | 500 |
| `labware_not_given` | -- | 500 |
| `layers_unresolved` | -- | 500 |
| `route_shadowed` | -- | 500 |
| `scope_not_composed` | -- | 500 |
| `settings_not_prepared` | -- | 500 |
| `tail_id_mode_unknown` | -- | 500 |
| `zstack_reference_unchecked` | -- | 500 |

<a id="ControlSessionRefusedError"></a>
### `ControlSessionRefusedError`

A refusal; bases `Refusal`, `Exception`.

Control of the scope was not taken or used: someone else's work holds it.

Raised by the activity claim when a control session is asked for while
another one is held, while an activity holds the scope, while a setting
or file write is still under way, or while work that has let the scope
go is still finishing -- so a client never takes the scope over the
operator's work, or over a write half done. While a session is held,
it is the one refusal of anything asked for by anyone but the
holder (`'control_session_held'`): a guarded write (a setting, a
file), a command on a hardware lane, and the start of a run, a
recording, a diagnostic or a home. Raised too for a request naming a control
session that is not the live one: it was ended, or never was. Nothing
was taken or written.

**Attributes**

- `reason`: `'control_session_held'` -- a control session is held, and the request is not its holder's. `'write_in_flight'` -- a setting or a file is still being written, so control is not taken over it half done. `'work_finishing'` -- a run or a recording that has ended is still finishing. `'control_session_ended'` -- the request names a control session that is not the live one: it has ended, or was never issued. `'exclusive_activity_running'`, an activity holding the scope, is `ProtocolRunRefusedError`'s to explain.
- `title`: The heading the person reads, the reason's.
- `holder`: The kind of activity holding the scope, for `'exclusive_activity_running'`; None otherwise.
- `holder_trigger`: Who holds the scope: the control session's taker, or the holding run's starter (`'gui'`, `'rest'`, `'api'`); None when neither is known.
- `control_session`: The name of the control session that holds, for `'control_session_held'`; None otherwise.

`holder`, `holder_trigger` and `control_session` are published.

**Fields**

- <a id="ControlSessionRefusedError.control_session"></a>`control_session`: `str | None` -- The name of the control session that holds, for `'control_session_held'`; None otherwise.
- <a id="ControlSessionRefusedError.holder"></a>`holder`: `Holder | None` -- The kind of activity holding the scope, for `'exclusive_activity_running'`; None otherwise.
- <a id="ControlSessionRefusedError.holder_trigger"></a>`holder_trigger`: `Caller | None` -- Who holds the scope: the control session's taker, or the holding run's starter (`'gui'`, `'rest'`, `'api'`); None when neither is known.

| Reason | Cause | REST status |
|---|---|---|
| `control_session_ended` | request | 422 |
| `control_session_held` | state | 409 |
| `exclusive_activity_running` | state | 409 |
| `work_finishing` | state | 409 |
| `write_in_flight` | state | 409 |

<a id="DiagnosticRefusedError"></a>
### `DiagnosticRefusedError`

A refusal; bases `Refusal`, `Exception`.

A diagnostic could not take the scope: another activity holds it.

Raised by `ScopeSession.diagnostic_claim()` when a run, a recording, a
home or another diagnostic already holds the session's activity claim;
a control session held by someone else refuses it with
`ControlSessionRefusedError`, as it does every lane. A diagnostic drives the
hardware directly (homes, LED modes, forced grabs), so it runs only on a
scope nothing else is using, and a caller told why can wait for the
holder or stop it. Nothing was committed.

**Attributes**

- `reason`: Machine-readable refusal code for callers that map refusals to responses (REST status codes, SDK branches): `'exclusive_activity_running'`, as `ProtocolRunRefusedError` explains it.
- `title`: Short user-facing refusal title, the reason's.
- `message`: One-sentence user-facing refusal body.
- `holder`: The activity kind holding the claim at refusal time.
- `holder_trigger`: Who started the holding run (`'gui'`, `'rest'`, `'api'`) when the holder is a run; None otherwise.

`holder` and `holder_trigger` are published.

**Fields**

- <a id="DiagnosticRefusedError.holder"></a>`holder`: `Holder | None` -- The activity kind holding the claim at refusal time.
- <a id="DiagnosticRefusedError.holder_trigger"></a>`holder_trigger`: `Caller | None` -- Who started the holding run (`'gui'`, `'rest'`, `'api'`) when the holder is a run; None otherwise.

| Reason | Cause | REST status |
|---|---|---|
| `exclusive_activity_running` | state | 409 |

<a id="DriverNotLiveError"></a>
### `DriverNotLiveError`

A fault; bases `Fault`, `RuntimeError`.

A driver selected by name constructed, but its hardware is not live.

| Reason | Cause | REST status |
|---|---|---|
| `driver_not_live` | -- | 500 |

<a id="DuplicateCaptureFilenamesNotice"></a>
### `DuplicateCaptureFilenamesNotice`

A notice; bases `Notice`, `Exception`.

A loaded protocol has steps that would save to the same file name.

The file still loads, so the steps can be renamed in the app; the run is
refused at start until each step's file name is unique.

| Reason | Cause | REST status |
|---|---|---|
| `duplicate_capture_filenames` | -- | 409 |

<a id="ExposureAtMaximumNotice"></a>
### `ExposureAtMaximumNotice`

A notice; bases `Notice`, `Exception`.

Auto-gain reached its exposure ceiling and the scene was still too dark.

The setting keeps the ceiling; the person can add light or raise it.

| Reason | Cause | REST status |
|---|---|---|
| `exposure_at_maximum` | -- | 409 |

<a id="ExposureAtMinimumNotice"></a>
### `ExposureAtMinimumNotice`

A notice; bases `Notice`, `Exception`.

Auto-gain settled at or below the channel's usable exposure floor.

The setting keeps the floor, since it cannot hold the raw value.

| Reason | Cause | REST status |
|---|---|---|
| `exposure_at_minimum` | -- | 409 |

<a id="FalsifyingChangeInFlightError"></a>
### `FalsifyingChangeInFlightError`

A refusal; bases `Refusal`, `Exception`.

A recording was asked for while a write that would falsify it runs.

Raised by `try_claim(WorkKind.RECORDING)`. A recording writes nothing to a
lane, so unlike a run it does not queue behind a turret or frame change
already under way: started then, its first frames would record the
change and its file would claim the settings it read before it. The
change ends, and the same start may then succeed.

**Attributes**

- `reason`: `'falsifying_change_in_flight'`.

| Reason | Cause | REST status |
|---|---|---|
| `falsifying_change_in_flight` | state | 409 |

<a id="Fault"></a>
### `Fault`

The fault kind; bases `LumascopeError`.

Something was attempted and failed.

A fault's words are a developer's unless its type says they are written
for the person: the reporter then shows them, and otherwise shows a
generic sentence and keeps the words in the log. Its reasons map to no
cause: a fault answers 500 whatever was asked.

**Attributes**

- `words_for_person`: True when the class writes its message for the person who will read it. Declared, never defaulted here: a default on this mixin would sit ahead of a value a fault inherits from its subject's class, and hide it. Unset reads as False.

<a id="FileWriterNotStuckError"></a>
### `FileWriterNotStuckError`

A refusal; bases `Refusal`, `Exception`.

File-writer recovery was asked for while nothing is stuck.

Recovery discards the images still waiting to be written, so it runs
only when the write in flight has stopped making progress. A writer that
is behind but moving finishes on its own; discarding its images then
would be data loss for nothing.

**Attributes**

- `reason`: `'file_writer_not_stuck'`.
- `pending`: The writes still outstanding when asked.

| Reason | Cause | REST status |
|---|---|---|
| `file_writer_not_stuck` | state | 409 |

<a id="FileWriterStalledError"></a>
### `FileWriterStalledError`

A fault; bases `Fault`, `CaptureError`.

A finished run's file writer stopped making progress on one of its files.

Nobody is waiting on a run's files once it has ended, so the run engine
reports the stall when it sees it, offering the recovery that answers it.
The run refusal for the same stall makes the same offer: the sentences
and the remedy below are the one wording of both.

**Attributes**

- `remedy`: Recovering the file writer, which gives up on the unsaved images.

| Reason | Cause | REST status |
|---|---|---|
| `file_writer_stalled` | -- | 500 |

<a id="FocusNotSavedError"></a>
### `FocusNotSavedError`

A refusal; bases `Refusal`, `ConfigError`.

A channel's saved focus was asked for, and none was ever saved.

A channel with no saved focus is imaged at the stage's current Z when a
step is built for it, so there is no height to go to; answering with a
stored placeholder is what sent every such channel to one fixed height.

**Attributes**

- `reason`: `'focus_not_saved'`.
- `layer`: The channel asked about.

| Reason | Cause | REST status |
|---|---|---|
| `focus_not_saved` | state | 409 |

<a id="FocusNotWrittenError"></a>
### `FocusNotWrittenError`

A refusal; bases `Refusal`, `ProtocolError`.

The focus a scan found was not written, because the protocol's steps changed under it.

Raised by `Protocol.adopt_focus_from` when the protocol no longer has
the steps the scan focused: a different number of them, or a step at a
different position, channel or objective. Each focused Z belongs to the
step it was found at, so writing it into a different step would give that
step a focus nobody measured. The protocol is left unchanged; scanning
again focuses the steps it has now.

**Attributes**

- `reason`: `'focus_not_written'`.
- `difference`: What no longer matches, in the words the person reads.

| Reason | Cause | REST status |
|---|---|---|
| `focus_not_written` | state | 409 |

<a id="FrameDepthError"></a>
### `FrameDepthError`

A fault; bases `Fault`, `Exception`.

A frame carries a payload value above its declared significant-bits depth.

The downconvert to 8-bit scales against `significant_bits` -- the meaningful
payload range the frame was captured under. A pixel larger than that range is
a depth-contract violation: the significant-bits value is wrong for the data.
Raised explicitly so the failure is loud and typed regardless of the
downconvert arithmetic underneath -- a value-indexed LUT happens to raise
IndexError today, but a scale-and-clip converter would silently map the
over-range frame to white instead.

**Attributes**

- `value`: The offending pixel value.
- `significant_bits`: The declared depth it exceeded.

| Reason | Cause | REST status |
|---|---|---|
| `frame_depth_exceeded` | -- | 500 |

<a id="FrameHandlerRemovedError"></a>
### `FrameHandlerRemovedError`

A refusal; bases `Refusal`, `Exception`.

The API stopped calling a frame handler that kept failing or kept running too long.

Nothing raises it: the frame thread that removes the handler has no
caller to raise to, so it is built to be reported. A refusal -- the API
declining to go on calling the handler, which the plugin's author can fix
-- so it is shown as a warning and logged without a traceback.

**Attributes**

- `name`: The handler's display name.
- `reason`: `'over_budget'` -- it ran past the per-frame budget for the drop count of frames in a row; `'raised'` -- it raised for the drop count of frames in a row.

| Reason | Cause | REST status |
|---|---|---|
| `over_budget` | state | 409 |
| `raised` | state | 409 |

<a id="FrameListenerNotRegisteredError"></a>
### `FrameListenerNotRegisteredError`

A fault; bases `Fault`, `CaptureError`.

The camera driver would not take a frame listener, so it will receive no frames.

Raised by `add_frame_listener` to whoever registered it -- a plugin, a
recording, a script -- because each of them has something to undo or to
refuse: a recording that started without its frames records nothing, and
a plugin listed as loaded never runs. The driver's error is the cause.

**Attributes**

- `name`: The listener's display name.

| Reason | Cause | REST status |
|---|---|---|
| `frame_listener_refused` | -- | 500 |

<a id="FrameRefittedNotice"></a>
### `FrameRefittedNotice`

A notice; bases `Notice`, `Exception`.

The saved frame is larger than this scope delivers; bring-up ran at the largest it does.

The refitted frame is stored in the saved frame's place, as the
substituted binning is, so the next bring-up has nothing to refit.

| Reason | Cause | REST status |
|---|---|---|
| `frame_refitted` | -- | 409 |

<a id="HardwareCommandRefusedError"></a>
### `HardwareCommandRefusedError`

A refusal; bases `Refusal`, `Exception`.

A hardware command was refused: something else has the scope, the lane is closed, or nothing is connected to take it.

Raised to whoever made the command -- a public hardware member (LED,
camera and motion commands), a raw task on a lane, the Session's
objective writers -- and never dropped: a command refused without a
raise reaches no hardware and reports success. While a run, a
diagnostic or a home holds the scope, a lane refuses any task not made under
the holder's taking with this; so do the run's own executor fences,
which cannot say who closed them. While a control session is held, a
lane refuses any task not made for its holder with
`ControlSessionRefusedError`, as a guarded write is refused.

The Session's objective writers (select, slot assign and slot clear)
raise it too while a run, a diagnostic or a home holds the scope: the run
stamps the active objective's scale into each capture, so a change
mid-run is a command against the run's hardware state.

A motion or LED command for hardware the scope does not have raises it
too, and `missing` names the part: `'not_connected'` when the model
has a motor controller and none is connected, whose remedy is the cable,
or when no LED controller is connected; `'axis_absent'` when the
scope does not have the motion part -- a Z-only scope asked for X, a
scope with no turret asked for one, a manual scope asked for any
motion; `'not_on_this_scope'` when it does not have another part -- an
LS560 asked to light Ex589, a motor controller without fan control asked
to drive the fan. Nothing was driven. So does a move whose drive needs a
position the controller did not report (`'position_unread'`): the
relative base, or Z for the backlash approach. Nothing was driven and
the axis keeps its state.
So does a move or a home the stage's own interlock refused before
anything moved (`INTERLOCK_REASONS`): the lid is open, or the stage
has no power; the refused axis is where it was, its position known.
A capture asked of a camera that is connected but not streaming is
refused `'camera_not_streaming'`: nothing can deliver its frame.

A declined request, not a fault, so it is a `Refusal`: the lane shows
it as a warning in its own words and logs one line without a
traceback. The message is written for the person at the scope and
names the holder; `reason` and `member` are for code that maps a
refusal to a response (REST status codes, SDK branches) and for the
log.

**Attributes**

- `reason`: Machine-readable refusal code. `'exclusive_activity_running'`, `'not_connected'`, `'camera_not_streaming'` and `'lid_open_before_motion'` are explained by `ProtocolRunRefusedError`, which carries them as the same fact. `'capture_in_flight'` -- an earlier still has not finished. `'home_in_flight'` -- another home holds the scope. `'home_not_stoppable'` -- a home drives on a motor controller that cannot stop one. `'scope_disconnected'` -- the scope's lanes are closed, so it takes no command until the application restarts. `'monitor_ended'` -- the motion monitor has ended, so nothing would judge the move. `'position_unread'` -- the move's drive needs a position the motor controller did not report: the relative base, or Z for the backlash approach. `'stage_unpowered_before_motion'` -- the stage has no power, so its interlock refuses to move it. `'axis_absent'` -- the scope does not have the motion part asked for. `'not_on_this_scope'` -- the scope does not have another part asked for, such as an LED channel or fan control. `'activity_ended'` -- the command was made under an activity that has ended.
- `member`: The member or task that was refused, for the log.
- `holder`: The kind of activity holding the scope, when known.
- `holder_trigger`: The holding run's starter, for a run (`'gui'`, `'rest'`, `'api'`); None otherwise.
- `missing`: The part the command needed and the scope does not have, for `'not_connected'`, `'axis_absent'` and `'not_on_this_scope'`; None otherwise.
- `title`: The heading shown with the sentence, which follows the reason: nothing connected is not a busy microscope.

`holder` and `holder_trigger` are published.

**Fields**

- <a id="HardwareCommandRefusedError.holder"></a>`holder`: `Holder | None` -- The kind of activity holding the scope, when known.
- <a id="HardwareCommandRefusedError.holder_trigger"></a>`holder_trigger`: `Caller | None` -- The holding run's starter, for a run (`'gui'`, `'rest'`, `'api'`); None otherwise.

| Reason | Cause | REST status |
|---|---|---|
| `activity_ended` | request | 422 |
| `axis_absent` | request | 422 |
| `camera_not_streaming` | state | 409 |
| `capture_in_flight` | state | 409 |
| `exclusive_activity_running` | state | 409 |
| `home_in_flight` | state | 409 |
| `home_not_stoppable` | state | 409 |
| `lid_open_before_motion` | state | 409 |
| `monitor_ended` | restart | 503 |
| `not_connected` | restart | 503 |
| `not_on_this_scope` | request | 422 |
| `position_unread` | state | 409 |
| `scope_disconnected` | restart | 503 |
| `stage_unpowered_before_motion` | state | 409 |

<a id="HardwareError"></a>
### `HardwareError`

A subject: never raised, its children are; bases `Exception`.

A fault in the hardware (motor, LED, camera): the subject, never raised itself.

`except HardwareError` catches every one; the child raised says which
fault it was, with its own reason and title.

<a id="HomingFailedError"></a>
### `HomingFailedError`

A subject: never raised, its children are; bases `Exception`.

A home ended without establishing a reference position: the subject, never raised itself.

`except HomingFailedError` catches every way a home ends short; the
child raised says which, and its kind with it: `HomingFaultError`
when the home failed, `HomingStoppedError` when a Stop ended it.

**Attributes**

- `home`: What was homed: `'ALL'`, `'Z'` or `'T'`.
- `reason`: The child's code for how the home ended.
- `axes`: The axes left without a known position.

<a id="HomingFaultError"></a>
### `HomingFaultError`

A fault; bases `Fault`, `HomingFailedError`.

A home failed: the driver raised, or a homed axis's position could not be read.

Each leaves the axes UNKNOWN, so no caller may take the scope as
knowing where it is. The motors may have moved. Chained from the
driver's exception when there is one.

**Attributes**

- `reason`: `'error'` -- the home raised; `'unread'` -- homed, but `axes` could not be read; `'lid_open'` -- the lid was opened while the home was moving.

| Reason | Cause | REST status |
|---|---|---|
| `error` | -- | 500 |
| `lid_open` | -- | 500 |
| `stage_unpowered` | -- | 500 |
| `unread` | -- | 500 |

<a id="HomingStoppedError"></a>
### `HomingStoppedError`

A notice; bases `Notice`, `HomingFailedError`.

A home was ended by a Stop: nothing failed.

A notice: the person whose home someone else stopped is told so, as
information, and a REST caller is answered 409. A Stop that landed
before the driver was asked to home names no axis: nothing it homes
had moved.

**Attributes**

- `reason`: `'stopped'` -- a Stop ended the home.
- `axes`: The axes left without a known position; empty for a home a Stop ended before it was driven.

| Reason | Cause | REST status |
|---|---|---|
| `stopped` | state | 409 |

<a id="HyperstackRefusedError"></a>
### `HyperstackRefusedError`

A fault; bases `Fault`, `CaptureError`.

The hyperstack builder declined to build a recording's frames into one file.

Raised by the manual recording's finish when the builder answers
status=False, or raises `InputFramesUnusableError` because a frame
cannot be read or cannot share the stack: the frames are on disk as
recorded.
The builder's own sentence rides in `message` so the notification the
user sees says why, in the builder's words, rather than telling them to
read a log.

**Attributes**

- `message`: The builder's one-paragraph reason, user-facing.
- `reason`: `'hyperstack_refused'` -- the builder answered status=False, or the frames are not all one size; `'unreadable_input_frame'` -- a frame could not be read; `'mixed_input_pixel_types'` -- frames of different pixel types cannot share one stack.

| Reason | Cause | REST status |
|---|---|---|
| `hyperstack_refused` | -- | 500 |
| `mixed_input_pixel_types` | -- | 500 |
| `unreadable_input_frame` | -- | 500 |

<a id="HyperstacksSavedNotice"></a>
### `HyperstacksSavedNotice`

A notice; bases `Notice`, `Exception`.

A run's hyperstacks were built: how many and where, or what was degraded.

| Reason | Cause | REST status |
|---|---|---|
| `hyperstacks_saved` | -- | 409 |

<a id="HyperstacksSavingNotice"></a>
### `HyperstacksSavingNotice`

A notice; bases `Notice`, `Exception`.

A run's hyperstacks are being built, which can take minutes.

| Reason | Cause | REST status |
|---|---|---|
| `hyperstacks_saving` | -- | 409 |

<a id="ImageSaveError"></a>
### `ImageSaveError`

A fault; bases `Fault`, `CaptureError`.

An image could not be written to disk.

Raised by `save_image` from the `OSError` of the write -- a missing
or read-only folder, a full disk, a denied permission -- so the words
point at the disk. A failure before the write (encoding, metadata)
is not this and propagates as itself.

**Attributes**

- `file_loc`: The path the image was to be written to.

| Reason | Cause | REST status |
|---|---|---|
| `image_save_failed` | -- | 500 |

<a id="InputFramesUnusableError"></a>
### `InputFramesUnusableError`

A refusal; bases `Refusal`, `CaptureError`, `ValueError`.

Frames handed to an image operation cannot be used, naming each frame.

Raised by the image reader for a frame that is missing or will not
decode, by the canvas check for frames of different sizes, and by the
hyperstack builder for frames whose pixel types cannot share one stack.
A post-processing build answers it as the one group's failure and goes
on with the others, so a damaged frame costs its group, not the run's
other outputs. A `ValueError`, so a caller that skips a frame it
cannot read keeps doing so.

**Attributes**

- `reason`: `'unreadable'` -- a frame is missing or will not decode; `'geometry'` -- the frames are not all one size; `'mixed_pixel_types'` -- the frames' pixel types cannot share one output.

| Reason | Cause | REST status |
|---|---|---|
| `geometry` | request | 422 |
| `mixed_pixel_types` | request | 422 |
| `unreadable` | request | 422 |

<a id="InstallationFileError"></a>
### `InstallationFileError`

A fault; bases `Fault`, `Exception`.

A file the installation ships is missing, unreadable, or not the shape its reader needs.

The installation is at fault, not the user's settings, so this is
deliberately not a `ConfigError`: a host that answers a `ConfigError`
by falling back to the shipped settings template would replace good
settings and still fail on the same file. The reader raises it and logs
nothing; whoever catches it logs it once.

**Attributes**

- `file_path`: The file that could not be used.

| Reason | Cause | REST status |
|---|---|---|
| `installation_file_unusable` | -- | 500 |

<a id="LaneCallStillRunningError"></a>
### `LaneCallStillRunningError`

A fault; bases `Fault`, `Exception`.

The API's own bound on a call ran out before the call finished.

The hardware has stopped answering or is slower than the bound allows,
and the call may still finish: nothing says it failed, nor that it
will not change the scope after this. A fault, never a `TimeoutError`,
which means only that the caller's own wait ran out.

**Attributes**

- `reason`: `'call_overran'`.
- `member`: The API member whose call is still running.
- `timeout_s`: The bound that ran out.

| Reason | Cause | REST status |
|---|---|---|
| `call_overran` | -- | 500 |

<a id="LayerSettingCappedNotice"></a>
### `LayerSettingCappedNotice`

A notice; bases `Notice`, `Exception`.

Saved layer gains or exposures above this camera's range; each runs at the camera's limit.

The saved value is the person's intent and is kept: a camera that can
reach it applies it again. One notice names every capped layer value
one bring-up found, as `StoredSettingReplacedNotice` names every
replaced one, since the notification centre shows one notice of a kind
at a time.

**Attributes**

- `capped`: `(layer, setting, stored, applied)` for each capped value: the layer, `'gain_db'` or `'exposure_ms'`, the saved value and the camera's limit it runs at.

| Reason | Cause | REST status |
|---|---|---|
| `layer_setting_capped` | -- | 409 |

<a id="LedBoardUnavailableError"></a>
### `LedBoardUnavailableError`

A fault; bases `Fault`, `BringUpError`.

The LED board did not come up on a scope whose other parts did.

Said once at bring-up rather than once per failed illumination command:
without it the first symptom is a sample under a dark objective and
controls that appear to do nothing. The advice follows the cause.

**Attributes**

- `reason`: The hardware registry's cause, `'not_detected'` and `'port_in_use'` as `CameraNotAvailableError` explains them, or: `'not_responding'` -- the board's port opened and the board answered nothing. `'connect_failed'` -- the board's driver raised while connecting. `'no_driver'` -- no real LED board driver is registered on this installation.

| Reason | Cause | REST status |
|---|---|---|
| `connect_failed` | -- | 500 |
| `no_driver` | -- | 500 |
| `not_detected` | -- | 500 |
| `not_responding` | -- | 500 |
| `port_in_use` | -- | 500 |

<a id="LedFirmwareWedgedError"></a>
### `LedFirmwareWedgedError`

A fault; bases `Fault`, `HardwareError`.

The LED controller's firmware is stuck and did not recover on a soft reset.

Only a power cycle brings it back, and its words say so.

| Reason | Cause | REST status |
|---|---|---|
| `led_firmware_wedged` | -- | 500 |

<a id="LedSafetyOffNotTakenError"></a>
### `LedSafetyOffNotTakenError`

A fault; bases `Fault`, `BringUpError`.

The LED board connected but did not confirm the LEDs-off sent on connect.

Sample safety: firmware before the confirmation existed can leave channels
on, photobleaching whatever is on the stage.

**Attributes**

- `reason`: `'safety_off_failed'`.

| Reason | Cause | REST status |
|---|---|---|
| `safety_off_failed` | -- | 500 |

<a id="LiveFolderFileNotWrittenError"></a>
### `LiveFolderFileNotWrittenError`

A fault; bases `Fault`, `Exception`.

A file put into the live folder could not be written.

Raised by `ScopeSession.put_live_folder_file`, chained from the
`OSError` of the write, so the words name the file and the operating
system's own reason -- a read-only or full disk, a denied permission.
Nothing is left at the name.

**Attributes**

- `name`: The name under the live folder the file was to be written at.

| Reason | Cause | REST status |
|---|---|---|
| `live_folder_file_not_written` | -- | 500 |

<a id="LiveFolderPathRefusedError"></a>
### `LiveFolderPathRefusedError`

A refusal; bases `Refusal`, `ValueError`.

A path the caller gave was refused: it cannot be used as the member uses it.

Raised by `ScopeSession.live_folder_path` for a wire caller's name,
which may reach only the live folder: a name that is a path of its own,
or that climbs out, reaches the rest of the machine. Raised by the
`path_utils` checks for any caller's path that is missing or of the
wrong kind, before the member reads or writes it.

**Attributes**

- `reason`: `'outside_live_folder'` -- the name is empty, absolute, carries a drive or a network share, or leads outside the live folder through `..` or a link. `'capture_location_unusable'` -- the live folder itself is missing or not a folder. `'path_not_found'` -- nothing is at the path. `'not_a_file'` -- a folder is where a file is expected. `'not_a_folder'` -- a file is where a folder is expected. `'wrong_extension'` -- a file to write is named without its kind's extension (a protocol's `.tsv`); the name given is the name written, so nothing adds it. `'file_exists'` -- a file to create is already there; an upload never replaces a file.
- `name`: The name or path that was given.

| Reason | Cause | REST status |
|---|---|---|
| `capture_location_unusable` | state | 409 |
| `file_exists` | request | 422 |
| `not_a_file` | request | 422 |
| `not_a_folder` | request | 422 |
| `outside_live_folder` | request | 422 |
| `path_not_found` | request | 422 |
| `wrong_extension` | request | 422 |

<a id="MotionInterlockError"></a>
### `MotionInterlockError`

A fault; bases `Fault`, `HardwareError`.

The board refused a motion command because a hardware interlock is open.

Raised by a driver whose board guards motion with its own inputs (a
lid switch, the stage's power), read on the wire with the command they
guard. The API turns it into its refusal or its failed home, so it
carries what the API needs to say which one happened: whether any part
of the refused command had already moved, and whether the driver tried
to stop the motors in refusing it.

**Attributes**

- `reason`: `'stage_unpowered'` -- the stage has no power. Or `'lid_open'`, the stage's lid is open, or was opened while the command moved, as `HomingFaultError` explains it for a home.
- `moved`: Whether anything of the refused command had moved.
- `stopped`: Whether the driver sent every motor a stop in refusing it, settled or not; False when no stop was sent, as for a stage without power. The API counts a sent stop as a Stop.

| Reason | Cause | REST status |
|---|---|---|
| `lid_open` | -- | 500 |
| `stage_unpowered` | -- | 500 |

<a id="MotionNotCompletedError"></a>
### `MotionNotCompletedError`

A fault; bases `Fault`, `HardwareError`.

A driven motion did not reach its end.

An axis that did not home, arrive or settle in the time it is given, or
a home that ended before its phases did. Whether a Stop ended it is the
API's stop generation to say, never this type.

| Reason | Cause | REST status |
|---|---|---|
| `motion_not_completed` | -- | 500 |

<a id="MotorStopFailedError"></a>
### `MotorStopFailedError`

A fault; bases `Fault`, `Exception`.

The motor STOP command failed, so the stage may still be moving.

Raised by `stop_motion` when the board did not take the STOP, chained
from the driver's error. A failure, not a refusal: the STOP was sent
and nothing vouches that it landed. The stop generation has already
moved, so no waited move reads itself as arrived.

| Reason | Cause | REST status |
|---|---|---|
| `motor_stop_failed` | -- | 500 |

<a id="MoveFaultError"></a>
### `MoveFaultError`

A fault; bases `Fault`, `MoveNotCompletedError`.

A move failed: the motor may have travelled any part of the way.

**Attributes**

- `reason`: `'driver_failed'` -- the motor board did not take the command (chained from the driver's error). `'stalled'` -- the motion monitor gave the axis up: its position stopped changing short of its target, and the stage was halted. `'monitor_ended_mid_move'` -- the motion monitor ended while the axis moved, so nothing could judge the move. `'board_lost'` -- the monitor lost the motor board while the axis moved. `'faulted'` -- something else set the axis UNKNOWN during the move, carrying no fault of its own, with no stop before it. `'position_unread_after_move'` -- the board did not report where the axis is: within the stall window after it reported the axis arrived, or when a home read every axis's position. `'status_unread'` -- the board did not report whether the axis arrived, within the stall window. Each leaves the axis UNKNOWN.
- `title`: The heading shown with the sentence, which follows the reason.

| Reason | Cause | REST status |
|---|---|---|
| `board_lost` | -- | 500 |
| `driver_failed` | -- | 500 |
| `faulted` | -- | 500 |
| `monitor_ended_mid_move` | -- | 500 |
| `position_unread_after_move` | -- | 500 |
| `stalled` | -- | 500 |
| `status_unread` | -- | 500 |

<a id="MoveNotCompletedError"></a>
### `MoveNotCompletedError`

A subject: never raised, its children are; bases `Exception`.

A move ended without its axis arriving at the target: the subject, never raised itself.

A waited move that returns means the axis arrived; this move could not
say that, so it raises instead of reporting a position nobody reached.
`except MoveNotCompletedError` catches every way a move ends short;
the child raised says which, and its kind with it: `MoveFaultError`
when the move failed, `MoveStoppedError` when a stop or a later
command ended it.

Distinct from `AxisStateUnknownError`, which refuses a move before
anything is driven and offers `force=True` -- advice that is wrong
for a move that already happened.

**Attributes**

- `axis`: The axis whose move did not complete.
- `reason`: The child's code for how the move ended.
- `title`: The heading shown with the sentence.

<a id="MoveStoppedError"></a>
### `MoveStoppedError`

A notice; bases `Notice`, `MoveNotCompletedError`.

A move was ended before it arrived by a stop or a later command: nothing failed.

A notice: the person whose move someone else stopped is told so, as
information, and a REST caller is answered 409. The axis is where the
stop or the later command left it, which its position reports; after
the scope is closed its position is unknown.

**Attributes**

- `reason`: `'stopped'` -- a stop sent while it moved ended it: `stop_motion`, the stage's own stop in refusing a move, or the close's, even one the board did not take (that failure is reported once, as itself); a turret is then in no known slot. `'superseded'` -- another move or a home took the axis before this one arrived; that command's own outcome says where the axis went.

| Reason | Cause | REST status |
|---|---|---|
| `stopped` | state | 409 |
| `superseded` | state | 409 |

<a id="MoveWaitExpiredError"></a>
### `MoveWaitExpiredError`

A quiet outcome; bases `MoveNotCompletedError`, `Quiet`, `TimeoutError`.

The caller's own wait ran out with the axis still moving: nothing failed.

A `TimeoutError`, as a caller's own wait expiring always is, and quiet:
the move goes on, and the axis keeps the state its own move gives it.
`TimeoutError` is the last base, so its `OSError` constructor never
reads the axis and reason as an errno.

**Attributes**

- `reason`: `'still_moving'`.

| Reason | Cause | REST status |
|---|---|---|
| `still_moving` | state | 409 |

<a id="NoHardwareDetectedNotice"></a>
### `NoHardwareDetectedNotice`

A notice; bases `Notice`, `Exception`.

Nothing came up at bring-up: no LED board, no motor board, no camera.

One notice for the whole scope, in place of one per part: the person
with no instrument attached needs to be told once, not three times.

| Reason | Cause | REST status |
|---|---|---|
| `no_hardware` | -- | 409 |

<a id="NoLiveDriverError"></a>
### `NoLiveDriverError`

A fault; bases `Fault`, `RuntimeError`.

'auto' found no live driver, and the kind has no null driver to hand back.

The camera's kind has none: bring-up records the cause this carries, as
a board's `DriverFallback` is recorded beside its null driver.

**Attributes**

- `fallback`: The cause, the drivers tried, and the drivers declined.

| Reason | Cause | REST status |
|---|---|---|
| `no_live_driver` | -- | 500 |

<a id="NoWireFormError"></a>
### `NoWireFormError`

A fault; bases `Fault`, `TypeError`.

A value whose type has no wire form reached `encode`.

| Reason | Cause | REST status |
|---|---|---|
| `no_wire_form` | -- | 500 |

<a id="NotHandedOutError"></a>
### `NotHandedOutError`

A fault; bases `NoWireFormError`.

A parameter takes only a live object no wire member hands out, so no client can name one.

| Reason | Cause | REST status |
|---|---|---|
| `not_handed_out` | -- | 500 |

<a id="Notice"></a>
### `Notice`

The notice kind; bases `LumascopeError`.

An outcome that tells the person something: nothing failed and nothing was declined.

Mixed into an exception whose message is written for the person -- a
capture saved without its position, a long operation starting, a move
someone else stopped. Most are reported, never raised: nothing waits on
them, so the reporter is the whole of their flight. One that ends a
caller's request is raised, as a stopped move is, and reported where
its flight stops. It is shown as a notice under `title` in its own
words, and logged at NOTICE. Its kind belongs to its type, as a
refusal's does, so every client that hears it reads the same answer.

A notice that is raised maps its reason to whether the same request
could succeed later, as a refusal's does; one that is only reported maps
it to None.

**Attributes**

- `title`: The heading the person reads above the message.

<a id="ObjectiveUnknownError"></a>
### `ObjectiveUnknownError`

A refusal; bases `Refusal`, `ConfigError`.

No one can say which objective is in the light path.

On a turreted scope the active objective is the objective assigned to
the slot in the light path, so it is unknown when the slot is unknown
(the turret has not been homed or moved since it was last lost), when
the slot has no assignment, or when its assignment names nothing in
the objective catalogue. Raised instead of answering with a stored
objective, because an objective that is not in the light path puts a
wrong scale into every image it names.

On a scope with no turret it is unknown only before any objective was
selected.

**Attributes**

- `reason`: `'slot_unknown'` -- the turret is in no known slot: it has not been homed or moved since its slot was last lost. `'slot_unassigned'` -- the slot in the light path has no objective assigned. `'not_in_catalogue'` -- the slot in the light path is assigned an objective that is not in the objective catalogue. `'none_selected'` -- no objective has been selected, as on a scope with no turret before its first selection.
- `slot`: The slot in the light path, or None when that is what is unknown.

| Reason | Cause | REST status |
|---|---|---|
| `none_selected` | state | 409 |
| `not_in_catalogue` | state | 409 |
| `slot_unassigned` | state | 409 |
| `slot_unknown` | state | 409 |

<a id="OperationSettleRefusedError"></a>
### `OperationSettleRefusedError`

A refusal; bases `Refusal`, `RuntimeError`.

An operation's `set_result` or `set_exception` was called by code holding it.

Only its producer settles an operation: a holder's settle would forge
the outcome every other holder reads, and make the producer's own settle
fail where it is made -- on a run's delivery, a recording's finish, a
move's monitor. It refuses the accident and the casual forge, not a
determined caller: code in the same process can settle through the base
class, as the producers do. Still a `RuntimeError`, as a Future's own
refusal of a second settle is.

**Attributes**

- `reason`: `'settled_by_its_producer'`.
- `member`: The member that was called.

| Reason | Cause | REST status |
|---|---|---|
| `settled_by_its_producer` | request | 422 |

<a id="PartAbsentError"></a>
### `PartAbsentError`

A fault; bases `Fault`, `HardwareError`.

A driver was asked for a part it does not have: an axis, a turret, a
board, an active camera or an SDK node.

A fault, not a refusal: the API refuses a command for a part the scope
lacks before the driver is asked (`axis_absent`, `not_on_this_scope`,
`not_connected`), so only a caller below the API reaches this.

| Reason | Cause | REST status |
|---|---|---|
| `part_absent` | -- | 500 |

<a id="PartialHardwareError"></a>
### `PartialHardwareError`

A fault; bases `Fault`, `BringUpError`.

Parts this scope's model has did not come up; the rest of the scope runs.

**Attributes**

- `reason`: `'partial_hardware'`.
- `missing`: Each missing part with its cause, as `PartStatus.describe` writes it.

| Reason | Cause | REST status |
|---|---|---|
| `partial_hardware` | -- | 500 |

<a id="PluginError"></a>
### `PluginError`

A subject: never raised, its children are; bases `Exception`.

A plugin failed: it did not load, or it failed after loading. The subject, never raised itself.

Nothing raises these: the plugin host catches the plugin's own failure
and has no caller to raise to, so each is built to be reported, chained
from the plugin's exception when there is one so the log record carries
that traceback. A plugin is separately versioned and may not be ours,
so its failure is a fault for the person to hear about while the rest
of the application carries on.

Each kind's title is its class's; the words name the plugin.

<a id="PluginFailedError"></a>
### `PluginFailedError`

A fault; bases `Fault`, `PluginError`.

A loaded plugin failed while the host was calling it.

**Attributes**

- `plugin_name`: The plugin that failed.
- `hook`: What the host was calling the plugin for.
- `detail`: The plugin's own account of the failure, when it gave one.

| Reason | Cause | REST status |
|---|---|---|
| `plugin_failed` | -- | 500 |

<a id="PluginNotLoadedError"></a>
### `PluginNotLoadedError`

A fault; bases `Fault`, `PluginError`.

Plugins found at discovery did not load, every one of them in one outcome.

Discovery reports once, when it ends: the reporter merges notices that
share a title within its window, so one outcome per plugin would show
only the first of several failing together. The host chains every
plugin's own exception, as an `ExceptionGroup`, so the one log record
carries each traceback.

**Attributes**

- `failures`: What did not load, each with its name and why, in the order discovery found them.

| Reason | Cause | REST status |
|---|---|---|
| `plugin_not_loaded` | -- | 500 |

<a id="PluginProcessorSkippedError"></a>
### `PluginProcessorSkippedError`

A fault; bases `Fault`, `PluginError`.

A run's post-processing by a plugin did not run: the plugin was unloaded first.

The session's close unloads plugins before it waits for the work under
way, because a plugin's `unregister` is what stops its own work; a
run that finishes during that wait -- or after a host unloaded the
plugins itself -- then hands its folder to processors no longer loaded. The folder and its images are complete; only the
plugin's processing of it is missing, so the person is told which run
to process again.

**Attributes**

- `run`: The run whose folder was not processed, by its protocol name or its folder.
- `folder`: The run's folder.
- `plugin_name`: The plugin whose processing did not run.

| Reason | Cause | REST status |
|---|---|---|
| `processor_skipped` | -- | 500 |

<a id="PluginRegistrationError"></a>
### `PluginRegistrationError`

A refusal; bases `Refusal`, `Exception`.

A plugin asked a namespace for something this LumaViewPro does not give.

The plugin is NOT loaded and the app continues. The loader catches the
raise from the plugin's register(ctx) and reports it as that plugin's
load failure. The same registration fails again unchanged, so its cause
is the request.

**Attributes**

- `reason`: `'name_taken'` -- the namespace already holds a plugin of that name; `'unknown_mount_point'` -- no UI mount point has that name; `'rest_namespace_reserved'` -- a plugin cannot add REST routes in this release.

| Reason | Cause | REST status |
|---|---|---|
| `name_taken` | request | 422 |
| `rest_namespace_reserved` | request | 422 |
| `unknown_mount_point` | request | 422 |

<a id="PositionOutOfRangeError"></a>
### `PositionOutOfRangeError`

A refusal; bases `Refusal`, `ValueError`.

An absolute move was commanded beyond the axis's travel.

The driver's own response to an out-of-travel target is to clamp it
to the nearest limit and drive there, which reports success at a
position nobody asked for: a protocol step saved beyond this scope's
travel images the wrong place, and nothing in the log distinguishes
that from a step that went where it was told. Refusing by name makes
the substitution impossible rather than silent.

Subclasses ValueError because an out-of-travel target is the same
kind of bad argument as a non-numeric one, and callers already
written to catch ValueError from this call keep working.

The message reaches the user verbatim, so it names the axis, the
request, and the range that refused it.

`bound` names WHICH limit refused, because two of them can: the
axis's own travel, and the coarse safety ceiling that rejects a
nonsense magnitude before any axis is consulted. Telling someone
their entry is "outside the travel range 0.0 to 80000.0" when it was
really refused as absurd points them at the wrong number. `quantity`
likewise distinguishes a position from a relative distance. One
optional argument each rather than a second exception class: the
refusal is the same event, and only the sentence differs.

**Attributes**

- `axis`: The axis whose position or distance was refused.
- `position`: What was asked for.
- `limits`: The lowest and highest value `bound` allows, in the quantity's own unit.
- `bound`: Which limit refused it.
- `quantity`: What was refused: a position, a distance, a plate position.

`axis` and `limits` are published, so a client corrects the move
without reading the words.

**Fields**

- <a id="PositionOutOfRangeError.axis"></a>`axis`: `Axis` -- The axis whose position or distance was refused.
- <a id="PositionOutOfRangeError.limits"></a>`limits`: `tuple[float, float]` -- The lowest and highest value `bound` allows, in the quantity's own unit.

| Reason | Cause | REST status |
|---|---|---|
| `position_out_of_range` | request | 422 |

<a id="PostProcessingFailedError"></a>
### `PostProcessingFailedError`

A fault; bases `Fault`, `CaptureError`.

A post-processing build did not produce everything it was asked for.

Raised whether nothing or only part was produced: a group that failed,
a group refused because its output would share a name with another's,
or frames a video could not add. What WAS produced rides along, so a
caller that keeps artifacts keeps them, and no caller can read an
incomplete build as a complete one.

**Attributes**

- `operation`: The operation's name as a person reads it ("Stitch").
- `produced_paths`: Every artifact that was written.
- `output_root`: The folder the artifacts went under, or None.
- `errors`: Every failed group's error, in order.
- `excluded_inputs`: The files of each kind the build did not take, as a complete build's result names them; empty for an operation that reads no protocol folder (Quick Enhance, a cell count).
- `reason`: `'post_processing_failed'` -- the build produced nothing. `'post_processing_incomplete'` -- the build produced part of what it was asked for, named in `produced_paths`.

**Fields**

- <a id="PostProcessingFailedError.excluded_inputs"></a>`excluded_inputs`: `tuple[ExcludedInput, ...]` -- The files of each kind the build did not take, as a complete build's result names them; empty for an operation that reads no protocol folder (Quick Enhance, a cell count).

| Reason | Cause | REST status |
|---|---|---|
| `post_processing_failed` | -- | 500 |
| `post_processing_incomplete` | -- | 500 |

<a id="PostProcessingRefusedError"></a>
### `PostProcessingRefusedError`

A refusal; bases `Refusal`, `CaptureError`.

What a post-processing operation was given cannot yield the output asked of it.

Nothing broke: the folder holds no images, no groups this operation can
combine, only derived outputs, only groups whose outputs would share a
name, source images in a format the operation cannot re-read, protocol
data that could not be loaded, or outputs the folder's protocol would
name outside the folder; or the cell-count method it was
given cannot be used. The message says which, in words written for the
person, and what to do. Or the scope's state refused it: a build
started while something holds the scope or a run's files are still
being written would share the process's cores, interpreter lock and
disk with that work (`'scope_held'`, `'files_writing'`), and the
same request may succeed once it ends.

**Attributes**

- `operation`: The operation's name as a person reads it ("Stitch").
- `reason`: Machine-readable refusal code. `'axis_not_plottable'` -- a graph's X or Y names a column of the results file that axis cannot take. `'collision'` -- every image group would write an output under a name another group's output shares, so nothing could be built. `'excluded_inputs'` -- the folder holds only derived outputs, which the build does not take as input. `'fit_impossible'` -- the graphed values cannot give the trendline asked for. `'graph_incomplete'` -- a graph was asked for without a results file, an X axis and a Y axis. `'invalid_path'` -- no folder was given to process. `'invalid_setting'` -- a setting of the build is not one it can use, such as no build chosen, an image shape or scale that is not one, or a playback rate below 1 frame per second. `'method_invalid'` -- the cell-count method holds something the count cannot use. `'method_unreadable'` -- the cell-count method file cannot be read, is not JSON, or has no metadata naming its type and version. `'no_data'` -- the folder holds images, but not the data this operation needs: no Z-stack for a z-projection, no recording frames for a video, nothing a count could count, or a recording where stills are needed. `'no_images'` -- the folder holds no image files. `'no_scale'` -- the image states no scale, so it is counted in pixels, and the method filters a size in microns. `'output_outside_folder'` -- the folder's protocol names a step whose output would be written outside the folder. `'protocol_data_unreadable'` -- the folder's protocol data could not be loaded. `'results_unreadable'` -- the cell-count results file cannot be read as one LumaViewPro writes. `'unsupported_source_format'` -- the source images are not TIFF or OME-TIFF, which the build re-reads. `'scope_held'` -- something holds the scope, and a build now would share the computer with its work. `'files_writing'`, as `ProtocolRunRefusedError` explains it, and `'unreadable'`, as `InputFramesUnusableError` explains it, are the same facts.

| Reason | Cause | REST status |
|---|---|---|
| `axis_not_plottable` | request | 422 |
| `collision` | request | 422 |
| `excluded_inputs` | request | 422 |
| `files_writing` | state | 409 |
| `fit_impossible` | request | 422 |
| `graph_incomplete` | request | 422 |
| `invalid_path` | request | 422 |
| `invalid_setting` | request | 422 |
| `method_invalid` | request | 422 |
| `method_unreadable` | request | 422 |
| `no_data` | request | 422 |
| `no_images` | request | 422 |
| `no_scale` | request | 422 |
| `output_outside_folder` | request | 422 |
| `protocol_data_unreadable` | request | 422 |
| `results_unreadable` | request | 422 |
| `scope_held` | state | 409 |
| `unreadable` | request | 422 |
| `unsupported_source_format` | request | 422 |

<a id="ProtocolError"></a>
### `ProtocolError`

A subject: never raised, its children are; bases `Exception`.

A protocol's file, steps or run: the subject, never raised itself.

Its children's words are written for the person unless a child says
otherwise.

<a id="ProtocolFormatError"></a>
### `ProtocolFormatError`

A refusal; bases `Refusal`, `ProtocolError`.

A protocol LumaViewPro cannot take: a file it cannot parse, or steps that do not validate.

A refusal: the file or the configuration is the person's, nothing broke,
and the answer is to fix or choose another. The words say what is wrong
and, for a file, name it, so the one report -- a warning in these words,
one log line -- needs no log line of its own beside each check.

**Attributes**

- `reason`: `'protocol_invalid'`.
- `file`: The file refused, or None for a protocol built in memory.

| Reason | Cause | REST status |
|---|---|---|
| `protocol_invalid` | request | 422 |

<a id="ProtocolNotLoadedError"></a>
### `ProtocolNotLoadedError`

A fault; bases `Fault`, `ProtocolError`.

A protocol file could not be read.

Raised by `Protocol.from_file`, chained from the `OSError` of the
read, so the words name the file and the operating system's own reason
-- a missing file, a denied permission, a drive that went away. A file
that was read but is not a protocol LumaViewPro can take is
`ProtocolFormatError` instead.

**Attributes**

- `file`: The path that could not be read.

| Reason | Cause | REST status |
|---|---|---|
| `protocol_not_loaded` | -- | 500 |

<a id="ProtocolNotSavedError"></a>
### `ProtocolNotSavedError`

A fault; bases `Fault`, `ProtocolError`.

A protocol could not be written to its file.

Raised by `Protocol.to_file`, chained from the `OSError` of the write,
so the words name the file and the operating system's own reason -- a
missing or read-only folder, a full disk, a file another program holds --
rather than guessing one. The write goes to a file beside the target
first, so a file already there is unchanged.

**Attributes**

- `file`: The path the protocol was to be written to.

| Reason | Cause | REST status |
|---|---|---|
| `protocol_not_saved` | -- | 500 |

<a id="ProtocolRunRefusedError"></a>
### `ProtocolRunRefusedError`

A refusal; bases `Refusal`, `ProtocolError`.  
Held in: [`Status.run_refused`](#Status.run_refused).

A sequenced run, or the protocol it would run, was refused before any state was committed.

Raised by SequencedCaptureRunner.prepare() when a run cannot start
(already running, files still writing, empty protocol, validation
errors, hardware not connected), by the protocols API when a protocol
or a step names something this scope cannot do, and by the protocol
builder for a z-stack asked for with no range. Each of those reports it
through the one reporter as it raises, so it has been logged and shown
once already; any later report of the same exception is a no-op, and a
caller reconciles its own state without telling anyone again.

Its message is the sentence a person reads, so a headless caller that
prints it prints what the GUI shows.

**Attributes**

- `reason`: Machine-readable refusal code for callers that map refusals to responses (REST status codes, UI branches). `'already_running'` -- a run already holds the scope. `'exclusive_activity_running'` -- another exclusive activity, such as a diagnostic, a home or a run, holds the scope, and the work asked for is not made under it. `'recording_active'` -- a video recording is in progress, or a previous one is still finishing. `'files_writing'` -- the last run is still writing its files. `'files_writing_stalled'` -- the last run's file writer has stopped making progress, and has to be recovered before a new run starts. `'post_processing_running'` -- a build is on the post-processing lane, and a capture never overlaps one. `'autofocus_running'` -- an autofocus sweep from a run that has ended did not stop, and still occupies the autofocus thread. `'not_connected'` -- a part the work needs is not connected, and none reconnects during a session. `'camera_not_streaming'` -- the camera is connected and delivers no frames. `'position_unknown'` -- an axis this scope has does not know its position, so the scope has to be homed first. `'lid_open_before_motion'` -- the stage's lid is open, so the stage refuses to move. `'no_acquiring_layer'` -- no channel, or not the step's, is set to acquire, so there is no step to make. `'channel_already_in_add'` -- another step added with this one already takes the channel, and steps added together are one composite with one image per channel. `'composite_two_transmitted_layers'` -- a composite run has more than one transmitted channel set to capture. `'turret_slot_ambiguous'` -- a step recorded no turret slot, and its objective is assigned to two or more. `'turret_slot_reassigned'` -- the turret slot a step recorded now holds another objective. `'step_position_unknown'` -- a step cannot be added or updated while an axis does not know its position. `'turret_objectives_unassigned'` -- the protocol uses an objective no turret slot is assigned. `'objective_not_mounted'` -- on a scope with no turret, the protocol names an objective other than the selected one. `'sequence_name_invalid'` -- the run's name is a path: it holds a separator or a drive, or is `..`. `'empty_protocol'` -- the protocol has no steps. `'validation_failed'` -- the protocol's steps fail the pre-run validation; the message names the errors. `'composite_needs_two_channels'` -- a composite run has fewer than two channels set to capture. `'tiling_unknown'` -- the tile grid asked for is not one this installation offers. `'tiling_scale_unknown'` -- the scope cannot report its pixel size, so a tile grid's spacing is unknown. `'zstack_not_configured'` -- a z-stack's range or step size is not greater than zero. `'already_tiled'` -- the protocol is tiled already, and a second grid cannot be laid over it. `'tiles_outside_travel'` -- tiles of the grid fall outside the stage's travel. `'zslices_outside_travel'` -- z-slices of the stack fall outside Z's travel. `'positions_outside_travel'` -- steps of the protocol fall outside the stage's travel. `'camera_setting_out_of_range'` -- a step's gain or exposure is outside this camera's range. `'positions_unreachable'` -- the work needs motion on an axis this scope has no motor for. `'step_position_missing'` -- steps have no position on an axis this scope moves, having been recorded on a scope without it. `'layer_not_on_scope'` -- steps take or stimulate a layer this scope does not have, or this scope's layers could not be resolved. `'objectives_require_turret'` -- the protocol changes objective between steps, and this scope has no turret to change it. `'objective_not_given'` -- a protocol with a channel set to acquire was built from a configuration that names no objective. `'focus_not_given'` -- a layer has no saved focus, and the configuration names no current Z to image it at. `'overlap_out_of_range'` -- the tile overlap is outside 0 to 50 percent. `'values_out_of_range'` -- layer settings the protocol would store hold values no layer setting takes; the message names each.
- `title`: The heading shown above the sentence, the reason's.
- `message`: The sentence; the same text as the exception's message.
- `holder`: What holds the microscope at refusal time (the exclusive-activity claim's kind, or `AUTOFOCUS` for a sweep in flight), or None when the refusal is not holder-shaped (validation, hardware, file drain).
- `holder_trigger`: Who started the run that holds the scope -- `'gui'`, `'rest'` or `'api'` (`modules.caller`); for a file-drain refusal the just-finished run's, for an autofocus sweep the run that dispatched it. None when no run is behind the holder -- a recording's kind IS the holder.
- `remedy`: The action that answers this refusal, or None. A stalled file writer has one (recover it); a writer still making progress does not, since its files will land.

`holder` and `holder_trigger` are published, so a REST problem
carries who holds the scope beside the words.

**Fields**

- <a id="ProtocolRunRefusedError.holder"></a>`holder`: `Holder | None` -- What holds the microscope at refusal time (the exclusive-activity claim's kind, or `AUTOFOCUS` for a sweep in flight), or None when the refusal is not holder-shaped (validation, hardware, file drain).
- <a id="ProtocolRunRefusedError.holder_trigger"></a>`holder_trigger`: `Caller | None` -- Who started the run that holds the scope -- `'gui'`, `'rest'` or `'api'` (`modules.caller`); for a file-drain refusal the just-finished run's, for an autofocus sweep the run that dispatched it. None when no run is behind the holder -- a recording's kind IS the holder.

| Reason | Cause | REST status |
|---|---|---|
| `already_running` | state | 409 |
| `already_tiled` | request | 422 |
| `autofocus_running` | state | 409 |
| `camera_not_streaming` | state | 409 |
| `camera_setting_out_of_range` | request | 422 |
| `capture_location_unusable` | state | 409 |
| `channel_already_in_add` | request | 422 |
| `channel_unknown` | request | 422 |
| `composite_needs_two_channels` | request | 422 |
| `composite_two_transmitted_layers` | request | 422 |
| `empty_protocol` | request | 422 |
| `exclusive_activity_running` | state | 409 |
| `files_writing` | state | 409 |
| `files_writing_stalled` | state | 409 |
| `focus_not_given` | request | 422 |
| `layer_not_on_scope` | request | 422 |
| `lid_open_before_motion` | state | 409 |
| `no_acquiring_layer` | state | 409 |
| `not_connected` | restart | 503 |
| `objective_not_given` | request | 422 |
| `objective_not_in_catalogue` | request | 422 |
| `objective_not_mounted` | state | 409 |
| `objectives_require_turret` | request | 422 |
| `overlap_out_of_range` | request | 422 |
| `position_unknown` | state | 409 |
| `positions_outside_travel` | request | 422 |
| `positions_unreachable` | request | 422 |
| `post_processing_running` | state | 409 |
| `recording_active` | state | 409 |
| `sequence_name_invalid` | request | 422 |
| `step_position_missing` | request | 422 |
| `step_position_unknown` | state | 409 |
| `tiles_outside_travel` | request | 422 |
| `tiling_scale_unknown` | request | 422 |
| `tiling_unknown` | request | 422 |
| `turret_objectives_unassigned` | state | 409 |
| `turret_slot_ambiguous` | state | 409 |
| `turret_slot_reassigned` | state | 409 |
| `validation_failed` | request | 422 |
| `values_out_of_range` | request | 422 |
| `zslices_outside_travel` | request | 422 |
| `zstack_not_configured` | request | 422 |

<a id="ProtocolScheduleRefusedError"></a>
### `ProtocolScheduleRefusedError`

A refusal; bases `ProtocolFormatError`.

A period or duration the scope cannot run, refused rather than changed.

**Attributes**

- `reason`: `'schedule_not_runnable'`.
- `key`: `'period'` or `'duration'`.
- `value`: What was given, as given.

| Reason | Cause | REST status |
|---|---|---|
| `schedule_not_runnable` | request | 422 |

<a id="ProtocolStepsInvalidNotice"></a>
### `ProtocolStepsInvalidNotice`

A notice; bases `Notice`, `Exception`.

A protocol holds a step the run will refuse, after a load or an edit.

The load and the edit still succeed, so the step can be fixed in the
app; the run is refused at start until every step is valid. The words
are the validator's own lines, one per field.

| Reason | Cause | REST status |
|---|---|---|
| `protocol_steps_invalid` | -- | 409 |

<a id="Quiet"></a>
### `Quiet`

The quiet outcome kind; bases `LumascopeError`.

An outcome that is recorded and never shown: nothing failed and nothing was declined.

Mixed into an exception that tells a caller something it may need to act
on -- a script learning its handle is stale -- but that the person at the
instrument has no reason to see. It is logged at INFO in its own words and
never becomes a notification. Whether an outcome is quiet belongs to its
type, never to the code that raises or catches it, so every client reads
the same answer. A message on a quiet exception is written for the log.

Every reason maps to its `RefusalCause` in `causes`, as a refusal's
does, since a quiet outcome answers a caller's request too.

<a id="RecordIncompleteError"></a>
### `RecordIncompleteError`

A fault; bases `Fault`, `CaptureError`.

Captures a run made are missing from its execution record.

Post-processing reads the record to find a run's images, so an image
with no row is skipped by stitching and video builds.

**Attributes**

- `missing`: Captures with no row.
- `attempted`: Captures the run attempted.

| Reason | Cause | REST status |
|---|---|---|
| `record_incomplete` | -- | 500 |

<a id="RecordingDetailsNotSavedError"></a>
### `RecordingDetailsNotSavedError`

A fault; bases `Fault`, `CaptureError`.

A recording's details file could not be written; its frames are whole.

Chained from the write's error. The details file is the only record of
the recording's channel color and measured rate, so a video built from
these frames later falls back to grayscale and a default rate.

| Reason | Cause | REST status |
|---|---|---|
| `recording_details_not_saved` | -- | 500 |

<a id="RecordingFinalizeError"></a>
### `RecordingFinalizeError`

A fault; bases `Fault`, `CaptureError`.

A recording finished but its output could not be fully assembled.

Raised around what failed in the finish of a manual recording or of a
protocol video step, and chained from it. The frames already written
are on disk.

**Attributes**

- `reason`: `'video_step_finalize_failed'` -- a protocol video step's output could not be fully assembled. `'recording_finalize_failed'` -- a manual recording's output could not be fully assembled.
- `protocol_step`: True for a protocol video step, False for a manual recording; each is named in its own words.

| Reason | Cause | REST status |
|---|---|---|
| `recording_finalize_failed` | -- | 500 |
| `video_step_finalize_failed` | -- | 500 |

<a id="RecordingPositionNotRecordedNotice"></a>
### `RecordingPositionNotRecordedNotice`

A notice; bases `Notice`, `Exception`.

A frames recording started that cannot yet record where its frames are.

An unknown axis is recorded from the moment it becomes known; with no
labware or stage offset there is no plate frame to record X and Y in.

**Attributes**

- `unknown_axes`: The axes whose position is unknown at the start.
- `has_plate`: Whether a plate frame exists to record X and Y in.

| Reason | Cause | REST status |
|---|---|---|
| `position_not_recorded` | -- | 409 |

<a id="RecordingRefusedError"></a>
### `RecordingRefusedError`

A refusal; bases `Refusal`, `CaptureError`.

A video recording start was refused before any state was committed.

Raised when a recording cannot begin: by VideoRecordingEngine.start()
when an exclusive activity -- a protocol run or another recording --
already holds the session's activity claim or the engine is still
draining, and by the recording controllers for the caller-shaped
refusals they own (a previous recording still finishing, an inactive
camera, an unknown exposure, insufficient disk). Mirrors the
ProtocolRunRefusedError shape so callers reconcile state the same way
in both directions. Nothing reports it as it is raised: the caller that
asked for the recording reports it where its flight ends.

**Attributes**

- `reason`: Machine-readable refusal code for callers that map refusals to responses (REST status codes, UI branches). A reason it shares with another refusal is explained there: with `ProtocolRunRefusedError`, and `'capture_location_unusable'` and `'names_exhausted'` with `CaptureLocationError`. Its own: `'camera_exposure_unknown'` -- the camera has not reported its exposure time, from which the recording's dead-feed bound derives. `'insufficient_disk'` -- free disk at the save location is below the floor a recording needs to start. `'falsifying_change_in_flight'` -- an objective or camera frame change is under way, which a recording would capture through.
- `title`: Short user-facing refusal title, the reason's.
- `message`: One-sentence user-facing refusal body; the same text as the exception's message.
- `holder`: The exclusive-activity claim owner at refusal time, or None when the refusal is not claim-shaped.
- `holder_trigger`: Who started the holding run (`'gui'`, `'rest'`, `'api'`) when the holder is 'protocol'; None for a recording holder.

`holder` and `holder_trigger` are published.

**Fields**

- <a id="RecordingRefusedError.holder"></a>`holder`: `Holder | None` -- The exclusive-activity claim owner at refusal time, or None when the refusal is not claim-shaped.
- <a id="RecordingRefusedError.holder_trigger"></a>`holder_trigger`: `Caller | None` -- Who started the holding run (`'gui'`, `'rest'`, `'api'`) when the holder is 'protocol'; None for a recording holder.

| Reason | Cause | REST status |
|---|---|---|
| `camera_exposure_unknown` | state | 409 |
| `capture_location_unusable` | state | 409 |
| `exclusive_activity_running` | state | 409 |
| `falsifying_change_in_flight` | state | 409 |
| `insufficient_disk` | state | 409 |
| `names_exhausted` | state | 409 |
| `post_processing_running` | state | 409 |
| `recording_active` | state | 409 |

<a id="RecordingStoppedError"></a>
### `RecordingStoppedError`

A fault; bases `Fault`, `CaptureError`.

A manual recording was stopped before anyone asked it to stop.

Built by the recording's own watch when the camera stops delivering
frames or the free disk falls below the floor, and reported where it is
detected: no caller is waiting on a recording that is running. The
frames written so far are on disk.

**Attributes**

- `reason`: `'camera_disconnected'` -- the camera is no longer active. `'camera_stalled'` -- the camera stopped delivering frames for longer than the recording's exposure allows. `'disk_floor'` -- free disk fell below the safety floor.

| Reason | Cause | REST status |
|---|---|---|
| `camera_disconnected` | -- | 500 |
| `camera_stalled` | -- | 500 |
| `disk_floor` | -- | 500 |

<a id="Refusal"></a>
### `Refusal`

The refusal kind; bases `LumascopeError`.

A request the scope declined: nothing broke, and the person who asked can act on it.

Mixed into an exception whose message is written for that person. The
background executor shows a refusal as a warning under `title`, in the
exception's own words, and logs one line for it with no traceback: an
ERROR and a traceback say something went wrong, and a refusal is a
designed outcome. Unmarked, a refusal raised inside a background task
read as a crash -- "Background operation failed" over its message, and a
traceback in the errors log.

Every reason a refusal carries maps to its `RefusalCause` in
`causes`: whether the same request could succeed later with the
caller changing nothing.

**Attributes**

- `title`: The heading the person reads above the message.
- `remedy`: The action that answers this refusal, when one exists; the reporter then shows the refusal as an offer to take it. Set per instance: one reason of a refusal type can have a remedy that its others do not.

<a id="RestServerNotStartedError"></a>
### `RestServerNotStartedError`

A fault; bases `Fault`, `Exception`.

The REST server served beside the application could not take its port.

Raised by `rest.server.serve_beside`, chained from the `OSError` of
the bind, so the words name the address and the operating system's own
reason -- most often another program on the port. Nothing is served; the
application runs on without it.

**Attributes**

- `port`: The port that could not be taken.

| Reason | Cause | REST status |
|---|---|---|
| `rest_server_not_started` | -- | 500 |

<a id="RunCheckFailedError"></a>
### `RunCheckFailedError`

A fault; bases `Fault`, `ProtocolError`.

A run could not be checked before it started: a check itself crashed.

Raised by SequencedCaptureRunner.prepare() when validating the protocol,
reading whether the camera is streaming, or reading the stage's
interlocks, raised instead of answering. Not a refusal: nothing was
declined -- the question could not be asked, and the crash that stopped
it is chained as `__cause__` so its traceback is logged with this.
Nothing is committed and nothing needs unwinding, as for a refusal. Not
reported where it is raised: the caller that asked for the run reports
it where its flight ends.

**Attributes**

- `reason`: Machine-readable cause. `'validation_crashed'` -- the protocol's pre-run validation raised before it produced an answer. `'hardware_state_unknown'` -- a hardware read the check needs, whether the camera is streaming or which of the stage's interlocks are open, raised instead of answering.
- `title`: Short heading for the user, the reason's.
- `message`: The sentence a user reads; the same text as the exception's message.

| Reason | Cause | REST status |
|---|---|---|
| `hardware_state_unknown` | -- | 500 |
| `validation_crashed` | -- | 500 |

<a id="RunCleanupFailedError"></a>
### `RunCleanupFailedError`

A fault; bases `Fault`, `CaptureError`.

Steps that put the scope back after a run did not finish.

The run's own ending stands; what failed is the restore after it, so
the LEDs, the camera's gain and exposure or the stage may not be where
the person expects. The run's outcome names the steps too.

**Attributes**

- `steps`: Each failed step's name, in the order they failed.

| Reason | Cause | REST status |
|---|---|---|
| `cleanup_failed` | -- | 500 |

<a id="RunEndedByStopNotice"></a>
### `RunEndedByStopNotice`

A notice; bases `Notice`, `Exception`.

STOP was pressed while a run was live, so the run ended.

The run's outcome carries it as status `'aborted'`, reason
`'stage_stopped'`, in these words, and the person is told once when the
run ends: the run's owner may not be the one who pressed STOP (a REST
client, the GUI's button). The words say what was pressed, which is true
whether or not the board took the STOP; a STOP it did not take is
reported as that failure, where it was pressed.

| Reason | Cause | REST status |
|---|---|---|
| `stage_stopped` | -- | 409 |

<a id="RunFailedError"></a>
### `RunFailedError`

A fault; bases `Fault`, `CaptureError`.

The instrument ended a run: the fault that ended it, in its own words.

Fatal: the person is told even during an unattended run, since the run
they left is no longer running. The title and words are the run's
ending's, so a client reading the run's outcome and one hearing this
read the same.

**Attributes**

- `title`: The ending's heading, its reason's (`ENDING_TITLES`).
- `reason`: The ending's machine-readable cause, a run ending's reason; `RunStartError` and `RunFailedToStartError` carry the same. `'captures_failed'` -- the run reached its end without every capture it was asked for. `'start_failed'` -- the run's setup raised a fault of no other reason after the run was committed. `'dispatch_refused'` -- the run loop was refused when it was dispatched, so it never ran. `'capture_location_failed'` -- the run's folder could not be made at the capture location. `'run_dir_init_failed'` -- the run's folder was made and could not be initialized. `'camera_lane_stalled'` -- a camera command did not finish, so the run could not take the camera. `'camera_read_failed'` -- the camera's settings could not be read when the run took its snapshot of them. `'run_loop_crashed'` -- the run loop raised, or stopped without saying how the run ended. `'hardware_disconnected'` -- a part the run needs was found disconnected. `'consecutive_scan_failures'` -- the scan failed too many times in a row with the hardware still present. `'interlock'` -- the stage's interlock refused a move during a scan. `'position_lost'` -- an axis lost its position during a scan, and nothing in a run homes it again. `'disk_space_critical'` -- free disk at the save location fell below what the run needs, so the run stopped to save what it has. `'file_writer_stalled'` -- saving stopped making progress, so the run stopped before it lost more captures. `'video_writer_died'` -- the writer of a video step failed. `'led_channel_unavailable'` -- steps of the run take an LED channel this scope does not have, and their captures kept failing. `'camera_failure'` -- the camera failed too many steps in a row. `'autofocus_failed'` -- a single-autofocus run's autofocus did not find focus. `'autofocus_not_started'` -- a single-autofocus run's autofocus was refused before the stage moved, its first window reaching past Z's travel.

| Reason | Cause | REST status |
|---|---|---|
| `autofocus_failed` | -- | 500 |
| `autofocus_not_started` | -- | 500 |
| `camera_failure` | -- | 500 |
| `camera_lane_stalled` | -- | 500 |
| `camera_read_failed` | -- | 500 |
| `capture_location_failed` | -- | 500 |
| `captures_failed` | -- | 500 |
| `consecutive_scan_failures` | -- | 500 |
| `disk_space_critical` | -- | 500 |
| `dispatch_refused` | -- | 500 |
| `file_writer_stalled` | -- | 500 |
| `hardware_disconnected` | -- | 500 |
| `interlock` | -- | 500 |
| `led_channel_unavailable` | -- | 500 |
| `position_lost` | -- | 500 |
| `run_dir_init_failed` | -- | 500 |
| `run_loop_crashed` | -- | 500 |
| `start_failed` | -- | 500 |
| `video_writer_died` | -- | 500 |

<a id="RunFailedToStartError"></a>
### `RunFailedToStartError`

A fault; bases `Fault`, `CaptureError`.

A run was committed to and did not start, in its ending's words.

Not fatal: it is reported after the run's cleanup has lifted the
unattended mute, so the person who started it sees it.

**Attributes**

- `title`: The ending's heading, its reason's (`ENDING_TITLES`).
- `reason`: The ending's machine-readable cause, as `RunFailedError` explains it.

| Reason | Cause | REST status |
|---|---|---|
| `autofocus_failed` | -- | 500 |
| `autofocus_not_started` | -- | 500 |
| `camera_failure` | -- | 500 |
| `camera_lane_stalled` | -- | 500 |
| `camera_read_failed` | -- | 500 |
| `capture_location_failed` | -- | 500 |
| `captures_failed` | -- | 500 |
| `consecutive_scan_failures` | -- | 500 |
| `disk_space_critical` | -- | 500 |
| `dispatch_refused` | -- | 500 |
| `file_writer_stalled` | -- | 500 |
| `hardware_disconnected` | -- | 500 |
| `interlock` | -- | 500 |
| `led_channel_unavailable` | -- | 500 |
| `position_lost` | -- | 500 |
| `run_dir_init_failed` | -- | 500 |
| `run_loop_crashed` | -- | 500 |
| `start_failed` | -- | 500 |
| `video_writer_died` | -- | 500 |

<a id="RunFilesNotWrittenError"></a>
### `RunFilesNotWrittenError`

A fault; bases `Fault`, `CaptureError`.

A build that reads a run's images back found them not all written.

Raised by the wait a post-run build makes on its run's writes -- the
composite merge and the hyperstack build -- so neither builds from a
folder that is still filling or that lost images: the artifact would be
silently incomplete. Nothing is built.

**Attributes**

- `reason`: `'write_batch_timeout'` -- the run's writes did not finish within the build's bound. `'write_batch_abandoned'` -- some of the run's writes never ran: the file writer was recovered, or the app shut down, while they were outstanding. `'write_batch_not_taken'` -- some of the run's images never reached the file writer: it was stuck, or had stopped taking work, when they were handed over. `'write_batch_save_failed'` -- saving some of the run's images failed on disk. `'write_batch_disk_full'` -- some of the run's images were refused because the save drive was nearly full. `'write_batch_video_unfinished'` -- a video step's file did not finish. `RunImagesNotSavedError` carries all but the first as the same facts.

| Reason | Cause | REST status |
|---|---|---|
| `write_batch_abandoned` | -- | 500 |
| `write_batch_disk_full` | -- | 500 |
| `write_batch_not_taken` | -- | 500 |
| `write_batch_save_failed` | -- | 500 |
| `write_batch_timeout` | -- | 500 |
| `write_batch_video_unfinished` | -- | 500 |

<a id="RunImagesNotSavedError"></a>
### `RunImagesNotSavedError`

A fault; bases `Fault`, `CaptureError`.

Images a run captured are not on disk.

Reported once, when the run's last write lands -- the one moment the
count is known, since a run's images keep landing after it ends.

**Attributes**

- `reason`: Why images are not on disk, as `RunFilesNotWrittenError` explains it.
- `not_written`: Images the run captured that are not on disk.
- `written`: Images that landed.

| Reason | Cause | REST status |
|---|---|---|
| `write_batch_abandoned` | -- | 500 |
| `write_batch_disk_full` | -- | 500 |
| `write_batch_not_taken` | -- | 500 |
| `write_batch_save_failed` | -- | 500 |
| `write_batch_video_unfinished` | -- | 500 |

<a id="RunIncompleteError"></a>
### `RunIncompleteError`

A fault; bases `Fault`, `CaptureError`.

A run reached its end without every capture it was asked for.

The run's outcome carries it as status `'incomplete'`, reason
`'captures_failed'`, in these words; the person is told once when the
run ends. The images that were captured are saved.

**Attributes**

- `asked`: Captures the run was asked for.
- `captured`: Captures that produced an image.
- `failed_steps`: The name of each step whose capture failed, in the order they failed.

| Reason | Cause | REST status |
|---|---|---|
| `captures_failed` | -- | 500 |

<a id="RunStartError"></a>
### `RunStartError`

A fault; bases `Fault`, `ProtocolError`.

A sequenced run failed after it was committed but before it ran.

The counterpart of ProtocolRunRefusedError on the other side of the
commit line: a refusal means nothing started and nothing needs
unwinding, while this means the run was committed, the terminal
callback will fire and cleanup will run. Carries the same three
fields so both sides deliver one shape to a caller, a REST handler
and the popup.

**Attributes**

- `reason`: Machine-readable cause, a run ending's reason, as `RunFailedError` explains it.
- `title`: Short heading for the user, the run ending's.
- `message`: The sentence a user reads -- never a raw exception string; those belong in the log.

| Reason | Cause | REST status |
|---|---|---|
| `autofocus_failed` | -- | 500 |
| `autofocus_not_started` | -- | 500 |
| `camera_failure` | -- | 500 |
| `camera_lane_stalled` | -- | 500 |
| `camera_read_failed` | -- | 500 |
| `capture_location_failed` | -- | 500 |
| `captures_failed` | -- | 500 |
| `consecutive_scan_failures` | -- | 500 |
| `disk_space_critical` | -- | 500 |
| `dispatch_refused` | -- | 500 |
| `file_writer_stalled` | -- | 500 |
| `hardware_disconnected` | -- | 500 |
| `interlock` | -- | 500 |
| `led_channel_unavailable` | -- | 500 |
| `position_lost` | -- | 500 |
| `run_dir_init_failed` | -- | 500 |
| `run_loop_crashed` | -- | 500 |
| `start_failed` | -- | 500 |
| `video_writer_died` | -- | 500 |

<a id="RunWaitExpiredError"></a>
### `RunWaitExpiredError`

A quiet outcome; bases `ProtocolError`, `Quiet`, `TimeoutError`.

A run's waiting form ran out of its `timeout_s` with the run still going: nothing failed.

Raised by `ProtocolRunner.run_X`. A `TimeoutError`, as a caller's own
wait expiring always is, and quiet: the run goes on, and `operation`
is the caller's hold on it, to wait on again or to stop. `TimeoutError`
is the last base, so its `OSError` constructor never reads the words
as an errno.

**Attributes**

- `reason`: `'still_running'`.
- `member`: The waiting form that was called.
- `operation`: The run's `Operation`, still running when this was raised.

| Reason | Cause | REST status |
|---|---|---|
| `still_running` | state | 409 |

<a id="RunWriteRefusedError"></a>
### `RunWriteRefusedError`

A refusal; bases `Refusal`, `CaptureError`.

A write was handed to a run whose writes have ended.

A run's writes end when its cleanup closes them, or when a writer
recovery or a shutdown abandons them. A write arriving after that
belongs to no run: taken, it would count toward the next run's files or
land after the finished run said its files were written. The caller
says what was not saved. Nothing broke in declining it, and the same
write into a live run is taken, so its cause is the state.

**Attributes**

- `reason`: `'run_writes_ended'` -- the run's cleanup closed its writes. `'writes_abandoned'` -- a writer recovery or a shutdown abandoned the run's remaining writes. `'writer_shut_down'` -- the file lane itself no longer takes work.

| Reason | Cause | REST status |
|---|---|---|
| `run_writes_ended` | state | 409 |
| `writer_shut_down` | state | 409 |
| `writes_abandoned` | state | 409 |

<a id="ScopeDisconnectError"></a>
### `ScopeDisconnectError`

A fault; bases `Fault`, `Exception`.

One or more parts of the microscope did not shut down cleanly.

Raised by `Lumascope.disconnect` only after every teardown step has
run, so one part's failure never leaves the others connected. Names
each part that failed, in the order the teardown reached it, and is
chained from the first part's error. Each part's error is kept private
and logged where it was caught: an exception has no wire form, so what
a client reads is each part's reason. A failure, not a refusal.

**Attributes**

- `reason`: `'disconnect_failed'`.
- `parts`: The parts that failed, in teardown order.
- `part_reasons`: Each failed part's reason, keyed by part; None for an error that carries none.

**Fields**

- <a id="ScopeDisconnectError.part_reasons"></a>`part_reasons`: `dict[DisconnectPart, str | None]` -- Each failed part's reason, keyed by part; None for an error that carries none.
- <a id="ScopeDisconnectError.parts"></a>`parts`: `tuple[DisconnectPart, ...]` -- The parts that failed, in teardown order.

| Reason | Cause | REST status |
|---|---|---|
| `disconnect_failed` | -- | 500 |

<a id="ScopeInUseError"></a>
### `ScopeInUseError`

A refusal; bases `Refusal`, `Exception`.

A real scope was asked for while another is driving the microscope.

Raised by a real scope's construction, before any of its hardware is
opened, when another process holds the installation's instance lock --
a LumaViewPro, a REST server or a script -- or another real scope in
this process holds it: a process drives one real scope. Nothing was
opened. Unlike `BringUpError`'s port notices, which report one part
that did not come up beside the rest, this refuses the whole scope.

**Attributes**

- `reason`: `'scope_in_use'`.

| Reason | Cause | REST status |
|---|---|---|
| `scope_in_use` | state | 409 |

<a id="ScopeModelDeferredNotice"></a>
### `ScopeModelDeferredNotice`

A notice; bases `Notice`, `Exception`.

A scope model saved for the next start, not the one running.

The running scope keeps its model until it is next brought up, so a
saved selection changes nothing yet; and a microscope that reports its
own model overrides the selection then.

| Reason | Cause | REST status |
|---|---|---|
| `scope_model_deferred` | -- | 409 |

<a id="ScopeModelUnknownError"></a>
### `ScopeModelUnknownError`

A refusal; bases `Refusal`, `ValueError`.

A scope model was selected that this release's catalogue does not list.

The selection is the whole identity of a scope that cannot report its
own model, so an unlisted one would start the scope with no layers, no
optics and no scale. Refused before it is saved, rather than saved and
found at the next start.

Subclasses ValueError because it is a bad argument, like an unknown
axis name. The message reaches the person verbatim and names the
models the catalogue does list.

**Attributes**

- `reason`: `'model_unknown'`.
- `model`: The model that was refused.

| Reason | Cause | REST status |
|---|---|---|
| `model_unknown` | request | 422 |

<a id="ServerRefusedError"></a>
### `ServerRefusedError`

A refusal; bases `Refusal`, `Exception`.

A request the REST server itself answers: it never reached a member.

Raised by `rest/` for a request it answers itself: from its request
reader, its handle and job stores, its idempotency keys, its live view
and its close. What the refusal is lives here; how it is sent -- each
reason's status, the `Retry-After` header -- is `rest.problems`'s,
read from this record.

**Attributes**

- `reason`: The machine-readable code. `'not_found'` -- the route, the handle or the job asked for is not there. `'invalid_request'` -- the request's arguments do not fit its member. `'method_not_allowed'` -- the route does not answer the method it was asked with. `'unsupported_media_type'` -- the request's body is not JSON. `'handle_shared'` -- the handle asked to be forgotten is one every client shares, and it is kept. `'overloaded'` -- the server holds as many handles or jobs as it keeps; asking again later may succeed. `'server_closing'` -- the server is closing: it finishes what is under way, answers nothing new, and exits. `'no_frame_yet'` -- the live view has no frame to show, because the camera sent none in time; asking again later may succeed. `'not_cancellable'` -- the job is a call held as a job when it outlived its wait, and holds no operation to stop. `'idempotency_key_reused'` -- the `Idempotency-Key` was sent before with a different request; a key is sent again only to retry the same request.
- `title`: The heading, the reason's.
- `errors`: For `'invalid_request'` from the request reader, each argument that does not fit; None otherwise.
- `retry_after_s`: Seconds to wait before asking again, for a refusal that waiting may lift (`'overloaded'`, `'no_frame_yet'`); None when no wait helps.

**Fields**

- <a id="ServerRefusedError.errors"></a>`errors`: `tuple[UnfitArgument, ...] | None` -- For `'invalid_request'` from the request reader, each argument that does not fit; None otherwise.

| Reason | Cause | REST status |
|---|---|---|
| `handle_shared` | request | 422 |
| `idempotency_key_reused` | request | 422 |
| `invalid_request` | request | 422 |
| `method_not_allowed` | request | 405 |
| `no_frame_yet` | state | 503 |
| `not_cancellable` | state | 409 |
| `not_found` | request | 404 |
| `overloaded` | state | 503 |
| `server_closing` | restart | 503 |
| `unsupported_media_type` | request | 415 |

<a id="SessionClosingError"></a>
### `SessionClosingError`

A refusal; bases `Refusal`, `Exception`.

An activity was asked to take the scope while the session is closing.

Raised by the activity claim for a run, a recording, a home or a
diagnostic while `ScopeSession.shutdown` runs: the close finishes the
work already under way, then releases the hardware, so nothing new may
start in between. Once the close has finished, nothing takes the scope
again: a start whose own checks come first is refused by them, finding
the scope disconnected, and any other is refused here when it takes the
scope -- a home, or a run that passed its checks before the close
finished. Work inside an activity already under way -- a run's video
step under its run -- is not refused. Nothing was taken.

**Attributes**

- `reason`: `'session_closing'`.
- `activity`: The kind of activity that was refused.

| Reason | Cause | REST status |
|---|---|---|
| `session_closing` | restart | 503 |

<a id="SettingRefusedError"></a>
### `SettingRefusedError`

A refusal; bases `Refusal`, `ConfigError`.

A settings write was refused; nothing was written.

Raised by `ScopeSession.update_settings` and `set_live_folder`, by
the members that own a ranged setting (`set_channel_acquire`,
`set_image_mode`, through `settings_paths.refuse_outside_range`) and
for a settings dict handed to a session, at `create`'s door
(`settings_init.prepare_loaded`, through
`settings_paths.refuse_handed_values`). A caller that maps refusals to
responses branches on `reason`; a person reads the message.

**Attributes**

- `reason`: `'has_member'` -- the setting is changed by its own member, named in `member` by its path from the session (the camera's as `scope.camera.<name>`). `'host_only'` -- that member is in-process: only the application hosting the scope changes the setting, and a REST client reads it. `'not_a_setting'` -- no setting has this path. `'block'` -- the path names a block of settings, which is written one setting at a time. `'installation_only'` -- the setting is read only from the installation's settings file. `'derived'` -- the setting is derived from the release when the template is read (`step_channel_order`); a handed dict's differing copy is refused for it too. `'wrong_kind'` -- the value is not the kind the setting holds. `'out_of_range'` -- the value is outside the setting's range. `'not_offered'` -- the setting holds a closed set (a `StrEnum`, such as `image_mode`'s `ImageMode`) and the value is not one of its values.
- `path`: The dotted path that was written.
- `detail`: Why, as a clause about the value (`'it must be above 0, not 0'`), for a caller that names the setting its own way: a protocol step's column.
- `member`: The member that changes the setting, for `'has_member'` and `'host_only'`; otherwise None.
- `limits`: The lowest and highest value the setting takes, for an `'out_of_range'` setting with both ends; otherwise None.

`path`, `member` and `limits` are published, so a REST problem
names the setting refused, the member to call instead and the range,
beside the words.

**Fields**

- <a id="SettingRefusedError.limits"></a>`limits`: `tuple[float, float] | None` -- The lowest and highest value the setting takes, for an `'out_of_range'` setting with both ends; otherwise None.
- <a id="SettingRefusedError.member"></a>`member`: `str | None` -- The member that changes the setting, for `'has_member'` and `'host_only'`; otherwise None.
- <a id="SettingRefusedError.path"></a>`path`: `str` -- The dotted path that was written.

| Reason | Cause | REST status |
|---|---|---|
| `block` | request | 422 |
| `derived` | request | 422 |
| `has_member` | request | 422 |
| `host_only` | request | 422 |
| `installation_only` | request | 422 |
| `not_a_setting` | request | 422 |
| `not_offered` | request | 422 |
| `out_of_range` | request | 422 |
| `wrong_kind` | request | 422 |

<a id="SettingsFileError"></a>
### `SettingsFileError`

A fault; bases `Fault`, `ValueError`.

A settings file exists but could not be turned into a settings dict.

Subclasses ValueError deliberately: `load_lvp_settings` routes a bad
`current.json` to the shipped template by catching
`(JSONDecodeError, ValueError)`, and a sibling of that family would
escape the catch and turn every recoverable failure into a startup
crash.

| Reason | Cause | REST status |
|---|---|---|
| `settings_file_unreadable` | -- | 500 |

<a id="SettingsFileNotReplacedError"></a>
### `SettingsFileNotReplacedError`

A fault; bases `Fault`, `ConfigError`.

The unreadable settings file could not be moved aside after the user chose to start over.

Raised by the retire, chained from the `OSError` the rename raised, so
the words name the file and the operating system's own reason rather
than guessing one: a lock by another program is the common cause on
Windows, but a permissions or disk fault reads the same to a guess. The
settings stay provisional, so nothing is lost and the question can be
answered again.

**Attributes**

- `file`: The settings file that is still in place.

| Reason | Cause | REST status |
|---|---|---|
| `settings_file_not_replaced` | -- | 500 |

<a id="SettingsFileNotWrittenError"></a>
### `SettingsFileNotWrittenError`

A fault; bases `Fault`, `ConfigError`.

The session's settings could not be written to its settings file.

Reported by the session's save, chained from the `OSError` of the
write, so the words name the file and the operating system's own reason.
The write goes beside the file first, so the file already there is
unchanged; the next accepted write, and the close, save again.

**Attributes**

- `file`: The settings file that was to be written.

| Reason | Cause | REST status |
|---|---|---|
| `settings_file_not_written` | -- | 500 |

<a id="SettingsNotSavedNotice"></a>
### `SettingsNotSavedNotice`

A notice; bases `Notice`, `Exception`.

The session's accepted settings writes are kept in memory, not saved.

Told once per session and reason, at the first save the session
withholds: the write itself was accepted and is in effect, so nothing
was refused, but a person who expects it at the next start must hear
that it will not be there.

**Attributes**

- `reason`: `'settings_provisional'` -- the app is running on the shipped template because the user's file could not be read, and that file is left as it is until they decide. `'settings_unmeasured'` -- no hardware came up this session, so the per-channel values are defaults, not values measured on this scope.
- `file`: The file the writes would have been saved to.

| Reason | Cause | REST status |
|---|---|---|
| `settings_provisional` | -- | 409 |
| `settings_unmeasured` | -- | 409 |

<a id="SingleScanNotice"></a>
### `SingleScanNotice`

A notice; bases `Notice`, `Exception`.

A timed run will perform one scan, because of its period and duration.

| Reason | Cause | REST status |
|---|---|---|
| `single_scan` | -- | 409 |

<a id="SlowFileWritesNotice"></a>
### `SlowFileWritesNotice`

A notice; bases `Notice`, `Exception`.

A run spent long enough waiting on the save disk to say so at its end.

| Reason | Cause | REST status |
|---|---|---|
| `slow_file_writes` | -- | 409 |

<a id="StepEditRefusedError"></a>
### `StepEditRefusedError`

A refusal; bases `Refusal`, `ProtocolError`.

A value a step writer cannot take; the protocol is unchanged.

A name with nothing left once cleaned, a value not of its column's type,
or an insert that names no place or two. A refusal, as a missing step is.

| Reason | Cause | REST status |
|---|---|---|
| `step_value_refused` | request | 422 |

<a id="StepNotFoundError"></a>
### `StepNotFoundError`

A refusal; bases `Refusal`, `ProtocolError`.

A step index the protocol does not have, refused before anything is read or written.

A refusal: the caller named a step, and the words say which steps there
are. Unmarked, a Delete on an empty protocol was reported as a fault --
"Operation failed", an ERROR and a traceback for a designed answer.

The words name no index: the index is the API's, zero-based, and a
person who typed a step number into the panel counts from 1. A caller
that wants it reads it here.

**Attributes**

- `reason`: `'no_such_step'`.
- `index`: The index refused, as the caller gave it.
- `num_steps`: How many steps the protocol has.

| Reason | Cause | REST status |
|---|---|---|
| `no_such_step` | request | 422 |

<a id="StoredSettingReplacedNotice"></a>
### `StoredSettingReplacedNotice`

A notice; bases `Notice`, `Exception`.

Saved settings no write could store, replaced by the shipped values at load.

One notice names every value one load replaced: the notification centre
shows one notice of a kind at a time, so a notice per value showed the
first and hid the rest. Each value is replaced for its key alone, in the
settings the app runs on and so in the file at its next save.

**Attributes**

- `replacements`: `(path, saved, used)` for each replaced setting: its dotted path, the value the settings file held, and the shipped value now in its place.

| Reason | Cause | REST status |
|---|---|---|
| `stored_setting_replaced` | -- | 409 |

<a id="SupportReportNotSavedError"></a>
### `SupportReportNotSavedError`

A fault; bases `Fault`, `Exception`.

A support report or a logs zip was not saved.

A fault, not a refusal: the report ran and something under it failed.
Chained from that failure, whose words the message carries, so the
person can say what went wrong when they write to support without the
report.

**Attributes**

- `report`: Which ZIP was not saved.

| Reason | Cause | REST status |
|---|---|---|
| `support_report_not_saved` | -- | 500 |

<a id="UpdateError"></a>
### `UpdateError`

A fault; bases `Fault`, `Exception`.

Firmware update failure with stage and recoverability info.

| Reason | Cause | REST status |
|---|---|---|
| `update_failed` | -- | 500 |

<a id="VideoFramesDroppedError"></a>
### `VideoFramesDroppedError`

A fault; bases `Fault`, `CaptureError`.

Frames a recording selected could not be written, so its video is short.

Each dropped frame is a write that failed; the recording's manifest
carries the counts too.

**Attributes**

- `dropped`: Frames that could not be written.
- `selected`: Frames the recording selected.
- `protocol_step`: True for a protocol video step, False for a manual recording.

| Reason | Cause | REST status |
|---|---|---|
| `video_frames_dropped` | -- | 500 |

<a id="VideoWriterFailedError"></a>
### `VideoWriterFailedError`

A fault; bases `Fault`, `CaptureError`.

The recording's video writer stopped working, so the recording ended.

Chained from what escaped the writer. The frames already written are on
disk. On a protocol video step the run is stopped too, and the run's
ending carries these same words, so a caller reading the run's outcome
reads what the person was shown.

**Attributes**

- `protocol_step`: True for a protocol video step, False for a manual recording; each is named in its own words.

| Reason | Cause | REST status |
|---|---|---|
| `video_writer_died` | -- | 500 |

## Routes

| Method | Path | Member |
|---|---|---|
| `GET` | `/api/v1/acceleration_limit_pct` | [`ScopeSession.acceleration_limit_pct`](#ScopeSession.acceleration_limit_pct) |
| `PUT` | `/api/v1/acceleration_limit_pct` | [`ScopeSession.set_acceleration_limit`](#ScopeSession.set_acceleration_limit) |
| `POST` | `/api/v1/add_step` | [`ScopeSession.add_step`](#ScopeSession.add_step) |
| `GET` | `/api/v1/app_runtime` | [`ScopeSession.app_runtime`](#ScopeSession.app_runtime) |
| `GET` | `/api/v1/app_version` | [`ScopeSession.app_version`](#ScopeSession.app_version) |
| `POST` | `/api/v1/apply_channel_settings` | [`ScopeSession.apply_channel_settings`](#ScopeSession.apply_channel_settings) |
| `POST` | `/api/v1/apply_focus_to_channel_steps` | [`ScopeSession.apply_focus_to_channel_steps`](#ScopeSession.apply_focus_to_channel_steps) |
| `POST` | `/api/v1/apply_remedy` | [`ScopeSession.apply_remedy`](#ScopeSession.apply_remedy) |
| `POST` | `/api/v1/apply_tiling` | [`ScopeSession.apply_tiling`](#ScopeSession.apply_tiling) |
| `POST` | `/api/v1/apply_zstacking` | [`ScopeSession.apply_zstacking`](#ScopeSession.apply_zstacking) |
| `GET` | `/api/v1/bookmark` | [`ScopeSession.bookmark`](#ScopeSession.bookmark) |
| `GET` | `/api/v1/bring_up_record` | [`ScopeSession.bring_up_record`](#ScopeSession.bring_up_record) |
| `GET` | `/api/v1/channel_acquire` | [`ScopeSession.channel_acquire`](#ScopeSession.channel_acquire) |
| `PUT` | `/api/v1/channel_acquire` | [`ScopeSession.set_channel_acquire`](#ScopeSession.set_channel_acquire) |
| `GET` | `/api/v1/channel_auto_gain` | [`ScopeSession.channel_auto_gain`](#ScopeSession.channel_auto_gain) |
| `PUT` | `/api/v1/channel_auto_gain` | [`ScopeSession.set_channel_auto_gain`](#ScopeSession.set_channel_auto_gain) |
| `GET` | `/api/v1/channel_auto_gain_lock` | [`ScopeSession.channel_auto_gain_lock`](#ScopeSession.channel_auto_gain_lock) |
| `GET` | `/api/v1/channel_colors` | [`ScopeSession.channel_colors`](#ScopeSession.channel_colors) |
| `POST` | `/api/v1/clear_current_turret_objective` | [`ScopeSession.clear_current_turret_objective`](#ScopeSession.clear_current_turret_objective) |
| `POST` | `/api/v1/clear_turret_objective` | [`ScopeSession.clear_turret_objective`](#ScopeSession.clear_turret_objective) |
| `POST` | `/api/v1/confirm_objective` | [`ScopeSession.confirm_objective`](#ScopeSession.confirm_objective) |
| `GET` | `/api/v1/control` | [`ScopeSession.control`](#ScopeSession.control) |
| `POST` | `/api/v1/create_empty_protocol` | [`ScopeSession.create_empty_protocol`](#ScopeSession.create_empty_protocol) |
| `POST` | `/api/v1/create_protocol_runner` | [`ScopeSession.create_protocol_runner`](#ScopeSession.create_protocol_runner) |
| `GET` | `/api/v1/current_slot_objective_question` | [`ScopeSession.current_slot_objective_question`](#ScopeSession.current_slot_objective_question) |
| `POST` | `/api/v1/delete_step` | [`ScopeSession.delete_step`](#ScopeSession.delete_step) |
| `POST` | `/api/v1/discard_close_drain` | [`ScopeSession.discard_close_drain`](#ScopeSession.discard_close_drain) |
| `POST` | `/api/v1/end_control` | [`ScopeSession.end_control`](#ScopeSession.end_control) |
| `GET` | `/api/v1/engineering_mode` | [`ScopeSession.engineering_mode`](#ScopeSession.engineering_mode) |
| `GET` | `/api/v1/get_current_plate_position` | [`ScopeSession.get_current_plate_position`](#ScopeSession.get_current_plate_position) |
| `GET` | `/api/v1/get_setting` | [`ScopeSession.get_setting`](#ScopeSession.get_setting) |
| `POST` | `/api/v1/go_to_bookmark` | [`ScopeSession.go_to_bookmark`](#ScopeSession.go_to_bookmark) |
| `POST` | `/api/v1/go_to_step` | [`ScopeSession.go_to_step`](#ScopeSession.go_to_step) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/CURRENT_VERSION` | [`Protocol.CURRENT_VERSION`](#Protocol.CURRENT_VERSION) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/PROTOCOL_FILE_HEADER` | [`Protocol.PROTOCOL_FILE_HEADER`](#Protocol.PROTOCOL_FILE_HEADER) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/capture_prefix` | [`Protocol.capture_prefix`](#Protocol.capture_prefix) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/capture_root` | [`Protocol.capture_root`](#Protocol.capture_root) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/channel_settings` | [`Protocol.channel_settings`](#Protocol.channel_settings) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/duration_s` | [`Protocol.duration_s`](#Protocol.duration_s) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/estimate_write_mb` | [`Protocol.estimate_write_mb`](#Protocol.estimate_write_mb) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/labware_id` | [`Protocol.labware_id`](#Protocol.labware_id) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/num_scans` | [`Protocol.num_scans`](#Protocol.num_scans) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/num_steps` | [`Protocol.num_steps`](#Protocol.num_steps) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/period_s` | [`Protocol.period_s`](#Protocol.period_s) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/size_advisory` | [`Protocol.size_advisory`](#Protocol.size_advisory) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/step_list_revision` | [`Protocol.step_list_revision`](#Protocol.step_list_revision) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/step_records` | [`Protocol.step_records`](#Protocol.step_records) |
| `GET` | `/api/v1/handles/Protocol/{handle_id}/tiling` | [`Protocol.tiling`](#Protocol.tiling) |
| `POST` | `/api/v1/handles/ProtocolRunner/{handle_id}/start_autofocus` | [`ProtocolRunner.start_autofocus`](#ProtocolRunner.start_autofocus) |
| `POST` | `/api/v1/handles/ProtocolRunner/{handle_id}/start_autofocus_all_steps` | [`ProtocolRunner.start_autofocus_all_steps`](#ProtocolRunner.start_autofocus_all_steps) |
| `POST` | `/api/v1/handles/ProtocolRunner/{handle_id}/start_composite` | [`ProtocolRunner.start_composite`](#ProtocolRunner.start_composite) |
| `POST` | `/api/v1/handles/ProtocolRunner/{handle_id}/start_protocol` | [`ProtocolRunner.start_protocol`](#ProtocolRunner.start_protocol) |
| `POST` | `/api/v1/handles/ProtocolRunner/{handle_id}/start_single_scan` | [`ProtocolRunner.start_single_scan`](#ProtocolRunner.start_single_scan) |
| `POST` | `/api/v1/handles/ProtocolRunner/{handle_id}/start_zstack` | [`ProtocolRunner.start_zstack`](#ProtocolRunner.start_zstack) |
| `GET` | `/api/v1/held_by_other` | [`ScopeSession.held_by_other`](#ScopeSession.held_by_other) |
| `GET` | `/api/v1/labware_id` | [`ScopeSession.labware_id`](#ScopeSession.labware_id) |
| `PUT` | `/api/v1/labware_id` | [`ScopeSession.select_labware`](#ScopeSession.select_labware) |
| `POST` | `/api/v1/last_run` | [`ScopeSession.last_run`](#ScopeSession.last_run) |
| `POST` | `/api/v1/light_shown_channel` | [`ScopeSession.light_shown_channel`](#ScopeSession.light_shown_channel) |
| `GET` | `/api/v1/live_folder_listing` | [`ScopeSession.live_folder_listing`](#ScopeSession.live_folder_listing) |
| `POST` | `/api/v1/load_protocol` | [`ScopeSession.load_protocol`](#ScopeSession.load_protocol) |
| `POST` | `/api/v1/make_logs_zip` | [`ScopeSession.make_logs_zip`](#ScopeSession.make_logs_zip) |
| `POST` | `/api/v1/make_support_report` | [`ScopeSession.make_support_report`](#ScopeSession.make_support_report) |
| `POST` | `/api/v1/manual_capture/capture` | [`ManualCaptureController.capture`](#ManualCaptureController.capture) |
| `GET` | `/api/v1/manual_capture/in_flight` | [`ManualCaptureController.in_flight`](#ManualCaptureController.in_flight) |
| `POST` | `/api/v1/manual_recording/discard_pending` | [`ManualRecordingController.discard_pending`](#ManualRecordingController.discard_pending) |
| `GET` | `/api/v1/manual_recording/elapsed_s` | [`ManualRecordingController.elapsed_s`](#ManualRecordingController.elapsed_s) |
| `GET` | `/api/v1/manual_recording/end_reason` | [`ManualRecordingController.end_reason`](#ManualRecordingController.end_reason) |
| `GET` | `/api/v1/manual_recording/is_busy` | [`ManualRecordingController.is_busy`](#ManualRecordingController.is_busy) |
| `GET` | `/api/v1/manual_recording/is_draining` | [`ManualRecordingController.is_draining`](#ManualRecordingController.is_draining) |
| `GET` | `/api/v1/manual_recording/is_recording` | [`ManualRecordingController.is_recording`](#ManualRecordingController.is_recording) |
| `GET` | `/api/v1/manual_recording/pending_writes` | [`ManualRecordingController.pending_writes`](#ManualRecordingController.pending_writes) |
| `GET` | `/api/v1/manual_recording/save_folder` | [`ManualRecordingController.save_folder`](#ManualRecordingController.save_folder) |
| `POST` | `/api/v1/manual_recording/start` | [`ManualRecordingController.start`](#ManualRecordingController.start) |
| `POST` | `/api/v1/manual_recording/stop` | [`ManualRecordingController.stop`](#ManualRecordingController.stop) |
| `GET` | `/api/v1/model` | [`ScopeSession.model`](#ScopeSession.model) |
| `PUT` | `/api/v1/model` | [`ScopeSession.select_model`](#ScopeSession.select_model) |
| `GET` | `/api/v1/model_at_next_start` | [`ScopeSession.model_at_next_start`](#ScopeSession.model_at_next_start) |
| `POST` | `/api/v1/new_protocol` | [`ScopeSession.new_protocol`](#ScopeSession.new_protocol) |
| `GET` | `/api/v1/objective_id` | [`ScopeSession.objective_id`](#ScopeSession.objective_id) |
| `PUT` | `/api/v1/objective_id` | [`ScopeSession.select_objective`](#ScopeSession.select_objective) |
| `GET` | `/api/v1/objective_question` | [`ScopeSession.objective_question`](#ScopeSession.objective_question) |
| `GET` | `/api/v1/plugin_api_level` | [`ScopeSession.plugin_api_level`](#ScopeSession.plugin_api_level) |
| `GET` | `/api/v1/plugin_health` | [`ScopeSession.plugin_health`](#ScopeSession.plugin_health) |
| `POST` | `/api/v1/point_steps_at_turret_slot` | [`ScopeSession.point_steps_at_turret_slot`](#ScopeSession.point_steps_at_turret_slot) |
| `POST` | `/api/v1/post_processing/build` | [`PostProcessingAPI.build`](#PostProcessingAPI.build) |
| `POST` | `/api/v1/post_processing/cell_count_size_scale` | [`PostProcessingAPI.cell_count_size_scale`](#PostProcessingAPI.cell_count_size_scale) |
| `POST` | `/api/v1/post_processing/count_cells` | [`PostProcessingAPI.count_cells`](#PostProcessingAPI.count_cells) |
| `GET` | `/api/v1/post_processing/default_cell_count_method` | [`PostProcessingAPI.default_cell_count_method`](#PostProcessingAPI.default_cell_count_method) |
| `POST` | `/api/v1/post_processing/enhance` | [`PostProcessingAPI.enhance`](#PostProcessingAPI.enhance) |
| `POST` | `/api/v1/post_processing/graph` | [`PostProcessingAPI.graph`](#PostProcessingAPI.graph) |
| `POST` | `/api/v1/post_processing/graph_options` | [`PostProcessingAPI.graph_options`](#PostProcessingAPI.graph_options) |
| `POST` | `/api/v1/post_processing/load_cell_count_method` | [`PostProcessingAPI.load_cell_count_method`](#PostProcessingAPI.load_cell_count_method) |
| `GET` | `/api/v1/post_processing/projection_methods` | [`PostProcessingAPI.projection_methods`](#PostProcessingAPI.projection_methods) |
| `POST` | `/api/v1/post_processing/save_cell_count_method` | [`PostProcessingAPI.save_cell_count_method`](#PostProcessingAPI.save_cell_count_method) |
| `GET` | `/api/v1/post_processing/stitch_modes` | [`PostProcessingAPI.stitch_modes`](#PostProcessingAPI.stitch_modes) |
| `GET` | `/api/v1/protocol_files_stalled` | [`ScopeSession.protocol_files_stalled`](#ScopeSession.protocol_files_stalled) |
| `GET` | `/api/v1/protocol_files_stuck_write` | [`ScopeSession.protocol_files_stuck_write`](#ScopeSession.protocol_files_stuck_write) |
| `GET` | `/api/v1/protocol_size_advisory` | [`ScopeSession.protocol_size_advisory`](#ScopeSession.protocol_size_advisory) |
| `POST` | `/api/v1/recover_file_writer` | [`ScopeSession.recover_file_writer`](#ScopeSession.recover_file_writer) |
| `POST` | `/api/v1/rename_step` | [`ScopeSession.rename_step`](#ScopeSession.rename_step) |
| `POST` | `/api/v1/repair_camera_driver` | [`ScopeSession.repair_camera_driver`](#ScopeSession.repair_camera_driver) |
| `POST` | `/api/v1/retire_rejected_settings` | [`ScopeSession.retire_rejected_settings`](#ScopeSession.retire_rejected_settings) |
| `POST` | `/api/v1/save_all_bookmarks` | [`ScopeSession.save_all_bookmarks`](#ScopeSession.save_all_bookmarks) |
| `POST` | `/api/v1/save_bookmark` | [`ScopeSession.save_bookmark`](#ScopeSession.save_bookmark) |
| `POST` | `/api/v1/save_focus` | [`ScopeSession.save_focus`](#ScopeSession.save_focus) |
| `POST` | `/api/v1/save_protocol` | [`ScopeSession.save_protocol`](#ScopeSession.save_protocol) |
| `GET` | `/api/v1/saved_focus_z_um` | [`ScopeSession.saved_focus_z_um`](#ScopeSession.saved_focus_z_um) |
| `PUT` | `/api/v1/saved_focus_z_um` | [`ScopeSession.save_channel_focus`](#ScopeSession.save_channel_focus) |
| `GET` | `/api/v1/scale_bar_enabled` | [`ScopeSession.scale_bar_enabled`](#ScopeSession.scale_bar_enabled) |
| `PUT` | `/api/v1/scale_bar_enabled` | [`ScopeSession.set_scale_bar`](#ScopeSession.set_scale_bar) |
| `GET` | `/api/v1/scope/camera/applied_auto_gain_for` | [`CameraAPI.applied_auto_gain_for`](#CameraAPI.applied_auto_gain_for) |
| `GET` | `/api/v1/scope/camera/applied_exposure_ms_for` | [`CameraAPI.applied_exposure_ms_for`](#CameraAPI.applied_exposure_ms_for) |
| `GET` | `/api/v1/scope/camera/applied_gain_db_for` | [`CameraAPI.applied_gain_db_for`](#CameraAPI.applied_gain_db_for) |
| `POST` | `/api/v1/scope/camera/apply` | [`CameraAPI.apply`](#CameraAPI.apply) |
| `GET` | `/api/v1/scope/camera/auto_exposure` | [`CameraAPI.auto_exposure`](#CameraAPI.auto_exposure) |
| `PUT` | `/api/v1/scope/camera/auto_exposure` | [`CameraAPI.set_auto_exposure`](#CameraAPI.set_auto_exposure) |
| `GET` | `/api/v1/scope/camera/auto_gain` | [`CameraAPI.auto_gain`](#CameraAPI.auto_gain) |
| `GET` | `/api/v1/scope/camera/binning` | [`CameraAPI.binning`](#CameraAPI.binning) |
| `PUT` | `/api/v1/scope/camera/binning` | [`CameraAPI.set_binning`](#CameraAPI.set_binning) |
| `GET` | `/api/v1/scope/camera/black_level` | [`CameraAPI.black_level`](#CameraAPI.black_level) |
| `PUT` | `/api/v1/scope/camera/black_level` | [`CameraAPI.set_black_level`](#CameraAPI.set_black_level) |
| `GET` | `/api/v1/scope/camera/exposure_ms` | [`CameraAPI.exposure_ms`](#CameraAPI.exposure_ms) |
| `PUT` | `/api/v1/scope/camera/exposure_ms` | [`CameraAPI.set_exposure_ms`](#CameraAPI.set_exposure_ms) |
| `GET` | `/api/v1/scope/camera/frame_at_binning` | [`CameraAPI.frame_at_binning`](#CameraAPI.frame_at_binning) |
| `GET` | `/api/v1/scope/camera/frame_size` | [`CameraAPI.frame_size`](#CameraAPI.frame_size) |
| `PUT` | `/api/v1/scope/camera/frame_size` | [`CameraAPI.set_frame_size`](#CameraAPI.set_frame_size) |
| `GET` | `/api/v1/scope/camera/gain_db` | [`CameraAPI.gain_db`](#CameraAPI.gain_db) |
| `PUT` | `/api/v1/scope/camera/gain_db` | [`CameraAPI.set_gain_db`](#CameraAPI.set_gain_db) |
| `GET` | `/api/v1/scope/camera/get_black_level_range` | [`CameraAPI.get_black_level_range`](#CameraAPI.get_black_level_range) |
| `GET` | `/api/v1/scope/camera/get_pixel_alignment` | [`CameraAPI.get_pixel_alignment`](#CameraAPI.get_pixel_alignment) |
| `GET` | `/api/v1/scope/camera/high_conversion_gain` | [`CameraAPI.high_conversion_gain`](#CameraAPI.high_conversion_gain) |
| `PUT` | `/api/v1/scope/camera/high_conversion_gain` | [`CameraAPI.set_high_conversion_gain`](#CameraAPI.set_high_conversion_gain) |
| `GET` | `/api/v1/scope/camera/image_mode` | [`CameraAPI.image_mode`](#CameraAPI.image_mode) |
| `PUT` | `/api/v1/scope/camera/image_mode` | [`CameraAPI.set_image_mode`](#CameraAPI.set_image_mode) |
| `GET` | `/api/v1/scope/camera/line_noise_reduction` | [`CameraAPI.line_noise_reduction`](#CameraAPI.line_noise_reduction) |
| `PUT` | `/api/v1/scope/camera/line_noise_reduction` | [`CameraAPI.set_line_noise_reduction`](#CameraAPI.set_line_noise_reduction) |
| `POST` | `/api/v1/scope/camera/lock_auto_gain` | [`CameraAPI.lock_auto_gain`](#CameraAPI.lock_auto_gain) |
| `GET` | `/api/v1/scope/camera/longest_exposure_ms` | [`CameraAPI.longest_exposure_ms`](#CameraAPI.longest_exposure_ms) |
| `GET` | `/api/v1/scope/camera/pixel_format` | [`CameraAPI.pixel_format`](#CameraAPI.pixel_format) |
| `GET` | `/api/v1/scope/camera/read_auto_exposure` | [`CameraAPI.read_auto_exposure`](#CameraAPI.read_auto_exposure) |
| `GET` | `/api/v1/scope/camera/read_auto_gain` | [`CameraAPI.read_auto_gain`](#CameraAPI.read_auto_gain) |
| `GET` | `/api/v1/scope/camera/read_binning` | [`CameraAPI.read_binning`](#CameraAPI.read_binning) |
| `GET` | `/api/v1/scope/camera/read_black_level` | [`CameraAPI.read_black_level`](#CameraAPI.read_black_level) |
| `GET` | `/api/v1/scope/camera/read_exposure_min_ms` | [`CameraAPI.read_exposure_min_ms`](#CameraAPI.read_exposure_min_ms) |
| `GET` | `/api/v1/scope/camera/read_exposure_ms` | [`CameraAPI.read_exposure_ms`](#CameraAPI.read_exposure_ms) |
| `GET` | `/api/v1/scope/camera/read_frame_rate_fps` | [`CameraAPI.read_frame_rate_fps`](#CameraAPI.read_frame_rate_fps) |
| `GET` | `/api/v1/scope/camera/read_frame_size` | [`CameraAPI.read_frame_size`](#CameraAPI.read_frame_size) |
| `GET` | `/api/v1/scope/camera/read_gain_db` | [`CameraAPI.read_gain_db`](#CameraAPI.read_gain_db) |
| `GET` | `/api/v1/scope/camera/read_pixel_format` | [`CameraAPI.read_pixel_format`](#CameraAPI.read_pixel_format) |
| `POST` | `/api/v1/scope/camera/set_auto_gain` | [`CameraAPI.set_auto_gain`](#CameraAPI.set_auto_gain) |
| `POST` | `/api/v1/scope/camera/update_auto_gain_target_brightness` | [`CameraAPI.update_auto_gain_target_brightness`](#CameraAPI.update_auto_gain_target_brightness) |
| `GET` | `/api/v1/scope/capabilities` | [`Lumascope.capabilities`](#Lumascope.capabilities) |
| `GET` | `/api/v1/scope/channel_identity` | [`Lumascope.channel_identity`](#Lumascope.channel_identity) |
| `GET` | `/api/v1/scope/diagnostics/get_camera_link_info` | [`DiagnosticsAPI.get_camera_link_info`](#DiagnosticsAPI.get_camera_link_info) |
| `GET` | `/api/v1/scope/diagnostics/get_camera_temperatures_degc` | [`DiagnosticsAPI.get_camera_temperatures_degc`](#DiagnosticsAPI.get_camera_temperatures_degc) |
| `GET` | `/api/v1/scope/diagnostics/get_camera_usb_bindings` | [`DiagnosticsAPI.get_camera_usb_bindings`](#DiagnosticsAPI.get_camera_usb_bindings) |
| `GET` | `/api/v1/scope/diagnostics/get_led_info` | [`DiagnosticsAPI.get_led_info`](#DiagnosticsAPI.get_led_info) |
| `GET` | `/api/v1/scope/diagnostics/get_motor_info` | [`DiagnosticsAPI.get_motor_info`](#DiagnosticsAPI.get_motor_info) |
| `GET` | `/api/v1/scope/illumination/get_led_states` | [`IlluminationAPI.get_led_states`](#IlluminationAPI.get_led_states) |
| `POST` | `/api/v1/scope/illumination/led_off` | [`IlluminationAPI.led_off`](#IlluminationAPI.led_off) |
| `POST` | `/api/v1/scope/illumination/led_on` | [`IlluminationAPI.led_on`](#IlluminationAPI.led_on) |
| `POST` | `/api/v1/scope/illumination/leds_off` | [`IlluminationAPI.leds_off`](#IlluminationAPI.leds_off) |
| `POST` | `/api/v1/scope/illumination/restore_led_state` | [`IlluminationAPI.restore_led_state`](#IlluminationAPI.restore_led_state) |
| `GET` | `/api/v1/scope/illumination/save_led_state` | [`IlluminationAPI.save_led_state`](#IlluminationAPI.save_led_state) |
| `POST` | `/api/v1/scope/imaging/disable_streaming` | [`ImagingAPI.disable_streaming`](#ImagingAPI.disable_streaming) |
| `POST` | `/api/v1/scope/imaging/enable_streaming` | [`ImagingAPI.enable_streaming`](#ImagingAPI.enable_streaming) |
| `GET` | `/api/v1/scope/imaging/frames_until_valid` | [`ImagingAPI.frames_until_valid`](#ImagingAPI.frames_until_valid) |
| `GET` | `/api/v1/scope/imaging/get_delivered_rate` | [`ImagingAPI.get_delivered_rate`](#ImagingAPI.get_delivered_rate) |
| `GET` | `/api/v1/scope/imaging/is_streaming` | [`ImagingAPI.is_streaming`](#ImagingAPI.is_streaming) |
| `GET` | `/api/v1/scope/imaging/scale_bar_color` | [`ImagingAPI.scale_bar_color`](#ImagingAPI.scale_bar_color) |
| `PUT` | `/api/v1/scope/imaging/scale_bar_color` | [`ImagingAPI.set_scale_bar_color`](#ImagingAPI.set_scale_bar_color) |
| `GET` | `/api/v1/scope/imaging/significant_bits` | [`ImagingAPI.significant_bits`](#ImagingAPI.significant_bits) |
| `GET` | `/api/v1/scope/labware` | [`Lumascope.labware`](#Lumascope.labware) |
| `GET` | `/api/v1/scope/motion/axes_without_position` | [`MotionAPI.axes_without_position`](#MotionAPI.axes_without_position) |
| `GET` | `/api/v1/scope/motion/axis_positions` | [`MotionAPI.axis_positions`](#MotionAPI.axis_positions) |
| `GET` | `/api/v1/scope/motion/get_actual_position` | [`MotionAPI.get_actual_position`](#MotionAPI.get_actual_position) |
| `GET` | `/api/v1/scope/motion/get_axis_limits` | [`MotionAPI.get_axis_limits`](#MotionAPI.get_axis_limits) |
| `GET` | `/api/v1/scope/motion/get_limit_switch_status` | [`MotionAPI.get_limit_switch_status`](#MotionAPI.get_limit_switch_status) |
| `GET` | `/api/v1/scope/motion/get_target_positions` | [`MotionAPI.get_target_positions`](#MotionAPI.get_target_positions) |
| `POST` | `/api/v1/scope/motion/home` | [`MotionAPI.home`](#MotionAPI.home) |
| `GET` | `/api/v1/scope/motion/interlocks` | [`MotionAPI.interlocks`](#MotionAPI.interlocks) |
| `GET` | `/api/v1/scope/motion/jog_step_um` | [`MotionAPI.jog_step_um`](#MotionAPI.jog_step_um) |
| `POST` | `/api/v1/scope/motion/move_to_plate` | [`MotionAPI.move_to_plate`](#MotionAPI.move_to_plate) |
| `POST` | `/api/v1/scope/motion/move_to_well` | [`MotionAPI.move_to_well`](#MotionAPI.move_to_well) |
| `POST` | `/api/v1/scope/motion/move_turret` | [`MotionAPI.move_turret`](#MotionAPI.move_turret) |
| `POST` | `/api/v1/scope/motion/move_xy` | [`MotionAPI.move_xy`](#MotionAPI.move_xy) |
| `POST` | `/api/v1/scope/motion/move_xy_by` | [`MotionAPI.move_xy_by`](#MotionAPI.move_xy_by) |
| `POST` | `/api/v1/scope/motion/move_z` | [`MotionAPI.move_z`](#MotionAPI.move_z) |
| `POST` | `/api/v1/scope/motion/move_z_by` | [`MotionAPI.move_z_by`](#MotionAPI.move_z_by) |
| `GET` | `/api/v1/scope/motion/positions_to_keep` | [`MotionAPI.positions_to_keep`](#MotionAPI.positions_to_keep) |
| `POST` | `/api/v1/scope/motion/set_precision_mode` | [`MotionAPI.set_precision_mode`](#MotionAPI.set_precision_mode) |
| `POST` | `/api/v1/scope/motion/start_home` | [`MotionAPI.start_home`](#MotionAPI.start_home) |
| `POST` | `/api/v1/scope/motion/start_move_to_plate` | [`MotionAPI.start_move_to_plate`](#MotionAPI.start_move_to_plate) |
| `POST` | `/api/v1/scope/motion/start_move_to_well` | [`MotionAPI.start_move_to_well`](#MotionAPI.start_move_to_well) |
| `POST` | `/api/v1/scope/motion/start_move_xy` | [`MotionAPI.start_move_xy`](#MotionAPI.start_move_xy) |
| `POST` | `/api/v1/scope/motion/start_move_xy_by` | [`MotionAPI.start_move_xy_by`](#MotionAPI.start_move_xy_by) |
| `POST` | `/api/v1/scope/motion/start_move_z` | [`MotionAPI.start_move_z`](#MotionAPI.start_move_z) |
| `POST` | `/api/v1/scope/motion/start_move_z_by` | [`MotionAPI.start_move_z_by`](#MotionAPI.start_move_z_by) |
| `POST` | `/api/v1/scope/motion/stop_motion` | [`MotionAPI.stop_motion`](#MotionAPI.stop_motion) |
| `GET` | `/api/v1/scope/motion/turret_status` | [`MotionAPI.turret_status`](#MotionAPI.turret_status) |
| `POST` | `/api/v1/scope/motion/wait_until_finished_moving` | [`MotionAPI.wait_until_finished_moving`](#MotionAPI.wait_until_finished_moving) |
| `GET` | `/api/v1/scope/motion_expected` | [`Lumascope.motion_expected`](#Lumascope.motion_expected) |
| `GET` | `/api/v1/scope/no_hardware` | [`Lumascope.no_hardware`](#Lumascope.no_hardware) |
| `GET` | `/api/v1/scope/objectives` | [`Lumascope.objectives`](#Lumascope.objectives) |
| `GET` | `/api/v1/scope/runtime_state/get_field_of_view` | [`RuntimeState.get_field_of_view`](#RuntimeState.get_field_of_view) |
| `GET` | `/api/v1/scope/runtime_state/get_pixel_size_um` | [`RuntimeState.get_pixel_size_um`](#RuntimeState.get_pixel_size_um) |
| `GET` | `/api/v1/scope/runtime_state/get_stage_offset` | [`RuntimeState.get_stage_offset`](#RuntimeState.get_stage_offset) |
| `GET` | `/api/v1/scope/runtime_state/get_well_label` | [`RuntimeState.get_well_label`](#RuntimeState.get_well_label) |
| `GET` | `/api/v1/scope/runtime_state/plate_to_stage` | [`RuntimeState.plate_to_stage`](#RuntimeState.plate_to_stage) |
| `GET` | `/api/v1/scope/runtime_state/resolve_current_objective` | [`RuntimeState.resolve_current_objective`](#RuntimeState.resolve_current_objective) |
| `GET` | `/api/v1/scope/runtime_state/stage_to_plate` | [`RuntimeState.stage_to_plate`](#RuntimeState.stage_to_plate) |
| `GET` | `/api/v1/scope/scope_models` | [`Lumascope.scope_models`](#Lumascope.scope_models) |
| `GET` | `/api/v1/scope/settings_template` | [`Lumascope.settings_template`](#Lumascope.settings_template) |
| `POST` | `/api/v1/scope/simulation/answer_model` | [`SimulationAPI.answer_model`](#SimulationAPI.answer_model) |
| `POST` | `/api/v1/scope/simulation/bind_camera_usb_driver` | [`SimulationAPI.bind_camera_usb_driver`](#SimulationAPI.bind_camera_usb_driver) |
| `POST` | `/api/v1/scope/simulation/clear` | [`SimulationAPI.clear`](#SimulationAPI.clear) |
| `POST` | `/api/v1/scope/simulation/delay_delivery` | [`SimulationAPI.delay_delivery`](#SimulationAPI.delay_delivery) |
| `POST` | `/api/v1/scope/simulation/delay_reply` | [`SimulationAPI.delay_reply`](#SimulationAPI.delay_reply) |
| `POST` | `/api/v1/scope/simulation/drop_replies` | [`SimulationAPI.drop_replies`](#SimulationAPI.drop_replies) |
| `POST` | `/api/v1/scope/simulation/fail_home` | [`SimulationAPI.fail_home`](#SimulationAPI.fail_home) |
| `GET` | `/api/v1/scope/simulation/injected` | [`SimulationAPI.injected`](#SimulationAPI.injected) |
| `POST` | `/api/v1/scope/simulation/interlock` | [`SimulationAPI.interlock`](#SimulationAPI.interlock) |
| `POST` | `/api/v1/scope/simulation/reboot` | [`SimulationAPI.reboot`](#SimulationAPI.reboot) |
| `POST` | `/api/v1/scope/simulation/refuse_camera_read` | [`SimulationAPI.refuse_camera_read`](#SimulationAPI.refuse_camera_read) |
| `POST` | `/api/v1/scope/simulation/refuse_camera_write` | [`SimulationAPI.refuse_camera_write`](#SimulationAPI.refuse_camera_write) |
| `POST` | `/api/v1/scope/simulation/refuse_led_write` | [`SimulationAPI.refuse_led_write`](#SimulationAPI.refuse_led_write) |
| `POST` | `/api/v1/scope/simulation/refuse_stream_start` | [`SimulationAPI.refuse_stream_start`](#SimulationAPI.refuse_stream_start) |
| `POST` | `/api/v1/scope/simulation/remove_camera` | [`SimulationAPI.remove_camera`](#SimulationAPI.remove_camera) |
| `POST` | `/api/v1/scope/simulation/replug` | [`SimulationAPI.replug`](#SimulationAPI.replug) |
| `POST` | `/api/v1/scope/simulation/silence` | [`SimulationAPI.silence`](#SimulationAPI.silence) |
| `POST` | `/api/v1/scope/simulation/stall_stream` | [`SimulationAPI.stall_stream`](#SimulationAPI.stall_stream) |
| `POST` | `/api/v1/scope/simulation/unplug` | [`SimulationAPI.unplug`](#SimulationAPI.unplug) |
| `POST` | `/api/v1/scope/simulation/unsilence` | [`SimulationAPI.unsilence`](#SimulationAPI.unsilence) |
| `GET` | `/api/v1/scope/unconnected_parts` | [`Lumascope.unconnected_parts`](#Lumascope.unconnected_parts) |
| `POST` | `/api/v1/set_protocol_capture_root` | [`ScopeSession.set_protocol_capture_root`](#ScopeSession.set_protocol_capture_root) |
| `POST` | `/api/v1/set_protocol_labware` | [`ScopeSession.set_protocol_labware`](#ScopeSession.set_protocol_labware) |
| `POST` | `/api/v1/set_protocol_schedule` | [`ScopeSession.set_protocol_schedule`](#ScopeSession.set_protocol_schedule) |
| `POST` | `/api/v1/set_step_autofocus` | [`ScopeSession.set_step_autofocus`](#ScopeSession.set_step_autofocus) |
| `GET` | `/api/v1/settings_are_provisional` | [`ScopeSession.settings_are_provisional`](#ScopeSession.settings_are_provisional) |
| `POST` | `/api/v1/show_channel` | [`ScopeSession.show_channel`](#ScopeSession.show_channel) |
| `GET` | `/api/v1/shown_channel` | [`ScopeSession.shown_channel`](#ScopeSession.shown_channel) |
| `GET` | `/api/v1/source_path` | [`ScopeSession.source_path`](#ScopeSession.source_path) |
| `POST` | `/api/v1/start_go_to_bookmark` | [`ScopeSession.start_go_to_bookmark`](#ScopeSession.start_go_to_bookmark) |
| `POST` | `/api/v1/start_go_to_step` | [`ScopeSession.start_go_to_step`](#ScopeSession.start_go_to_step) |
| `GET` | `/api/v1/status` | [`ScopeSession.status`](#ScopeSession.status) |
| `POST` | `/api/v1/take_control` | [`ScopeSession.take_control`](#ScopeSession.take_control) |
| `GET` | `/api/v1/tiling_config` | [`ScopeSession.tiling_config`](#ScopeSession.tiling_config) |
| `GET` | `/api/v1/turret_objective` | [`ScopeSession.turret_objective`](#ScopeSession.turret_objective) |
| `PUT` | `/api/v1/turret_objective` | [`ScopeSession.assign_turret_objective`](#ScopeSession.assign_turret_objective) |
| `GET` | `/api/v1/turret_slot_questions` | [`ScopeSession.turret_slot_questions`](#ScopeSession.turret_slot_questions) |
| `POST` | `/api/v1/update_settings` | [`ScopeSession.update_settings`](#ScopeSession.update_settings) |
| `POST` | `/api/v1/update_step` | [`ScopeSession.update_step`](#ScopeSession.update_step) |

## Events

Each event, the member or `RunEvents` field that delivers it, its record (whose fields are the callback's arguments, in order), and what carries it to a REST client.

| Event | Delivered by | Record | Over REST |
|---|---|---|---|
| `exposure` | [`CameraAPI.add_exposure_listener`](#CameraAPI.add_exposure_listener) | [`ExposureChanged`](#ExposureChanged) | event stream |
| `files_written` | [`RunEvents.files_written`](#RunEvents.files_written) | [`FilesWritten`](#FilesWritten) | event stream |
| `frame` | [`ImagingAPI.add_frame_listener`](#ImagingAPI.add_frame_listener) | [`FrameDelivered`](#FrameDelivered) | live view |
| `frame_captured` | [`RunEvents.frame_captured`](#RunEvents.frame_captured) | [`FrameCaptured`](#FrameCaptured) | live view |
| `gain` | [`CameraAPI.add_gain_listener`](#CameraAPI.add_gain_listener) | [`GainChanged`](#GainChanged) | event stream |
| `led` | [`IlluminationAPI.add_led_listener`](#IlluminationAPI.add_led_listener) | [`LedReading`](#LedReading) | event stream |
| `outcome` | [`ScopeSession.add_outcome_listener`](#ScopeSession.add_outcome_listener) | [`Notification`](#Notification) | event stream |
| `position` | [`MotionAPI.add_position_listener`](#MotionAPI.add_position_listener) | [`PositionChanged`](#PositionChanged) | event stream |
| `run_ended` | [`RunEvents.run_ended`](#RunEvents.run_ended) | [`RunEnded`](#RunEnded) | event stream |
| `run_state` | [`ScopeSession.add_run_state_listener`](#ScopeSession.add_run_state_listener) | [`Status`](#Status) | event stream |
| `scan_ended` | [`RunEvents.scan_ended`](#RunEvents.scan_ended) | [`ScanEnded`](#ScanEnded) | event stream |
| `scan_started` | [`RunEvents.scan_started`](#RunEvents.scan_started) | [`ScanStarted`](#ScanStarted) | event stream |
| `settings` | [`ScopeSession.add_settings_listener`](#ScopeSession.add_settings_listener) | [`SettingsChanged`](#SettingsChanged) | event stream |
| `shown_channel` | [`ScopeSession.add_shown_channel_listener`](#ScopeSession.add_shown_channel_listener) | [`ShownChannel`](#ShownChannel) | event stream |
| `step_started` | [`RunEvents.step_started`](#RunEvents.step_started) | [`StepStarted`](#StepStarted) | event stream |
| `turret` | [`MotionAPI.add_turret_listener`](#MotionAPI.add_turret_listener) | [`TurretChanged`](#TurretChanged) | event stream |
| `video_progress` | [`RunEvents.video_progress`](#RunEvents.video_progress) | [`VideoProgress`](#VideoProgress) | event stream |

## Saved-file facts

Every fact a saved file's record can state, by its key (`modules/image_utils.FILE_FACTS`), in the order the writer writes them: what it describes (a derived file states a `capture` fact never, a `channel` fact only on its inputs' channel, a `shared` fact as all its inputs state it), its key in the file's Plane block, the unit written beside it, and whether a sum of frames drops it.

| Key | Kind | Plane key | Unit | Dropped by a sum |
|---|---|---|---|---|
| `objective` | shared | `Objective` | -- | no |
| `plate_pos_mm` | shared | -- | -- | no |
| `z_pos_um` | shared | `PositionZ` | um | no |
| `illumination_ma` | channel | `Illumination` | mA | no |
| `illumination_by_channel_ma` | channel | `IlluminationByChannel` | mA | no |
| `illumination_timing` | channel | `IlluminationTiming` | -- | no |
| `exposure_time_ms` | channel | `ExposureTime` | ms | no |
| `gain_db` | channel | `Gain` | dB | no |
| `black_level` | channel | `BlackLevel` | -- | no |
| `frames_summed` | channel | `FramesSummed` | -- | yes |
| `binning_size` | shared | `Binning` | -- | no |
| `timestamp_iso` | capture | `Timestamp` | -- | no |
| `timestamp_source` | capture | `TimestampSource` | -- | no |
| `timestamp_camera_ticks` | capture | `TimestampCameraTicks` | -- | no |
| `timestamp_camera_tick_hz` | capture | `TimestampCameraTickHz` | -- | no |
| `frame_id` | capture | `FrameID` | -- | no |
| `frame_validity` | capture | `FrameValidity` | -- | no |
| `saturated_fraction` | capture | `SaturatedFraction` | -- | no |
| `dark` | capture | `Dark` | -- | no |
| `camera_held_a_channel` | capture | `CameraHeldAChannel` | -- | no |
| `camera_held_channel` | capture | `CameraHeldChannel` | -- | no |
| `fallback_focus` | capture | `FallbackFocus` | -- | no |
| `recording_frame` | capture | `RecordingFrame` | -- | no |
| `stage_moving` | capture | `StageMoving` | -- | no |
| `channel` | file | -- | -- | no |
| `channel_display` | channel | -- | -- | no |
| `excitation_nm` | channel | -- | -- | no |
| `filterset` | shared | -- | -- | no |
| `pixel_size_um` | shared | -- | -- | no |
| `instrument` | shared | -- | -- | no |
| `plate` | shared | -- | -- | no |
| `well_label` | shared | -- | -- | no |
| `datetime` | file | -- | -- | no |
| `significant_bits` | file | -- | -- | no |
| `algorithm` | file | -- | -- | no |
