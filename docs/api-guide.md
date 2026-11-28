# The Lumascope API guide

This guide shows how to drive an Etaluma Lumascope from your own program, in
Python or over REST: how the API fits together, and how to do each task with
it. Every member, field, closed set and outcome class is in the
[API reference](api-reference.md), generated from the code; this guide links to
it rather than repeating it. The guide is this page and the files under
[`guide/`](guide): load the ones [the map](#the-map) names for your task.

> **Pre-release.** Until LumaViewPro `4.0.0` ships, the API may change without a
> deprecation cycle: members, routes, records and outcome reasons may be
> renamed, moved or removed. Most likely to move are the REST surface (its
> routes, handles, jobs, problems and event stream) and the protocol members
> and file columns. If you build on the API before `4.0.0`, **contact Etaluma
> support**, so we can consult you before a structural change. From `4.0.0`, a
> 4.x release changes the API only as the [Changelog](#changelog)'s rules
> allow: an addition or a behavior change at a minor version, never a rename
> or removal without a deprecation cycle. Each change is recorded there. What freezes is what the
> [API reference](api-reference.md) lists, the [setting paths](api-reference.md#settings)
> among them, and the plugin contract
> ([Plugins](#plugins)), with one exception: the simulator's surface,
> [`SimulationAPI`](api-reference.md#SimulationAPI), may change in any 4.x
> release ([The simulator](guide/simulator.md#the-simulator)). Etaluma's bench and support
> members are not in the reference and are outside the freeze.

**Contents.** [What LumaViewPro is](#what-lumaviewpro-is) ·
[Supported Lumascopes](#supported-lumascopes) ·
[The architecture at a glance](#the-architecture-at-a-glance) ·
[The map](#the-map) · [Plugins](#plugins) ·
[Writing this guide](#writing-this-guide) · [Changelog](#changelog)

## What LumaViewPro is

LumaViewPro controls Etaluma's Lumascope microscopes: the LEDs that light the
sample, the stage, focus and objective turret that move it, and the camera that
images it. It runs as a desktop application, and the API it is built on is open
to your own programs:

- **Python**, in the same process: a script, a notebook, MATLAB calling Python,
  or a plugin loaded into LumaViewPro.
- **REST**, over HTTP from any language: a server LumaViewPro starts beside its
  window, or one you start without a window at all.

This guide is for researchers who script an experiment and for developers who
integrate a Lumascope into a larger system. LumaViewPro runs on Python 3.12 and
3.13, on Windows, macOS and Linux.

A Python program runs from LumaViewPro's source, with the Python its install
script set up ([Where it runs](guide/python.md#where-it-runs)). A REST client
needs only a running server.
To start, bring a session up on the simulator, which needs no microscope
([Bringing a session up in Python](guide/python.md#bringing-a-session-up-in-python),
`simulate=True`), or start a simulated server
([Serving it](guide/rest.md#serving-it), `--simulate`). Where your settings,
captures and logs live is in the same two places.

## Supported Lumascopes

Every shipped Lumascope is supported. The models differ in what they have, and
the API tells your program what the attached scope has, so it never needs to
know models by name. Here, as in every example, `session` is the
`ScopeSession` your program brought up and `scope` is `session.scope`:

```python
caps = scope.capabilities                   # read once, when the scope came up
print(caps.axes, caps.has_xy_stage, caps.has_turret)
print(caps.camera_model, caps.sensor_pitch_um, caps.led_max_ma)
```

Over REST it is `GET /api/v1/scope/capabilities`. `camera_model` names the
sensor, not the scope: every FX2 model answers `MT9P031-LS620`.

| Model | Focus (Z) | XY stage | Turret | Channels | Controller |
|---|---|---|---|---|---|
| LS820 | yes | -- | -- | BF, PC, DF, Ex405, Ex488, Ex589 | EL-0940 |
| LS850 | yes | yes | -- | BF, PC, DF, Ex405, Ex488, Ex589 | EL-0940 |
| LS850-0 | yes | yes | -- | BF, PC, DF | EL-0940 |
| LS850T | yes | yes | yes | BF, PC, DF, Ex405, Ex488, Ex589 | EL-0940 |
| LS850T-0 | yes | yes | yes | BF, PC, DF | EL-0940 |
| Lumi | yes | -- | -- | BF, PC, DF, Lumi | EL-0940 |
| LS560 | -- | -- | -- | BF, Ex488 | FX2 |
| LS460 | -- | -- | -- | BF | FX2 |
| LS620 | -- | -- | -- | BF, Ex405, Ex488, Ex589 | FX2 |
| LS720 | yes | yes | -- | BF, Ex405, Ex488, Ex589 | FX2, with a TMCM-6110 motor board |

- **Channels.** `BF` brightfield, `PC` phase contrast, `DF` darkfield, `Ex405`,
  `Ex488` and `Ex589` fluorescence named for its excitation wavelength in nm, and
  `Lumi` luminescence, which lights no LED. The FX2 models label brightfield
  "BF-Phase". The release also knows `Ex550` and `Ex640`. Which channels the
  attached scope has, and which LEDs each drives, is `scope.channel_identity`
  ([Illumination](guide/light-and-camera.md#illumination)).
- **EL-0940 models** have one mainboard carrying the LED controller and the
  motor controller, and a Basler or IDS camera, either on any model
  (`scope.capabilities.camera_model` says which).
- **FX2 models** have one USB device that serves both the camera and the LEDs.
  The LS720 adds a TMCM-6110 motor board for X, Y and focus; the LS460, LS560
  and LS620 have no motion, and present identically, so the user selects which.

## The architecture at a glance

| Layer | What it is | How a caller reaches it |
|---|---|---|
| The session, `ScopeSession` | One per microscope, brought up by `ScopeSession.create` the same way in every host. It owns the settings, protocols, runs, captures, the control session and outcomes. | In Python, the session your program brought up, or the one a plugin is handed; over REST, `/api/v1/<member>`. |
| The scope, `session.scope` | The instrument, one area per part: `motion`, `illumination`, `camera`, `imaging` and the rest. | In Python, `session.scope.<area>`; over REST, `/api/v1/scope/<area>/<member>`. |
| The drivers and the hardware | The motor board, the LED board and the camera, or the simulator, which stands in for every board. | Only through the scope. |

LumaViewPro's window is one client of the same API, and its REST server, beside
the window or run without one, has one route per member.

- **Python and REST are one API**: `scope.motion.start_move_z` is
  `POST /api/v1/scope/motion/start_move_z`, with the same refusals, faults and
  events; the form that waits, `move_z`, is in-process, and a REST client reads
  the start form's job instead. Each part's commands, from every caller, run one at a time, in order.
- **A call that works returns its value; every other outcome is a typed
  exception** with a `reason`. It is raised to the caller or reported to
  listeners. It is a refusal (nothing was done), a fault (something broke) or a
  notice (something the caller should know, a stopped move among them).
- **One client at a time may take the control session.** Work holds the scope
  for its length: a run, a home, a recording or a diagnostic
  ([The holders](guide/concepts.md#the-holders)).

The detail is [concepts.md](guide/concepts.md).

## The map

For every task, read [concepts.md](guide/concepts.md) first. Then read
[python.md](guide/python.md) if your program calls the API from Python, in its
own process or as a plugin, or [rest.md](guide/rest.md) if it calls over HTTP.
Then read the files the task's row names; each links to the others where it
needs them. Every member, field and outcome is in the
[API reference](api-reference.md), which the files link into by anchor.
[`api_examples/`](api_examples) holds a Python script per task, each with a
`.sh` twin that does the same over REST with `curl`; its
[README](api_examples/README.md) says how to run them. LumaViewPro's test suite runs
every example, script and twin on the simulator, with the REST walkthrough and
every Python block of this guide. The one block it does not run is the server in
[Serving it](guide/rest.md#serving-it), which serves until it is stopped.

| Task | Start from | Example | Read |
|---|---|---|---|
| Bring a session up; read and write its settings | `ScopeSession.create` (`simulate=True` for the simulator), `session.run_application_session()`, `session.update_settings` | `first_still` | [concepts.md](guide/concepts.md), [python.md](guide/python.md), [rest.md](guide/rest.md) |
| Move the stage, the focus or the turret, by stage position or on the plate | `home`, `move_z`, `move_xy`, `move_to_well`, `move_turret`, `start_move_z`; `axis_limits` bounds every move | | [motion.md](guide/motion.md) |
| Light a channel; set the camera | `session.show_channel`, the channel a capture is named for; `led_on`, `leds_off`; `session.set_channel_settings`; `set_exposure_ms`, `set_gain_db` | `channel_settings` | [light-and-camera.md](guide/light-and-camera.md) |
| Take a still or a recording; download a file | `session.capture_still`, `session.manual_capture.start_save_still`, `session.manual_recording.start`; the record a saved file carries | `first_still` | [light-and-camera.md](guide/light-and-camera.md), [captures-and-files.md](guide/captures-and-files.md) |
| Run a scan or a protocol; build a stack, stitch, composite or video from its folder | `create_empty_protocol`, `add_step`, `session.protocol_runner()` (`start_single_scan`, `start_protocol`), `post_processing.build` | `scan`, `z_stack_saved`, `stop_a_run` | [runs.md](guide/runs.md) |
| Take the control session, so no other client writes ([Who may act](guide/concepts.md#who-may-act)) | `take_control`, `acting_in_control`, `end_control`; over REST, the `Control-Session` header ([The control session](guide/rest.md#the-control-session)) | `control_session` | [concepts.md](guide/concepts.md), [python.md](guide/python.md), [rest.md](guide/rest.md) |
| Hear outcomes, events and live frames ([Hearing outcomes](guide/python.md#hearing-outcomes), [The event stream](guide/rest.md#the-event-stream)) | `add_outcome_listener`, `add_frame_listener`; `GET /api/v1/events`, `GET /api/v1/live` | | [python.md](guide/python.md), [rest.md](guide/rest.md) |
| Test against the simulator, or inject a fault | `simulate=True`, `scope.simulation` | `refusals_and_faults` | [simulator.md](guide/simulator.md) |
| Drive the scope from another acquisition engine (Micro-Manager, pymmcore-plus, a lab's own sequencer) | `capture_still(exposed_after_call=True)`, `set_exposure_ms`, `led_on` | | [outside-engines.md](guide/outside-engines.md), [light-and-camera.md](guide/light-and-camera.md) |
| Serve REST, on this computer or for a client on another; learn it from a worked session with `curl` ([Serving it](guide/rest.md#serving-it), [A walkthrough](guide/rest.md#a-walkthrough), [Serving another computer](guide/rest.md#serving-another-computer)) | `python -m lumascope.rest`, `build_server` | the walkthrough | [rest.md](guide/rest.md) |
| Write a plugin | `PluginSpec`, `ctx.plugins.<namespace>.register` | | [plugin-tutorial.md](plugin-tutorial.md), [python.md](guide/python.md) |

## Plugins

A plugin extends LumaViewPro: a post-processor, a live frame processor, a
panel in the window. It is a Python package loaded at startup, handed the
session, and it uses the API this guide describes. How to write one, its
lifecycle, and which plugin API level a host offers are in
[`plugin-tutorial.md`](plugin-tutorial.md). The plugin contract -- the session's
`plugins` and the namespaces' members the tutorial teaches -- freezes with the
API at `4.0.0`. A plugin cannot add REST routes in this release, and that
namespace is outside the freeze
([The reserved namespace](plugin-tutorial.md#the-reserved-namespace)).

## Writing this guide

How the hub, the files under `guide/` and the examples' README are written. A
change to them keeps to these lines.

- **Identifiers verbatim.** A class, member, field, reason, route or setting is
  spelled as the code spells it, in backticks, whatever the lines below say.
- **US spelling** in prose: color, gray, center, behavior, catalog.
- **"You" for the reader**, the person writing the program.
- **One name for one thing**, the name
  [Words this guide uses](guide/concepts.md#words-this-guide-uses) gives it. A
  new term goes there before the guide uses it.
- **Short sentences**: one idea each, about 20 words. A long sentence splits at
  its semicolons and dashes, and keeps every qualifier it had.
- **Plain words first, then the contract.** A call that is refused is
  "refused", with its class and reason, never "won't": the reason is what your
  program reads.

## Changelog

Every change to the API after `4.0.0`, recorded in the same commit as the
change. Before `4.0.0` the API is fluid by design, and nothing is recorded here.

| Version | Date | Type | Member | Change |
|---|---|---|---|---|

- **Type** `additive` (a new optional argument, field or member): a minor
  version bump (4.x.0); a patch bump is fixes only.
- **Type** `behavior-change`: a minor version bump.
- **Type** `rename` / `removal`: a deprecation cycle (the old name kept with a
  `FutureWarning`, removed in the next major version), or a major version bump.

(No entries yet: 4.0.0 has not shipped.)
