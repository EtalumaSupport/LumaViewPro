# REST

The API over HTTP: serving it, routes, handles, jobs, limits, the control session, problems, events, the live view and files, a walkthrough with `curl`, and serving a client on another computer.

A server is LumaViewPro's, beside its window, or `python -m lumascope.rest`
from its source ([Serving it](#serving-it)). `GET /api` answers
the versions served. The OpenAPI 3.1 description of every route is at
`/api/v1/openapi.json`, with an interactive reference at `/docs`, served
whole by the server, so it works with no internet. Each route's
description lists the refusals and faults its member declares, under the
status each answers.

## Serving it

Two hosts serve the API, the same API from either:

- **LumaViewPro**, beside its window, when `rest_api.enabled` is `true` in its
  settings (shipped `false`). An installed LumaViewPro serves this way.
- **`python -m lumascope.rest`**, with no window, run from LumaViewPro's
  source by the Python its install script installed into ([Where it
  runs](python.md#where-it-runs)). An installed LumaViewPro has no
  such command. It serves even when `enabled` is `false`, but `enabled` must
  still be `true` or `false`.

```bash
python -m lumascope.rest --simulate      # a simulated scope; leave out --simulate for the hardware
```

It brings the scope up as LumaViewPro does -- the user's settings, the startup
home, the plugins -- and serves where `rest_api` says (shipped
`127.0.0.1:8000`, this computer only). It takes the single-instance lock, so
it will not start while LumaViewPro or another server runs on this computer,
simulated or not ([One session per scope](concepts.md#one-session-per-scope)). A failed startup home is
reported and the server still serves, so a client can home again. What
bring-up found -- each part, why one is not up, and the remedy it offers -- is
`GET /api/v1/bring_up_record`, for a client that connects after it was told. Ctrl-C or
SIGTERM closes it: new calls are refused (`'session_closing'`), the session
shuts down with the event stream open, and the stream's last event is
`closing`. It exits `0`, or `1` when it could not take the lock or serve
(printing why), stopped before it served, or the session did not close cleanly
(the log says why). The log is `logs/LVP_Log/` under the source folder:
`lumaviewpro.log` and `lumaviewpro_errors.log`. Captures go to the live
folder, `capture/` under the source folder as shipped (`live_folder`
`./capture`, a relative path being the source folder's).

`--source-path FOLDER` runs it on another folder's `data/` instead: its
settings, its catalogs, and the live folder they name. A simulated server
on a copy of `data/` leaves your own settings and captures untouched. The
copy carries the same single-instance lock, so one server still runs at a
time. A folder with no `data/settings.json` in it is refused, and the
server exits `1`.

```bash
mkdir -p /tmp/lvp-sim && cp -R data /tmp/lvp-sim/    # from the source folder
python -m lumascope.rest --simulate --source-path /tmp/lvp-sim
```

To serve a session your own program has brought up, its startup motion
included (`build_server` and `build_app` run none), build its server, the one
every host serves through:

```python norun
from lumascope import build_server

server = build_server(session, shut_down=session.shutdown)
server.run()    # until Ctrl-C, whose close shuts the session down
if not server.close_begun:    # it stopped before it served
    session.shutdown()
elif server.close_failed is not None:    # what the session's shutdown raised
    print(server.close_failed)
```

To mount the API in an ASGI application of your own, take it from
`build_app(session)`. Its `allowed_clients` and its body deadline apply
whoever serves it. The header deadline is `build_server`'s, so your program
sets its own server's. The address check below is on with
`build_app(session, bound=True)`, for a server on `rest_api`'s address and
port. Mounted, its
startup is not run for you: run it inside your own application's lifespan,
`async with api.router.lifespan_context(api):` where `api` is what
`build_app` returned, or it has no event stream.

**Addressed to it.** The server answers only a request addressed to its own
address and port. The `Host` a client sends names them, in any spelling of the
address, or `localhost` at that port when it serves on `127.0.0.1` or `::1`.
Any other is refused `421` (`'wrong_host'`), so address the scope by its IP
address, never by a name, which any site can point at it. A request sent by a
page from another site, its `Origin` not the server's own, is refused `403`
(`'foreign_origin'`): no web page you open can drive the scope. A program sends
no `Origin`, and `/docs` sends the server's own.

**The settings.** The server reads `rest_api` from the session's settings. It
is installation-only: no call changes it, from any caller
(`'installation_only'`). So it is edited in the
[installation](concepts.md#words-this-guide-uses)'s `data/current.json`
([Settings](python.md#settings)),
with LumaViewPro and the server closed. Each of them writes the whole settings
file back as it runs and when it closes, over an edit made meanwhile. A source
folder neither has run in has no `current.json`, and serves from the template's
`rest_api`. To change it there, copy `data/settings.json` to
`data/current.json` and edit the copy. The port is
`rest_api.port`, 1024 to 65535, and the command takes no port of its own: to
serve on another, change the setting.

A `rest_api` the server cannot use, or
an address and port it cannot take, is refused `RestServerNotStartedError`
before anything is served, from every host: LumaViewPro shows it and runs on
without a server, and `python -m lumascope.rest` prints it and exits `1`. A
`current.json` the server cannot use -- an edit that left it short of a key or
holding a block of the wrong kind -- is `SettingsFileError`, printed the same
way, naming the file and what is wrong. The file is not changed. So is a
`microscope` setting naming a model this version does not know, on a scope
whose motor board reports no model: `CatalogNameRefusedError` (reason
`model_unknown`), naming the setting and the models it does know. (A board that
reports a model, an OEM name included, says what the scope is.) While it
serves, `server.reach` says where and to whom (`python -m lumascope.rest`
prints it, and LumaViewPro shows it in Advanced Settings).

### Testing a client without a server

To test a client with no server, no port and no instance lock, send its
requests through FastAPI's `TestClient`:
`with fastapi.testclient.TestClient(build_app(session)) as client:`, the
session a simulated one on a copy of `data/`, so the test writes nothing into
your own. The `with` runs the application's startup, which the event stream
needs. A job is read ended once its work ends, by any client of the
application, inside a `with` or not. Make every client, a second one
included, with its own `with`, each after the one before it has ended.
`TestClient` returns a response only when its body ends, so it never returns
from `GET /api/v1/events` or `/live`: read the events from `/events/held` and
a frame from `/live.jpg`. `TestClient` needs `httpx2`, at the version
`requirements-dev.txt` pins.

## Routes

The session is the root: its member `<member>` is `/api/v1/<member>`, and each sub-object is
a path segment, so `session.scope.motion.start_move_xy` is
`POST /api/v1/scope/motion/start_move_xy`. As in Python, what is the session's
is a session route (`POST /api/v1/show_channel`) and the hardware is under
`/api/v1/scope/`. Each capability has one route.

- **A read is `GET`**: a property, a published field, or a method that changes
  nothing. A read method's arguments are its query string
  (`GET /api/v1/scope/motion/axis_limits?axis=Z`); a list is its key
  repeated, and a record one key per field, `<parameter>.<field>`.
- **A setting is `PUT` on its read's URL**, with the setter's arguments as a
  JSON object; the answer is the read after the write, the value the scope took
  (`PUT /api/v1/scope/camera/binning` with `{"size": 2}` answers `2`). A
  setting kept per channel takes the channel in the body and the query
  (`PUT /api/v1/channel_settings` with `{"channel": "BF", "acquire": "image"}`,
  then `GET /api/v1/channel_settings?channel=BF`). There a key left out is a
  setting not written, and `null` is the value. Such a setter has no route
  of its own. Not every setter is one: `set_auto_gain` keeps its own `POST`
  beside its read. Nor is every one named `set_`: `select_labware` is
  `PUT /api/v1/labware_id`, and `select_model` and `assign_turret_objective`
  likewise. Each member's [reference](../api-reference.md) entry names
  its route.
- **Every other call is `POST`**, its arguments a JSON object by parameter
  name; `null` is None, and a parameter left out takes its default. A member
  that returns an operation is a `POST` even when it changes nothing
  (`last_run`). The parameter names are the member's, in its
  [reference](../api-reference.md) entry and its route's body in `/docs`:
  `move_to_well(well)` is `{"well": "B2"}`.
- **A body carries only the keys its record names.** A key it does not name,
  in a nested record too, is refused `422` `invalid_request`. A key the
  record defaults may be left out and takes that default, as its field's
  `= value` in the [reference](../api-reference.md) shows; leaving out any
  other is refused the same way. A record is one
  schema in the OpenAPI description, whether it is sent or answered, and an
  answer is open: a record may gain a key in a later release. So send a
  record back built from its schema, not as the answer it came in.
- **The answer** to a finished call is `200` with its value as the body, `null`
  for None.
- **Arguments are checked before the member runs**: a key it does not take, a
  value of the wrong type, a missing one, or a number that is not finite is
  refused `422` (`'invalid_request'`), its `errors` naming each argument. A
  number where a channel is named is refused here too, as `'invalid_request'`.
- **Paths are names in the live folder**, `/`-separated
  (`"ProtocolData/run1"`). A name that leaves it is refused
  (`'outside_live_folder'`); `"."` is the live folder itself. An answered path is
  `{"name": ..., "host_path": ...}`, `name` null for a path outside it;
  `host_path` is the path on the scope PC in its own form, for a program
  running there.
- **Units are the API's**: an axis position crosses as
  `{"position_um": ..., "state": ...}`, a duration as a number of seconds, and
  a `datetime` as ISO 8601 with its offset. A field whose name ends in
  `_epoch_s` is a number, seconds since the epoch.

In-process members -- those taking a callback, returning a live object, or
waiting on the caller's thread -- have no route. Each has a wire form beside
it. A listener is an event, and `acting_in_control` is the `Control-Session`
header. A call that waits for a long action -- a move, a home, a go-to, a
build, a cell count, a report -- is its `start_` form, read as a job.

## Handles

A live object -- a protocol, a protocol runner -- crosses as a handle,
`{"handle": "1", "type": "Protocol"}`. Its members are at
`/api/v1/handles/<type>/<id>/<member>` (`GET /api/v1/handles/Protocol/1/period_s`),
and a parameter that takes it is sent its id (`{"protocol": "1"}`).
`GET /api/v1/handles` lists every handle held, oldest first.
`DELETE /api/v1/handles/<type>/<id>` forgets the id, never the object (`204`).
The session's protocol runner is every client's and is kept (`422`,
`'handle_shared'`). A forget is a write: under another's control session it is
refused. The handles held are bounded ([Limits](#limits)), so forget the
handles a client is done with.

## Jobs

Every call runs on its own thread, and its route waits for it as long as the
client says: `Prefer: wait=<seconds>`, 2 s when it says nothing, a longer wait
cut to 60 s. `Prefer: respond-async` asks for no wait at all. A preference
taken is echoed in `Preference-Applied`. A call still running then answers
`202 Accepted` with `Location: /api/v1/jobs/<id>`, `Retry-After: 1` and the
job:

```json
{"id": "3", "member": "handles/ProtocolRunner/2/start_single_scan", "status": "running",
 "requested": "2026-10-29T11:47:56.043-07:00", "ended": null,
 "progress": {"is_last_run": true, "is_live": true, "is_scanning": false,
              "is_stopping": false, "next_scan_at": "2026-10-29T11:47:56-07:00",
              "num_steps": 1, "period_s": 1200.0,
              "run_dir": {"name": "ProtocolData/20261029_114756", "host_path": ...},
              "scans_remaining": 1, "step_index": 0},
 "files": {"id": "4", "member": "handles/ProtocolRunner/2/start_single_scan/files",
           "status": "running", "requested": "2026-10-29T11:47:56.043-07:00",
           "ended": null, "progress": null}}
```

A member that returns an operation -- a run's start, `start_move_to_well`,
`start_home`, a saved still's `start_save_still` -- is answered `202` with the operation's
job as soon as it has started, whatever the wait. The route's description
names the job's model, and its `result` the record the operation ends with
(`start_build`'s `BuildsResultJob`, a run's `RunOutcomeJob`). Its twin that
waits for the operation, `move_to_well` or `home`, is in-process and has no
route. Any other call is answered `200` with its result within the wait, a
`202` job past it. `GET /api/v1/jobs/<id>` reads a job and answers `200`
whatever its status. It waits as `Prefer: wait` says. With
`Prefer: respond-async` it answers at once, which is how you poll the job's
`progress`. Once it has ended it carries `result` or `error` (the problem). A job's
`status` says whether the call answered a problem, not how the operation went:
a stopped run's job is `completed` with its outcome `aborted`, a stopped move's
`failed` with its stop notice. A run's job also carries `files`, the job of
the run's images, its `result` the run's `RunFiles`
([A run's files](runs.md#a-runs-files)). The next run is refused
`'files_writing'` until it ends. `GET /api/v1/jobs` lists every job held,
newest first.

`DELETE /api/v1/jobs/<id>` stops a running operation (`202`, the job) or
forgets a finished job (`204`). A stop passes any control session. A forget is
a write: under another's control session it is refused, and the job is kept.
A still has no stop, and its job ends as it would have. A call that became a
job only because it ran past its wait has no operation to stop: deleting it
while it runs is refused `409` (`'not_cancelable'`). How many jobs are kept, how many calls run at once, and
how large and how slow a body may be are [Limits](#limits). Take a job's id from the answer that made it, never count it: every call held
as a job takes the next id.

A `POST` or `PUT` that may be sent twice, an upload among them, carries
`Idempotency-Key`. Sent again with the same key, it runs nothing and is
answered what the first was, with `Idempotent-Replayed: true`. A key sent with
another request is refused `422` (`'idempotency_key_reused'`). How long a key
is held is [Limits](#limits).

## Limits

- **Calls.** At most 256 run at once, an upload counted among them while its
  body arrives; past that a call is refused `503` (`'overloaded'`).
- **Jobs.** Finished jobs are kept up to 1000 and 24 hours.
- **Handles.** At most 1000 are held; past that, a call whose answer could
  carry a new one is refused `503` (`'overloaded'`, `Retry-After: 5`) before
  it runs.
- **Bodies.** A call's JSON body is at most 64 KiB, an upload's 64 MiB, refused
  `413` (`'body_too_large'`) past it: on its `Content-Length` before a byte is
  read, a chunked one at the first byte past it. The server still reads a
  refused body to its end.
- **Uploads.** Four are read at once, each counted among the calls while its
  body arrives: a fifth, or one past the 256 calls, is refused `503`
  (`'overloaded'`, `Retry-After`) before its body is read, whichever clients
  hold the four.
- **Time.** A body arrives within 20 s plus its length at 64 KiB a second,
  else it is refused `408` (`'body_too_slow'`), its connection closed and its
  place given back: an upload whose client's link dropped is refused so. A
  connection has 20 s for each request's headers, from its opening and from
  each answer, or is closed. A request that has arrived is never ended by it,
  however long its answer takes.
- **Idempotency keys.** A key is held for 24 hours, and at most 10,000, the
  oldest let go first.

## The control session

The wire form of [the control session](concepts.md#the-control-session).
`POST /api/v1/take_control` with `{"name": ...}` answers the id, to this
client alone. `GET /api/v1/control` reads who holds it, and since when. While
it is held, every write carries its id in the `Control-Session` header. Any
other write is refused `409` (`'control_session_held'`), and a write naming
an ended session `422` (`'control_session_ended'`). Reads, stops and the
display preferences (the settings the reference's
[Settings](../api-reference.md#settings) marks a preference) pass without it,
and a read or a stop naming an ended session is answered as one with no header: a client stops what it started
whatever became of its session. `POST /api/v1/end_control` with
`{"session_id": ...}` ends it, stopping a run or recording taken in it.
Every request carrying the header renews the session, a read or a job poll
included. When a silent one ends is
[The control session](concepts.md#the-control-session).
Taking back control without the id is the window's alone: it has no route.

## Problems

Every answer that is not a result is an RFC 9457 problem,
`application/problem+json`:

```json
{"type": "urn:lumascope:problem:not_offered", "title": "Not One of the Values Offered",
 "detail": "axis must be one of X, Y, Z, T; 'W' is not one.", "status": 422,
 "instance": "urn:uuid:3faa1296-...", "kind": "refusal", "reason": "not_offered",
 "cause": "request", "remedy": null, "argument": "axis", "offered": ["X", "Y", "Z", "T"],
 "board": null, "labware_id": null, "limits": null, "member": null, "takes": null}
```

`title` and `detail` are the words a person is shown. `reason` is the code to
branch on, and `kind` is the outcome's. A problem names no class: a reason
means one thing whichever class carries it, and the Outcomes by REST status
in its member's reference entry name the class. `cause` is whether the same request
could succeed later, the [cause](concepts.md#reasons-and-causes) a Python
caller reads (`state`, `request` or `restart`; null for a fault). `remedy` is
taken by sending its `member` back as `POST /api/v1/apply_remedy` with
`{"remedy": "recover_file_writer"}`. A refusal carries every
field its class publishes -- `argument`, `offered`, `limits`, `holder` --
null where unused, so a client corrects its request without reading the
words. An outcome that declares no reason is `type` `about:blank`.

The status is the outcome's [cause](concepts.md#outcomes), or the server's own
refusal's. Every reason and its status is in the reference's
[table](../api-reference.md#ServerRefusedError) and
[Outcomes](../api-reference.md#outcomes).

| Status | Means |
|---|---|
| `409` | A refusal the scope's state gave: the same request may succeed later. Also a move or home a stop ended (`kind` `notice`), and `'not_cancelable'`. |
| `422` | A refusal of the request as sent: change it. Also `'invalid_request'`, `'handle_shared'` and `'idempotency_key_reused'`. |
| `403` | `'client_not_allowed'`: a client [`allowed_clients`](#serving-another-computer) does not list; its connection is closed. `'foreign_origin'`: a page from another site sent it ([Serving it](#serving-it)). |
| `421` | `'wrong_host'`: a request addressed to another server than this one's address and port ([Serving it](#serving-it)). |
| `500` | A fault: the call failed. |
| `404` | No such route, handle, job or file. |
| `405` / `408` / `413` / `415` | The wrong method; a body that did not arrive in time (`'body_too_slow'`; the connection is closed); a body larger than its route takes (`'body_too_large'`), both in [Limits](#limits); a body that is not JSON. |
| `503` | With `Retry-After`, try again later (`'overloaded'`, `'no_frame_yet'`). Without it, nothing succeeds until the application restarts (`'not_connected'`, `'session_closing'`, and every other restart cause). |

## The event stream

`GET /api/v1/events` is a server-sent event stream. Its first event is
`status`, the scope as `GET /api/v1/status` reads it. After that every change
is an event, named as in the [events table](../api-reference.md#events), its data
the event's record -- but `frame`, the camera's frames, which are the live
view's, and `frame_captured`, which is not sent. A
quiet stream sends a keep-alive comment every 15 s and asks a client to
reconnect after 1 s (`retry: 1000`).

A run's events carry `run`, `{"job": "<id>"}`, the job of the run that sent
them. Only a run started over REST sends them: a run started in the window or
by Python in the same process shows on the stream as the scope's own events
alone (`status`, `position`, `led` and the like). They come as
`scan_started`, a `step_started` per step (and `video_progress` while a video
step records), `scan_ended`, scan after scan; then `run_ended`,
`files_written` and the two jobs' `job_ended`, in no fixed order. No event
marks a step's end: it has ended at the next `step_started` or `scan_ended`.
Every job's end is a
`job_ended` event, its data the job. A run's `files` job ends with its own, so
match a `job_ended` by its `id`. Every notice, and every outcome nobody asked
for, is an `outcome` event (a model set for the next start among them). A
refusal or fault that answered a call is not, its asker having had it.

A client that reconnects with `Last-Event-ID` is sent what it missed. When the
server no longer holds it (more than 1024 events ago, or a restarted server),
it is sent `reset` and then `status`, where it starts again. So is a
connected reader that falls 1024 events behind. A client that
cannot hold a stream open polls `GET /api/v1/events/held?after=<id>`. It gets
the events a stream resumed from `after` is sent, as
`{"events": [{"id", "event", "data"}, ...], "next": <id>}`. Send `next` as the
next `after`. With no `after`, it gets `status` and its id. The last event is
`closing`.

## The live view and files

Every frame below -- `/live`, `/live.jpg`, `/still`'s pixels -- has row 0 at
the top of the picture, as LumaViewPro's window and its saved files have it.

- `GET /api/v1/live` -- the camera's frames as MJPEG, for a browser's `<img>`;
  `max_width` bounds a frame's width, half the camera's frame when not given.
  Each part carries `X-Frame-Ordinal`, a count of the camera's frames received
  while anyone watched (a gap is frames this client skipped; it is not
  `Frame.seq`), and `X-Frame-Timestamp`, the host's clock when the frame
  arrived, ISO 8601. The camera's delivered frame rate is
  `GET /api/v1/scope/imaging/delivered_rate`.
- `GET /api/v1/live.jpg` -- one frame, with the same two headers, `max_width`
  as for `/live`; `503` (`'no_frame_yet'`, with `Retry-After`) when the camera
  sends none in time.
- `GET /api/v1/live_folder_listing?name=<folder>` -- what a folder in the live
  folder holds, one level, each entry by its full name. A download sends that
  name percent-encoded, its `/` kept as `/`: a run's `sequence_name` can put a
  space, `#` or `%` in it (`curl` needs `%20` for a space). A run's folder is its job's `progress.run_dir`
  ([`RunProgress`](../api-reference.md#RunProgress)), there from the run's
  start, so list it once the run's `files` job has ended.
- `GET /api/v1/files/<name>` -- a file in the live folder, as an attachment; a
  `Range` request resumes a large download.
- `PUT /api/v1/files/<name>` -- a new file in the live folder, answered `201`
  with its path; a file already at the name is refused (`'file_exists'`), so
  an upload never replaces a capture. It is a write, carrying
  `Control-Session` under a held session. Its size, how many are read at once
  and its deadline are [Limits](#limits).
- `GET /api/v1/previews/<name>` -- a saved image downscaled to a JPEG for
  display (`max_width` as for `/live`).
- `POST /api/v1/still` -- a still at the camera's own depth, saved nowhere
  (`ScopeSession.capture_still`): the body is the still's record as JSON,
  `X-Record-Length` bytes, then its pixels, row by row, little-endian, shaped
  by `X-Image-Shape` (`rows,columns`), `X-Image-Dtype` (a NumPy dtype name,
  such as `uint16`) and `X-Significant-Bits`. Its JSON body
  takes the member's arguments, each its own default when left out. With
  `{"channel": "BF"}` the camera holds the shown channel's stored settings for
  the frame, unless a raw camera write came after the channel was shown: the
  written values stand, and the record's `held_channel` is null. With no body
  it takes the camera as set.
  `{"frames_summed": 4, "sum_delay_s": 0.1}` sums four frames. `accept_dark`,
  `all_ones_check` and `timeout_s` are the member's.
  `{"exposed_after_call": true}` keeps only a frame exposed after the call, for
  a light or shutter LumaViewPro did not switch. It answers when the still is
  taken, never as a job.

The live view and previews are display: 8-bit gray, lossy, with no metadata.
The data is the file. Every answer carries `X-Request-ID`, which a problem
names as its `instance`, and each request is one line of
`lumaviewpro_rest_api.log`.

## A walkthrough

A session over REST with `curl`: the status, a move and a light, a one-step run and its files, and the control session.

Against a simulated server on a copy of `data/`, the `--source-path` form in
[Serving it](#serving-it): the walkthrough confirms an objective and writes a
channel's settings, which a server on your own `data/` would keep. It runs in
`bash` (on Windows, Git Bash or WSL:
[the examples' README](../api_examples/README.md#running-a-rest-example)),
with `curl` 7.82 or newer, and reads the answers with `jq`, as the sections
above describe them. Each line's
comment is the status it answers; a handle's or a job's id is read from the
answer that made it. Every line here is run, in this order, in LumaViewPro's
test suite.

```bash
API=http://127.0.0.1:8000/api/v1

curl -s $API/status                                         # 200: the scope's work, axes and camera
curl -s "$API/scope/motion/axis_limits?axis=W"          # 422 not_offered: offered X, Y, Z, T
MOVE=$(curl -s --json '{"z_um": 4000.0}' $API/scope/motion/start_move_z | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$MOVE               # 200: "status": "completed"
curl -s --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0
curl -s --json '{}' $API/scope/illumination/leds_off        # 200
```

Build a one-step protocol and run it. A run is answered `202` at once with its
job. Poll the job at once with `Prefer: respond-async`, or read it with
`Prefer: wait` until it has ended, then its `files` job until the run's images
are on disk. A run is refused until the session knows which objective is
mounted ([The plate, the objective and the
model](concepts.md#the-plate-the-objective-and-the-model)), so the first lines
say which it is. `GET $API/current_slot_objective_question` always answers,
proposing the objective a scope already knows, so the lines change nothing
there. Its `turret_slot` is `null` without a turret.

```bash
OBJECTIVE=$(curl -s $API/current_slot_objective_question | jq -r .proposed)   # 200
SLOT=$(curl -s $API/current_slot_objective_question | jq -r .turret_slot)    # 200: null without a turret
curl -s --json "{\"objective_id\": \"$OBJECTIVE\", \"slot\": $SLOT}" $API/confirm_objective   # 200
curl -s -X PUT --json '{"channel": "BF", "acquire": "image"}' $API/channel_settings   # 200: BF's settings
PROTOCOL=$(curl -s --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
MOVE=$(curl -s --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$MOVE               # 200: "status": "completed"
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200: ["custom0000_BF"]
RUNNER=$(curl -s --json '{}' $API/protocol_runner | jq -r .handle)         # 200
JOB=$(curl -s --json "{\"protocol\": \"$PROTOCOL\"}" $API/handles/ProtocolRunner/$RUNNER/start_single_scan | jq -r .id)   # 202
curl -s -H 'Prefer: respond-async' $API/jobs/$JOB           # 200: the job as it stands, its "progress" where the run has got
FILES=$(curl -s -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r .files.id)   # 200: "status": "completed", "result" the RunOutcome
curl -s -H 'Prefer: wait=60' $API/jobs/$FILES               # 200: "result" the RunFiles, once the images are on disk
curl -s "$API/live_folder_listing?name=ProtocolData"        # 200: the run's folder
```

Take the control session, and see another write refused while it is held.

```bash
CONTROL=$(curl -s --json '{"name": "walkthrough"}' $API/take_control | jq -r .)   # 200: the id, for this client alone
curl -s --json '{"z_um": 3000.0}' $API/scope/motion/start_move_z                    # 409 control_session_held
MOVE=$(curl -s -H "Control-Session: $CONTROL" --json '{"z_um": 3000.0}' $API/scope/motion/start_move_z | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$MOVE               # 200: "status": "completed"
curl -s --json "{\"session_id\": \"$CONTROL\"}" $API/end_control                    # 200
curl -s $API/events/held                                    # 200: where a poll of the events begins
```

A client that holds a connection open reads the events as they happen with
`curl -N $API/events`.

## Serving another computer

Shipped, the server answers this computer only (`127.0.0.1`). To drive the
scope from another computer over a dedicated cable, or a private network the
lab trusts, name the scope PC's address on that link:

```json
"rest_api": {
    "enabled": true,
    "port": 8000,
    "address": "192.168.10.2",
    "allowed_clients": ["192.168.10.3"]
}
```

- `address` is one address of the scope PC, IPv4 or IPv6, never `0.0.0.0` or
  `::` (every network the PC is on), never a name such as `localhost` (write
  `127.0.0.1`), and not a link-local IPv6 address with a zone (`fe80::1%eth0`).
- `allowed_clients`, optional, lists the client addresses or networks served
  (`"192.168.10.3"`, `"192.168.10.0/24"`); a request from any other address is
  refused `403` (`client_not_allowed`, [Problems](#problems)). Empty, every client that can reach the
  address is served. A network is written as its network address: `192.168.10.0/24`,
  not `192.168.10.5/24`. Each entry is the address's family, and a list beside
  `127.0.0.1` is refused, since no other computer reaches it.

This is plain HTTP with no credential: anyone who can reach the address drives
the scope, and the control session's id travels unencrypted. The allowlist
guards against a client on the wrong network, not an attacker on the same one.
Use it on a cable or a network you trust. Nor does it bound connections: a
machine that can reach the address can open them faster than their header
deadline closes them, until the server, and LumaViewPro with it, has no more
files it can open.

On the scope PC:

- **Connect the network before LumaViewPro starts.** An unplugged cable takes
  its address with it, so a server started before the cable is in cannot take
  the address. LumaViewPro says so and runs with REST off until it restarts,
  and `python -m lumascope.rest` prints it and exits `1`.
- **The firewall.** The installer adds no firewall rule: only REST on a
  network address needs one. Windows asks, the first time LumaViewPro serves
  on a network, whether to allow it, and answering needs an administrator.
  Allow it on Public networks too: a direct cable with fixed addresses is an
  unidentified network, which Windows counts as Public. Canceling the prompt
  makes Windows block LumaViewPro, and a block outranks any allow. Where no
  one can answer the prompt, or after a cancel, an administrator opens it in
  a Command Prompt run as administrator, the first line removing any block:

  ```
  netsh advfirewall firewall delete rule name=all dir=in program="C:\Program Files\LumaViewPro\lumaviewpro.exe"
  netsh advfirewall firewall add rule name="LumaViewPro REST" dir=in action=allow program="C:\Program Files\LumaViewPro\lumaviewpro.exe" enable=yes
  ```

  `netsh advfirewall firewall delete rule name="LumaViewPro REST"` removes
  it. A site whose policy ignores local rules needs its IT to add the rule.
  A server run from source serves through its `python.exe`, and Windows asks
  for that program instead. On Linux, a firewall that is on (firewalld on Fedora and RHEL, ufw when enabled
  on Ubuntu) drops the connection with no prompt and no message: open the port,
  for example `sudo firewall-cmd --permanent --add-port=8000/tcp && sudo
  firewall-cmd --reload`, or `sudo ufw allow from 192.168.10.3 to any port 8000
  proto tcp`.
