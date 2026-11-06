# REST

The API over HTTP: serving it, reaching it from another computer, routes, handles, jobs, the control session, problems, events, the live view and files, and a walkthrough.

## Serving it

```bash
python -m lumascope.rest --simulate      # a simulated scope; leave out --simulate for the hardware
```

The server brings the scope up as LumaViewPro does -- the user's settings, the
startup home, the plugins -- and serves where `rest_api` in its settings says
(shipped `127.0.0.1:8000`, this computer only; see "Reaching it from another
computer"). It takes the same single-instance lock as
LumaViewPro, so it will not start while LumaViewPro or another server drives
the scope. A failed startup home is reported and the server still serves, so a
client can home again. Ctrl-C closes it: new calls are refused
(`'server_closing'`), the session shuts down with the event stream open, the
stream's last event is `closing`, and the process exits `0`, or `1` when the
session did not close cleanly.

LumaViewPro serves the same API beside its window when `rest_api.enabled` is
set in its settings (shipped `false`). To serve a session your own program has
brought up, build its server, the one every host serves through:

```python norun
from lumascope import build_server

server = build_server(session, shut_down=session.shutdown)
server.run()    # until Ctrl-C, whose close shuts the session down
if not server.close_begun:    # it stopped before it served
    session.shutdown()
```

The server reads `rest_api` from the session's settings. A `rest_api` setting
it cannot use, or an address and port it cannot take, is refused with
`RestServerNotStartedError` before anything is served, from every host:
LumaViewPro shows it and runs on without a server, and `python -m lumascope.rest`
prints it and exits `1`. While it serves, `server.reach` says where and to
whom (`python -m lumascope.rest` prints it, and LumaViewPro shows it in Advanced
Settings). The settings are read from the file as written, and LumaViewPro
writes its whole settings file back as it runs, so edit `rest_api` with
LumaViewPro closed.

## Reaching it from another computer

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
  refused `403` (`client_not_allowed`). Empty, every client that can reach the
  address is served. A network is written as its network address: `192.168.10.0/24`,
  not `192.168.10.5/24`. Each entry is the address's family, and a list beside
  `127.0.0.1` is refused, since no other computer reaches it.

This is plain HTTP with no credential: anyone who can reach the address drives
the scope, and the control session's id travels unencrypted. The allowlist
guards against a client on the wrong network, not an attacker on the same one;
use it on a cable or a network you trust.

On the scope PC:

- **Connect the network before LumaViewPro starts.** An unplugged cable takes
  its address with it, so a server started before the cable is in cannot take
  the address; it says so, and REST is off until LumaViewPro restarts.
- **The firewall.** The installer adds no firewall rule: only REST on a
  network address needs one. Windows asks, the first time LumaViewPro serves
  on a network, whether to allow it, and answering needs an administrator.
  Allow it on Public networks too: a direct cable with fixed addresses is an
  unidentified network, which Windows counts as Public. Cancelling the prompt
  makes Windows block LumaViewPro, and a block outranks any allow. Where no
  one can answer the prompt, or after a cancel, an administrator opens it in
  a Command Prompt run as administrator, the first line removing any block:

  ```
  netsh advfirewall firewall delete rule name=all dir=in program="C:\Program Files\LumaViewPro\lumaviewpro.exe"
  netsh advfirewall firewall add rule name="LumaViewPro REST" dir=in action=allow program="C:\Program Files\LumaViewPro\lumaviewpro.exe" enable=yes
  ```

  `netsh advfirewall firewall delete rule name="LumaViewPro REST"` removes
  it. A site whose policy ignores local rules needs its IT to add the rule.
  A copy run from source serves through its `python.exe`, and Windows asks
  for that program instead. On Linux, a firewall that is on (firewalld on Fedora and RHEL, ufw when enabled
  on Ubuntu) drops the connection with no prompt and no message: open the port,
  for example `sudo firewall-cmd --permanent --add-port=8000/tcp && sudo
  firewall-cmd --reload`, or `sudo ufw allow from 192.168.10.3 to any port 8000
  proto tcp`.

`GET /api` answers the versions served; the OpenAPI 3.1 description of every
route is at `/api/v1/openapi.json`, with an interactive reference at `/docs`.
Each route's description lists the refusals and faults its member declares,
under the status each answers.

## Routes

The session is the root: its member `<member>` is `/api/v1/<member>`, and each sub-object is
a path segment, so `session.scope.motion.move_xy` is
`POST /api/v1/scope/motion/move_xy`.

- **A read is `GET`**: a property, a published field, or a method that changes
  nothing. A read method's arguments are its query string
  (`GET /api/v1/scope/motion/get_axis_limits?axis=Z`); a list is its key
  repeated, and a record one key per field, `<parameter>.<field>`.
- **A setting is `PUT` on its read's URL**, with the setter's arguments as a
  JSON object; the answer is the read after the write, the value the scope took
  (`PUT /api/v1/scope/camera/binning` with `{"size": 2}` answers `2`). A
  setting kept per channel takes the channel in the body and the query
  (`PUT /api/v1/channel_acquire` with `{"channel": "BF", "mode": "image"}`,
  then `GET /api/v1/channel_acquire?channel=BF`). The setter itself has no
  route.
- **Every other call is `POST`**, its arguments a JSON object by parameter
  name; `null` is None, and a parameter left out takes its default. A member
  that hands out a running call is a `POST` even when it changes nothing
  (`last_run`).
- **The answer** to a finished call is `200` with its value as the body, `null`
  for None.
- **Arguments are checked before the member runs**: a key it does not take, a
  value of the wrong type, a missing one, or a number that is not finite is
  refused `422` (`'invalid_request'`), its `errors` naming each argument. A
  number where a channel is named is refused here too, as `'invalid_request'`.
- **Paths are names in the live folder**, `/`-separated
  (`"ProtocolData/run1"`). A name that leaves it is refused
  (`'outside_live_folder'`); `"."` is the live folder itself. An answered path is
  `{"name": ..., "host_path": ...}`.
- **Units are the API's**: an axis position crosses as
  `{"position_um": ..., "state": ...}`, and a duration as a number of seconds.

In-process members -- those taking a callback, returning a live object, or
waiting on the caller's thread -- have no route; each has a wire form beside
it (a listener is an event, a waiting call is a job, `acting_in_control` is
the `Control-Session` header).

## Handles

A live object -- a protocol, a protocol runner -- crosses as a handle,
`{"handle": "1", "type": "Protocol"}`. Its members are at
`/api/v1/handles/<type>/<id>/<member>` (`GET /api/v1/handles/Protocol/1/period_s`),
and a parameter that takes it is sent its id (`{"protocol": "1"}`).
`GET /api/v1/handles` lists every handle held, oldest first.
`DELETE /api/v1/handles/<type>/<id>` forgets the id, never the object; the
session's protocol runner is every client's and is kept (`422`,
`'handle_shared'`). At most 1000 handles are held.

## Jobs

Every call runs on its own thread, and its route waits for it as long as the
client says: `Prefer: wait=<seconds>`, 2 s when it says nothing, at most 60 s.
`Prefer: respond-async` asks for no wait at all. A call still running then
answers `202 Accepted` with `Location: /api/v1/jobs/<id>` and the job:

```json
{"id": "3", "member": "handles/ProtocolRunner/2/start_single_scan", "status": "running",
 "requested": "2026-10-29T11:47:56.043-07:00", "ended": null, "progress": null}
```

A member that starts an operation -- a run, a move, a home, a still -- is
answered `202` with the operation's job as soon as it has started, whatever
the wait. `GET /api/v1/jobs/<id>`, itself waiting by `Prefer: wait`, reads a
job, `200` whatever its status; with `Prefer: respond-async` it answers at
once, a poll of the job's `progress`. Once it has ended it carries `result` or `error` (the problem). A job's
`status` says whether the call answered a problem, not how the operation went:
a stopped run's job is `completed` with its outcome `aborted`, a stopped move's
`failed` with its stop notice. A run's job also carries `files`, the job of
the run's images: the run's job ends as the run lets go of the scope, and its
`files` job once the images are on disk, its `result` the run's `RunFiles`.
Wait on it before reading the run's folder or starting the next run, which is
refused `'files_writing'` until then. `DELETE /api/v1/jobs/<id>` stops a running
operation (`202`) or forgets a finished job (`204`). Finished jobs are kept up
to 1000 and 24 hours. Take a job's id from the answer that made it, never count
it: every call held as a job takes the next id.

A `POST` or `PUT` that may be sent twice carries `Idempotency-Key`: sent again
with the same key, it runs nothing and is answered what the first was, with
`Idempotent-Replayed: true`. A key is held for 24 hours.

## The control session

While a control session is held, every write carries its id in the
`Control-Session` header; any other write is refused `409`
(`'control_session_held'`), and a request naming an ended session `422`
(`'control_session_ended'`). `POST /api/v1/take_control` with `{"name": ...}`
answers the id; `POST /api/v1/end_control` with `{"session_id": ...}` ends it.
Every request carrying the header renews the session, a read or a job poll
included; one silent for 180 s is ended by the next client it would refuse.

## Problems

Every answer that is not a result is an RFC 9457 problem,
`application/problem+json`:

```json
{"type": "urn:lumascope:problem:not_offered", "title": "Not One of the Values Offered",
 "detail": "axis must be one of X, Y, Z; 'W' is not one.", "status": 422,
 "instance": "urn:uuid:3faa1296-...", "kind": "refusal", "reason": "not_offered",
 "remedy": null, "argument": "axis", "offered": ["X", "Y", "Z"]}
```

`title` and `detail` are the words a person is shown; `reason` is the code to
branch on; `kind` is the outcome's; `remedy` is sent back as
`POST /api/v1/apply_remedy` with `{"remedy": ...}`. A refusal carries the
fields its class publishes -- `argument`, `offered`, `limits`, `holder` -- so a
client corrects its request without reading the words.

| Status | Means |
|---|---|
| `409` | A refusal the scope's state gave: the same request may succeed later. Also a move or home a stop ended (`kind` `notice`). |
| `422` | A refusal of the request as sent: change it. Also `'invalid_request'` and `'handle_shared'`. |
| `500` | A fault: the call failed. |
| `404` | No such route, handle, job or file. |
| `405` / `415` | The wrong method; a body that is not JSON. |
| `503` | With `Retry-After`, try again later (`'overloaded'`, `'no_frame_yet'`). Without it, nothing succeeds until the application restarts (`'not_connected'`, `'scope_disconnected'`, `'session_closing'`, `'server_closing'`). |

## The event stream

`GET /api/v1/events` is a server-sent event stream. Its first event is
`status`, the scope as `GET /api/v1/status` reads it; after that every change
is an event, named as in the [events table](../api-reference.md#events), its data
the event's record. A run's events carry `run`, `{"job": "<id>"}`, the job of
the run that sent them, and every job's end is a `job_ended` event, its data
the job: a run's `files` job ends with its own, so match a `job_ended` by its
`id`. An outcome nobody asked for is an `outcome` event.

A client that reconnects with `Last-Event-ID` is sent what it missed; when the
server no longer holds it (more than 1024 events ago, or a restarted server),
it is sent `reset` and then `status`, where it starts again. A client that
cannot hold a stream open polls `GET /api/v1/events/held?after=<id>`. The last
event is `closing`.

## The live view and files

- `GET /api/v1/live` -- the camera's frames as MJPEG, for a browser's `<img>`.
- `GET /api/v1/live.jpg` -- one frame.
- `GET /api/v1/files/<name>` -- a file in the live folder, as an attachment; a
  `Range` request resumes a large download.
- `PUT /api/v1/files/<name>` -- a new file in the live folder; a file already
  at the name is refused (`'file_exists'`), so an upload never replaces a
  capture.
- `GET /api/v1/previews/<name>` -- a saved image downscaled to a JPEG for
  display (`max_width`).
- `POST /api/v1/still` -- a still with the camera as set, at its own depth,
  saved nowhere (`ScopeSession.capture_still`): the body is the still's record
  as JSON, `X-Record-Length` bytes, then its pixels, row by row, little-endian,
  shaped by `X-Image-Shape`, `X-Image-Dtype` and `X-Significant-Bits`. Its
  JSON body, `{"exposed_after_call": true}`, keeps only a frame exposed after
  the call, for a light or shutter LVP did not switch. It answers when the
  still is taken, never as a job.

The live view and previews are display: 8-bit grey, lossy, with no metadata.
The data is the file. Every answer carries `X-Request-ID`, which a problem
names as its `instance`, and each request is one line of
`lumaviewpro_rest_api.log`.

## A walkthrough

A session with `curl` against a server started with `--simulate`, reading the
answers with `jq`. Each line's comment is the status it answers; a handle's or
a job's id is read from the answer that made it. Every line here is run, in
this order, in LumaViewPro's test suite.

```bash
API=http://127.0.0.1:8000/api/v1

curl -s $API/status                                         # 200: the scope's work, axes, parts and camera
curl -s "$API/scope/motion/get_axis_limits?axis=W"          # 422 not_offered: offered X, Y, Z
curl -s -H 'Prefer: wait=60' --json '{"z_um": 4000.0}' $API/scope/motion/move_z    # 200
curl -s --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0
curl -s --json '{}' $API/scope/illumination/leds_off        # 200
```

Build a one-step protocol and run it. A run is answered `202` at once with its
job; poll the job at once with `Prefer: respond-async`, or read it with
`Prefer: wait` until it has ended, then its `files` job until the run's images
are on disk.

```bash
curl -s -X PUT --json '{"channel": "BF", "mode": "image"}' $API/channel_acquire   # 200: "image"
PROTOCOL=$(curl -s --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200
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
curl -s --json '{"z_um": 3000.0}' $API/scope/motion/move_z                          # 409 control_session_held
curl -s -H "Control-Session: $CONTROL" -H 'Prefer: wait=60' --json '{"z_um": 3000.0}' $API/scope/motion/move_z   # 200
curl -s --json "{\"session_id\": \"$CONTROL\"}" $API/end_control                    # 200
curl -s $API/events/held                                    # 200: where a poll of the events begins
```

A client that holds a connection open reads the events as they happen with
`curl -N $API/events`.
