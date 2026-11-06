# REST

The API over HTTP: routes, handles, jobs, the control session, problems, events, and the live view and files. [A walkthrough](rest-walkthrough.md) drives a session with `curl`.

A server is LumaViewPro's, beside its window, or `python -m lumascope.rest`
from its source; [Serving REST](rest-serving.md) says how each starts, its
settings, and how a client on another computer reaches it. `GET /api` answers
the versions served; the OpenAPI 3.1 description of every route is at
`/api/v1/openapi.json`, with an interactive reference at `/docs`. Each route's
description lists the refusals and faults its member declares, under the
status each answers.

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
  `{"name": ..., "host_path": ...}`, `name` null for a path outside it.
- **Units are the API's**: an axis position crosses as
  `{"position_um": ..., "state": ...}`, a duration as a number of seconds, and
  a time as ISO 8601 with its offset.

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
`DELETE /api/v1/handles/<type>/<id>` forgets the id, never the object (`204`);
the session's protocol runner is every client's and is kept (`422`,
`'handle_shared'`). A forget is a write: under another's control session it is
refused. At most 1000 handles are held; past that, a call whose answer could
carry a new one is refused `503` (`'overloaded'`, `Retry-After: 5`) before it
runs, so forget the handles a client is done with.

## Jobs

Every call runs on its own thread, and its route waits for it as long as the
client says: `Prefer: wait=<seconds>`, 2 s when it says nothing, a longer wait
cut to 60 s. `Prefer: respond-async` asks for no wait at all. A preference
taken is echoed in `Preference-Applied`. A call still running then answers
`202 Accepted` with `Location: /api/v1/jobs/<id>`, `Retry-After: 1` and the
job:

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
refused `'files_writing'` until then. `GET /api/v1/jobs` lists every job held,
newest first.

`DELETE /api/v1/jobs/<id>` stops a running operation (`202`, the job) or
forgets a finished job (`204`). It is a stop, so no control session refuses it.
A still has no stop, and its job ends as it would have; a call held as a job
only because it outlived its wait holds no operation, and is refused `409`
(`'not_cancellable'`). Finished jobs are kept up to 1000 and 24 hours; at most
256 calls run at once, past which a call is refused `503` (`'overloaded'`).
Take a job's id from the answer that made it, never count it: every call held
as a job takes the next id.

A `POST` or `PUT` that may be sent twice, an upload among them, carries
`Idempotency-Key`: sent again with the same key, it runs nothing and is
answered what the first was, with `Idempotent-Replayed: true`. A key sent with
another request is refused `422` (`'idempotency_key_reused'`). A key is held
for 24 hours, and at most 10,000, the oldest let go first.

## The control session

The wire form of [the control session](concepts.md#who-may-act).
`POST /api/v1/take_control` with `{"name": ...}` answers the id, to this
client alone; `GET /api/v1/control` reads who holds it, and since when. While
it is held, every write carries its id in the `Control-Session` header; any
other write is refused `409` (`'control_session_held'`), and a request naming
an ended session `422` (`'control_session_ended'`). Reads, stops and the
display preferences pass without it. `POST /api/v1/end_control` with
`{"session_id": ...}` ends it, stopping a run or recording taken in it.
Every request carrying the header renews the session, a read or a job poll
included; one silent for 180 s is ended by the next client it would refuse.
Taking back control without the id is the window's alone: it has no route.

## Problems

Every answer that is not a result is an RFC 9457 problem,
`application/problem+json`:

```json
{"type": "urn:lumascope:problem:not_offered", "title": "Not One of the Values Offered",
 "detail": "axis must be one of X, Y, Z, T; 'W' is not one.", "status": 422,
 "instance": "urn:uuid:3faa1296-...", "kind": "refusal", "reason": "not_offered",
 "remedy": null, "argument": "axis", "offered": ["X", "Y", "Z", "T"],
 "board": null, "limits": null, "member": null, "plate": null, "takes": null}
```

`title` and `detail` are the words a person is shown; `reason` is the code to
branch on; `kind` is the outcome's; `remedy` is sent back as
`POST /api/v1/apply_remedy` with `{"remedy": ...}`. A refusal carries every
field its class publishes -- `argument`, `offered`, `limits`, `holder` --
null where unused, so a client corrects its request without reading the
words. An outcome that declares no reason is `type` `about:blank`.

The status is the outcome's [cause](concepts.md#outcomes), or the server's own
refusal's; every reason and its status is in the reference's
[table](../api-reference.md#ServerRefusedError) and
[Outcomes](../api-reference.md#outcomes).

| Status | Means |
|---|---|
| `409` | A refusal the scope's state gave: the same request may succeed later. Also a move or home a stop ended (`kind` `notice`), and `'not_cancellable'`. |
| `422` | A refusal of the request as sent: change it. Also `'invalid_request'`, `'handle_shared'` and `'idempotency_key_reused'`. |
| `403` | `'client_not_allowed'`: a client [`allowed_clients`](rest-serving.md#serving-another-computer) does not list. |
| `500` | A fault: the call failed. |
| `404` | No such route, handle, job or file. |
| `405` / `415` | The wrong method; a body that is not JSON. |
| `503` | With `Retry-After`, try again later (`'overloaded'`, `'no_frame_yet'`). Without it, nothing succeeds until the application restarts (`'not_connected'`, `'session_closing'`, `'server_closing'`, and every other restart cause). |

## The event stream

`GET /api/v1/events` is a server-sent event stream. Its first event is
`status`, the scope as `GET /api/v1/status` reads it; after that every change
is an event, named as in the [events table](../api-reference.md#events), its data
the event's record -- but the frame events, which are the live view's. A
quiet stream sends a keep-alive comment every 15 s and asks a client to
reconnect after 1 s (`retry: 1000`). A run's events carry `run`, `{"job": "<id>"}`, the job of
the run that sent them, and every job's end is a `job_ended` event, its data
the job: a run's `files` job ends with its own, so match a `job_ended` by its
`id`. An outcome nobody asked for is an `outcome` event.

A client that reconnects with `Last-Event-ID` is sent what it missed; when the
server no longer holds it (more than 1024 events ago, or a restarted server),
it is sent `reset` and then `status`, where it starts again. A client that
cannot hold a stream open polls `GET /api/v1/events/held?after=<id>`. The last
event is `closing`.

## The live view and files

- `GET /api/v1/live` -- the camera's frames as MJPEG, for a browser's `<img>`,
  each part carrying `X-Frame-Ordinal` and `X-Frame-Timestamp`; `max_width`
  bounds a frame's width, half the camera's frame when not given.
- `GET /api/v1/live.jpg` -- one frame, `max_width` as for `/live`; `503`
  (`'no_frame_yet'`, with `Retry-After`) when the camera sends none in time.
- `GET /api/v1/files/<name>` -- a file in the live folder, as an attachment; a
  `Range` request resumes a large download.
- `PUT /api/v1/files/<name>` -- a new file in the live folder, answered `201`
  with its path; a file already at the name is refused (`'file_exists'`), so
  an upload never replaces a capture. It is a write, carrying
  `Control-Session` under a held session.
- `GET /api/v1/previews/<name>` -- a saved image downscaled to a JPEG for
  display (`max_width` as for `/live`).
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
