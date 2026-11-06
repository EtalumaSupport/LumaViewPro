# A REST walkthrough

A session over REST with `curl`: the status, a move and a light, a one-step run and its files, and the control session.

Against a server [started](rest-serving.md#serving-it) with `--simulate`,
reading the answers with `jq`, as [REST](rest.md) describes them. Each line's
comment is the status it answers; a handle's or a job's id is read from the
answer that made it. Every line here is run, in this order, in LumaViewPro's
test suite.

```bash
API=http://127.0.0.1:8000/api/v1

curl -s $API/status                                         # 200: the scope's work, axes, parts and camera
curl -s "$API/scope/motion/get_axis_limits?axis=W"          # 422 not_offered: offered X, Y, Z, T
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
