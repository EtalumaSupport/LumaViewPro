#!/usr/bin/env bash
# Stopping a run, over REST: stop_a_run.py's run,
# on a simulated LumaViewPro serving the API -- the installed app or
# `python -m rest --simulate`; how to start one and run this: README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

API=http://127.0.0.1:8000/api/v1

# A brightfield step at each of three wells, a scan every 60 s for an hour.
curl -s -X PUT --json '{"channel": "BF", "mode": "image"}' $API/channel_acquire   # 200: "image"
curl -s --json '{"path": "BF.illumination_ma", "value": 5}' $API/update_settings  # 200
PROTOCOL=$(curl -s --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B3"}' $API/scope/motion/move_to_well   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B4"}' $API/scope/motion/move_to_well   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"period_s\": 60.0, \"duration_s\": 3600.0}" $API/set_protocol_schedule   # 200

RUNNER=$(curl -s --json '{}' $API/create_protocol_runner | jq -r .handle)         # 200
JOB=$(curl -s --json "{\"protocol\": \"$PROTOCOL\", \"sequence_name\": \"stopped\"}" $API/handles/ProtocolRunner/$RUNNER/start_protocol | jq -r .id)   # 202
curl -s $API/jobs/$JOB                                      # 200: running, its progress
# DELETE on a running job is its stop: it only asks, and the run ends on its own.
curl -s -X DELETE $API/jobs/$JOB                            # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$JOB                 # 200: the job completed, its "result" the run aborted, "reason" stopped
