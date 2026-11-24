#!/usr/bin/env bash
# Stopping a run, over REST: stop_a_run.py's run, on a simulated server
# running on a copy of LumaViewPro's data/; how to start one and run this:
# README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

set -euo pipefail

# The server: this computer's, on the port LVP_PORT names (8000 as shipped).
API="http://127.0.0.1:${LVP_PORT:-8000}/api/v1"
# A job's read answers 200 however the job ended, so its end is checked:
# this passes the job on, or fails naming how it ended.
COMPLETED='if .status == "completed" then . else error("the job ended \(.status): \(.error)") end'

# A new install has not been told which objective is mounted, and a saved
# still and a run are refused until it has. Confirm the objective the question
# proposes for the slot in the light path (null without a turret); once the
# scope has been told, it proposes what it was told, and this changes nothing.
OBJECTIVE=$(curl --fail-with-body -sS $API/current_slot_objective_question | jq -r .proposed)   # 200
SLOT=$(curl --fail-with-body -sS $API/current_slot_objective_question | jq -r .turret_slot)    # 200: null without a turret
curl --fail-with-body -sS -w '\n' --json "{\"objective_id\": \"$OBJECTIVE\", \"slot\": $SLOT}" $API/confirm_objective   # 200

# A brightfield step at each of three wells, a scan every 60 s for an hour.
curl --fail-with-body -sS -w '\n' -X PUT --json '{"channel": "BF", "acquire": "image", "illumination_ma": 5}' $API/channel_settings   # 200: BF's settings
PROTOCOL=$(curl --fail-with-body -sS --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
MOVE=$(curl --fail-with-body -sS --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
MOVE=$(curl --fail-with-body -sS --json '{"well": "B3"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
MOVE=$(curl --fail-with-body -sS --json '{"well": "B4"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
curl --fail-with-body -sS -w '\n' --json "{\"protocol\": \"$PROTOCOL\", \"period_s\": 60.0, \"duration_s\": 3600.0}" $API/set_protocol_schedule   # 200

# Where a poll of the events begins, read before the run starts so none of its events is missed.
AFTER=$(curl --fail-with-body -sS $API/events/held | jq -r .next)   # 200
RUNNER=$(curl --fail-with-body -sS --json '{}' $API/protocol_runner | jq -r .handle)         # 200
JOB=$(curl --fail-with-body -sS --json "{\"protocol\": \"$PROTOCOL\", \"sequence_name\": \"stopped\"}" $API/handles/ProtocolRunner/$RUNNER/start_protocol | jq -r .id)   # 202

# Poll the events, each poll after the last, until this run's first scan_ended:
# for a minute at most, then the run is stopped all the same.
for _ in {1..60}; do
    HELD=$(curl --fail-with-body -sS "$API/events/held?after=$AFTER")   # 200
    jq -e --arg job "$JOB" 'any(.events[]; .event == "scan_ended" and .data.run.job == $job)' <<< "$HELD" > /dev/null && break
    AFTER=$(jq -r .next <<< "$HELD")
    sleep 1
done

# DELETE on a running job is its stop: it only asks, and the run ends on its own.
curl --fail-with-body -sS -w '\n' -X DELETE $API/jobs/$JOB                            # 202
# The run's images keep writing after it ends: wait on its files job too.
FILES=$(curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r "$COMPLETED | .files.id")   # 200: the job completed, its "result" the run aborted, "reason" stopped
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$FILES | jq -c "$COMPLETED"   # 200: "result" the RunFiles, once the images are on disk
