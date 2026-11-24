#!/usr/bin/env bash
# A scan of two wells whose images are kept, over REST: scan.py's scan,
# on a simulated server running on a copy of LumaViewPro's data/; how to
# start one and run this: README.md here.
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

curl --fail-with-body -sS -w '\n' -X PUT --json '{"channel": "BF", "acquire": "image", "illumination_ma": 5}' $API/channel_settings   # 200: BF's settings
PROTOCOL=$(curl --fail-with-body -sS --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
MOVE=$(curl --fail-with-body -sS --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
MOVE=$(curl --fail-with-body -sS --json '{"well": "B3"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$MOVE | jq -c "$COMPLETED"   # 200
curl --fail-with-body -sS -w '\n' --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200

# A run is answered 202 with its job, which ends as the run lets go of the
# scope; the job's files job ends once the run's images are on disk.
RUNNER=$(curl --fail-with-body -sS --json '{}' $API/protocol_runner | jq -r .handle)         # 200
JOB=$(curl --fail-with-body -sS --json "{\"protocol\": \"$PROTOCOL\", \"sequence_name\": \"example_scan\"}" $API/handles/ProtocolRunner/$RUNNER/start_single_scan | jq -r .id)   # 202
FILES=$(curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r "$COMPLETED | .files.id")   # 200: "result" the RunOutcome
curl --fail-with-body -sS -H 'Prefer: wait=60' $API/jobs/$FILES | jq -c "$COMPLETED"   # 200: "result" the RunFiles, once the images are on disk

# The run's folder, by its name under the live folder, and its first image.
RUN_DIR=$(curl --fail-with-body -sS $API/jobs/$JOB | jq -r .progress.run_dir.name)   # 200
curl --fail-with-body -sS -w '\n' "$API/live_folder_listing?name=$RUN_DIR"            # 200: the images, the protocol, the record
curl --fail-with-body -sS -o first.tiff "$API/files/$RUN_DIR/custom0000_BF_0000.tiff"   # 200
