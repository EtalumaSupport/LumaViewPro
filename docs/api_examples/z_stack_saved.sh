#!/usr/bin/env bash
# A saved z-stack, over REST: z_stack_saved.py's stack,
# on a simulated LumaViewPro serving the API -- the installed app or
# `python -m lumascope.rest --simulate`; how to start one and run this: README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

API=http://127.0.0.1:8000/api/v1

# A new install has not been told which objective is mounted, and a saved
# still and a run are refused until it has. Confirm the objective the question
# proposes for the slot in the light path (null without a turret); once the
# scope has been told, it proposes what it was told, and this changes nothing.
OBJECTIVE=$(curl -s $API/current_slot_objective_question | jq -r .proposed)   # 200
SLOT=$(curl -s $API/current_slot_objective_question | jq -r .turret_slot)    # 200: null without a turret
curl -s -w '\n' --json "{\"objective_id\": \"$OBJECTIVE\", \"slot\": $SLOT}" $API/confirm_objective   # 200

curl -s -w '\n' -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200

# The stack's extent and step, and where the stage sits in it.
curl -s -w '\n' --json '{"path": "zstack.range", "value": 20.0}' $API/update_settings        # 200
curl -s -w '\n' --json '{"path": "zstack.step_size", "value": 5.0}' $API/update_settings     # 200
curl -s -w '\n' --json '{"path": "zstack.position", "value": "center"}' $API/update_settings # 200
curl -s -w '\n' --json '{"path": "BF.illumination_ma", "value": 5}' $API/update_settings     # 200

# A z-stack is a run: its job, then its files' job once the slices are on disk.
RUNNER=$(curl -s --json '{}' $API/protocol_runner | jq -r .handle)         # 200
JOB=$(curl -s --json '{"channel": "BF", "sequence_name": "example_stack"}' $API/handles/ProtocolRunner/$RUNNER/start_zstack | jq -r .id)   # 202
FILES=$(curl -s -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r .files.id)   # 200: "result" the RunOutcome
curl -s -w '\n' -H 'Prefer: wait=60' $API/jobs/$FILES               # 200: "result" the RunFiles, once the slices are on disk

RUN_DIR=$(curl -s $API/jobs/$JOB | jq -r .progress.run_dir.name)   # 200
curl -s -w '\n' "$API/live_folder_listing?name=$RUN_DIR"            # 200: a file per slice
