#!/usr/bin/env bash
# A first saved still, over REST: first_still.py's still,
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

MOVE=$(curl -s --json '{"well": "B2"}' $API/scope/motion/start_move_to_well | jq -r .id)   # 202
curl -s -w '\n' -H 'Prefer: wait=60' $API/jobs/$MOVE               # 200: "status": "completed"
curl -s -w '\n' --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0

# A still is answered 202 with its job; the job's result is the saved files.
JOB=$(curl -s --json '{}' $API/manual_capture/capture | jq -r .id)   # 202
FILE=$(curl -s -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r '.result[0].file.name')   # 200
curl -s -o still.tiff "$API/files/$FILE"                    # 200: the TIFF, its record inside
curl -s -w '\n' --json '{}' $API/scope/illumination/leds_off        # 200
