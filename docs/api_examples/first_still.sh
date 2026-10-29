#!/usr/bin/env bash
# A first saved still, over REST: first_still.py's still,
# on a simulated LumaViewPro serving the API -- the installed app or
# `python -m rest --simulate`; how to start one and run this: README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

API=http://127.0.0.1:8000/api/v1

curl -s -w '\n' -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200
curl -s -w '\n' --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0

# A still is answered 202 with its job; the job's result is the saved files.
JOB=$(curl -s --json '{"false_color_on": false}' $API/manual_capture/capture | jq -r .id)   # 202
FILE=$(curl -s -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r '.result[0].file.name')   # 200
curl -s -o still.tiff "$API/files/$FILE"                    # 200: the TIFF, its record inside
curl -s -w '\n' --json '{}' $API/scope/illumination/leds_off        # 200
