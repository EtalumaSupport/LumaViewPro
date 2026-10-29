#!/usr/bin/env bash
# A scan of two wells whose images are kept, over REST: scan.py's scan,
# on a simulated LumaViewPro serving the API -- the installed app or
# `python -m rest --simulate`; how to start one and run this: README.md here.
# The scope has already run the startup motion.
# Each line's comment is the status it answers.

API=http://127.0.0.1:8000/api/v1

curl -s -X PUT --json '{"channel": "BF", "mode": "image"}' $API/channel_acquire   # 200: "image"
curl -s --json '{"path": "BF.illumination_ma", "value": 5}' $API/update_settings  # 200
PROTOCOL=$(curl -s --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B3"}' $API/scope/motion/move_to_well   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200

# A run is answered 202 with its job, which ends as the run lets go of the
# scope; the job's files job ends once the run's images are on disk.
RUNNER=$(curl -s --json '{}' $API/create_protocol_runner | jq -r .handle)         # 200
JOB=$(curl -s --json "{\"protocol\": \"$PROTOCOL\", \"sequence_name\": \"example_scan\"}" $API/handles/ProtocolRunner/$RUNNER/start_single_scan | jq -r .id)   # 202
FILES=$(curl -s -H 'Prefer: wait=60' $API/jobs/$JOB | jq -r .files.id)   # 200: "result" the RunOutcome
curl -s -H 'Prefer: wait=60' $API/jobs/$FILES               # 200: "result" the RunFiles, once the images are on disk

# The run's folder, by its name under the live folder, and its first image.
RUN_DIR=$(curl -s $API/jobs/$JOB | jq -r .progress.run_dir.name)   # 200
curl -s "$API/live_folder_listing?name=$RUN_DIR"            # 200: the images, the protocol, the record
curl -s -o first.tiff "$API/files/$RUN_DIR/custom0000_BF_0000.tiff"   # 200
