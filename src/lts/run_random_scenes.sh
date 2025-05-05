#!/bin/bash
SCENARIOS_DIR="/home/jlidard/kubric/examples/lts/scenarios"
DEBUG=true

for SCENE_FILE in $(ls "$SCENARIOS_DIR" | sort -f)
do
    echo "Processing scene file: $SCENE_FILE"
    SCENE_NAME=$(basename "$SCENE_FILE" .txt)
    JOB_DIR="./examples/lts/scenarios/$SCENE_NAME"
    mkdir -p "$JOB_DIR"

    
    SCENE_FILE_PATH="/kubric/examples/lts/$(basename "$SCENARIOS_DIR")/$(basename "$SCENE_FILE")"
    SCENE_FILE_PATH="${SCENE_FILE_PATH%.txt}.txt"

    echo "Job directory: $JOB_DIR"
    echo "Scene filename: $SCENE_FILE_PATH"
    sudo docker run --rm --interactive \
        --user 0 \
        --volume "/home/jlidard/kubric:/kubric" \
        kubricdockerhub/kubruntu \
        /bin/bash -c "python3 -m pip install pyyaml && python3 /kubric/examples/lts/generate_scene.py --job-dir=$JOB_DIR --scene-filename=$SCENE_FILE_PATH"

    if [ "$DEBUG" = true ]; then
        echo "Debug mode enabled. Terminating after first docker run."
        break
    fi

    trap "exit" INT
done