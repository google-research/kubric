#!/bin/bash -x
# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

JOB_NAME="`date +"%b%d_%H%M%S"`"
PROJECT_ID=${1:-`gcloud config get-value project`}
TAG="gcr.io/$PROJECT_ID/kubric"
REGION="us-central1"

# --- The container configuration
cat > /tmp/Dockerfile <<EOF
  FROM kubruntu:latest
  COPY . /
  WORKDIR /
  ENTRYPOINT ["blender", "-noaudio", "--background", "--python", "viewer/helloworld.py"]
EOF

# --- Build the container
docker build -f /tmp/Dockerfile -t $TAG $PWD

# --- Specify the hypertune configuration
cat > /tmp/config.yml << EOF
  trainingInput:
    hyperparameters:
      goal: MAXIMIZE
      hyperparameterMetricTag: "generated_images"
      maxTrials: 4
      maxParallelTrials: 4
      enableTrialEarlyStopping: False

      # --- each of these become an argparse argument
      params:
      - parameterName: parameter
        type: DISCRETE
        discreteValues: [1,2,3,4]
EOF

# --- Run the container
if [ "${1}" == "local" ]; then
  docker run $TAG -- --output "$JOB_NAME/#####/frame_" --parameter 5

else
  # --- Launches the job on aiplatform
  # see: https://cloud.google.com/ai-platform/training/docs/using-gpus
  docker push $TAG
  # the first "--" separates glcoud parameters from blender parameters
  # the second "--" separates blender parameters from script parameters
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $TAG \
    --scale-tier "basic" \
    --config /tmp/config.yml \
    -- \
    -- \
    --output "$JOB_NAME/#####/frame_"

  # --- Streams the job logs to local terminal
  # https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/stream-logs
  # gcloud ai-platform jobs stream-logs $JOB_NAME
fi