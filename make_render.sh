#!/bin/bash -x
# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

JOB_NAME="kubric_`date +"%b%d_%H%M%S"`"
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

# --- Run the container
if [ "${1}" == "local" ]
then
  docker run $TAG
else
  # --- Launches the job on aiplatform
  # see: https://cloud.google.com/ai-platform/training/docs/using-gpus
  docker push $TAG
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --master-image-uri $TAG \
    --scale-tier "basic"
    # -- \
    # --job_name $JOB_NAME

  # --- Streams the job logs to local terminal
  # gcloud ai-platform jobs stream-logs $JOB_NAME
fi