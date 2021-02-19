#!/bin/bash -x
# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

JOB_NAME="kubric_`date +"%b%d_%H%M%S"`"
PROJECT_ID="kubric-xgcp"
REGION="us-central1"  #< WARNING: match region of bucket!

run_mode=${1}
shift # shifts all arguments to the left thus removing ${1}
worker_file=${1}
shift # shifts all arguments to the left thus removing ${1}


if [[ "${run_mode}" == "dev" ]]
then
  SOURCE_TAG=gcr.io/kubric-xgcp/kubruntudev:latest
else
  SOURCE_TAG=gcr.io/kubric-xgcp/kubruntu:latest
fi


# --- The container configuration
cat > /tmp/Dockerfile <<EOF
FROM ${SOURCE_TAG}

COPY ${worker_file} /worker/worker.py
WORKDIR /kubric
ENTRYPOINT ["python3", "/worker/worker.py"]
EOF



if [[ "${run_mode}" == "local" ]] || [[ "${run_mode}" == "dev" ]]
then
  # --- Launches the job locally
  TAG="local"
  docker build -f /tmp/Dockerfile -t $TAG $PWD
  docker run -v "`pwd`:/kubric" -v "`pwd`/../assets:/assets" $TAG  "$@"
else
  # --- Parameters for the launch
  TAG="gcr.io/$PROJECT_ID/worker"
  docker build -f /tmp/Dockerfile -t $TAG $PWD
  docker push $TAG
  # --- Launches the job on aiplatform
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --scale-tier basic \
    --master-image-uri $TAG \
    --job-dir "gs://research-brain-kubric-xgcp/jobs/klevr/$JOB_NAME" \
    -- "$@"

  # --- Streams the job logs to local terminal
  gcloud ai-platform jobs stream-logs $JOB_NAME
fi


