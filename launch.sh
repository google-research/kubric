#!/bin/bash -x
# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

#JOB_NAME="kubric_$(date +"%b%d_%H%M%S" | tr A-Z a-z)"
PROJECT_ID="kubric-xgcp"
REGION="us-central1"  #< WARNING: match region of bucket!

run_mode=${1}
shift # shifts all arguments to the left thus removing ${1}
worker_file=${1}
shift # shifts all arguments to the left thus removing ${1}
JOB_NAME=${1}
shift
NR_VIDEOS=${1}
shift
NR_WORKERS=${1}
shift


# --- Specify the hypertune configuration
cat > /tmp/hypertune.yml << EOF
  trainingInput:
    hyperparameters:
      goal: MAXIMIZE
      hyperparameterMetricTag: "answer"
      maxTrials: $NR_VIDEOS
      maxParallelTrials: $NR_WORKERS
      maxFailedTrials: 100
      enableTrialEarlyStopping: False

      # --- each of these become an argparse argument
      params:
      - parameterName: seed
        type: INTEGER
        minValue: 0
        maxValue: 52423
EOF


# --- The container configuration
cat > Dockerfile <<EOF
FROM kubricdockerhub/kubruntudev:latest
COPY . /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "/workspace/${worker_file}"]
EOF


if [[ "${run_mode}" == "local" ]] || [[ "${run_mode}" == "dev" ]]
then
  # --- Launches the job locally
  TAG="local"
  docker build -f Dockerfile -t $TAG $PWD
  docker run $TAG  "$@"
elif [[ "${run_mode}" == "remote" ]]
then
  # --- Parameters for the launch
  TAG="gcr.io/$PROJECT_ID/worker"
  docker build -f Dockerfile -t $TAG $PWD
  docker push $TAG
  # --- Launches the job on aiplatform
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --scale-tier basic \
    --master-image-uri $TAG \
    --job-dir "gs://research-brain-kubric-xgcp/jobs/$JOB_NAME" \
    -- "$@"

  # --- Streams the job logs to local terminal
  gcloud ai-platform jobs stream-logs $JOB_NAME
else   # hyper
  # --- Parameters for the launch
  TAG="gcr.io/$PROJECT_ID/$JOB_NAME"
  docker build -f Dockerfile -t $TAG $PWD
  docker push $TAG
  # --- Launches the job on aiplatform
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --scale-tier basic \
    --master-image-uri $TAG \
    --config /tmp/hypertune.yml \
    --job-dir "gs://research-brain-kubric-xgcp/jobs/$JOB_NAME" \
    -- "$@"
  gcloud ai-platform jobs describe $JOB_NAME
fi


