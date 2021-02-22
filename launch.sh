#!/bin/bash -x
# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

#JOB_NAME="kubric_`date +"%b%d_%H%M%S"`"
JOB_NAME="klevr_v0"
PROJECT_ID="kubric-xgcp"
REGION="us-central1-a"  #< WARNING: match region of bucket!

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

# --- Specify the hypertune configuration
cat > /tmp/hypertune.yml << EOF
  trainingInput:
    hyperparameters:
      goal: MAXIMIZE
      hyperparameterMetricTag: "generated_images"
      maxTrials: 1000
      maxParallelTrials: 100
      enableTrialEarlyStopping: False

      # --- each of these become an argparse argument
      params:
      - parameterName: seed
        type: INTEGER
        minValue: 1
        maxValue: 1000000
EOF


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
elif [[ "${run_mode}" == "remote" ]]
then
  # --- Parameters for the launch
  TAG="gcr.io/$PROJECT_ID/worker"
  docker build -f /tmp/Dockerfile -t $TAG $PWD
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
  docker build -f /tmp/Dockerfile -t $TAG $PWD
  docker push $TAG
  # --- Launches the job on aiplatform
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --scale-tier basic \
    --master-image-uri $TAG \
    --config /tmp/hypertune.yml \
    --job-dir "gs://research-brain-kubric-xgcp/jobs/$JOB_NAME" \
    -- "$@"
fi


