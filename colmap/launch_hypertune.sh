# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

PROJECT_ID="kubric-xgcp"
REGION="us-central1"  #< WARNING: match region of bucket!
JOB_NAME="colmap_hypertune_`date +"%b%d_%H%M%S" | tr A-Z a-z`"

# --- Container configuration
cat > /tmp/Dockerfile <<EOF
  FROM kubricdockerhub/colmap:latest
  COPY colmap_auto.sh /workspace/colmap.sh
  WORKDIR /workspace
  ENTRYPOINT ["./colmap.sh"]
EOF

# --- Specify the hypertune configuration
cat > /tmp/hypertune.yml << EOF
  trainingInput:
    hyperparameters:
      goal: MAXIMIZE
      hyperparameterMetricTag: "answer"
      maxTrials: 8
      maxParallelTrials: 8
      maxFailedTrials: 8
      enableTrialEarlyStopping: False
      # --- each of these become an argparse argument
      params:
      - parameterName: sceneid
        type: INTEGER
        minValue: 8
        maxValue: 15
EOF

# --- Parameters for the launch
TAG="gcr.io/$PROJECT_ID/colmap"
docker build -f /tmp/Dockerfile -t $TAG $PWD
docker push $TAG

echo "hypertune job on: '$1'"
echo "saving results to: '$2'"

# --- Parameters for the launch
TAG="gcr.io/$PROJECT_ID/$JOB_NAME"
docker build -f /tmp/Dockerfile -t $TAG $PWD
docker push $TAG
# --- Launches the job on aiplatform
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --scale-tier custom \
  --master-machine-type n1-highcpu-16 \
  --master-accelerator count=1,type=nvidia-tesla-p100 \
  --master-image-uri $TAG \
  --config /tmp/hypertune.yml \
  -- $@
gcloud ai-platform jobs describe $JOB_NAME

# For faster testing use "--scale-tier basic"
# https://cloud.google.com/ai-platform/training/docs/using-gpus