# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

PROJECT_ID="kubric-xgcp"
REGION="us-central1"  #< WARNING: match region of bucket!
JOB_NAME="colmap_`date +"%b%d_%H%M%S"`"

# --- Container configuration
cat > Dockerfile <<EOF
FROM kubricdockerhub/colmap:latest
COPY script.sh /workspace/script.sh
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "script.sh"]
EOF

# --- Parameters for the launch
TAG="gcr.io/$PROJECT_ID/colmap"
docker build -f Dockerfile -t $TAG $PWD
docker push $TAG

# --- Launches the job on aiplatform
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --scale-tier custom --master-machine-type standard_v100 \
  --master-image-uri $TAG \
  -- $1 $2

# gcloud ai-platform jobs stream-logs $JOB_NAME