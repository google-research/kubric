# WARNING: verify credentials are enabled "gcloud auth configure-docker"
# WARNING: verify credentials are enabled "gcloud auth application-default login"

PROJECT_ID="kubric-xgcp"
REGION="us-central1"  #< WARNING: match region of bucket!
JOB_NAME="colmap_`date +"%b%d_%H%M%S"`"

# --- Container configuration
cat > /tmp/Dockerfile <<EOF
FROM kubricdockerhub/colmap:latest
COPY colmap.sh /workspace/colmap.sh
WORKDIR /workspace
ENTRYPOINT ["./colmap.sh"]
EOF

# --- Parameters for the launch
TAG="gcr.io/$PROJECT_ID/colmap"
docker build -f /tmp/Dockerfile -t $TAG $PWD
docker push $TAG

echo "reading images from: '$1'"
echo "saving results to: '$2'"

# --- Launches the job on aiplatform
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --scale-tier custom --master-machine-type standard_v100 \
  --master-image-uri $TAG \
  -- $1 $2

# gcloud ai-platform jobs stream-logs $JOB_NAME