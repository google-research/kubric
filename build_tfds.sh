#!/bin/bash -x

DATASET_NAME=klevr
DATASET_CONFIG=kubric/datasets/klevr.py
GCP_PROJECT=kubric-xgcp
GCS_BUCKET=gs://research-brain-kubric-xgcp
REGION=us-central1

echo "tensorflow_datasets" > /tmp/beam_requirements.txt


tfds build $DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options="runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen,staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,requirements_file=/tmp/beam_requirements.txt,region=$REGION"

