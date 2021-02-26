#!/bin/bash -x

DATASET_NAME=klevr  # has to be the same as the filename of DATASET_CONFIG
DATASET_CONFIG=kubric/datasets/klevr.py
GCP_PROJECT=kubric-xgcp
GCS_BUCKET=gs://research-brain-kubric-xgcp
REGION=us-central1

# create a pseudo-package in a temporary directory to ship the dataset code to dataflow workers
# https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/
TEMP=$(mktemp -d)
mkdir "$TEMP/$DATASET_NAME"
cp "$DATASET_CONFIG" "$TEMP/$DATASET_NAME/__init__.py"
echo "Dummy package to ship dataset code to worker nodes" > "$TEMP/README"
cat > "$TEMP/setup.py" <<EOF
import setuptools

setuptools.setup(
    name='$DATASET_NAME',
    version='0.0.1',
    url="https://github.com/google-research/kubric",
    author="kubric authors",
    author_email="kubric@google.com",
    install_requires=['tensorflow_datasets'],
    packages=setuptools.find_packages(),
)
EOF

#
tfds build $DATASET_CONFIG \
  --data_dir=$GCS_BUCKET/tensorflow_datasets \
  --beam_pipeline_options="runner=DataflowRunner,project=$GCP_PROJECT,job_name=$DATASET_NAME-gen-subsampling-splits,staging_location=$GCS_BUCKET/binaries,temp_location=$GCS_BUCKET/temp,region=$REGION,setup_file=$TEMP/setup.py,machine_type=n1-highmem-32,num_workers=20"


# clean-up: delete the pseudo-package
rm -rf "$TEMP"
