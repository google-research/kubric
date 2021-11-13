FROM colmap/colmap:latest

# --- prereqs
RUN apt-get update --fix-missing
RUN apt-get install -y curl

# --- download and install GCP gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud
RUN tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
RUN /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH "$PATH:/usr/local/gcloud/google-cloud-sdk/bin"

# --- hypertune support
RUN apt-get install -y python3-pip
RUN pip3 install cloudml-hypertune

# --- last entry
WORKDIR /workspace