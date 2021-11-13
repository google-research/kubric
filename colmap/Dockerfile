# ---
FROM ubuntu:xenial

# --- helps "apt install -y tzdata"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# --- install C++ dependencies
RUN apt-get update --fix-missing
RUN apt-get install -y git
RUN apt-get install -y cmake
RUN apt-get install -y build-essential
RUN apt-get install -y libboost-program-options-dev
RUN apt-get install -y libboost-filesystem-dev
RUN apt-get install -y libboost-graph-dev
RUN apt-get install -y libboost-regex-dev
RUN apt-get install -y libboost-system-dev
RUN apt-get install -y libboost-test-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libfreeimage-dev
RUN apt-get install -y libgoogle-glog-dev
RUN apt-get install -y libgflags-dev
RUN apt-get install -y libglew-dev
RUN apt-get install -y qtbase5-dev
RUN apt-get install -y libqt5opengl5-dev
RUN apt-get install -y libcgal-dev
RUN apt-get install -y g++-6
RUN apt-get install -y libcgal-qt5-dev
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libsuitesparse-dev

# --- download and install GCP gcloud package
WORKDIR /
RUN apt-get install curl
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud
RUN tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz
RUN /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH "$PATH:/usr/local/gcloud/google-cloud-sdk/bin"

# --- install ceres
RUN git clone https://ceres-solver.googlesource.com/ceres-solver /ceres
WORKDIR /ceres
RUN git checkout --detach facb199
RUN mkdir build
WORKDIR /ceres/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
RUN make -j16
RUN make install

# --- install colmap
RUN git clone https://github.com/colmap/colmap.git /colmap
WORKDIR /colmap
RUN git checkout --detach c9f0b5674754e8c48040b9d4fb5995f77d6ea105
RUN mkdir build
WORKDIR /colmap/build
ENV CMAKE_C_FLAGS="-fext-numeric-literals"
ENV CMAKE_CXX_FLAGS="-fext-numeric-literals"
RUN CC=/usr/bin/clang-6.0 CXX=/usr/bin/clang++-6.0 cmake .. -DOPENGL_ENABLED=OFF
RUN make -j16 
RUN make install

# # --- download and install gsutil
# # https://cloud.google.com/storage/docs/gsutil_install
# RUN apt-get install -y wget
# RUN apt-get install -y python3
# WORKDIR /downloads
# RUN mkdir /gsutil
# RUN wget https://storage.googleapis.com/pub/gsutil.tar.gz
# RUN tar xfz gsutil.tar.gz -C /gsutil
# RUN chmod +x /gsutil/gsutil/gsutil
# ENV PATH "$PATH:/gsutil/gsutil"

# --- copies a dummy file to shared bucket
# RUN gsutil ls -l gs://kubric-public/colmap
# RUN echo `date '+%m%d%y %H%M%S'` > dummy.txt 
# RUN gsutil cp dummy.txt gs://kubric-public/colmap

# --- final call
WORKDIR /workspace