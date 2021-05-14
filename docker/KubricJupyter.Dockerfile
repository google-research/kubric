# --- Image with an active jupyterlab server exposed it to port 8888
FROM kubricdockerhub/kubruntu:latest

EXPOSE 8888
WORKDIR /kubric
RUN apt-get update && \
    apt-get install --yes --quiet --no-install-recommends imagemagick ffmpeg
RUN pip3 install jupyterlab seaborn Pillow

ENTRYPOINT ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
