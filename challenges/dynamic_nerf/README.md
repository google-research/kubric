# Dynamic NeRF Challenge
An example of rendering animation from blender is given in examples/articulation.py
To render the following example, run:
```
mkdir examples/KuBasic/rain_v22
gcloud cp gs://research-brain-kubric-xgcp/articulation/* examples/KuBasic/rain_v22/
docker build -f docker/KubruntuDev.Dockerfile -t kubricdockerhub/kubruntudev:latest .
docker run --rm --interactive --user `id -u`:`id -g` \
    --volume `pwd`:/workspace \
    kubricdockerhub/kubruntudev \
    python3 challenges/dynamic_nerf/worker.py
```

![](/docs/images/articulation.gif)
