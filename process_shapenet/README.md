# Steps to process shapenet


## 1. Download Dataset

- Download ShapeNetCore v2 release from https://shapenet.org/download/shapenetcore

## 2. Download Kubric Repository

- Download Kubric Repository

```
git clone git@github.com:google-research/kubric.git
cd kubric
```

- Visualize an example of a shapenet Kubric asset at `process_shapenet/example`

## 3. Run Docker and Install Required Libraries

- Create Docker Image

```
cd process_shapenet
bash make_docker.sh
```

- Install Manifold Plus within the Docker container

```
cd ..
git clone git@github.com:hjwdzh/ManifoldPlus.git
cd ManifoldPlus
bash compile.sh

# test
./build/manifold --input data/bathtub.obj --output results/bathtub_manifold.obj --depth 8
```

## 4. Run ShapeNet Script

```
python3.7 process_shapenet/process_shapenet.py -d ../datasets/ShapeNetCore.v2
```

The Kubric assets will be saved at process_shapenet/output