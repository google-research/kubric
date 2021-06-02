# Steps to process shapenet

## 1. Download Dataset

- Download ShapeNetCore v2 release from https://shapenet.org/download/shapenetcore

## 2. Create and Enter Docker

```
bash make_docker.sh
```

## 3. 

- Download Repo

```
git clone 
```

- Install Manifold Plus

```
git clone git@github.com:hjwdzh/ManifoldPlus.git
cd ManifoldPlus
bash compile.sh
# test
./build/manifold --input data/bathtub.obj --output results/bathtub_manifold.obj --depth 8
```

## Run ShapeNet Script

```
python3.7 process_shapenet -d ../datasets/ShapeNetCore.v2
```