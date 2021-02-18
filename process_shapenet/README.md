# Steps to process shapenet

## Installation

- Install Manifold Plus

```
git clone git@github.com:hjwdzh/ManifoldPlus.git
cd ManifoldPlus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
rm -rf .git
```

- Install NPM from `https://www.npmjs.com/get-npm`

- Install `obj2gltf` with the following command

```
npm install -g obj2gltf
```

## Run ShapeNet Script

```
python process_shapenet -d {path_to_shapenet} -c {category_id}
```