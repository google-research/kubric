# Compile
mkdir build
cd build
/mnt/home/projects/kubric/ManifoldPlus/cmake-3.19.0-rc1-Linux-x86_64/bin/cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..
mkdir results