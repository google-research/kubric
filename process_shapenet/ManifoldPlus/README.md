# ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups
Advanced version of my previous Manifold algorithm from this [**repo**](https://github.com/hjwdzh/Manifold).

![Plane Fitting Results](https://github.com/hjwdzh/ManifoldPlus/raw/master/res/manifold-teaser.jpg)

### Dependencies
1. Eigen
2. LibIGL

### Installing prerequisites
```
git submodule update --init --recursive
```

### Quick examples
```
sh compile.sh
sh examples.sh
```

### Build
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Run
The input is a random triangle mesh in obj/off format. The output is a watertight manifold mesh in obj format.
```
./ManifoldPlus --input input.obj --output output.obj --depth 8
```
An example script is provided so that you can try several provided models. We convert inputs in data folder to outputs in results folder.

Copyright:
This software is distributed for free for non-commercial use only.


**IMPORTANT**: If you use this code please cite the following (to provide) in any resulting publication:
```
@article{huang2020manifoldplus,
  title={ManifoldPlus: A Robust and Scalable Watertight Manifold Surface Generation Method for Triangle Soups},
  author={Huang, Jingwei and Zhou, Yichao and Guibas, Leonidas},
  journal={arXiv preprint arXiv:2005.11621},
  year={2020}
}
```

### Copyright
This software is distributed for free for non-commercial use only.

- [Jingwei Huang](mailto:jingweih@stanford.edu)

&copy; 2020 Jingwei Huang All Rights Reserved

```
This software is provided by the copyright holders and the contributors 
"as is" and any express or implied warranties, including, but not limited 
to, the implied warranties of merchantability and fitness for a particular 
purpose are disclaimed. In no event shall the copyright holders or 
contributors be liable for any direct, indirect, incidental, special, 
exemplary, or consequential damages (including, but not limited to, 
procurement of substitute goods or services; loss of use, data, or profits;
or business interruption) however caused and on any theory of liability, 
whether in contract, strict liability, or tort (including negligence or 
otherwise) arising in any way out of the use of this software, even if 
advised of the possibility of such damage.

The views and conclusions contained in the software and documentation are 
those of the authors and should not be interpreted as representing official 
policies, either expressed or implied, of Jingwei Huang.
```
