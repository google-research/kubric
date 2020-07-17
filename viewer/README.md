# general python style guide
http://google.github.io/styleguide/pyguide.html

# example of threejs in colab
https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/mesh_viewer.py

# derek's tutorial
https://github.com/HTDerekLiu/BlenderToolbox/blob/master/cycles/tutorial.py

# threejs (javascript) starting example
https://threejs.org/docs/#manual/en/introduction/Creating-a-scene

# meshcat backend (webgl)
https://github.com/rdeits/meshcat-python

# installing mathutils on blender's python (macOS)
/Applications/Blender.app/Contents/Resources/2.83/python/bin/python3.7m -m pip install mathutils
/Applications/Blender.app/Contents/Resources/2.83/python/bin/python3.7m -m pip install webcolors
/Applications/Blender.app/Contents/Resources/2.83/python/bin/python3.7m -m pip install tensorflow
/Applications/Blender.app/Contents/Resources/2.83/python/bin/python3.7m -m pip install trimesh


# Blender on a CloudVM (install from SNAP package manager)
**WARNING: failed because cannot install pip packages inside blender (readonly)**

```
sudo apt update
sudo apt-get install snapd
sudo snap install blender --classic # TODO: how pick 2.83.2
```

Figure out which python you need to use (`/snap/blender/41/2.83/python` below):
```
blender --background --python-console
Blender 2.83.2 (hash 239fbf7d936f built 2020-07-09 06:21:45)
/run/user/899989266/snap.blender/gvfs/ non-existent directory
found bundled python: /snap/blender/41/2.83/python
Python 3.7.4 (default, Oct  8 2019, 15:23:02) 
[GCC 6.3.1 20170216 (Red Hat 6.3.1-3)] on linux
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
```

Pip is missing, it seems:
```
(blender) ~/Downloads: /snap/blender/41/2.83/python/bin/python3.7m -m ensurepip
Looking in links: /tmp/tmpp142f39b
Collecting setuptools
Collecting pip
Installing collected packages: setuptools, pip
Could not install packages due to an EnvironmentError: [Errno 30] Read-only file system: '/snap/blender/41/2.83/python/lib/python3.7/site-packages/easy_install.py'
```

FAIL :(

# Blender on a CloudVM (TAR + custom python)
**WARNING: failed because python is not recognized**

Following [this instructions](https://docs.blender.org/api/current/info_tips_and_tricks.html#bundled-python-extensions)

Blender needs to work on the same python as it was compiled with (for Blender 2.83):
```
sudo apt-get install python3.7
```

Create a virtualenv for blender
```
virtualenv -p python3.7 ~/Envs/blender
source ~/Envs/blender/bin/activate
```

URL fetched from [here](https://download.blender.org/release/Blender2.83).
1) on Ubuntu 18.4 only blender 2.7 is available)
2) on snapd, one cannot modify the blender snap subdirectory

```
wget https://download.blender.org/release/Blender2.83/blender-2.83.2-linux64.tar.xz
md5sum blender-2.83.2-linux64.tar.xz  # expected: 96194bd9d0630686765d7c12d92fcaeb
tar -xf blender-2.83.2-linux64.tar.xz
blender-2.83.2-linux64/blender --background --python-expr "import os; print(os.__file__)"
rm -rf blender-2.83.2-linux64/2.83/python
ln -s ~/Envs/blender blender-2.83.2-linux64/2.83/python
blender-2.83.2-linux64/blender -noaudio --background --python-console
```

Unfortunately virtualenv python still not found:
```
(blender) ~/Downloads: blender-2.83.2-linux64/blender -noaudio --background --python-console
Blender 2.83.2 (hash 239fbf7d936f built 2020-07-09 06:21:45)
/run/user/899989266/gvfs/ non-existent directory
found bundled python: /home/atagliasacchi_google_com/Downloads/blender-2.83.2-linux64/2.83/python
Fatal Python error: initfsencoding: Unable to get the locale encoding
ModuleNotFoundError: No module named 'encodings'

Current thread 0x00007f69cd931040 (most recent call first):
Aborted (core dumped)
```

# Blender on a CloudVM (TAR + blender REPL)

```
wget https://download.blender.org/release/Blender2.83/blender-2.83.2-linux64.tar.xz
md5sum blender-2.83.2-linux64.tar.xz  # expected: 96194bd9d0630686765d7c12d92fcaeb
tar -xvf blender-2.83.2-linux64.tar.xz
export PATH="$PWD/blender-2.83.2-linux64:$PATH"
```

```
blender -noaudio --background --python-expr "import os; print(os.__file__)"
blender -noaudio --background --python-console
```

```
sudo apt-get install python3.7
sudo apt-get install python3.7-dev
```

Install the dependencies (but first pip!)
```
blender-2.83.2-linux64/2.83/python/bin/python3.7m -m ensurepip
blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install --upgrade pip
blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install wheel
blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install numpy
blender-2.83.2-linux64/2.83/python/bin/python3.7m -m pip install trimesh
```

Test out the helloworld
```
blender -noaudio --background --python helloworld.py
```


# Logging not working when launched inside Blender's REPL? 
import logging
logging.basicConfig(level="INFO")
logging.info("testing logging on ai-platform")
logging.critical("shoult terminate program")

# replace gfile with a non-TF solution?
Docs: https://docs.pyfilesystem.org/en/latest/
Dependencies
```
pip install google-api-core
pip install fs-gcsfs
```
Usage
```
from fs import open_fs
gcsfs = open_fs("gs://mybucket/root_path")
```