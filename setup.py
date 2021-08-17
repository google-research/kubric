# Copyright 2021 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
import setuptools

try:
  with open("README.md", "r", encoding="utf-8") as fh:
    README = fh.read()
except IOError:
  README = ""

# --- compute the version (for both nightly and normal)
now = datetime.datetime.now()
VERSION = "{}.{}.{}".format(now.year, now.month, now.day)


setuptools.setup(
    name="kubric",
    version=VERSION,
    author="Kubric team",
    author_email="kubric+dev@google.com",
    description="A data generation pipeline for creating semi-realistic synthetic multi-object "
                "videos with rich annotations such as instance segmentation, depth maps, "
                "and optical flow.",
    license="Apache 2.0",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/kubric",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved ::  Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
)

# --- add a version to the built file
curr_path = os.path.dirname(__file__)
ini_file_path = os.path.join(curr_path, "build/lib/kubric/__init__.py")
ini_file_lines = list(open(ini_file_path))
with open(ini_file_path, "w") as f:
  for line in ini_file_lines:
    f.write(line.replace("__version__ = \"HEAD\"",
                         "__version__ = \"{}\"".format(VERSION)))