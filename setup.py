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

import setuptools

try:
  with open("README.md", "r", encoding="utf-8") as fh:
    README = fh.read()
except IOError:
  README = ""


__version__ = None

with open('kubric/version.py') as f:
  exec(f.read(), globals())


setuptools.setup(
    name="kubric",
    version="0.1",
    author="Kubric team",
    author_email="klausg@google.com",  # TODO: create a kubric-dev group
    description="A data generation pipeline for creating semi-realistic synthetic multi-object "
                "videos with rich annotations such as instance segmentation, depth maps, "
                "and optical flow.",
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
