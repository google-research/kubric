# Copyright 2022 The Kubric Authors.
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

import argparse
import sys
import setuptools
from datetime import datetime
from pathlib import Path
from packaging import version

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--tag', type=str, default=None, help="use v1.2.3 syntax")
parser.add_argument('--nightly', action='store_true', help="automatic version, e.g. '2021.8.18'")
parser.add_argument('--secondly', action='store_true', help="automatic version, e.g. '2021.8.18.16.13.12'")
# --- remove known args from sys.argv (setup.py compliant)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# --- Compute the version (for both nightly and normal)
now = datetime.now()
VERSION = None
if args.tag is not None:
  assert len(args.tag)>1 and args.tag[0]=="v"
  VERSION = args.tag[1:]
  NAME="kubric"
if args.nightly:
  VERSION = f"{now.year}.{now.month}.{now.day}"
  NAME="kubric-nightly"
if args.secondly or VERSION is None:  #< contingency plan
  VERSION = f"{now.year}.{now.month}.{now.day}.{now.hour}.{now.minute}.{now.second}"
  NAME="kubric-secondly"
try:
  version.Version(VERSION)  #< assert if regex fails
except version.InvalidVersion as err:
  print(str(err) + f"\nVersion must match the regex: {version.VERSION_PATTERN}")

def set_version_in_file(version="HEAD"):
  ini_file_path = Path(__file__).parent / "kubric" / "__init__.py"
  ini_file_lines = list(open(ini_file_path))
  with open(ini_file_path, "w") as f:
    for line in ini_file_lines:
      if line.startswith("__version__"):
        f.write("__version__ = \"{}\"\n".format(version))
      else:
        f.write(line)

# --- Auto-update the build version in the library
set_version_in_file(VERSION)

# --- cache readme into a string
README = ""
if Path("README.md").exists():
  with open("README.md", "r", encoding="utf-8") as fh:
    README = fh.read()

# --- Extract the dependencies
REQS = [line.strip() for line in open("requirements.txt")]
INSTALL_PACKAGES = [line for line in REQS if not line.startswith("#")]

# --- Build the whl file
setuptools.setup(
    name=NAME,
    version=VERSION,
    author="Kubric team",
    author_email="kubric+dev@google.com",
    description="A data generation pipeline for creating semi-realistic synthetic multi-object "
                "videos with rich annotations such as instance segmentation, depth maps, "
                "and optical flow.",
    license="Apache 2.0",
    install_requires=INSTALL_PACKAGES,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/kubric",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
)

# --- Revert the version in the local folder
set_version_in_file("HEAD")