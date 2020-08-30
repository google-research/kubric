# Kubric

![Unittests](https://github.com/google-research/kubric/workflows/Unittests/badge.svg)
[![Coverage](https://badgen.net/codecov/c/github/google-research/kubric)](https://codecov.io/github/google-research/kubric)

A data generation pipeline for creating semi-realistic synthetic multi-object 
videos with rich annotations such as instance segmentation masks, depth maps, 
and optical flow.

> :warning: This project is pre-alpha work in progress and subject to extensive change.

## Motivation
We need better data for training and evaluating machine learning systems, especially in the context of unsupervised multi-object video understanding.
Current systems succeed on [toy datasets](https://github.com/deepmind/multi_object_datasets), but fail on real-world data.
Progress is could be greatly accelerated if we had the ability to create suitable datasets of varying complexity on demand.

## Requirements
- A pipeline for conveniently generating video data. 
- Physics simulation for automatically generating physical interactions between multiple objects.
- Good control over the complexity of the generated data, so that we can evaluate individual aspects such as variability of objects and textures.
- Realism: Ideally, the ability to span the entire complexity range from CLEVR all the way to real-world video such as YouTube8. This is clearly not feasible, but we would like to get as close as possible. 
- Access to rich ground truth information about the objects in a scene for the purpose of evaluation (eg. object segmentations and properties)
- Control the train/test split to evaluate compositionality and systematic generalization (for example on held-out combinations of features or objects)

## Getting Started
To run locally:
* install Blender2.83
* install requirements in the Blender-internal python
* extract KLEVR.zip 
* `blender -noaudio --background --python worker.py -- --assets='/PATH/TO/KLEVR'`
* (Results are stored in `./output/`)

To run on GCP using docker:
* `make_kubruntu.sh` to build the required docker image
* `make_render.sh local` to run the docker container locally
* or `make_render.sh remote` to submit a run using the ai-platform
* or `make_render.sh hypertune` to launch parallel jobs using the ai-platform

## Design
Mainly built on-top of pybullet for physics simulation and Blender for rendering the video.
But the code is kept modular to support different rendering backends.

## Contributors
[Klaus Greff](https://github.com/qwlouse) (Google), [Andrea Tagliasacchi](https://github.com/taiya) (Google and University of Toronto), Derek Liu (University of Toronto), Cinjon Resnick (NYU), Francis Williams (NYU), Issam Laradji (McGill and MILA), Or Litany (Stanford and NVIDIA), Luca Prasso (Google)

## Disclaimer
This is not an official Google Product
