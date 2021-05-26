# Kubric

![Build container and run tests](https://github.com/google-research/kubric/workflows/Build%20container%20and%20run%20tests/badge.svg)
[![Coverage](https://badgen.net/codecov/c/github/google-research/kubric)](https://codecov.io/github/google-research/kubric)
[![Docs](https://readthedocs.org/projects/kubric/badge/?version=latest)](https://kubric.readthedocs.io/en/latest/)

A data generation pipeline for creating semi-realistic synthetic multi-object 
videos with rich annotations such as instance segmentation masks, depth maps, 
and optical flow.

> :warning: This project is pre-alpha work in progress and subject to extensive change.

## Motivation and design
We need better data for training and evaluating machine learning systems, especially in the context of unsupervised multi-object video understanding.
Current systems succeed on [toy datasets](https://github.com/deepmind/multi_object_datasets), but fail on real-world data.
Progress could be greatly accelerated if we had the ability to create suitable datasets of varying complexity on demand.
Kubric is mainly built on-top of pybullet (for physics simulation) and Blender (for rendering); however, the code is kept modular to potentially support different rendering backends.

## Getting started
For instructions, please refer to [https://kubric.readthedocs.io](https://kubric.readthedocs.io)

Assuming you have docker installed, to generate the data above simply execute:
```
docker pull docker pull kubricdockerhub/kubruntu
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 examples/klevr.py
```

![KLEVR: a CLEVR scene rendered by Kubric](https://kubric.readthedocs.io/en/latest/_images/KLEVR.gif)


## Requirements
- A pipeline for conveniently generating video data. 
- Physics simulation for automatically generating physical interactions between multiple objects.
- Good control over the complexity of the generated data, so that we can evaluate individual aspects such as variability of objects and textures.
- Realism: Ideally, the ability to span the entire complexity range from CLEVR all the way to real-world video such as YouTube8. This is clearly not feasible, but we would like to get as close as possible. 
- Access to rich ground truth information about the objects in a scene for the purpose of evaluation (eg. object segmentations and properties)
- Control the train/test split to evaluate compositionality and systematic generalization (for example on held-out combinations of features or objects)

## Contributors
[Klaus Greff](https://github.com/qwlouse) (Google), [Andrea Tagliasacchi](https://github.com/taiya) (Google and University of Toronto), Derek Liu (University of Toronto), Issam Laradji (McGill and MILA)

## Disclaimer
This is not an official Google Product
