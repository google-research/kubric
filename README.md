# Kubric

[![Blender](https://github.com/google-research/kubric/actions/workflows/blender.yml/badge.svg?branch=main)](https://github.com/google-research/kubric/actions/workflows/blender.yml)
[![Kubruntu](https://github.com/google-research/kubric/actions/workflows/kubruntu.yml/badge.svg?branch=main)](https://github.com/google-research/kubric/actions/workflows/kubruntu.yml)
[![Test](https://github.com/google-research/kubric/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/google-research/kubric/actions/workflows/test.yml)
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
git clone https://github.com/google-research/kubric.git
cd kubric
docker pull kubricdockerhub/kubruntu
docker run --rm --interactive \
           --user $(id -u):$(id -g) \
           --volume "$(pwd):/kubric" \
           kubricdockerhub/kubruntu \
           /usr/bin/python3 examples/helloworld.py
ls output
```

## Requirements
- A pipeline for conveniently generating video data. 
- Physics simulation for automatically generating physical interactions between multiple objects.
- Good control over the complexity of the generated data, so that we can evaluate individual aspects such as variability of objects and textures.
- Realism: Ideally, the ability to span the entire complexity range from CLEVR all the way to real-world video such as YouTube8. This is clearly not feasible, but we would like to get as close as possible. 
- Access to rich ground truth information about the objects in a scene for the purpose of evaluation (eg. object segmentations and properties)
- Control the train/test split to evaluate compositionality and systematic generalization (for example on held-out combinations of features or objects)

## Bibtex
```
@article{greff2021kubric,
    title = {Kubric: a scalable dataset generator}, 
    author = {Klaus Greff and Francois Belletti and Lucas Beyer and Carl Doersch and
              Yilun Du and Daniel Duckworth and David J Fleet and Dan Gnanapragasam and
              Florian Golemo and Charles Herrmann and Thomas Kipf and Abhijit Kundu and
              Dmitry Lagun and Issam Laradji and Hsueh-Ti (Derek) Liu and Henning Meyer and
              Yishu Miao and Derek Nowrouzezahrai and Cengiz Oztireli and Etienne Pot and
              Noha Radwan and Daniel Rebain and Sara Sabour and Mehdi S. M. Sajjadi and Matan Sela and
              Vincent Sitzmann and Austin Stone and Deqing Sun and Suhani Vora and Ziyu Wang and
              Tianhao Wu and Kwang Moo Yi and Fangcheng Zhong and Andrea Tagliasacchi},
    booktitle = {arXiv preprint},
    year = {2021},
}
```




## Disclaimer
This is not an official Google Product
