# Kubric

A data generation pipeline for creating semi-realistic synthetic multi-object 
videos with rich annotations such as instance segmentation masks, depth maps, 
and optical flow.

NOTE: This project is pre-alpha work in progress and subject to extensive change. Use at your own risk.



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


## Design
Mainly built on-top of pybullet for physics simulation and Blender for rendering the video.
But the code is kept modular to support different rendering backends.



## Disclaimer
This is not an official Google Product
