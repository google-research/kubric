# Single View Reconstruction

This dataset is based on ShapeNetCoreV2 which consists of 55 categories with a total of approximately 51,300 object models. Each model is projected  into 24 random viewpoints, all while maintaining consistent meta information (camera pose and object category) that allows us to train single-view reconstructiion methods (like SoftRas) efficiently.

Please see the worker.py file to get a glimpse of how the data was generated. However, the pre-rendered dataset available at:
`gs://kubric-public/data/single_view_reconstruction`

The dataset is divided into two difficulties levels: in-distribution and out-of-distribution. For in-distribution, swe follow the training regimen of SoftRas, train on 80% of each category, and test and report performance on the remaining 20% of each category, while in out-of-distribution we train on all categories except 4 classes that we leave out for testing. They are {\it train, tower, washer and vessel}.

See example of predicted images by SoftRas as well as ground truth masks:


<p align="center" width="100%">
    <img width="50%" src="teaser.png"> 
</p>
