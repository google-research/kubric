# Multi-view object matting

This dataset creates a scene where a foreground object is to be distinguished
from the background. Foreground objects are borrowed from shapnet. Backgrounds
are from indoor scenes of polyhaven. All foreground objects are situated on top
of a "table" which is gernated to be random in color. Instead of background
removal with a single image. This dataset is special in that multiple images of
the foreground object (taken from different camera poses) are given. This
"multi-view" persepctive should be very helpful for background removal but is
currently underexplored in the literature.

The dataset is divided into two difficulties levels: *easy* and *hard*.
For the easy challenge, scenes only contain one salient object within the scene,
while in the hard challenge we additionally insert clutter.

Please see the `worker.py` file to get a glimpse of how the data was generated.

See example training images as well as ground truth masks
from *easy* and *hard*:
![](teaser.jpg)