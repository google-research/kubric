# Robust NeRF


Dataset available at `gs://kubric-public/data/robust_nerf`

Neural Radiance Fields or NeRF, trains a representation of a static 3D scene via volume rendering by minimizing a photometric reconstruction loss.
The nature of this loss implies that when the scene is not perfectly static across views, the recovered representation is corrupted.

This challenge demonstrates that further research is still needed to fully address this problem.
In the teleport challenge, while most of the scene remains rigid, we add impostor non-static object (i.e. the monkey head) randomly within the scene bounds, while in the jitter challenge the impostor position jitters around a fixed position.
In other words, the two datasets evaluate the sensitivity of unstructured (teleport) vs. structured (jitter) outliers in the training process. 

Please see `worker.py` to get a glimpse of how the data was generated.

<img src="teaser.png" width=50%>
