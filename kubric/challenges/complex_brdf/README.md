# Complex BRDFs

Renderered datasets available at `gs://kubric-public/data/complex_brdf`

An important vision problem is that of reconstructing a 3D scene from few observations. Current datasets mostly feature Lambertian scenes, i.e., scenes that consist of mostly diffuse surfaces, with few specular highlights. In this case, the only relevant scene parameters are the 3D geometry, as well as the diffuse surface color.  When scene surfaces are highly reflective, the number of scene properties required for accurate novel view synthesis grows significantly. Instead of just 3D shape and appearance, the model needs to address 3D geometry, the BRDF of every surface point, as well as a full characterization of the light incident onto the scene. 

To this end, we render out a highly specular version of the ShapeNet dataset as a challenge for few-shot novel view synthesis algorithms.
We follow the NMR dataset and render objects across 13 classes from the  same 24 views.
To each object, we randomly assign an RGB color. We place three light sources at randomized positions on the upper hemisphere.
In this challenge, we fix the material properties of each object to the properties of the specular CLEVR, and ray-trace each scene with 12 ray bounces. The specular version of the dataset may be found at `gs://kubric-public/data/complex_brdf/specular/`, while the Lambertian variant of the dataset may be found at `gs://kubric-public/data/complex_brdf/lambertian/`.

We benchmark PixelNeRF, a conditional 3D-structured neural scene representation on this dataset. In order for these models to successfully train and perform at test-time, they need to both model the view-dependent forward model correctly, and correctly infer the position of the light sources. We illustrate how PixelNERF struggle to represent inherent specularities in shapes.

![](teaser.png)

