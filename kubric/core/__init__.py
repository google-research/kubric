""" This package defines the basic object hierarchy that forms the center of Kubrics interface.

The root classes are Scene and Asset, which further specializes into:
 * Material
   - PrincipledBSDFMaterial
   - FlatMaterial
   - UndefinedMaterial
 * Object3D
   - PhysicalObject
     > FileBasedObject
     > Cube
     > Sphere
 * Light
   - DirectionalLight
   - RectAreaLight
   - PointLight
 * Camera
   - PerspectiveCamera
   - OrthographicCamera
   - UndefinedCamera

"""
