Simulator
=========

A slightly more complex example does not render static frames but instead uses the physics engine to run a simulation, and then renders the corresponding video.
Let us look at `examples/simulator.py <https://github.com/google-research/kubric/blob/main/examples/simulator.py>`_ (full source at the bottom of this page).

As we are generating a video, we also specify the framerate properties of renderer and simulator, how many frames to render, and attach a simulator to the scene:

.. code-block:: python
  :emphasize-lines: 5

  scene = kb.Scene(resolution=(256, 256))
  scene.frame_end = 48   #< numbers of frames to render
  scene.frame_rate = 24  #< rendering framerate
  scene.step_rate = 240  #< simulation framerate
  simulator = KubricSimulator(scene)
  renderer = KubricRenderer(scene, scratch_dir="./output")

And notice that when we specify the floor, the ``static=True`` argument ensures that the floor remains fixed during the simulation:

.. code-block:: python
  :emphasize-lines: 1

  scene += kb.Cube(scale=(1, 1, 0.1), position=(0, 0, -0.1), static=True)
  scene += kb.DirectionalLight(position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
  scene.camera = kb.PerspectiveCamera(position=(2, -0.5, 4), look_at=(0, 0, 0))

Let us add a couple of colorful balls (:class:`~kubric.core.objects.Sphere` primitives) that bounce around; we use ``rng.uniform(low, high)`` to ensures that each ball is initialized at its own random random position:

.. code-block:: python

  spawn_region = [[-1, -1, 0], [1, 1, 1]]
  rng = np.random.default_rng()
  for i in range(8):
    position = rng.uniform(*spawn_region)
    velocity = rng.uniform([-1, -1, 0], [1, 1, 0])
    material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
    sphere = kb.Sphere(scale=0.1, position=position, velocity=velocity, material=material)
    scene += sphere
    kb.move_until_no_overlap(sphere, simulator, spawn_region=spawn_region)

To color them, we use the :class:`~kubric.core.materials.PrincipledBSDFMaterial`.
This material is very versatile and can represent a wide range of materials including plastic, rubber, metal, wax, and glass (see e.g. `these examples from the blender documentation <https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html#examples>`_). 

The utility function :func:`~kubric.randomness.move_until_no_overlap` jitters the objects position (and rotation) until the simulator no longer detects any collisions.

Now that we have all the objects in place, it is time to run the simulation.
Once the simulation terminates (pybullet), the simulated object states are saved as keyframes within the renderer (blender).

.. TODO: start_frame, end_frame, frame_rate, step_rate

.. code-block:: python
  
  simulator.run()

The gif below is generated via the ``convert`` tool from the ImageMagick package:

.. code-block:: shell

  convert -delay 8 -loop 0 output/images/frame_*.png output/simulator.gif

.. image:: /images/simulator.gif
   :width: 250pt
   :align: center

-------

.. literalinclude:: /../examples/simulator.py
  :lineno-start: 1
  :lines: 15-