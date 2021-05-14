.. _installation:

Installing Kubric
=================

There are several ways of installing Kubric:

  1. Using Docker (recommended)
  2. Through the Blender CLI
  3. Native installation as Python package

Docker
------
The Docker image `kubruntu <https://hub.docker.com/r/kubricdockerhub/kubruntu>`_ features a fully functional installation of Kubric and all its dependencies.
This approach for using Kubric is by far the easiest, and the one we recommend.
Assuming a working `Docker installation <https://docs.docker.com/get-docker/>`_ the image can be downloaded simply by:

.. code-block:: console

    docker pull kubricdockerhub/kubruntu

This completes the "installation", and any Kubric worker file can now be run as:

.. code-block:: console

    docker run --user $(id -u):$(id -g) --volume "$PWD:/kubric" --interactive --rm  kubricdockerhub/kubruntu python3 worker.py

.. note::
    The flag ``--user $(id -u):$(id -g)`` ensures commands executed within the container use the host user information 
    in creating new files.
    The flag ``--volume "HOST_DIR:CONTAINER_DIR"`` mounts the current directory into the ``/kubric`` container directory; this is a convenient way to pass files (like ``worker.py``) into the container and also get the output files back.
    The flags ``--interactive`` allocates a pseudo-tty including STDIN for interactivity and finally ``--rm`` deletes the container (but not the image) after completion (optional to avoid clutter).


Blender
-------
In principle, Kubric scripts can also be run directly through the CLI of Blender, which makes use of the included Python version that Blender ships with (instead of the system Python).
For this approach, download and install the desired version of `Blender <https://www.blender.org/download/>`_ normally, and also clone Kubric as usual.

Then from the Kubric source directory install the requirements and Kubric inside Blender Python.
For example like this:

.. code-block:: console

    blender --python -m pip install -r requirements.txt
    blender --python -m pip install .

.. warning:: This may fail for some dependencies which have to be built (e.g. OpenEXR), even if the corresponding system-packages are installed because the Blender-internal pip fails to find the correct paths. We do not know of a good solution for this.

It should then be possible to run Kubric worker files using:

.. code-block:: console

    blender --factory-startup -noaudio --background --python worker.py


Native
------
In case that you need (or want) a Kubric installation directly on your host system, you need to first manually install ``bpy`` and the other dependencies.
Be warned though: this process can be VERY difficult and frustrating.

First clone the `Kubric git repository <https://github.com/google-research/kubric>`_:

.. code-block:: console

    git clone https://github.com/google-research/kubric.git

Then use ``pip`` to install the non-``bpy`` dependencies:

.. code-block:: console

    cd kubric
    pip install -r requirements.txt

.. note::
    This step may require installing additional (non-Python) packages.
    On Ubuntu, for example, the ``OpenEXR`` Python package depends on ``libopenexr-dev``.

Next, install the Blender python module ``bpy``.
We recommend to first try using the `blenderpy <https://github.com/TylerGubala/blenderpy>`_ project, which aims to make ``bpy`` pip-installable, and already works for most environments.
Note however, that this still requires installing the minimum build dependencies for Blender.
Just follow the `install instructions for blenderpy <https://github.com/TylerGubala/blenderpy#getting-started>`_, and make sure to install the correct version of Blender (currently 2.91).

If this approach fails, you can try to manually `build Blender as a Python module <https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule>`_.
Though be warned, that `"The option to build Blender as a Python module is not officially supported [...]"`.


Finally, install Kubric using pip from the source directory:

.. code-block:: console

    pip install .

Kubric worker files can then be run normally using python:

.. code-block:: console

    python worker.py
