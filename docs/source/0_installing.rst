Installing
==========

The Docker image `kubruntu <https://hub.docker.com/r/kubricdockerhub/kubruntu>`_ features a fully functional installation of Kubric and all its dependencies.
This approach for using Kubric is by far the easiest, and the one we recommend. 
Assuming a working `Docker installation <https://docs.docker.com/get-docker/>`_ the image can be downloaded simply by:

.. code-block:: console

    docker pull kubricdockerhub/kubruntu

This completes the "installation", and any Kubric worker file can now be run as:

.. code-block:: console

    docker run --rm --interactive \
        --user $(id -u):$(id -g) \
        --volume "$PWD:/workspace" \
        kubricdockerhub/kubruntu \
        python3 examples/helloworld.py

.. note::
    The flag ``--user $(id -u):$(id -g)`` ensures commands executed within the container use the host user information in creating new files.
    The flag ``--volume "HOST_DIR:CONTAINER_DIR"`` mounts the current directory into the ``/workspace`` container directory; this is a convenient way to pass files (like ``worker.py``) into the container and also get the output files back.
    The flags ``--interactive`` allocates a pseudo-tty including STDIN for interactivity and finally ``--rm`` deletes the container (but not the image) after completion (optional to avoid clutter).


.. warning:: Kubric scripts can also be run directly through the CLI of Blender, which makes use of the included Python version that Blender ships with (instead of the system Python). This may fail for some dependencies which have to be built (e.g. OpenEXR), even if the corresponding system-packages are installed because the Blender-internal pip fails to find the correct paths. Please refer to the instructions for `macOS <https://github.com/google-research/kubric/issues/99>`_ and `Debian <https://github.com/google-research/kubric/issues/98>`_.

.. warning:: You can also install Kubric directly on your host system (virtualenv). To achieve this, you need to first manually install ``bpy`` and the other dependencies. Be warned though: this process can be VERY difficult and frustrating. Please refer to `these instructions <https://github.com/google-research/kubric/issues/100>`_.