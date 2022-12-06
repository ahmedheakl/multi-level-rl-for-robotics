Installation
============

Prerequisites
-------------

Highrl requires python 3.8+, PyTorch >= 1.12, and pyglet >= 1.5

Windows 10
----------

Currently, the library is **neighter tested or built on Windows 10 platform**, however,
you can use `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_ to install Linux on Windows.

Another option is to create a `Docker <https://www.docker.com/>`_ container with *ubuntu >= 20.04*, and install our
library.

You can also visit the section :ref:`docker <dockerinstall>`.

Linux
-----

Creating a virtual env
~~~~~~~~~~~~~~~~~~~~~~


Install using pip
~~~~~~~~~~~~~~~~~

To use Highrl, first install it using pip:

.. code-block:: console

   (.venv) $ pip install highrl


.. _dockerinstall:

Development version
-------------------

To contribute to Highrl, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/ahmedheakl/multi-level-rl-for-robotics
    cd multi-level-rl-for-robotics
    pip install -e .[docs]

Docker
------