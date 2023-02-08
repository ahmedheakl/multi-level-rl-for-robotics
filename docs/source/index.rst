.. Highrl documentation master file, created by
   sphinx-quickstart on Tue Nov 29 03:46:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Highrl's documentation!
==================================

**Highrl** is a library for training robots using RL under the scheme of multi-level RL.
The library has *numerous* features from generating random environment, training agents to generate curriculum learning schemes,
or train robots in pre-defined environments.

Github repository: https://github.com/ahmedheakl/multi-level-rl-for-robotics

Check out the :doc:`guide/install` section for further information, including how to
install the project.

Main Features
--------------

- Unified state-of-the art environment
- PEP8 compliant (unified code style)
- Documented functions and classes
- Tests, high code coverage and type hints
- Clean code
- Wrapped environment style

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/configurations
   

.. toctree::
   :maxdepth: 1
   :caption: Environments

   envs/env
   envs/teacherenvdocs
   envs/robotenv
   envs/evalenv

.. toctree::
   :maxdepth: 1
   :caption: Utilities

   utils/convexhull



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
