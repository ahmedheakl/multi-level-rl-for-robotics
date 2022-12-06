.. _quickstart:

Getting Started
===============

Here is a quick example of how to train and run a teacher environment:

.. code-block:: console

   (.venv) $ highrl

The teacher will start training and will output the results (traning graphs for both teacher and robot on the Desktop).

You can view the command-line args:

.. code-block:: console

   (.venv) $ highrl -h

``USAGE: Parse arguments [-h] [--robot-config ROBOT_CONFIG_PATH] [--teacher-config TEACHER_CONFIG_PATH] [--mode {train,test}] [--env-mode {teacher,robot}] [--render-each RENDER_EACH] [--output-dir OUTPUT_DIR] [--lidar-mode {flat,rings}]``


Here is the full of command-line args:

-h, --help              show help message and exit
--robot-config          path of configurations file for robot env
--teacher-config        path of configurations file for teacher env
--mode=train            mode of operation
--env-mode=teacher      which env to use through training/testing
--render-each           frequency of rendering for robot environment
--output-dir           relative path to output results for robot mode
--lidar-mode=flat           mode to process lidar flat=1D, rings=2D


   


.. note::

    The library is built with default configurations, however, you can change those configurations by incorporating your own `.config` files. See section :ref:`configs <yourconfigs>` for more details.
