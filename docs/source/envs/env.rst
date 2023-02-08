Environments
============

Creating env instances
----------------------

To retrieve an env instance from the teacher,
you can use the ``TeacherEnv()`` function:

The ``args`` parameter should be a command-line args object.

.. py:function:: highrl.envs.teacher_env.TeacherEnv(robot_config)

    Return an instance of the teacher env.

    :param robot_config: Configurations object for robot env.
    :type robot_config: configparser.RawConfigParser
    :param teacher_config: Configurations object for teacher env.
    :type teacher_config: configparser.RawConfigParser
    :param args: Command-line arguments.
    :type args: argparse.Namespace
    :return: A new instance of the teacher env.
    :rtype: gym.Env

Compute the difficulty
----------------------

you can use the ``highrl.utils.planner_checker.convex_hull_compute()`` function:

autofunction:: highrl.utils.p