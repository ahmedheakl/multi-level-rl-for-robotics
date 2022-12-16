import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_train(x_label, y_label, graph_title, x, y, name):
    """Plots training graph and save it as svg graph.

    Args:
        x_label (str): x axis label for the graph
        y_label (str): y axis label for the graph
        graph_title (str): title for the graph
        x (list): data for the x axis of the graph (usually number of episode)
        y (list): data for the y axis of the graph (success rate or reward of training episodes)
        name (str): the name to save the graph with (name.svg)
    """
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    plt.plot(x, y)

    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(graph_title)

    plt.savefig(name, bbox_inches="tight", pad_inches=0.1)


def smooth(y, weight):
    """smooth the graphs data using function:
       smoothed_next_data_point = smoothed_current_data_point * weight + (1 - weight) * point

    Args:
        y (list): data to smooth
        weight (float): smoothing factor between 0 and 1

    Returns:
        list: data after being smoothed
    """
    last = y[0]
    smoothed = list()
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def average_x(df, x):
    """smooth the graphs data using averaging of x points.


    Args:
        df (Dataframe): training logs
        x (int): number of points to average

    Returns:
        iteration_n(list): contains the x axis data for the graphs (usually n of episodes)
        average_reward(list): contains the reward of episodes after being averaged
        average_succ(list): contains the success of episodes after being averaged
    """
    average_reward = []
    average_r_per_x = 0
    for i in range(len(df)):
        average_r_per_x = average_r_per_x + df.loc[i].at["episode_reward"]
        if i % x == 0:
            average_reward.append(average_r_per_x / x)
            average_r_per_x = 0
    average_succ = []
    average_s_per_x = 0
    for i in range(len(df)):
        if df.loc[i].at["goal_reached"] == True:
            average_s_per_x += 1
        if i % x == 0:
            average_succ.append(average_s_per_x / x)
            average_s_per_x = 0
    iteration_n = [i for i in range(1, len(average_reward) + 1)]

    return iteration_n, average_reward, average_succ


def min_x(df, x):
    """smooth the graphs data using minimum of x points.


    Args:
        df (Dataframe): training logs
        x (int): number of points to get the min of them

    Returns:
        iteration_n(list): contains the x axis data for the graphs (usually n of episodes)
        min_reward(list): contains the reward of episodes after chosing the min
    """
    min_reward = []

    for i in range(0, len(df), x):
        df_slice = df.loc[i : i + x]
        min_reward.append(df_slice["episode_reward"].min())
    iteration_n = [i for i in range(1, len(min_reward) + 1)]

    return iteration_n, min_reward


def max_x(df, x):
    """smooth the graphs data using maximum of x points.


    Args:
        df (Dataframe): training logs
        x (int): number of points to get the max of them

    Returns:
        iteration_n(list): contains the x axis data for the graphs (usually n of episodes)
        max_reward(list): contains the reward of episodes after chosing the max
    """
    max_reward = []
    for i in range(0, len(df), 20):
        df_slice = df.loc[i : i + 20]
        max_reward.append(df_slice["episode_reward"].max())
    iteration_n = [i for i in range(1, len(max_reward) + 1)]

    return iteration_n, max_reward


def train_stats_averaged_x(df, x, name):
    """Plots training graph and save it as svg graph after preproccessing the data using averaging.

    Args:
        df (Dataframe): training logs
        x (int): number of points to average
        name (str): the name to save the graph with (name.svg)

    """
    iteration_n, average_reward, average_succ = average_x(df, x)

    reward_name = name + "reward_graph.svg"
    success_name = name + "success_rate_graph.svg"
    plot_train(
        "# of training episodes",
        "Average reward",
        "Average reward in training",
        iteration_n,
        average_reward,
        reward_name,
    )
    plot_train(
        "# of training episodes",
        "Success rate",
        "Success rate in training",
        iteration_n,
        average_succ,
        success_name,
    )


def train_stats_max_x(df, x, name):
    """Plots training graph and save it as svg graph after preproccessing the data using maximum of x points.

    Args:
        df (Dataframe): training logs
        x (int): number of points to get the max of them
        name (str): the name to save the graph with (name.svg)

    """

    iteration_n, average_reward = max_x(df, x)

    reward_name = name + "reward_graph.svg"
    plot_train(
        "# of training episodes",
        "Average reward",
        "Average reward in training",
        iteration_n,
        average_reward,
        reward_name,
    )


def train_stats_min_x(df, x, name):
    """Plots training graph and save it as svg graph after preproccessing the data using minimum of x points.

    Args:
        df (Dataframe): training logs
        x (int): number of points to get the min of them
        name (str): the name to save the graph with (name.svg)

    """

    iteration_n, average_reward = min_x(df, x)

    reward_name = name + "reward_graph.svg"
    plot_train(
        "# of training episodes",
        "Average reward",
        "Average reward in training",
        iteration_n,
        average_reward,
        reward_name,
    )


def train_stats_smoothed_x(df, weight, name):
    """Plots training graph and save it as svg graph after preproccessing the data using smoothing function.

    Args:
        df (Dataframe): training logs
        weight (float): smoothing factor between 0 and 1
        name (str): the name to save the graph with (name.svg)

    """

    reward = df.reward
    reward_smoothed = smooth(reward, weight)

    df.goal_reached.replace({True: 1, False: 0}, inplace=True)

    goal = df.goal_reached
    goal_smoothed = smooth(goal, weight)
    iteration_n = [i for i in range(1, len(goal_smoothed) + 1)]

    reward_name = name + "reward_graph.svg"
    success_name = name + "success_rate_graph.svg"
    plot_train(
        "# of training episodes",
        "Average reward",
        "Average reward in training",
        iteration_n,
        reward_smoothed,
        reward_name,
    )
    plot_train(
        "# of training episodes",
        "Success rate",
        "Success rate in training",
        iteration_n,
        goal_smoothed,
        success_name,
    )
