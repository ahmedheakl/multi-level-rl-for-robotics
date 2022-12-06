import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_train(x_label,y_label,graph_title,x1,y1,name):
    fig = plt.figure() 
    fig.set_size_inches(10, 5)    
    plt.plot(x1, y1)
    
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(graph_title)
    
    plt.savefig(name,bbox_inches='tight',pad_inches =0.1)
    #plt.show()



def smooth(x, weight):
    """ Weight between 0 and 1 """
    last = x[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in x:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


def average_x(df,x):
    average_reward = [] 
    average_r_per_x = 0
    for i in range(len(df)):
        average_r_per_x= average_r_per_x + df.loc[i].at['episode_reward']
        if i%x == 0 : 
            average_reward.append(average_r_per_x/x)
            average_r_per_x =0
    average_succ = [] 
    average_s_per_x = 0
    for i in range(len(df)):
        if df.loc[i].at['goal_reached'] == True :
            average_s_per_x += 1
        if i%x == 0 : 
            average_succ.append(average_s_per_x/x)
            average_s_per_x =0
    iteration_n =[ i for i in range(1,len(average_reward)+1)]
    
    return iteration_n,average_reward,average_succ


def min_x(df,x):
    min_reward = [] 
    for i in range(0,len(df),20):
        df_slice = df.loc[i:i+20]
        min_reward.append(df_slice['episode_reward'].min())
    iteration_n =[ i for i in range(1,len(min_reward)+1)]


    return iteration_n,min_reward

def max_x(df,x):
    max_reward = [] 
    for i in range(0,len(df),20):
        df_slice = df.loc[i:i+20]
        max_reward.append(df_slice['episode_reward'].max())
    iteration_n =[ i for i in range(1,len(max_reward)+1)]
    
    return iteration_n,max_reward

def train_stats_averaged_x (df,x,name):
    iteration_n,average_reward,average_succ = average_x(df,x)

    reward_name = name + 'reward_graph.svg'
    success_name = name + 'success_rate_graph.svg'
    plot_train('# of training episodes','Average reward','Average reward in training'
               ,iteration_n,average_reward,reward_name)
    plot_train('# of training episodes','Success rate','Success rate in training'
               ,iteration_n,average_succ,success_name)

def train_stats_averaged_x (df,x,name):
    iteration_n,average_reward,average_succ = average_x(df,x)

    reward_name = name + 'reward_graph.svg'
    success_name = name + 'success_rate_graph.svg'
    plot_train('# of training episodes','Average reward','Average reward in training'
               ,iteration_n,average_reward,reward_name)
    plot_train('# of training episodes','Success rate','Success rate in training'
               ,iteration_n,average_succ,success_name)

def train_stats_max_x (df,x,name):
    iteration_n,average_reward = max_x(df,x)

    reward_name = name + 'reward_graph.svg'
    plot_train('# of training episodes','Average reward','Average reward in training'
               ,iteration_n,average_reward,reward_name)


def train_stats_min_x (df,x,name):
    iteration_n,average_reward = min_x(df,x)

    reward_name = name + 'reward_graph.svg'
    plot_train('# of training episodes','Average reward','Average reward in training'
               ,iteration_n,average_reward,reward_name)

def train_stats_smoothed_x (df,x,name):
    
    reward = df.reward
    reward_smoothed = smooth(reward,x)
    
    df.goal_reached.replace({True: 1, False: 0},inplace = True)

    goal = df.goal_reached
    goal_smoothed = smooth(goal,.99)
    iteration_n =[ i for i in range(1,len(goal_smoothed)+1)]

    reward_name = name + 'reward_graph.svg'
    success_name = name + 'success_rate_graph.svg'
    plot_train('# of training episodes','Average reward','Average reward in training'
               ,iteration_n,reward_smoothed,reward_name)
    plot_train('# of training episodes','Success rate','Success rate in training'
               ,iteration_n,goal_smoothed,success_name)
    