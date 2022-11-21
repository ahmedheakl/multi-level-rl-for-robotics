from highrl.graphs.model_training_plot_styles import train_stats_smoothed_x
import pandas as pd
import os  

log_path = input("please Enter The logs path: ")
save_path = input("please enter the path to save figures: ")
x = float(input("please enter the smoothing factor: "))
models_log_dir = os.path.expanduser(log_path)
save_graph_dir = os.path.expanduser(save_path)

#Training log Paths >> CSV
models_log_files = sorted(os.listdir(models_log_dir))
models_log_files_pathes = [os.path.join(models_log_dir, file) for file in models_log_files if file.endswith(".csv")]
files_names = [ file.split(".csv")[0] for file in models_log_files if file.endswith(".csv")]

#print(models_log_files_pathes)
for i,path in enumerate(models_log_files_pathes):
    print("generating graphs for file : ",  files_names[i])
    train_df =pd.read_csv(path,index_col=0)
    name = files_names[i]+"_trail_{}_smoothed_{}".format(i+1,int(x*100))
    name = os.path.join(save_graph_dir, name)
    train_stats_smoothed_x (train_df,x,name)



 