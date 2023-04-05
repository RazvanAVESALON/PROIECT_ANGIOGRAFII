import pandas as pd 
import matplotlib.pyplot as plt 

def hist_metrics_per_acq(dataf,path):

    mean_mse= dataf['MeanMSE'].unique()
    mean_distance=dataf['MeanDIstance'].unique()

    plt.hist(mean_mse, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Frames per Image')
    plt.xlabel('MSE Values per interval')
    plt.ylabel("Count of Acquisition per MSE value ")
    plt.savefig(f"{path}\\Histograma_mse_per_ac")

    plt.clf()

    plt.hist(mean_distance, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Frames per Image')
    plt.xlabel('Distance Values per interval')
    plt.ylabel("Count of Acquisition per Distance value ")
    plt.savefig(f"{path}\\Histograma_distance_per_ac")