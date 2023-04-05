import pandas as pd 
import numpy as np 


df=pd.read_csv(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\experiments\exp 3.04\Experiment_MSE04012023_1445\Test04032023_1601\CSV_TEST.csv")
print(df['MeanDice'].mean(),df['MeanDIstance'].mean() ) 
