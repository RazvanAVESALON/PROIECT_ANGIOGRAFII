
import pandas as pd 
import json 

df = pd.read_csv(r"D:\ai intro\Angiografii\PROIECT_ANGIOGRAFII\experiments\Segmentare\Experiment_Dice_index06122023_1538\Test06152023_1743\CSV_TEST.csv")

print(df['MeanDice'].mean() )# df['Distance'].mean()) 

list_d_2mm=[]
list_d_5mm=[]
list_d_10mm=[] 
list_mean_distance=[]

for d in df['Distance']:
    if len(d)==57 or len(d)==61:
        list_d_2mm.append(0)
        list_d_5mm.append(0)
        list_d_10mm.append(0)
        list_mean_distance.append(50)
    else:
       dr=json.loads(d)
    if len(dr)>1:
        list_d_2mm.append(0)
        list_d_5mm.append(0)
        list_d_10mm.append(0)
        list_mean_distance.append(50)
    if dr[0]<=2:
            
        list_d_2mm.append(1)
        list_mean_distance.append(dr[0])
    if dr[0]<=5:
        list_d_5mm.append(1)
        list_mean_distance.append(dr[0])
                
    if dr[0]<=10:
        list_d_10mm.append(1)
    else :
        list_d_2mm.append(0)
        list_d_5mm.append(0)
        list_d_10mm.append(0)
        list_mean_distance.append(dr[0])
    
print ((sum(list_d_2mm)/len(list_d_2mm))*100,(sum(list_d_5mm)/len(list_d_5mm))*100,(sum(list_d_10mm)/len(list_d_10mm))*100,(sum(list_mean_distance)/len(list_mean_distance)))

        
        
    
    