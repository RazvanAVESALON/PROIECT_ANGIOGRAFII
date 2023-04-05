import pandas as pd
import matplotlib.pyplot as plt 


def histogram_distance(csv,path):
    nr_ccdf=0
    nr_d=0

    dict2={'nr_ccdf_acquisitions':[]}

    for i in csv.index:
        
        if csv['Distance'][i]== "Can't calculate Distance For this frame ( No prediction )":
            nr_ccdf += 1
            dict2['nr_ccdf_acquisitions'].append(csv['Acquistion'][i])
        else: 
            nr_d += 1
            
        

    df2=pd.DataFrame(dict2)
    Statistici= {'Acquisition without Distance':[len(df2['nr_ccdf_acquisitions'].unique())],'Acquisition with Distance':[len(csv['Acquistion'].unique())-len(df2['nr_ccdf_acquisitions'].unique())],'Frames without Distance':nr_ccdf,'Frames with Distance':nr_d}
    Statistici=pd.DataFrame(Statistici)
    Statistici.to_csv(f"{path}\Statistici.csv")

    achizitii= csv['Acquistion'].unique()
    mean_distance_per_acquisition=[]
    for acquisition in achizitii:
        sum=0  
        nr =0
        for i in csv['Acquistion'].index:
            
            if acquisition == csv['Acquistion'][i]:
                if csv['Distance'][i] == "Can't calculate Distance For this frame ( No prediction )":
                    sum += 100
                    nr +=1 
                else: 
                    nr += 1
                    # print (csv['Distance'], type(csv['Distance']) )
                    # distante=csv['Distance'][i].split()
                    # print(distante)
                    for distance in csv['Distance'][i]:
                        print (distance)
                        sum += distance
        avrage=sum/nr
        mean_distance_per_acquisition.append(avrage)
        
    plt.clf()

    plt.hist(mean_distance_per_acquisition, [0,10,20,30,40,50,60,70,80,100])
    plt.title('Mean Distance per Acquisition')
    plt.xlabel('Distance values per interval')
    plt.ylabel("Count of acquisitions per dice value ")
    plt.savefig(f"{path}\\Histograma-Distance-per-Acquisition")


    
    
    
                    
                        
                                      







    