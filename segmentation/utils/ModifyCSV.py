import pandas as pd 
import yaml
import pathlib as pt 

def angio_modify (df):
 
    print (df , df.shape)
    angioloader_list=[]
    for ind  in df.index : 
        path = df['images_path'][ind]
        p = pt.Path(path).parents[0]
        angio_loader_path=pt.Path(p)/"angio_loader_header.json"
        
        angioloader_list.append(angio_loader_path)
    df["angio_loader_header"]=angioloader_list
    return df 
    
       
      

def main():
    
    config = None
    with open('config.yaml') as f:  # reads .yml/.yaml files
        config = yaml.safe_load(f)
        
    df=pd.read_csv(config['data']['dataset_csv']) 
    new=angio_modify(df)
    new.to_csv(config['data']['dataset_csv'])
    


if __name__ == "__main__":
    main()