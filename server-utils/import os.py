import os


def modify(df):
    print(df.head())
    df_2 = df.drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0',
                   'images_path', 'annotations_path', 'angio_loader_header'])
    image = []
    anot = []
    angio = []
    frames = []
    path_construct = (
        "/media/cuda/HDD 1TB  - DATE/AvesalonRazvanDate , Experimente/data/*")
    path_construct2 ="/media/cuda/HDD 1TB - DATE/AvesalonRazvanDate, Experimente/data
    patients = df['patient'].unique()
    for patient in patients:
        #print (image)
        # x=os.path.join(image,r"*")
        x = glob.glob(os.path.join(os.path.join(
            path_construct2, f'{patient}'), '*'))
        # print (x)
        for acquisiton in x:
            img = os.path.join(acquisiton, "frame_extractor_frames.npz")
            annotations = os.path.join(acquisiton, "clipping_points.json")
            angio_leader = os.path.join(acquisiton, "angio_loader_header.json")
            with open(annotations) as f:
                clipping_points = json.load(f)

            for frame in clipping_points:
                frame_int = int(frame)
                image.append(img)
                anot.append(annotations)
                angio.append(angio_leader)
    df_2['images_path'] = image
    df_2['annotations_path'] = anot
    df_2['angio_loader_header'] = angio_leader

    return df_2
