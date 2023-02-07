import random

import glob
import os
import json


def create_dataset_csv(path_construct):
    path_list = {"images_path": [], "annotations_path": [], "frames": [
    ], "patient": [], "acquisition": [], "angio_loader_header": []}
    # frame_list={"frames"}

    for patient in path_construct:
        #print (image)
        # x=os.path.join(image,r"*")

        x = glob.glob(os.path.join(patient, r"*"))
        # print (x)
        for acquisiton in x:
            img = os.path.join(acquisiton, "frame_extractor_frames.npz")
            annotations = os.path.join(acquisiton, "clipping_points.json")
            angio_leader = os.path.join(acquisiton, "angio_loader_header.json")
            with open(annotations) as f:
                clipping_points = json.load(f)

            for frame in clipping_points:
                frame_int = int(frame)

                path_list['images_path'].append(img)
                path_list['annotations_path'].append(annotations)
                path_list['frames'].append(frame_int)
                path_list['patient'].append(os.path.basename(patient))
                path_list['acquisition'].append(os.path.basename(acquisiton))
                path_list['angio_loader_header'].append(angio_leader)

    return path_list


def split_dataset(dataset_df, split_per, seed=1):
    """Impartirea setului de date in antrenare, validare si testare in mod aleatoriu

    Args:
        dataset_df (pandas.DataFrame): contine caile catre imaginile de input si mastile de segmentare
        split_per (dict): un dictionare de forma {"train": float, "valid": float, "test": float} ce descrie
            procentajele pentru fiecare subset
        seed (int, optional): valoarea seed pentru reproducerea impartirii setului de date. Defaults to 1.
    """
    # se amesteca aleatoriu indecsii DataFrame-ului
    # indexul este un numar (de cele mai multe ori) asociat fiecarui rand

    patients = dataset_df['patient'].unique()

    total = len(patients)

    random.seed(seed)
    random.shuffle(patients)

    # se impart indecsii in functie de procentele primite ca input
    train_idx = int(total * split_per["train"])
    valid_idx = train_idx + int(total * split_per["valid"])
    test_idx = train_idx + valid_idx + int(total * split_per["test"])

    train_patients = patients[:train_idx]
    valid_patients = patients[train_idx:valid_idx]
    test_patients = patients[valid_idx:test_idx]

    dataset_df['subset'] = ""

    dataset_df.loc[dataset_df['patient'].isin(
        train_patients), 'subset'] = 'train'
    dataset_df.loc[dataset_df['patient'].isin(
        valid_patients), 'subset'] = 'valid'
    dataset_df.loc[dataset_df['patient'].isin(
        test_patients), 'subset'] = 'test'

    return dataset_df
