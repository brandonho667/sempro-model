from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from PIL import Image
import torch

class semDataset(Dataset):
    def __init__(self,filepath="SEM_Final.xlsx", transform = None):
        sem_df=pd.read_excel("SEM_Final.xlsx")
        parsed_sem_df = sem_df[sem_df['measurement'].notna()]
        parsed_sem_df = parsed_sem_df[parsed_sem_df['SEM_img'].notna()]
        # print(parsed_sem_df['skip'].isna().sum())
        parsed_sem_df = parsed_sem_df[parsed_sem_df['skip'].isna()]
        parsed_sem_df['img_path'] = parsed_sem_df.apply(lambda x: os.path.join(x['SEM'],x['SEM_img']),axis=1)
        # print(parsed_sem_df.shape)
        parsed_sem_df['measurement']= parsed_sem_df['measurement'].apply(self.convert_measurement)
        self.sem_df = parsed_sem_df
        self.transform = transform
    
    def __len__(self):
        return self.sem_df.shape[0]
    
    def convert_measurement(self,string):
        string.replace(" ","")
        kpa = ["KPa","kPa", "Kpa", "kpa","KpA"]
        mpa = ["Mpa","MPa"]
        gpa = ["GPa","Gpa"]

        if any(KPA in string for KPA in kpa):
            for KPA in kpa:
                string = string.replace(KPA,"")
            string = string.strip()
            value = float(string)
            value *= 1000 
        elif any(MPA in string for MPA in mpa):
            for MPA in mpa:
                string = string.replace(MPA,"")
            string = string.strip()
            value = float(string)
            value *= 1000000
        elif any(GPA in string for GPA in gpa):
            for GPA in gpa:
                string = string.replace(GPA,"")
            string = string.strip()
            value = float(string)
            value *= 1000000000
        elif "Pa" in string:
            string = string.replace("Pa","")
            string = string.strip()
            value = float(string)
        else:
            print(string)
            return
        return np.log10(value)
    def __getitem__(self,index):
        image = Image.open(self.sem_df.iloc[index]["img_path"])
        label = self.sem_df.iloc[index]["measurement"]
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label

if 1==0:
    semdata = semDataset(transform=transform)
    print(f"Length: {len(semdata)}")
    image, label=semdata.__getitem__(0)
    plt.imshow(image.permute(1,2,0))