from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from utils import *
from scipy.ndimage import convolve1d


class semDataset(Dataset):
    def __init__(self, filepath="SEM_Final.xlsx", transform=None,
                 reweight='sqrt_inv', lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        sem_df = pd.read_excel("SEM_Final.xlsx")
        parsed_sem_df = sem_df[sem_df['measurement'].notna()]
        parsed_sem_df = parsed_sem_df[parsed_sem_df['SEM_img'].notna()]
        # print(parsed_sem_df['skip'].isna().sum())
        parsed_sem_df = parsed_sem_df[parsed_sem_df['skip'].isna()]
        parsed_sem_df['img_path'] = parsed_sem_df.apply(
            lambda x: os.path.join(x['SEM'], x['SEM_img']).replace("\\", "/"), axis=1)
        # print(parsed_sem_df.shape)
        parsed_sem_df['measurement'] = parsed_sem_df['measurement'].apply(
            self.convert_measurement)
        self.sem_df = parsed_sem_df
        self.transform = transform
        self.weights = self._prepare_weights(
            reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return self.sem_df.shape[0]

    def convert_measurement(self, string):
        string.replace(" ", "")
        kpa = ["KPa", "kPa", "Kpa", "kpa", "KpA"]
        mpa = ["Mpa", "MPa"]
        gpa = ["GPa", "Gpa"]

        if any(KPA in string for KPA in kpa):
            for KPA in kpa:
                string = string.replace(KPA, "")
            string = string.strip()
            value = float(string)
            value *= 1000
        elif any(MPA in string for MPA in mpa):
            for MPA in mpa:
                string = string.replace(MPA, "")
            string = string.strip()
            value = float(string)
            value *= 1000000
        elif any(GPA in string for GPA in gpa):
            for GPA in gpa:
                string = string.replace(GPA, "")
            string = string.strip()
            value = float(string)
            value *= 1000000000
        elif "Pa" in string:
            string = string.replace("Pa", "")
            string = string.strip()
            value = float(string)
        else:
            print(string)
            return
        return np.log10(value)

    def __getitem__(self, index):
        image = Image.open(self.sem_df.iloc[index]["img_path"])
        label = self.sem_df.iloc[index]["measurement"]
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label)
        weight = np.asarray(self.weights[index]).astype(
            'float32') if self.weights is not None else np.asarray(np.float32(1.))
        return image, label, weight

    def _prepare_weights(self, reweight, max_target=51, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.sem_df["measurement"].tolist()
        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            # clip weights for inverse re-weight
            value_dict = {k: np.clip(v, 5, 1000)
                          for k, v in value_dict.items()}
        num_per_label = [
            value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(
                lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [
                smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


if 1 == 0:
    semdata = semDataset(transform=transform)
    print(f"Length: {len(semdata)}")
    image, label = semdata.__getitem__(0)
    plt.imshow(image.permute(1, 2, 0))
