import torch
from torch.utils.data import Dataset, random_split, Subset
import os
import numpy as np
import pandas as pd
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}    

class CNN_Dataset(Dataset):
    def __init__(self, dir, year, ret_days=5):
        images_path = os.path.join(dir, f'20d_month_has_vb_[20]_ma_{year}_images.dat')
        images = np.memmap(images_path, dtype=np.uint8, mode='r')
        images = images.reshape((-1,1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20]))

        labels_path = os.path.join(dir, f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather')
        labels_df = pd.read_feather(labels_path)
        labels = labels_df[f'Ret_{ret_days}d']

        missing = labels.isna() 
        self.images = torch.tensor(images[~missing], dtype=torch.float)
        self.labels = torch.tensor(labels[~missing]>0, dtype=torch.long)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
def get_years_dataset(dir, year_start, year_end, ret_days=5):
    dataset = []
    for year in range(year_start, year_end):
        dataset = torch.utils.data.ConcatDataset([dataset, CNN_Dataset(dir, year, ret_days)])
    print(f"[INFO]Length of {year_start}-{year_end-1} datasets: {len(dataset)}") 
    return dataset

def train_val_split(dataset, ratio=0.7, generator=None, chronological=False):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    if chronological:
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, len(dataset)))
        return Subset(dataset, train_indices), Subset(dataset, val_indices)
    else:
        return random_split(dataset, [train_size, val_size], generator=generator)
