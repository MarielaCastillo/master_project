from operator import getitem
from turtle import pd
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import pathlib

class ThermalDataset(Dataset):
    def __init__(self, dataset_url, transform = False) -> None:
        super().__init__()
        self.dataset_url = dataset_url

        


        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) :
        
        return super().__getitem__(index)
        

