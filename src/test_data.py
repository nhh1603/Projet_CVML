import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'test-resized')
        self.img_names = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)
            
        return img
    

