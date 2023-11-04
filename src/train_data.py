import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'train-resized')
        self.label_dir = os.path.join(data_dir, 'train-labels.csv')
        self.img_labels = pd.read_csv(self.label_dir)
        # self.image_filenames = os.listdir(self.image_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.jpg')
        img = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
            
        return img, label
    

