import os
from train_data import CustomDataset

train_dataset = CustomDataset(data_dir=f'{os.getcwd()}\isic-2020-resized')

print(train_dataset.__getitem__(0))
