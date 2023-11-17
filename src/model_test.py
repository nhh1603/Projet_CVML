import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torchvision import models, transforms, DataLoader

import test_data

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'isic-2020-resized'))
test_dataset = test_data.CustomDataset(data_dir=data_dir, transform=data_transforms)
dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Lists to store predictions and corresponding file names
predictions = []
file_names = []

if __name__ == '__main__':
    model = models.resnet18()
    
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    
    with torch.no_grad():
        for batch in unlabeled_data_loader:
            inputs = batch['image']  # Assuming 'image' is the key for images in your dataset
            # Add any necessary preprocessing steps here if not included in the DataLoader
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Assuming a classification task
            predictions.extend(predicted.cpu().numpy())
            file_names.extend(batch['file_name'])  # Replace 'file_name' with the actual key in your dataset

    # Create a DataFrame to store results
    results_df = pd.DataFrame({'file_name': file_names, 'predicted_class': predictions})
    
    # Save the results to a CSV file
    results_df.to_csv('path_to_results.csv', index=False)