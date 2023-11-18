import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

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

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

# Lists to store predictions and corresponding file names
predictions = []
file_names = []

if __name__ == '__main__':
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for inputs, names in dataloader:
            # Add any necessary preprocessing steps here if not included in the DataLoader
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Assuming a classification task
            # probs = softmax(outputs, dim=1)  
            predictions.extend(predicted.cpu().numpy())
            file_names.extend(names)  # Replace 'file_name' with the actual key in your dataset
    
    # Create a DataFrame to store results
    results_df = pd.DataFrame({'file_name': file_names, 'predicted_class': predictions})
    
    # Save the results to a CSV file
    results_df.to_csv('submission.csv', index=False)