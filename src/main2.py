import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights

import train_data
# model
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# train_dataset = CustomDataset(data_dir=f'{os.getcwd()}\isic-2020-resized')

# print(train_dataset.__getitem__(0))

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'isic-2020-resized'))
custom_train_dataset = train_data.CustomDataset(data_dir=data_dir, transform=data_transforms)

# Define the size of the training and validation sets
train_size = int(0.8 * len(custom_train_dataset))
val_size = len(custom_train_dataset) - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(custom_train_dataset, [train_size, val_size])

# train_dataset = datasets.ImageFolder(root=f'{os.getcwd()}\isic-2020-resized\train_resized', transform=data_transforms)
# test_dataset = datasets.ImageFolder(root=f'{os.getcwd()}\isic-2020-resized\test_resized', transform=data_transforms)

# data_dir = f'{os.getcwd()}\isic-2020-resized'
# data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'isic-2020-resized'))
# train_dataset = train_data.CustomDataset(data_dir=data_dir, transform=data_transforms)
# test_dataset = test_data.CustomDataset(data_dir=data_dir, transform=data_transforms)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=64, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=64, shuffle=True)
}

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Replace the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Define loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            iter = 0
            print(f'Starting {phase} phase')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print('*')
                inputs = inputs.to(device)
                labels = labels.to(device)
                iter += 1
                print(iter, inputs.shape, labels.shape)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=5)
    torch.save(model.state_dict(), 'model_weights.pth')