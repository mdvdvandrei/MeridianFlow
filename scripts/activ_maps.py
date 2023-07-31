import os
import matplotlib.pyplot as plt
import torch
from torchvision import models
from torch.utils.data import DataLoader
from models.models import resnext50_32x4d, shufflenet_small
from DataSet import MyDataset
import numpy as np

# Load the pre-trained resnext50_32x4d model
model = resnext50_32x4d()
checkpoints_basepath = "/app/amedvedev/data/models"
filename = "resnext50_32x4d_run0009/best_model.pt"
filepath = os.path.join(checkpoints_basepath, filename)

# Load the saved model state dictionary
state_dict = torch.load(filepath)

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.eval()

# Dictionary to store activations of convolutional layers
activations = {}

# Define a hook function to store activations of each convolutional layer
def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register the hook function for each convolutional layer in the model
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        layer.register_forward_hook(get_activations(name))

# Create a generator function for the custom dataset
def my_dataset_gen(dataset):
    for data in dataset:
        yield data

# Prepare DataLoader for the custom dataset
dataset_test = MyDataset(
    root_dir='/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d',
    batch_size=1, val_mark=True)

dataset_test_gen = my_dataset_gen(dataset_test)

# Get an image and its label from the DataLoader
img_t, _ = next(dataset_test_gen)
img_t = torch.from_numpy(img_t)

# Run the image through the model and get the output (activations)
with torch.no_grad():
    out = model(img_t)

# Define the folder path to save the generated plots
folder_path = '/app/amedvedev/data/plots/resnext50_32x4d_run0009'

# Create the folder if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Visualize activation maps and their distributions for each convolutional layer
for name, act in activations.items():
    act = act.squeeze().cpu().numpy()
    random_map = act[np.random.randint(act.shape[0])]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(name)
    plt.imshow(random_map, cmap='viridis')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Distribution of {name}')
    plt.hist(random_map.ravel(), bins=50, color='blue', alpha=0.7)
    
    # Save the plot as an image file
    plt.savefig(os.path.join(folder_path, f'{name.replace(".", "_")}_distribution.png'))
    plt.close()
