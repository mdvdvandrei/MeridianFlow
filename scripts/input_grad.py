import torch
from DataSet import MyDataset
from models.models import resnext50_32x4d, mobilenet_v2
import os 
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnext50_32x4d()
checkpoints_basepath = "/app/amedvedev/data/models/resnext50_32x4d_run0030"

filename = "best_model.pt"
filepath = os.path.join(checkpoints_basepath, filename)

# Load the saved model state dictionary
state_dict = torch.load(filepath, map_location=device)

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.to(device)  # Move the model to the GPU if available

loss_fn = torch.nn.MSELoss()

dataset_test = MyDataset(
    root_dir='/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d',
    batch_size=8, val_mark=True)

mean = 8619244861.558443
std = 4320463472.088931

num_batches = len(dataset_test)  # Total number of batches in the dataset

# Initialize tqdm with the total number of iterations
# Set model to evaluation mode
model.eval()

# Initialize tqdm with the total number of iterations
pbar = tqdm(total=len(dataset_test), desc='Processing batches', unit='batch')

dataset_iter = iter(dataset_test)  # Create an iterator from the dataset

total_gradients = None

for _ in range(len(dataset_test)):
    try:
        batch_data, batch_y = next(dataset_iter)
    except StopIteration:
        break  # Stop the loop if the iterator is exhausted
    
    data = Variable(torch.from_numpy(batch_data).float().to(device), requires_grad=True)
    y = (batch_y - mean) / std
    y = Variable(torch.from_numpy(y).float().to(device))

    # Forward pass
    output = model(data)

    loss = loss_fn(output, y.view(-1, 1))

    # Backward pass
    model.zero_grad()  # Clear existing gradients
    loss.backward()

    # Accumulate gradients with respect to the input

    # Accumulate gradients with respect to the input
    if total_gradients is None:
        total_gradients = data.grad.clone().detach().sum(dim=0)
    else:
        total_gradients += data.grad.clone().detach().sum(dim=0)


    pbar.update(1)  # manually update the progress bar

# Close the progress bar
pbar.close()

# Average gradients
average_gradients = total_gradients / len(dataset_test)

# Reshape gradients to the desired shape
reshaped_gradients = abs(average_gradients.cpu().numpy().reshape((18, 402, 934)))

# Normalize the image
normalized_gradients = (reshaped_gradients - reshaped_gradients.min()) / (reshaped_gradients.max() - reshaped_gradients.min())

# Plot and save the image
for channel in range(18):
    plt.imshow(normalized_gradients[channel])
    print(normalized_gradients.min(), normalized_gradients.max())
    plt.axis('off')  # Optional: Turn off axis
    plt.savefig(f'//app/amedvedev/scripts/data/graphs/gradients_image_normalized_{channel}.png', bbox_inches='tight', pad_inches=0)  # Save the image


all_grads = (np.sum(normalized_gradients, axis = 0 ) - np.sum(normalized_gradients, axis = 0 ).min()) / (np.sum(normalized_gradients, axis = 0 ).max() - np.sum(normalized_gradients, axis = 0 ).min())

plt.imshow(all_grads)
plt.axis('off')  # Optional: Turn off axis
plt.savefig(f'//app/amedvedev/scripts/data/graphs/gradients_image_normalized_mean.png', bbox_inches='tight', pad_inches=0)  # Save the image