import torch
from DataSet import MyDataset
from models.models import resnext50_32x4d, mobilenet_v2
import os 
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

model = mobilenet_v2()
checkpoints_basepath = "/storage/kubrick/amedvedev/data/models"
epoch = 9

filename = 'model_ep{:04d}.pt'.format(epoch)
filepath = os.path.join(checkpoints_basepath, filename)

# Load the saved model state dictionary
state_dict = torch.load(filepath)

# Load the state dictionary into the model
model.load_state_dict(state_dict)
loss_fn = torch.nn.MSELoss()
dataset = MyDataset('/storage/kubrick/amedvedev/data/dataset/train', batch_size=1, shuffle=True)
for batch_data, batch_y in dataset:
    data = batch_data
    y = batch_y
    break

data = Variable(torch.from_numpy(data).float(), requires_grad=True)  # Set requires_grad=True
mean = 8619244861.558443
std = 4320463472.088931
y = (y - mean) / std
y = Variable(torch.from_numpy(y).float())

# Forward pass
output = model(data)
loss = loss_fn(output, y.view(-1, 1))

# Backward pass
loss.backward()

# Get the gradients of the input and print its shape
gradients = data.grad  # Use data.grad instead of input.grad

# Reshape gradients to the desired shape
reshaped_gradients = np.array(gradients).reshape((3, 402, 934))

# Plot and save the image
plt.imshow(reshaped_gradients[0])
print(reshaped_gradients.min(),reshaped_gradients.max())
plt.axis('off')  # Optional: Turn off axis
plt.savefig('/storage/kubrick/amedvedev/data/graphs/gradients_image.png', bbox_inches='tight', pad_inches=0)  # Save the image