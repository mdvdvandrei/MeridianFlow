import torch
from DataSet import MyDataset
from models.models import resnext50_32x4d, mobilenet_v2
import os 
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm  
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset



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
    batch_size=1, val_mark=True)

mean = 8619244861.558443
std = 4320463472.088931

num_batches = len(dataset_test)  # Total number of batches in the dataset

# Initialize tqdm with the total number of iterations
# Set model to evaluation mode
model.eval()

print(len(dataset_test))
# Initialize tqdm with the total number of iterations
pbar = tqdm(total=len(dataset_test), desc='Processing images', unit='image')

dataset_iter = iter(dataset_test)  # Create an iterator from the dataset

total_gradients = None
total_y_gradients = None


for _ in range(len(dataset_test)):
    try:
        batch_data, batch_y = next(dataset_iter)
    except StopIteration:
        break  # Stop the loop if the iterator is exhausted
    
    data = Variable(torch.from_numpy(batch_data).float().to(device), requires_grad=True)
    y = (batch_y - mean) / std
    y = Variable(torch.from_numpy(y).float().to(device), requires_grad=True)

    # Forward pass
    output = model(data)

    y_grads = torch.autograd.grad(output, inputs=data, create_graph=False)[0]

    # Accumulate gradients with respect to the input for y

    if total_y_gradients is None:
        total_y_gradients = y_grads.clone().detach().sum(dim=0)
    else:
        total_y_gradients += y_grads.clone().detach().sum(dim=0)

    model.zero_grad()

    pbar.update(1)  # manually update the progress bar

# Close the progress bar
pbar.close()


# For y gradients
average_y_gradients = total_y_gradients / len(dataset_test)
reshaped_y_gradients = (average_y_gradients.cpu().numpy().reshape((18, 402, 934)))
#normalized_y_gradients = (reshaped_y_gradients - reshaped_y_gradients.min()) / (reshaped_y_gradients.max() - reshaped_y_gradients.min())
normalized_y_gradients = reshaped_y_gradients


with Dataset('/storage/kubrick/amedvedev/data/NNATL-12/coordinates.nc', 'r') as ds:
    lons = ds.variables['nav_lon'][:]
    lats = ds.variables['nav_lat'][:]


def visualize_data( input_y_array, lats, lons, output_path):
    """
    Visualize and save input data and y gradients using Basemap.

    Parameters:
    input_y_array (numpy.array): Input data array with shape (channels, height, width) representing gradients for the target y.
    lats (numpy.array): 2D array of latitudes with shape (height, width).
    lons (numpy.array): 2D array of longitudes with shape (height, width).
    output_path (str): Path to save the output images.
    """

    m = Basemap(width=10_000_000,height=4_000_000,
            resolution='l',projection='eqdc',
            lat_1=40., lat_2=65,
            lat_0=60, lon_0=-30.)

    fig, axs = plt.subplots(6, 3, figsize=(15, 20), dpi=200)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    h = [1.5, 3.8, 6.5, 9.8, 2262, 3138]
    names = ['u', 'sin(phi)', 'cos(phi)']
    for channel in range(input_y_array.shape[0]):
        ax = axs[channel // 3, channel % 3]
        name = names[channel % 3] + ' на глубине ' + str(h[channel // 3]) + ' м'
        ax.set_title(name)

        m.ax = ax 

        m.drawcoastlines()
        m.fillcontinents(color='tan')


        m.drawparallels(np.arange(-80., 81., 10.), labels=[0,0,0,0], fontsize=10)
        m.drawmeridians(np.arange(-180., 181., 30.), labels=[0,0,0,0], fontsize=10)

        x, y = m(lons, lats)
        data = input_y_array[channel]

        c = m.contourf(x, y, data, cmap="RdBu_r")
        ax.text(0.05, 0.95, f"{channel + 1}", transform=ax.transAxes, fontsize=14, va='top', ha='left', color='black', weight='bold', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round4'))


    plt.tight_layout()
    plt.savefig(f'{output_path}/y_gradients_combined.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()


    fig, axs = plt.subplots(6, 3, figsize=(15, 20), dpi=200)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    h = [1.5, 3.8, 6.5, 9.8, 2262, 3138]
    names = ['u', 'sin(phi)', 'cos(phi)']

    for channel in range(input_y_array.shape[0]):
        ax = axs[channel // 3, channel % 3]
        name = names[channel % 3] + ' на глубине ' + str(h[channel // 3]) + ' м'
        ax.set_title(name)
        
        data = input_y_array[channel].flatten()
        
        ax.hist(data, bins=50, color='blue', alpha=0.7)
        ax.set_xlabel('Gradient Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ax.text(0.05, 0.95, f"{channel + 1}", transform=ax.transAxes, fontsize=14, va='top', ha='left', color='black', weight='bold', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round4'))
        


    plt.tight_layout()
    plt.savefig(f'{output_path}/y_gradients_histograms.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()

    #all_y_grads = (np.sum(reshaped_y_gradients, axis=0) - np.sum(reshaped_y_gradients, axis=0).min()) / (np.sum(reshaped_y_gradients, axis=0).max() - np.sum(reshaped_y_gradients, axis=0).min())
    

    all_y_grads =  np.sum(input_y_array, axis=0)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    m.drawcoastlines()
    m.fillcontinents(color='tan')
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1,1,0,0], fontsize=10)  # labels=[left,right,top,bottom]
    m.drawmeridians(np.arange(-180., 181., 30.), labels=[0,0,0,1], fontsize=10)
    x, y = m(lons, lats)
    data = all_y_grads
    
    c = m.contourf(x, y, data, cmap="RdBu_r")
    
    plt.savefig(f'{output_path}/y_gradients_image_normalized_sum.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# Usage example
visualize_data(normalized_y_gradients, lats, lons, '//app/amedvedev/scripts/data/graphs')
