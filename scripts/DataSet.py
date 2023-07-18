import threading
from sklearn.utils import shuffle
import numpy as np
from netCDF4 import Dataset
import os
from data_preproc import time_from_path, data_preproc
import datetime
from scipy.ndimage import gaussian_filter, affine_transform
from skimage.transform import rotate
import random
from skimage.transform import AffineTransform, warp


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator."""
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_objects_i(objects_count):
    """Cyclic generator of paths indices"""
    current_objects_id = 0
    while True:
        yield current_objects_id
        current_objects_id = (current_objects_id + 1) % objects_count
        

class MyDataset():
    """Dataset for handling large volumes of netCDF data."""
    
    def __init__(self, root_dir, batch_size = 32, val_mark = False, shuffle = True):
        self.root_dir = root_dir
        self.val_mark = val_mark
        self.file_list = self.read_file_list(root_dir)
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.objects_id_generator = threadsafe_iter(get_objects_i(len(self.file_list)))

        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.init_count = 0

    def __len__(self):
        return len(self.file_list)

    def read_file_list(self, root_dir):
        file_list = []
        if self.val_mark == False:
            for subdir, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('gridV.nc') and time_from_path(file) > datetime.datetime(1993, 2, 1, 0, 0) and time_from_path(file) < datetime.datetime(1996, 1, 1, 0, 0):
                        file_list.append(os.path.join(subdir, file))
        else:
            for subdir, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('gridV.nc') and time_from_path(file) > datetime.datetime(1996, 1, 1, 0, 0):
                        file_list.append(os.path.join(subdir, file))
            
        return file_list
    

    def read_my_data(self, fname):
        # Here we open the .nc file and return the Dataset object
        #nc = Dataset(fname, "r")

        return fname

    def get_data_by_id(self, id):
        # Here we get our data by the ID: index, filename, etc.

        filename = self.file_list[id]
        x_item, mask, y_item = data_preproc(self.read_my_data(filename))


        return x_item, mask, y_item



    def __iter__(self):
        while True:
            with self.lock:
                if (self.init_count == 0):
                    self.shuffle_data()
                    self.batch_data = []
                    self.batch_y = []
                    self.init_count = 1

            for obj_id in self.objects_id_generator:
                curr_data_x, curr_data_mask, curr_data_y = self.get_data_by_id(obj_id)

                means = [0.05746843,0.006585821,0.0285264,0.056652255,0.003588926,0.027109021,0.0559087,
                0.0014212164,0.02554415,0.055143178,-0.00026903558,0.023744814,0.052508943,-0.0024550243,0.018736638,
                0.013417877,-0.0073913042,0.0035079382] 
                std_devs = [0.10220336,0.46970904,0.45765448,0.101526916,0.4696368,0.45784783,0.10080856,
                0.4697133,0.45787156,0.099938385,0.46996266,0.4576818,0.097211756,0.4658524,0.4521642,0.044967487,0.33513236,0.32524768]

                 # Normalize each channel in the image
                for i in range(18):
                    curr_data_x[i, :, :] = (curr_data_x[i, :, :] - means[i]) / std_devs[i]

                if self.val_mark == False:
                    # Apply Gaussian noise
                    noise = np.random.normal(0, 0.01, curr_data_x.shape)
                    curr_data_x = curr_data_x + noise
                    _,rows,cols = curr_data_x.shape

                    # Small shift
                    shift_x = random.randint(-10, 10)  # Random shift in the x-axis
                    shift_y = random.randint(-10, 10)  # Random shift in the y-axis
                    transform_shift = AffineTransform(translation=(shift_x, shift_y))
                    curr_data_x_shifted = warp(curr_data_x, transform_shift.inverse)
                    curr_data_mask_shifted = warp(curr_data_mask, transform_shift.inverse)

                    # Zoom
                    zoom_factor = random.uniform(0.01, 1.01)  # Random zoom factor between 0.01 and 1.01
                    transform_zoom = AffineTransform(scale=(zoom_factor, zoom_factor))
                    curr_data_x_zoomed = warp(curr_data_x_shifted, transform_zoom.inverse)
                    curr_data_mask_zoomed = warp(curr_data_mask_shifted, transform_zoom.inverse)


                    # Small rotation
                    angle = random.uniform(-2, 2)  # Random angle between -10 and 10 degrees
                    # Rearrange axes for rotation. Here we want to rotate around axes 1 and 2.
                    curr_data_mask = np.expand_dims(curr_data_mask, axis=0)

                    curr_data_x = np.swapaxes(curr_data_x, 0, 1)
                    curr_data_x = np.swapaxes(curr_data_x, 1, 2)
                    curr_data_mask = np.swapaxes(curr_data_mask, 0, 1)
                    curr_data_mask = np.swapaxes(curr_data_mask, 1, 2)

                    # Perform rotation
                    curr_data_x = rotate(curr_data_x, angle, mode='reflect')
                    curr_data_mask = rotate(curr_data_mask, angle, mode='reflect')

                    # Swap axes back to original order
                    curr_data_x = np.swapaxes(curr_data_x, 2, 1)
                    curr_data_x = np.swapaxes(curr_data_x, 1, 0)
                    curr_data_mask = np.swapaxes(curr_data_mask, 2, 1)
                    curr_data_mask = np.swapaxes(curr_data_mask, 1, 0)
                    curr_data_mask = np.squeeze(curr_data_mask, axis=0)
                    
                # Ensure mask is the same shape as data
                mask_3d = np.repeat(curr_data_mask[np.newaxis, :, :], curr_data_x.shape[0], axis=0)
                curr_data_x = np.nan_to_num(curr_data_x)

                # Apply the mask to data
                curr_data_x *= mask_3d
                curr_data_x = np.nan_to_num(curr_data_x)

                with self.yield_lock:
                    if len(self.batch_data) < self.batch_size:
                        self.batch_data.append(curr_data_x)
                        self.batch_y.append(curr_data_y)

                    if len(self.batch_data) % self.batch_size == 0:
                        batch_x = np.stack(self.batch_data, axis=0)
                        batch_y = np.stack(self.batch_y, axis=0)
                        yield batch_x, batch_y
                        self.batch_data = []
                        self.batch_y = []


    def shuffle_data(self):
        if self.shuffle:
            self.file_list = shuffle(self.file_list)


def test():
    dataset = MyDataset('/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d', batch_size=1, shuffle=True)
    for batch_data, batch_y in dataset:
        print(batch_data.shape)
        break
    

if __name__ == "__main__":
    test()