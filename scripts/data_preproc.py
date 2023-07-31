import os
import numpy as np
from netCDF4 import Dataset
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def time_from_path(path):
    """
    Extracts the date from the file path.

    Args:
    - path (str): The path of the file.

    Returns:
    - date (datetime.datetime): The extracted date.
    """
    filename = os.path.basename(path)
    date_str = filename.split('_')[1][1:11]
    date = datetime.strptime(date_str, '%Ym%md%d')
    return date

def files_from_path(path, months):
    """
    Generates a list of file paths for the previous months based on the given path.

    Args:
    - path (str): The path of the file.
    - months (list): A list of integers representing the number of previous months.

    Returns:
    - new_paths (list): A list of file paths for the previous months.
    - date (datetime.datetime): The date extracted from the original file path.
    """
    filename = os.path.basename(path)
    date_str = filename.split('_')[1][1:11]
    date = datetime.strptime(date_str, '%Ym%md%d')
    new_paths = []
    for m in months:
        prev_month_date = date - relativedelta(months=m)
        new_filename = filename.replace(date_str, datetime.strftime(prev_month_date, '%Ym%md%d'))
        path_parts = path.split('/')
        path_parts[-2] = str(prev_month_date.year)
        new_path = os.path.join('/'.join(path_parts[:-1]), new_filename)
        new_paths.append(new_path)
    return new_paths, date

def data_preproc(file):
    """
    Performs data preprocessing on the given file.

    Args:
    - file (str): The file path.

    Returns:
    - x_items (numpy.ndarray): The preprocessed data items.
    - mask (numpy.ndarray): The mask for the latitude values.
    - y_item (float): The target variable value associated with the file.
    """
    heights = [1, 3, 5, 7, 55, 60]
    x_items = []
    with open('/storage/kubrick/amedvedev/scripts/data/fluxes/data.json', 'r') as f:
        json_data = json.load(f)
    y_item = json_data[os.path.basename(file)]
    new_files, date = files_from_path(file, [1])
    for f in new_files:
        grid_u = Dataset(f.replace('gridV', 'gridU'), mode='r')
        grid_v = Dataset(f, mode='r')
        grid_w = Dataset(f.replace('gridV', 'gridW'), mode='r')
        for h in heights:
            u = grid_u.variables['vozocrtx'][0, h, :, :].filled(np.nan)
            v = grid_v.variables['vomecrty'][0, h, :, :].filled(np.nan)
            w = grid_w.variables['vovecrtz'][0, h, :, :].filled(np.nan)
            speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
            sin_dir = v / speed
            cos_dir = u / speed
            x_item = np.array([speed, sin_dir, cos_dir])
            x_items.append(x_item)
        lat = grid_u.variables['nav_lat'][:]
        grid_u.close()
        grid_v.close()
        grid_w.close()
    x_items = np.concatenate(x_items, axis=0)
    mask = (lat <= 57) | (lat >= 63)
    return x_items, mask, y_item

def main():
    """
    Main function that performs the data preprocessing and saves the results to new NetCDF files.
    """
    base_folder = "/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d/"
    output_folder = '/storage/kubrick/amedvedev/data/dataset/train/'
    with open('/storage/kubrick/amedvedev/scripts/data/fluxes/data.json', 'r') as f:
        json_data = json.load(f)
    years = os.listdir(base_folder)
    h = 1
    for year in years:
        year_folder = os.path.join(base_folder, year)
        output_year_folder = os.path.join(output_folder, year)
        os.makedirs(output_year_folder, exist_ok=True)
        files = os.listdir(year_folder)
        for file in files:
            if 'gridV' in file:
                grid_u = Dataset(os.path.join(year_folder, file.replace('gridV', 'gridU')), mode='r')
                grid_v = Dataset(os.path.join(year_folder, file), mode='r')
                grid_w = Dataset(os.path.join(year_folder, file.replace('gridV', 'gridW')), mode='r')

                u = grid_u.variables['vozocrtx'][0, h, :, :].filled(np.nan)
                v = grid_v.variables['vomecrty'][0, h, :, :].filled(np.nan)
                w = grid_w.variables['vovecrtz'][0, h, :, :].filled(np.nan)
                lat = grid_u.variables['nav_lat'][:]
                
                mask = (lat <= 57) | (lat >= 63)
                u = np.where(mask, u, np.nan)
                v = np.where(mask, v, np.nan)
                w = np.where(mask, w, np.nan)

                speed = np.sqrt(u**2 + v**2 + w**2)
                sin_dir = v / speed
                cos_dir = u / speed

                result_dataset = Dataset(os.path.join(output_year_folder, file.replace('gridV', 'ds')), 'w', format='NETCDF4')

                x_dim = result_dataset.createDimension('x', u.shape[0])
                y_dim = result_dataset.createDimension('y', u.shape[1])

                if file in json_data:
                    json_var = result_dataset.createVariable('target_var', 'f8')
                    json_var[:] = json_data[file]

                speed_var = result_dataset.createVariable('speed', 'f8', ('x', 'y'))
                sin_dir_var = result_dataset.createVariable('sin_dir', 'f8', ('x', 'y'))
                cos_dir_var = result_dataset.createVariable('cos_dir', 'f8', ('x', 'y'))

                speed_var[:] = np.nan_to_num(speed)
                sin_dir_var[:] = np.nan_to_num(sin_dir)
                cos_dir_var[:] = np.nan_to_num(cos_dir)

                result_dataset.close()
                grid_u.close()
                grid_v.close()
                grid_w.close()

if __name__ == "__main__":
    data_preproc('/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d/1993/NNATL12-MP423c_y1993m03d29.1d_gridV.nc')

