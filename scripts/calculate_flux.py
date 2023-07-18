from netCDF4 import Dataset
import numpy as np
import os
from interpolate import interpolation_weights, interpolate_data
import json

def calc_mash(dataset,i):
    n = ['u','v','w']
    depthu_bounds = dataset.variables['depth'+n[i]+'_bounds'][:]
    depths_diff_array = depthu_bounds[:,1] - depthu_bounds[:,0]
    len_of_one_cell = 2005.6568521951
    
    return depths_diff_array, len_of_one_cell

def interpolation_and_computing_flux(data, depths_diff_array, len_of_one_cell, inds, wghts, shape):
    values_with_depth = np.zeros((75,512))
    for h in range(75):
        cape_interpolated = interpolate_data(data[h,:,:], inds, wghts, shape)
        values_with_depth[h,:] = cape_interpolated[0,:]
    depth_array = np.repeat(depths_diff_array[:, np.newaxis], values_with_depth.shape[1], axis=1)
    result = len_of_one_cell * depth_array * np.nan_to_num(values_with_depth)* 1025
    return result

def calculation(path, inds, wghts, shape):
    names_array = ['vozocrtx','vomecrty','vovecrtz']
    path_array = ['1d_gridU.nc','1d_gridV.nc','1d_gridW.nc']
    full_flux_array = np.zeros((3,75,512))
    base_data_path = "/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d/"
    for i,p in enumerate(path_array):
        full_path = path + p
        with Dataset(os.path.join(base_data_path, full_path), 'r') as ds:
            data = ds.variables[names_array[i]][0,:,:,:]
            depths_diff_array, len_of_one_cell = calc_mash(ds,i)
            one_axis_flux = interpolation_and_computing_flux(data, depths_diff_array, len_of_one_cell, inds, wghts, shape)
            full_flux_array[i,:,:] = one_axis_flux
    # axis [x,y,z]
    proj_vector = np.array([0,1,0]) 
    dot_product = np.einsum('ijk,i->jk', full_flux_array, proj_vector)
    magnitude_squared_B = np.einsum('i,i', proj_vector, proj_vector)

    P = np.einsum('jk,i->ijk', dot_product / magnitude_squared_B, proj_vector)
    
    return np.sum(P)


def main():
    inds, wghts, shape = None, None, None
    dir_path = "/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d/"

    with Dataset(os.path.join(dir_path, '1995/NNATL12-MP423c_y1995m01d26.1d_gridU.nc'), 'r') as ds:
        nav_lat = ds.variables['nav_lat'][:].data
        nav_lon = ds.variables['nav_lon'][:].data

    lons_proj = np.linspace(nav_lon.min(), nav_lon.max(), 512)
    lats_proj = np.repeat(60, 512)

    inds, wghts, shape = interpolation_weights(nav_lon, nav_lat, lons_proj, lats_proj)

    data = {}

    for year in os.listdir(dir_path):
        for file in os.listdir(dir_path + year):
            if file.endswith('1d_gridV.nc'):
                #key = file[11:21]  # This extracts 'y1995m01d01'
                #year = key[1:5]
                #month = key[6:8]
                #day = key[9:11]
                #date = f"{year}-{month}-{day}"
                flux = calculation(year + '/' + file[:-11], inds, wghts, shape)
                data[file] = flux
        print(year)

    data_path = 'data/fluxes/'

    # Check if the directory exists and create it if necessary
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    with open(data_path + 'data.json', 'w') as json_file:
        json.dump(data, json_file)

if __name__ == "__main__":
    main()


