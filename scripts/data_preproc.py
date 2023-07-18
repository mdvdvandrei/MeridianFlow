import os
import numpy as np
from netCDF4 import Dataset
import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta





def time_from_path(path):
    # Извлечь имя файла из пути
    filename = os.path.basename(path)

    # Извлечь дату из имени файла
    date_str = filename.split('_')[1][1:11]  # y1993m01d29
    date = datetime.strptime(date_str, '%Ym%md%d')

    return date


def files_from_path(path, months):
    # Извлечь имя файла из пути
    filename = os.path.basename(path)

    # Извлечь дату из имени файла
    date_str = filename.split('_')[1][1:11]  # y1993m01d29
    date = datetime.strptime(date_str, '%Ym%md%d')

    new_paths = []
    for m in months:
        # Вычислить дату на m месяцев ранее
        prev_month_date = date - relativedelta(months=m)

        # Создать новое имя файла с датой месяцем ранее
        new_filename = filename.replace(date_str, datetime.strftime(prev_month_date, '%Ym%md%d'))

        # Заменить год в подкаталоге
        path_parts = path.split('/')
        # Год, который мы хотим заменить, находится в предпоследнем подкаталоге перед именем файла
        path_parts[-2] = str(prev_month_date.year)

        # Создать новый путь с новым именем файла
        new_path = os.path.join('/'.join(path_parts[:-1]), new_filename)
        new_paths.append(new_path)

    return new_paths, date




def data_preproc(file):
    
    heights = [1,3,5,7,55,60] # 3 5 7 1,5km, 2,5km, 3,5km, 30 days 
    x_items = []

    with open('/storage/kubrick/amedvedev/scripts/data/fluxes/data.json', 'r') as f:
        json_data = json.load(f)

    y_item = json_data[os.path.basename(file)]
    new_files, date = files_from_path(file,[1])




    for f in new_files:
        grid_u = Dataset(f.replace('gridV', 'gridU'), mode='r')
        grid_v = Dataset(f, mode='r')
        grid_w = Dataset(f.replace('gridV', 'gridW'), mode='r')


        for h in heights:
            u = grid_u.variables['vozocrtx'][0, h, :, :].filled(np.nan)
            v = grid_v.variables['vomecrty'][0, h, :, :].filled(np.nan)
            w = grid_w.variables['vovecrtz'][0, h, :, :].filled(np.nan)
            speed = np.sqrt(u**2 + v**2 + w**2)
            sin_dir = v / speed
            cos_dir = u / speed
            x_item = np.array([speed, sin_dir, cos_dir]) #  (3, 402, 934)
            x_items.append(x_item)
            

        lat = grid_u.variables['nav_lat'][:] 
        grid_u.close()
        grid_v.close()
        grid_w.close()

    x_items = np.concatenate(x_items, axis = 0) #  (18, 402, 934)
    mask = (lat <= 57) | (lat >= 63)

    '''
    lat = grid_u.variables['nav_lat'][:] 
    mask = (lat <= 57) | (lat >= 63)
    # Применяем маску ко всем данным
    mask = (lat <= 57) | (lat >= 63)
    u = np.where(mask, u, np.nan)
    v = np.where(mask, v, np.nan)
    w = np.where(mask, w, np.nan)
    '''
    return x_items, mask, y_item










def main():
    # Папка, содержащая данные
    base_folder = "/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d/"
    output_folder = '/storage/kubrick/amedvedev/data/dataset/train/'

    with open('/storage/kubrick/amedvedev/scripts/data/fluxes/data.json', 'r') as f:
        json_data = json.load(f)

    # Получаем список всех годов
    years = os.listdir(base_folder)
    h = 1 # Заменить на нужное 3 5 7 11, 1,5km, 2,5km, 3,5km, 30 days 

    # Проходим по каждому году
    for year in years:
        year_folder = os.path.join(base_folder, year)
        output_year_folder = os.path.join(output_folder, year)
        os.makedirs(output_year_folder, exist_ok=True)
        # Получаем список всех файлов в папке года
        files = os.listdir(year_folder)

        # Проходим по каждому файлу
        for file in files:
            if 'gridV' in file: 

                grid_u = Dataset(os.path.join(year_folder, file.replace('gridV', 'gridU')), mode='r')
                grid_v = Dataset(os.path.join(year_folder, file), mode='r')
                grid_w = Dataset(os.path.join(year_folder, file.replace('gridV', 'gridW')), mode='r')

                u = grid_u.variables['vozocrtx'][0, h, :, :].filled(np.nan)
                v = grid_v.variables['vomecrty'][0, h, :, :].filled(np.nan)
                w = grid_w.variables['vovecrtz'][0, h, :, :].filled(np.nan)
                lat = grid_u.variables['nav_lat'][:]
                
                # Применяем маску ко всем данным
                mask = (lat <= 57) | (lat >= 63)
                u = np.where(mask, u, np.nan)
                v = np.where(mask, v, np.nan)
                w = np.where(mask, w, np.nan)

                # Вычисляем модуль скорости
                speed = np.sqrt(u**2 + v**2 + w**2)

                # Вычисляем sin и cos направления скорости
                sin_dir = v / speed
                cos_dir = u / speed



                # Создаем новый netCDF файл для сохранения результата
                result_dataset = Dataset(os.path.join(output_year_folder, file.replace('gridV', 'ds')), 'w', format='NETCDF4')
       
                # Определяем измерения 'x' и 'y' в новом файле netCDF
                x_dim = result_dataset.createDimension('x', u.shape[0])
                y_dim = result_dataset.createDimension('y', u.shape[1])

                if file in json_data:
                    json_var = result_dataset.createVariable('target_var', 'f8')  
                    json_var[:] = json_data[file]


                # Создаем переменные в новом netCDF файле
                speed_var = result_dataset.createVariable('speed', 'f8', ('x', 'y'))
                sin_dir_var = result_dataset.createVariable('sin_dir', 'f8', ('x', 'y'))
                cos_dir_var = result_dataset.createVariable('cos_dir', 'f8', ('x', 'y'))

                # Сохраняем данные в переменные
                speed_var[:] = np.nan_to_num(speed)
                sin_dir_var[:] = np.nan_to_num(sin_dir)
                cos_dir_var[:] = np.nan_to_num(cos_dir)

                # Закрываем файлы
                result_dataset.close()
                grid_u.close()
                grid_v.close()
                grid_w.close()

if __name__ == "__main__":
    data_preproc('/storage/kubrick/amedvedev/data/NNATL-12/NNATL12-MP423c-S/1d/1993/NNATL12-MP423c_y1993m03d29.1d_gridV.nc')
