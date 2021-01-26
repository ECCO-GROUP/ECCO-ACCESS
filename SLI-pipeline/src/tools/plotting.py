import os
import random
import requests
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib


# Creates plots of all cycles in a given directory
def plotting(files_dir, files, var):
    for filename in files:
        filename = filename.split('/')[-1]

        print(f'Plotting {filename}')
        filepath = files_dir + filename
        output_file = f'{filename[:-2]}png'
        output_dir = f'{files_dir}plots/'
        output_path = f'{output_dir}{output_file}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ds = xr.open_dataset(filepath)

        da = ds[var]

        try:
            lons = da.longitude.values.ravel()
            lats = da.latitude.values.ravel()
        except:
            lons = da.lon.values.ravel()
            lats = da.lat.values.ravel()

        vals = da.values.ravel()

        # for the purposes of removing a mean, remove the mean from the SSH points
        # between 40S and 40N
        vals_subset = vals[np.where(np.logical_and(lats > -40, lats < 40))]
        v_mean = np.nanmean(vals_subset)
        vals_anom = vals - v_mean

        # Downsample to 100000 values
        if vals_anom.shape[0] > 100000:
            sample_indeces = random.sample(
                range(1, vals_anom.shape[0]), 100000)
            vals_anom = vals_anom[sample_indeces]
            lons = lons[sample_indeces]
            lats = lats[sample_indeces]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(
            1, 1, 1, projection=ccrs.Robinson(central_longitude=-66))
        ax.set_global()
        ax.coastlines()
        sax = ax.scatter(lons, lats, c=vals_anom, vmin=-.3, vmax=0.3, cmap='jet',
                         s=.5, transform=ccrs.PlateCarree())

        fig.colorbar(sax, ax=ax)
        fig.suptitle(f'{filename[:-3]}', fontsize=20)
        plt.savefig(output_path)
        plt.close()

        # plt.show()


if __name__ == "__main__":
    plot_aggregated = True

    if plot_aggregated:
        files_dir = '/Users/kevinmarlis/Developer/JPL/sealevel_output/ssha_JASON_3_L2_OST_OGDR_GPS/aggregated_products/'
    else:
        files_dir = '/Users/kevinmarlis/Developer/JPL/sealevel_output/ssha_JASON_3_L2_OST_OGDR_GPS/harvested_granules/2016/'

    files = [f for f in os.listdir(files_dir) if '.nc' in f]
    files.sort()

    var = 'ssha'

    plotting(files_dir, files, var)
