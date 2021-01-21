import os
import random
import requests
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# Creates plots of all cycles in a given directory
def plotting(output_dir, files):

    for filename in files:
        filename = filename.split('/')[-1]

        print(f'Plotting {filename}')
        filepath = output_dir + filename
        output_file = f'{filename[:-2]}png'
        output_path = f'{output_dir}plots/{output_file}'
        var = 'gps_ssha'

        if not os.path.exists(f'{output_dir}plots/'):
            os.makedirs(f'{output_dir}plots/')

        ds = xr.open_dataset(filepath)

        start = ds.time.values[0]
        end = ds.time.values[-1]

        da = ds[var]

        mean = da.mean(skipna=True).values
        mean_plus = mean + 0.3
        mean_minus = mean - 0.3

        lons = da.longitude.values.ravel()
        lats = da.latitude.values.ravel()
        vals = da.values.ravel()

        # for the purposes of removing a mean, remove the mean from the SSH points
        # between 40S and 40N
        vals_subset = vals[np.where(np.logical_and(lats > -40, lats < 40))]
        v_mean = np.nanmean(vals_subset)
        vals_anom = vals - v_mean

        # Downsample to just values within +- 0.5m within mean
        # range_indeces = np.where(np.logical_and(
        #     vals >= mean_minus, vals <= mean_plus))
        # vals_anom = vals_anom[range_indeces]
        # lons = lons[range_indeces]
        # lats = lats[range_indeces]

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
        sax = ax.scatter(lons, lats, c=vals_anom, vmin=-0.3, vmax=0.3, cmap='jet',
                         s=.5, transform=ccrs.PlateCarree())

        fig.colorbar(sax, ax=ax)
        fig.suptitle(f'{filename[:-3]}', fontsize=20)
        plt.savefig(output_path)
        plt.close()

        # plt.show()


if __name__ == "__main__":
    output_dir = '/Users/kevinmarlis/Developer/JPL/sealevel_output/ssha_JASON_3_L2_OST_OGDR_GPS/aggregated_products/'
    files = [f for f in os.listdir(output_dir) if '.nc' in f]

    # cycle_id = 'bdf6aef4-1e37-4843-bb4e-610b151c069f'
    # fq = ['dataset_s:ssha*', 'type_s:harvested', f'cycle_id_s:{cycle_id}']
    # getVars = {'q': '*:*',
    #            'fq': fq,
    #            'rows': 300000}

    # url = f'http://localhost:8983/solr/sealevel_datasets/select?'
    # response = requests.get(url, params=getVars)
    # harvested_docs = response.json()['response']['docs']

    # files = [doc['granule_file_path_s'] for doc in harvested_docs]

    files.sort()

    plotting(output_dir, files)
