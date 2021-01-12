import random

import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

filepath = '/Users/kevinmarlis/Developer/JPL/sealevel_output/ssha_JASON_3_L2_OST_OGDR_GPS/aggregated_products/ssha_20201113T134415.nc'
var = 'gps_ssha'
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
sample_indeces = random.sample(range(1, vals_anom.shape[0]), 100000)
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
fig.suptitle(var, fontsize=20)
plt.savefig(
    f'/Users/kevinmarlis/Developer/JPL/sealevel_output/ssha_JASON_3_L2_OST_OGDR_GPS/aggregated_products/{var}_{start}_{end}.png')
plt.close()

# plt.show()
