#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:22:51 2020

@author: ifenty
"""

#%%

import sys
sys.path.append('/home/ifenty/ECCOv4-py/')
import ecco_v4_py as ecco
import numpy as np
import xarray as xr
from pathlib import Path
from cartopy.util import add_cyclic_point


sys.path.append('/home/ifenty/ECCO-GROUP/ECCO-ACCESS/ecco-cloud-utils')
import ecco_cloud_utils as ea

# available at nasa github simplegrid
sys.path.append('/home/ifenty/git_repo_others/simplegrid')
import simplegrid as sg

import matplotlib.path as mpath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pylab as plt
from cartopy.util import add_cyclic_point
import netCDF4 as nc4

def add_metadata_to_model_grid(model_grid):
    
    model_grid.attrs['geospatial_lat_min'] = np.min(model_grid.YC.values)
    model_grid.attrs['geospatial_lat_max'] = np.max(model_grid.YC.values)
    model_grid.attrs['geospatial_lon_min'] = np.min(model_grid.XC.values)
    model_grid.attrs['geospatial_lon_max'] = np.max(model_grid.XC.values)
    
    model_grid.attrs['geospatial_lat_units'] = 'degrees_north'
    model_grid.attrs['geospatial_lon_units'] = 'degrees_east'

    model_grid['YC'].attrs['standard_name'] = 'latitude'
    model_grid['YC'].attrs['long_name'] = 'latitude'
    model_grid['YC'].attrs['units'] = 'degrees_north'
    model_grid['YC'].attrs['valid_range'] = [-180.,  180.]
    model_grid['YC'].attrs['axis'] = 'X'
    
    model_grid['XC'].attrs['standard_name'] = 'longitude'
    model_grid['XC'].attrs['long_name'] = 'longitude'
    model_grid['XC'].attrs['units'] = 'degrees_east'
    model_grid['XC'].attrs['valid_range'] = [-180.,  180.]
    model_grid['XC'].attrs['axis'] = 'X'
    

    return model_grid

#%%
   
def make_model_grid_encodings(model_grid, netcdf_fill_value):
    
    encoding_each = {'zlib':True, \
                     'complevel':5,\
                     'fletcher32':True,\
                     '_FillValue':netcdf_fill_value}
    
    encoding_vars = {var:encoding_each for var in model_grid.data_vars}
    
    
    coord_encoding_each = {'zlib':True, \
                         'complevel':5,\
                         'fletcher32':True,\
                         '_FillValue':False}
      
    encoding_coords = {var:coord_encoding_each for var in model_grid.dims}
    
    encoding = {**encoding_vars, **encoding_coords}  
    
    
    return encoding



#%%

def plot_3panel(xc, yc, markersize, fignum, suptitle=[], field = []):
    
    fig = plt.figure(num=fignum, figsize=[8,12], clear=True);fig.clf()
      
    geo_axes_top_left  = plt.subplot(2, 2, 1, 
                                     projection = ccrs.NorthPolarStereo())
    geo_axes_top_right = plt.subplot(2, 2, 2, 
                                     projection = ccrs.SouthPolarStereo())
    geo_axes_bottom    = plt.subplot(2, 1, 2, 
                                     projection = ccrs.Robinson())
       
    axes = []
    axes.append(geo_axes_top_left)
    axes.append(geo_axes_top_right)
    axes.append(geo_axes_bottom)
    
    circle_boundary=True
    
    draw_labels = False
    data_projection_code = 4326
    
    for axis in range(3):
        ax = axes[axis]
            
        if isinstance(ax.projection, ccrs.NorthPolarStereo):
            ax.set_extent([-180, 180, 45, 90], ccrs.PlateCarree())
            
        elif isinstance(ax.projection, ccrs.SouthPolarStereo):
            ax.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
            
        else:
            circle_boundary=False
            ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    
        if circle_boundary:
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
        
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=draw_labels,
                      linewidth=1, color='black', alpha=0.5, linestyle='--')

        if data_projection_code == 4326: # lat lon does nneed to be projected
            data_crs =  ccrs.PlateCarree()
            
        else:
            data_crs=ccrs.epsg(data_projection_code)

        if len(field) == 0:
            ax.plot(xc, yc, 'b.', transform=data_crs,markersize=markersize)
            
        else:
            cmin = np.min(field)
            cmax = np.max(field)
            
            print(cmin, cmax)
            im = ax.pcolormesh(xc,yc,field, 
                               transform=data_crs, \
                               vmin = cmin-1, vmax=cmax+1)
            
            if axis == 2:
                fig.colorbar(im,  orientation='horizontal')
            
        ax.add_feature(cfeature.LAND)
        ax.coastlines('110m', linewidth=0.8)
        
    if len(suptitle) > 0:
        fig.suptitle(suptitle)
        
    return axes
    
#%%
        #%%
##################################################
if __name__== "__main__":

        
        
    source_dir = Path('/home/ifenty/data/grids/demos/source')
    grid_output_dir = Path( '/home/ifenty/data/grids/demos')
    
    # Define precision of output files, float32 is standard
    # ------------------------------------------------------
    array_precision = np.float32
       
    # Define fill values for binary and netcdf
    # ---------------------------------------------
    if array_precision == np.float32:
        binary_output_dtype = '>f4'
        netcdf_fill_value = nc4.default_fillvals['f4']
    
    elif array_precision == np.float64:
        binary_output_dtype = '>f8'
        netcdf_fill_value = nc4.default_fillvals['f8']
    
    # ECCO always uses -9999 for missing data.
    binary_fill_value = -9999
    
    
    
    
    #%% GRID 1: 2X2 DEMO
          
    xs = np.linspace(-179.5, 179.5, 180, dtype=array_precision)
    ys = np.linspace(-89.5, 89.5, 90,  dtype=array_precision)
    
    [xg, yg] = np.meshgrid(np.linspace(-180, 180, 181), np.linspace(-90, 90, 91))
    [xc, yc] = np.meshgrid(xs, ys)
    
    
    plt.close('all')
      
    cell_areas = ea.area_of_latlon_grid(-180, 180, -90, 90, 2, 2)
    
    effective_radius = 0.5 * np.sqrt(cell_areas)
    
    
    XC_da = xr.DataArray(xc, dims=['lat','lon'],\
                         coords={'lon': xs, 'lat': ys})
    
    YC_da = xr.DataArray(yc, dims=['lat','lon'],\
                      coords={'lon': xs, 'lat': ys})
        
    effective_radius = xr.DataArray(effective_radius, dims=['lat','lon'],\
                      coords={'lon': xs, 'lat': ys})
    
    XC_da.name = 'XC'
    YC_da.name = 'YC'
    effective_radius.name = 'effective_radius'
    
    XC_da = XC_da.astype(array_precision)
    YC_da = YC_da.astype(array_precision)
    effective_radius = effective_radius.astype(array_precision)
    
    model_grid = xr.merge((XC_da,YC_da, effective_radius))
    model_grid.attrs['name']= '2x2deg_demo'
    model_grid.attrs['type'] = 'latlon'
    
    model_grid = add_metadata_to_model_grid(model_grid)
    
    encoding = make_model_grid_encodings(model_grid, netcdf_fill_value)
    
    print(model_grid)
    model_grid.to_netcdf(str(grid_output_dir) + '/2x2deg_demo.nc',\
                         encoding=encoding)


#%%
axes = plot_3panel(model_grid.XC.values, model_grid.YC.values,\
                        .5, 1,'2-degree lat-lon grid: grid center points')

wrap_lat = add_cyclic_point(model_grid.YC)
wrap_lon = add_cyclic_point(model_grid.XC)
wrap_data = add_cyclic_point(model_grid.effective_radius)

axes = plot_3panel(wrap_lon, wrap_lat,\
                        .5, 2,'2-degree lat-lon grid: effective grid radius (m)', \
                        wrap_data)


#%% GRID 2: north polar stereographic

#sea_ice_data_dir = Path('/mnt/intraid/ian1/ifenty/data/observations/seaice/G02202_V3/north/monthly')

sid=xr.open_dataset(str(source_dir / 'polar_stereo_n_25km' / 'seaice_conc_monthly_nh_n07_198604_v03r01.nc'))
xs = np.arange(0, 304, dtype=np.int16)
ys = np.arange(0, 448, dtype=np.int16)

XC_da = xr.DataArray(sid.longitude.values,dims=['j','i'],\
                 coords={'j': ys, 'i': xs})

YC_da = xr.DataArray(sid.latitude.values, dims=['j','i'],\
                 coords={'j': ys, 'i': xs})

effective_grid_radius_da = XC_da.copy(deep=True)
effective_grid_radius_da.values[:,:] = 25e3 * np.sqrt(2)
effective_grid_radius_da.attrs = {}

XC_da.name = 'XC'
YC_da.name = 'YC'
effective_grid_radius_da.name = 'effective_grid_radius'

XC_da = XC_da.astype(array_precision)
YC_da = YC_da.astype(array_precision)
effective_radius = effective_radius.astype(array_precision)

model_grid = xr.merge((XC_da, YC_da, effective_grid_radius_da))
model_grid.attrs['name']= 'polar_stereo_n_25km_demo'
model_grid.attrs['type'] = 'latlon'

model_grid = add_metadata_to_model_grid(model_grid)
encoding = make_model_grid_encodings(model_grid, netcdf_fill_value)

print(model_grid)
model_grid.to_netcdf(str(grid_output_dir) + '/polar_stereo_n_25km_demo.nc',\
                     encoding=encoding)

#%%
axes = plot_3panel(model_grid.XC.values, model_grid.YC.values,\
                        .5, 10,'polar_stereo_n_25km_demo: grid center points')

axes = plot_3panel(model_grid.XC.values, model_grid.YC.values,\
                        .5, 20,'polar_stereo_n_25km_demo: effective grid radius (m)', \
                        model_grid.effective_grid_radius.values)




#%% GRID 3: fixed lat-lon-cap-90

llc90_dir = Path(source_dir / 'llc90')

tile_files = list(llc90_dir.glob('tile*mitgrid*'))
print(tile_files)

ny = [270, 270, 90, 90, 90]
nx = [90, 90, 90, 270, 270]

js = np.arange(90, dtype=np.int16)
ts = np.arange(13, dtype=np.int16)

tile_data = []
XC_faces = {}
YC_faces = {}
RAC_faces= {}
for t in range(5):
    tile_data = sg.gridio.read_mitgridfile(tile_files[t], nx[t], ny[t])
    
    XC_faces[t+1] = tile_data['XC'].T
    YC_faces[t+1] = tile_data['YC'].T
    RAC_faces[t+1] = tile_data['RAC'].T

XC_tiles = ecco.llc_faces_to_tiles(XC_faces)
YC_tiles = ecco.llc_faces_to_tiles(YC_faces)
RAC_tiles = ecco.llc_faces_to_tiles(RAC_faces)

effective_grid_radius = 0.5 * np.sqrt(RAC_tiles) * np.sqrt(2)

XC_da = xr.DataArray(XC_tiles, dims=['tile','j','i'],\
                     coords={'tile': ts,'j': js, 'i': js})

YC_da = xr.DataArray(YC_tiles, dims=['tile','j','i'],\
                     coords={'tile': ts,'j': js, 'i': js})

effective_grid_radius_da = xr.DataArray(effective_grid_radius,\
                                        dims=['tile','j','i'],\
                                        coords={'tile': ts,'j': js, 'i': js})


XC_da.name = 'XC'
YC_da.name = 'YC'
effective_grid_radius_da.name = 'effective_grid_radius'

XC_da = XC_da.astype(array_precision)
YC_da = YC_da.astype(array_precision)
effective_radius = effective_radius.astype(array_precision)

model_grid = xr.merge((XC_da,YC_da, effective_grid_radius_da))
model_grid.attrs['name']= 'ECCO_llc90'
model_grid.attrs['type'] = 'llc'

model_grid = add_metadata_to_model_grid(model_grid)
encoding = make_model_grid_encodings(model_grid, netcdf_fill_value)

print(model_grid)
model_grid.to_netcdf(str(grid_output_dir) + '/ECCO_llc90_demo.nc',\
                     encoding=encoding)

#%%
ecco.plot_tiles(model_grid.XC);
ecco.plot_tiles(model_grid.effective_grid_radius, show_colorbar=True);


#%% GRID 4: fixed lat-lon-cap-90

llc270_dir = Path(source_dir / 'llc270')

tile_files = list(llc270_dir.glob('tile*mitgrid*'))
print(tile_files)

ny = [270*3, 270*3, 90*3, 90*3, 90*3]
nx = [90*3, 90*3, 90*3, 270*3, 270*3]

js = np.arange(270,  dtype=np.int16)
ts = np.arange(13, dtype=np.int16)

tile_data = []
XC_faces = {}
YC_faces = {}
RAC_faces= {}
for t in range(5):
    tile_data = sg.gridio.read_mitgridfile(tile_files[t], nx[t], ny[t])
    
    XC_faces[t+1] = tile_data['XC'].T
    YC_faces[t+1] = tile_data['YC'].T
    RAC_faces[t+1] = tile_data['RAC'].T

XC_tiles = ecco.llc_faces_to_tiles(XC_faces)
YC_tiles = ecco.llc_faces_to_tiles(YC_faces)
RAC_tiles = ecco.llc_faces_to_tiles(RAC_faces)

effective_grid_radius = 0.5 * np.sqrt(RAC_tiles) * np.sqrt(2)

XC_da = xr.DataArray(XC_tiles, dims=['tile','j','i'], \
                     coords={'tile': ts,'j': js, 'i': js})
YC_da = xr.DataArray(YC_tiles, dims=['tile','j','i'], \
                     coords={'tile': ts,'j': js, 'i': js})

effective_grid_radius_da = xr.DataArray(effective_grid_radius,\
                                        dims=['tile','j','i'],\
                                        coords={'tile': ts,'j': js, 'i': js})

XC_da.name = 'XC'
YC_da.name = 'YC'
effective_grid_radius_da.name = 'effective_grid_radius'

XC_da = XC_da.astype(array_precision)
YC_da = YC_da.astype(array_precision)
effective_radius = effective_radius.astype(array_precision)

model_grid = xr.merge((XC_da,YC_da, effective_grid_radius_da))
model_grid.attrs['name']= 'ECCO_llc270'
model_grid.attrs['type'] = 'llc'

model_grid = add_metadata_to_model_grid(model_grid)
encoding = make_model_grid_encodings(model_grid, netcdf_fill_value)


print(model_grid)
model_grid.to_netcdf(str(grid_output_dir) + '/ECCO_llc270_demo.nc',\
                     encoding=encoding)

#%%
ecco.plot_tiles(model_grid.XC);
ecco.plot_tiles(model_grid.effective_grid_radius, show_colorbar=True);
