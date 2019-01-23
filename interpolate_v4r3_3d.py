#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import matplotlib.pylab as plt
import numpy as np
import sys
import xarray as xr
import pyresample as pr
import time
import os
import glob
import pyproj 
import copy
sys.path.append('/Users/owang/Documents/OuWang/Documents/cygwin/home/owang/CODE/projects/ECCO-SEALEVEL/ECCOv4-py/')
import ecco_v4_py as ecco
print('modules loaded')
import pyproj 
import datetime
import string
import numpy.ma as ma
import os.path
 
def interploate_one3d(infile, outfile, varnm, unitnm):
    # ## Load the Grid Data
    
    # In[2]:
    
    
    fdir = '/Volumes/data01/forum/user/owang/junk/nonblank/'
    fname = 'XC.data'
    # in compact format
    XC = ecco.load_llc_compact(fdir, fname)
    # in 13 tiles format
    XC_t = ecco.load_llc_compact_to_tiles(fdir,fname,less_output=True)
    
    fname = 'YC.data'
    YC = ecco.load_llc_compact(fdir, fname)
    YC_t = ecco.load_llc_compact_to_tiles(fdir,fname,less_output=True)


    fname = 'maskCtrlC.data'
    maskc = ecco.load_llc_compact(fdir, fname, nk=50)
    maskc_t = ecco.load_llc_compact_to_tiles(fdir,fname,less_output=True,nk=50)

    
    # This is a 1D array.
    RC = np.squeeze(-1*ecco.load_binary_array(fdir, 'RC.data', 1, 50))
    
    
    # In[3]:
    
    
    fname = infile
    fdir = '/Volumes/data01/forum/user/owang/junk/h8i_i48/diags/STATE/'
    
    # as compact
    data = ecco.load_llc_compact(fdir, fname, nk=50);
    # as tiles
    data_t = ecco.llc_compact_to_tiles(data, less_output=True);
    
    
    # In[4]:
    
    
#   print(data_t.shape)
#   print(data.shape)
    
    
    # # Plot level 11 of the data
    
    # In[5]:
    
    
    ecco.plot_tiles(data_t[:,10,:],layout='latlon', rotate_to_latlon=True,cmin=-.2,cmax=.2)
    
    
    # # Resample to nearest latlon point
    
    # In[6]:
    
    
    #help(ecco.resample_to_latlon)
    
    
    # In[7]:
    
    
    new_grid_delta_deg=0.25
    
    new_grid_min_lat = -90+new_grid_delta_deg
    new_grid_max_lat = 90-new_grid_delta_deg
    new_grid_min_lon = -180
    new_grid_max_lon = 180-new_grid_delta_deg
    
    radius_of_influence = 120000 #m
    
    
    # In[8]:
    
    
    RC.shape
    
    
    # In[9]:
    
    
    for num, name in np.ndenumerate(RC):
        print num,name
        print type(num)
    
    
    # In[10]:
    
    
    data_latlon_projection = []
    
    for num, name in np.ndenumerate(RC):
        print(num[0])

# create mask
        masktmp = 1.-maskc[num,:]
        datamsked = ma.array(data[num,:], mask=masktmp)
    
        new_grid_lon, new_grid_lat, tmp = ecco.resample_to_latlon(XC, YC, datamsked,
                                                  new_grid_min_lat,
                                                  new_grid_max_lat, new_grid_delta_deg, 
                                                  new_grid_min_lon, 
                                                  new_grid_max_lon, new_grid_delta_deg, 
                                                  nprocs_user=1,
                                                  mapping_method = 'nearest_neighbor',
                                                  radius_of_influence=radius_of_influence)
        
        print(tmp.shape)
        tmp = np.expand_dims(tmp, 0)
        if num[0] == 0:
            data_latlon_projection = tmp
        else:
            data_latlon_projection = np.append(data_latlon_projection,tmp,0)
    
        print data_latlon_projection.shape
    
    
    # In[11]:
    
    
    lons_1d = new_grid_lon[0,:]
    lats_1d = new_grid_lat[:,0]
    
    
    # In[12]:
    
    
    nz, ny, nx  = np.shape(data_latlon_projection)
    print(lons_1d[0])
    print(lons_1d[-1])
    
    print(nx,ny,nz)
    
    
    # In[13]:
    
    
    plt.imshow(data_latlon_projection[0,:],origin='lower', vmin=0,vmax=.2,cmap='jet')
    
    
    # In[14]:
    
    
    plt.imshow(data_latlon_projection[25,:],origin='lower', vmin=0,vmax=.2,cmap='jet')
    
    
    # In[16]:
    
    
    foo = xr.DataArray(data_latlon_projection, coords=[RC, lats_1d, lons_1d], dims=['depth', 'latitude', 'longitude'])
    foo.isel(depth=1).plot()
    
    
    # In[17]:
    
    
    foo.attrs['xmin'] = np.min(new_grid_lon)
    foo.attrs['ymax'] = np.max(new_grid_lat)
    foo.attrs['spacing'] = new_grid_delta_deg
    foo.attrs['no_data'] = -9999.0
    foo.attrs['proj4'] = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    foo.attrs['Projection'] = 'Lat Lon EPSG:4326'
    foo.attrs['Insitution'] = 'JPL'
    foo.attrs['original_filename'] = fname
    foo.attrs['author'] = 'Ou Wang'
    foo.attrs['product_version'] = '0.1'
    foo.attrs['nx'] = nx
    foo.attrs['ny'] = ny
    foo.attrs['nz'] = nz
    foo.attrs['units'] = unitnm
    foo.attrs['Conventions'] = 'CF-1.6'
    foo.attrs['Metadata_Conventions'] = 'CF-1.6, Unidata Dataset Discovery v1.0, GDS v2.0'
    foo.attrs['cdm_data_type'] = 'Grid'
    foo.attrs['grid_method'] = 'bin average neighbor'
    foo.attrs['radius_of_influence'] = radius_of_influence
    foo.attrs['depth_units'] = 'meters'
    foo.name  = varnm
    
    
    # In[18]:
    
    
    #foo
    
    
    # In[19]:
    
    
    foo.to_netcdf('/Volumes/data01/forum/user/owang/junk/output/'+varnm+'/'+outfile+'.nc')
    
    
    # In[ ]:

varnm = 'THETA'
unitnm = 'degC'
fileset='state_3d_set1'
fdir = '/Volumes/data01/forum/user/owang/junk/h8i_i48/diags/STATE/'

cwd = os.getcwd()
os.chdir(fdir)
filelist=glob.glob(fileset+".*.data")
os.chdir(cwd)
print(filelist)
for fname in filelist:
    print(fname)
    outfnametmp=string.replace(fname, fileset, varnm) 
    outfname=string.replace(outfnametmp, '.data', '') 
    print(outfname)
    outfileexist=os.path.isfile('/Volumes/data01/forum/user/owang/junk/output/'+varnm+'/'+outfname+'.nc') 
    if outfileexist == 0:
       interploate_one3d(fname, outfname, varnm, unitnm)    
#   interploate_one3d(fname, 'test'+fname, varnm, unitnm)    
    
    
    
