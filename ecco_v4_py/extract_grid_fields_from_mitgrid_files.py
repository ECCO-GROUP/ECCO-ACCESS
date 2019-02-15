#!/usr/bin/env python

#%%
import sys
import numpy as np

sys.path.append('/Users/ifenty/git_repo_mine/simplegrid/')
import simplegrid as sg

#%%
def extract_U_point_grid_fields_from_mitgrid_as_tiles(llc, grid_dir, 
                                                       grid_orientation='latlon'):
    #
    # grid_orientation can be 'latlon' or 'native'
    # if native, then keep rotation of i and j for tiles 8-13
    # if latlon, then rotate i and j for tiles 8-13 so that they 
    # are the same direction as tiles 1-6
    tile1_name = 'tile001.mitgrid'
    tile2_name = 'tile002.mitgrid'
    tile3_name = 'tile003.mitgrid'
    tile4_name = 'tile004.mitgrid'
    tile5_name = 'tile005.mitgrid'

    i=np.arange(llc*3)
    i1=i[llc*0  :llc+llc*0]
    i2=i[llc*1:llc+llc*1]
    i3=i[llc*2:llc+llc*2]

    print i1.shape, i2.shape, i3.shape
    print i1[0:3], i1[-3:]
    print i2[0:3], i2[-3:]
    print i3[0:3], i3[-3:]
    
    DXC = {}
    DYG = {}
    
    
    # Tile 1
    mg1 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile1_name,
                                     llc,llc*3, verbose=True)
    
    
    DXC[1] = mg1['DXC'][np.ix_(np.arange(91),i1)].T
    DXC[2] = mg1['DXC'][np.ix_(np.arange(91),i2)].T
    DXC[3] = mg1['DXC'][np.ix_(np.arange(91),i3)].T
    
    DYG[1] = mg1['DYG'][np.ix_(np.arange(91),i1)].T
    DYG[2] = mg1['DYG'][np.ix_(np.arange(91),i2)].T
    DYG[3] = mg1['DYG'][np.ix_(np.arange(91),i3)].T
    
    mg1 = []

    # Tile 2    
    mg2 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile2_name,
                                     llc,3*llc)
    
    DXC[4] = mg2['DXC'][np.ix_(np.arange(91),i1)].T
    DXC[5] = mg2['DXC'][np.ix_(np.arange(91),i2)].T
    DXC[6] = mg2['DXC'][np.ix_(np.arange(91),i3)].T
    
    DYG[4] = mg2['DYG'][np.ix_(np.arange(91),i1)].T
    DYG[5] = mg2['DYG'][np.ix_(np.arange(91),i2)].T
    DYG[6] = mg2['DYG'][np.ix_(np.arange(91),i3)].T
       
    # TILE 3
    mg3 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile3_name,
                                     llc,llc)
    
    DXC[7] = mg3['DXC'][np.ix_(np.arange(91),i1)].T
    DYG[7] = mg3['DYG'][np.ix_(np.arange(91),i1)].T

    mg3 = []

    # TILE 4
    mg4 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile4_name,
                                     llc*3,llc)

    if grid_orientation == 'latlon' :

        i=np.arange(llc*3)
        i1=i[llc*0  :llc+llc*0]
        i2=i[llc*1:llc+llc*1]
        i3=i[llc*2:llc+llc*2]
        print mg4['DYC'].shape
        
        DXC[8]  = mg4['DYC'][np.ix_(i1,np.arange(91))]
        DXC[9]  = mg4['DYC'][np.ix_(i2,np.arange(91))]
        DXC[10] = mg4['DYC'][np.ix_(i3,np.arange(91))]
        
        DYG[8]   = mg4['DXG'][np.ix_(i1,np.arange(91))]
        DYG[9]   = mg4['DXG'][np.ix_(i2,np.arange(91))]
        DYG[10]  = mg4['DXG'][np.ix_(i3,np.arange(91))]

        mg4 = []
        mg5 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile5_name,
                                         llc*3,llc)
        DXC[11] = mg5['DYC'][np.ix_(i1,np.arange(91))]
        DXC[12] = mg5['DYC'][np.ix_(i2,np.arange(91))]
        DXC[13] = mg5['DYC'][np.ix_(i3,np.arange(91))]
        
        DYG[11]  = mg5['DXG'][np.ix_(i1,np.arange(91))]
        DYG[12]  = mg5['DXG'][np.ix_(i2,np.arange(91))]
        DYG[13]  = mg5['DXG'][np.ix_(i3,np.arange(91))]
        
        mg5 = []

        
    elif grid_orientation == 'native':
        i=np.arange(llc*3)  
        i1=i[llc*0  :llc+llc*0]
        i2=i[llc*1:llc+llc*1]
        i3=i[llc*2:llc+llc*2]

        DXC[8]  = mg4['DXC'][np.ix_(i1,np.arange(90))].T
        DXC[9]  = mg4['DXC'][np.ix_(i2,np.arange(90))].T
        DXC[10] = mg4['DXC'][np.ix_(i3,np.arange(90))].T
        
        DYG[8]   = mg4['DYG'][np.ix_(i1,np.arange(90))].T
        DYG[9]   = mg4['DYG'][np.ix_(i2,np.arange(90))].T
        DYG[10]  = mg4['DYG'][np.ix_(i3,np.arange(90))].T

        mg4 = []

        mg5 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile5_name,
                                         llc*3,llc)
        DXC[11] = mg5['DXC'][np.ix_(i1,np.arange(90))].T
        DXC[12] = mg5['DXC'][np.ix_(i2,np.arange(90))].T
        DXC[13] = mg5['DXC'][np.ix_(i3,np.arange(90))].T
        
        DYG[11]  = mg5['DYG'][np.ix_(i1,np.arange(90))].T
        DYG[12]  = mg5['DYG'][np.ix_(i2,np.arange(90))].T
        DYG[13]  = mg5['DYG'][np.ix_(i3,np.arange(90))].T
    
        mg5 = []
    
    # Make array (if latlon orientation) 
    # or keep as dictionary (if native grid orientation)
    if grid_orientation == 'latlon' :
        DXC_arr = np.zeros((13, llc, llc+1))
        for tile in range(7):
            DXC_arr[tile,:,:] = DXC[tile+1]
        for tile in range(7,13):
            DXC_arr[tile,:,:] = np.flipud(DXC[tile+1])

        DYG_arr = np.zeros((13, llc, llc+1))
        for tile in range(7):
            DYG_arr[tile,:,:] = DYG[tile+1]
        for tile in range(7,13):
            DYG_arr[tile,:,:] = np.flipud(DYG[tile+1])
            
        DXC = DXC_arr
        DYG = DYG_arr 
    
        print DYG.shape
        print DXC.shape
        
        plt.figure(1);plt.clf();
        ecco.plot_tiles(DXC,layout='latlon')
        plt.figure(2);plt.clf();
        ecco.plot_tiles(DYG,layout='latlon')
    
    return DXC, DYG
 

########################
    
#%%
def extract_G_point_grid_fields_from_mitgrid_as_tiles(llc, grid_dir):

    tile1_name = 'tile001.mitgrid'
    tile2_name = 'tile002.mitgrid'
    tile3_name = 'tile003.mitgrid'
    tile4_name = 'tile004.mitgrid'
    tile5_name = 'tile005.mitgrid'
    
    XG_t = np.ones((13,llc+1,llc+1))*np.nan
    YG_t = np.ones((13,llc+1,llc+1))*np.nan
    RAZ_t = np.ones((13,llc+1,llc+1))*np.nan    
    
    i=np.arange(llc*3 + 1)
    i1=i[llc*0  :llc+llc*0+1]
    i2=i[llc*1:llc+llc*1+1]
    i3=i[llc*2:llc+llc*2+1]
    
    print i1.shape, i2.shape, i3.shape
    print i1[0:3], i1[-3:]
    print i2[0:3], i2[-3:]
    print i3[0:3], i3[-3:]
    
    # TILE 1
    mg1 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile1_name,
                                     llc,llc*3, verbose=True)
    
    XG_t[0,:,:] = mg1['XG'][np.ix_(np.arange(91),i1)].T
    XG_t[1,:,:] = mg1['XG'][np.ix_(np.arange(91),i2)].T
    XG_t[2,:,:] = mg1['XG'][np.ix_(np.arange(91),i3)].T
    
    YG_t[0,:,:] = mg1['YG'][np.ix_(np.arange(91),i1)].T
    YG_t[1,:,:] = mg1['YG'][np.ix_(np.arange(91),i2)].T
    YG_t[2,:,:] = mg1['YG'][np.ix_(np.arange(91),i3)].T
    
    RAZ_t[0,:,:] = mg1['RAZ'][np.ix_(np.arange(91),i1)].T
    RAZ_t[1,:,:] = mg1['RAZ'][np.ix_(np.arange(91),i2)].T
    RAZ_t[2,:,:] = mg1['RAZ'][np.ix_(np.arange(91),i3)].T
    
    mg1=[]
    
    # TILE 2
    mg2 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile2_name,
                                     llc,3*llc)
    
    XG_t[3,:,:] = mg2['XG'][np.ix_(np.arange(91),i1)].T
    XG_t[4,:,:] = mg2['XG'][np.ix_(np.arange(91),i2)].T
    XG_t[5,:,:] = mg2['XG'][np.ix_(np.arange(91),i3)].T
    
    YG_t[3,:,:] = mg2['YG'][np.ix_(np.arange(91),i1)].T
    YG_t[4,:,:] = mg2['YG'][np.ix_(np.arange(91),i2)].T
    YG_t[5,:,:] = mg2['YG'][np.ix_(np.arange(91),i3)].T

    RAZ_t[3,:,:] = mg2['RAZ'][np.ix_(np.arange(91),i1)].T
    RAZ_t[4,:,:] = mg2['RAZ'][np.ix_(np.arange(91),i2)].T
    RAZ_t[5,:,:] = mg2['RAZ'][np.ix_(np.arange(91),i3)].T
    
    mg2=[]
    
    # TILE 3
    mg3 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile3_name,
                                     llc,llc)
    XG_t[6,:,:]  = mg3['XG'].T
    YG_t[6,:,:]  = mg3['YG'].T
    RAZ_t[6,:,:] = mg3['RAZ'].T
    mg3=[]
    
    # TILE 4
    mg4 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile4_name,
                                     llc*3,llc)
    XG_t[7,:,:] = mg4['XG'][np.ix_(i1,np.arange(91))].T
    XG_t[8,:,:] = mg4['XG'][np.ix_(i2,np.arange(91))].T
    XG_t[9,:,:] = mg4['XG'][np.ix_(i3,np.arange(91))].T
    
    YG_t[7,:,:] = mg4['YG'][np.ix_(i1,np.arange(91))].T
    YG_t[8,:,:] = mg4['YG'][np.ix_(i2,np.arange(91))].T
    YG_t[9,:,:] = mg4['YG'][np.ix_(i3,np.arange(91))].T

    RAZ_t[7,:,:] = mg4['RAZ'][np.ix_(i1,np.arange(91))].T
    RAZ_t[8,:,:] = mg4['RAZ'][np.ix_(i2,np.arange(91))].T
    RAZ_t[9,:,:] = mg4['RAZ'][np.ix_(i3,np.arange(91))].T
    
    mg4 = []
    
    # TILE 5
    mg5 = sg.gridio.read_mitgridfile(grid_dir + '/' + tile5_name,
                                     llc*3,llc)
    XG_t[10,:,:] = mg5['XG'][np.ix_(i1,np.arange(91))].T
    XG_t[11,:,:] = mg5['XG'][np.ix_(i2,np.arange(91))].T
    XG_t[12,:,:] = mg5['XG'][np.ix_(i3,np.arange(91))].T
    
    YG_t[10,:,:] = mg5['YG'][np.ix_(i1,np.arange(91))].T
    YG_t[11,:,:] = mg5['YG'][np.ix_(i2,np.arange(91))].T
    YG_t[12,:,:] = mg5['YG'][np.ix_(i3,np.arange(91))].T
    
    RAZ_t[10,:,:] = mg5['RAZ'][np.ix_(i1,np.arange(91))].T
    RAZ_t[11,:,:] = mg5['RAZ'][np.ix_(i2,np.arange(91))].T
    RAZ_t[12,:,:] = mg5['RAZ'][np.ix_(i3,np.arange(91))].T
    

    mg5 = []

    return XG_t, YG_t, RAZ_t

#%%
if __name__ == '__main__':

    sys.path.append('/Users/ifenty/git_repo_mine/ECCOv4-py/')
    import ecco_v4_py as ecco

    llc = 90
    grid_dir = '/Volumes/ECCO_BASE/ECCO_v4r3/grid_llc90'

    XG, YG, RAZ = extract_G_point_grid_fields_from_mitgrid_as_tiles(llc, grid_dir)
    
    #%%
    lon_name = 'longitude'
    lat_name = 'latitude'
    
    ds = xr.Dataset({'RAZ': (['tile','j_g','i_g'], RAZ)}, 
                    coords={'tile': range(1,14),
                            'j_g': (('j_g',), range(1,92), {'axis':'Y',}),
                            'i_g': (('i_g',), range(1,92), {'axis':'X',}),
                            lon_name : (('tile','j_g','i_g'), XG),
                            lat_name : (('tile','j_g','i_g'), YG)})
    
    ds['i_g'].attrs['standard_name'] = 'i index of llc G-point'
    ds['i_g'].attrs['long_name'] = 'i index of llc G-point'
    ds['i_g'].attrs['axis'] = 'X'
    ds['i_g'].attrs['unit'] = 'non-dimensional'
    ds['i_g'].attrs['valid_range'] = [1, llc+1]

    ds['j_g'].attrs['standard_name'] = 'j index of llc G-point'
    ds['j_g'].attrs['long_name'] = 'j index of llc G-point'
    ds['j_g'].attrs['axis'] = 'Y'
    ds['j_g'].attrs['unit'] = 'non-dimensional'
    ds['j_g'].attrs['valid_range'] = [1, llc+1]

    ds['RAZ'].attrs['standard_name'] = 'area (m^2) of zeta point'

    ds[lat_name].name = 'latitude'
    ds[lat_name].attrs['standard_name']= 'latitude'
    ds[lat_name].attrs['long_name'] = 'latitude'
    ds[lat_name].attrs['units'] = 'degrees_north'
    ds[lat_name].attrs['valid_range'] = np.array([np.min(YG), np.max(YG)])
    ds[lat_name].valid_range
    
    ds[lon_name].name = 'longitude'
    ds[lon_name].attrs['standard_name']= 'longitude'
    ds[lon_name].attrs['long_name'] = 'longitude'
    ds[lon_name].attrs['units'] = 'degrees_east'
    ds[lon_name].attrs['valid_range'] = np.array([np.min(XG), np.max(XG)])
    ds[lon_name].valid_range
        
    ds.attrs['geospatial_lon_units'] = 'degrees_east'
    ds.attrs['geospatial_lat_units'] = 'degrees_north'
    ds.attrs['geospatial_lon_min'] = np.min(XG)
    ds.attrs['geospatial_lon_max'] = np.min(XG)
    ds.attrs['geospatial_lat_min'] = np.min(YG)
    ds.attrs['geospatial_lat_max'] = np.max(YG)
    ds.attrs['spatial_resolution'] = 'variable'
    ds.attrs['llc'] = llc
    ds.to_netcdf(grid_dir + '/' + 'G_point_grid_fields_as_tiles_latlon.nc')

    for tile_i in range(13):
        tmp = ds.isel(tile=tile_i)
        tmp[lon_name].attrs['valid_range'] = np.array([np.min(tmp[lon_name]), np.max(tmp[lon_name])])
        tmp[lat_name].attrs['valid_range'] = np.array([np.min(tmp[lat_name]), np.max(tmp[lat_name])])

        tmp.to_netcdf(grid_dir + '/' + 'G_point_grid_fields_as_tiles_latlon_tile_' + "{:02d}".format(tile_i) + '.nc')



    #%%
    DXC, DYG = extract_U_point_grid_fields_from_mitgrid_as_tiles(llc, grid_dir)
    
    dsu = xr.Dataset({'DXC': (['tile','j_u','i_u'], DXC),
                      'DYG': (['tile','j_u','i_u'], DYG)},
                    coords={'tile': range(1,14),
                            'j_u':   (('j_u',  ), range(1,91), {'axis':'Y',}),
                            'i_u': (('i_u',), range(1,92), {'axis':'X',})})
    
    dsu['i_u'].attrs['standard_name'] = 'i index of llc U-point'
    dsu['i_u'].attrs['long_name'] = 'i index of llc U-point'
    dsu['i_u'].attrs['axis'] = 'X'
    dsu['i_u'].attrs['unit'] = 'non-dimensional'
    dsu['i_u'].attrs['valid_range'] = [1, llc+1]

    dsu['j_u'].attrs['standard_name'] = 'j index of llc U-point'
    dsu['j_u'].attrs['long_name'] = 'j index of llc U-point'
    dsu['j_u'].attrs['axis'] = 'Y'
    dsu['j_u'].attrs['unit'] = 'non-dimensional'
    dsu['j_u'].attrs['valid_range'] = [1, llc+1]

    dsu.to_netcdf(grid_dir + '/' + 'U_point_grid_fields_as_tiles_latlon.nc')
    dsu.attrs['llc'] = llc
    for tile_i in range(13):
        tmp = dsu.isel(tile=tile_i)
        tmp.to_netcdf(grid_dir + '/' + 'U_point_grid_fields_as_tiles_latlon_tile_' + "{:02d}".format(tile_i) + '.nc')
    
