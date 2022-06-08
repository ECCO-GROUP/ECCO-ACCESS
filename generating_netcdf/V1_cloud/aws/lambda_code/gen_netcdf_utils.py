import os
import sys
import lzma
import uuid
import pickle
import datetime
import numpy as np
import xarray as xr
import pyresample as pr
from pprint import pprint
from pandas import read_csv
from collections import OrderedDict

# ==========================================================================================================================
# TIME STEPS
# ==========================================================================================================================
def find_all_time_steps(vars_to_load, var_directories, local, s3_files):
    """

    Collects list of all time steps. Time steps are collected from the
    file name of the data files for each variable.

    Parameters
    ----------
    vars_to_load : list[str]
        a list of variable names to load
    
    var_directories : dict{str:PosixPath}
        a dictionary, key:var; value:path to var
    

    Returns
    -------
    all_time_steps : numpy.array(dtype=numpy.int64)
        an array of all integer time steps to process

    """

    all_time_steps_var = {}
    all_time_steps_all_vars = []

    for var in vars_to_load:
        all_time_steps_var[var] = []

        if local:
            all_files = np.sort(list(var_directories[var].glob('*data')))
        else:
            all_files = s3_files[var]

        for file in all_files:
            ts = int(str(file).split('.')[-2])
            all_time_steps_all_vars.append(ts)
            all_time_steps_var[var].append(ts)


    unique_ts = np.unique(all_time_steps_all_vars)
    for var in vars_to_load:
        if np.array_equal(unique_ts, np.array(all_time_steps_var[var])):
            print('--- number of time steps equal ', var)
        else:
            print('--- number of time steps not equal ', var)
            sys.exit()

    # if we've made it this far, then we have a list of all
    # of the time steps to load in unique_ts
    # print(f'Time steps: {unique_ts}')
    all_time_steps = unique_ts

    return all_time_steps


# ==========================================================================================================================
# MAPPING FACTORS, LAND MASK, and LATLON GRID
# ==========================================================================================================================
def get_mapping_factors(dataset_dim, mapping_factors_dir, factors_to_get, debug_mode=False, extra_prints=False, k=0):
    # factors_to_get : factors to load in from the mapping_factors_dir
    # can be 'all', 'k', or 'both'
    grid_mappings_all = []
    grid_mappings_k = []

    if extra_prints: print('\nGetting Grid Mappings')
    grid_mapping_fname_all = mapping_factors_dir / 'ecco_latlon_grid_mappings_all.xz'
    grid_mapping_fname_2D = mapping_factors_dir / 'ecco_latlon_grid_mappings_2D.xz'
    grid_mapping_fname_3D = mapping_factors_dir / '3D' / f'ecco_latlon_grid_mappings_3D_{k}.xz'

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
        grid_mappings_all = []
        grid_mappings_k = []
    else:
        # Check to see that the mapping factors have been made
        if (dataset_dim == '2D' and grid_mapping_fname_2D.is_file()) or (dataset_dim == '3D' and grid_mapping_fname_3D.is_file()):
            # if so, load
            try:
                if factors_to_get == 'all' or factors_to_get == 'both':
                    if extra_prints: print(f'... loading ecco_latlon_grid_mappings_all.xz')
                    grid_mappings_all = pickle.load(lzma.open(grid_mapping_fname_all, 'rb'))

                if factors_to_get == 'k' or factors_to_get == 'both':
                    if dataset_dim == '2D':
                        if extra_prints: print(f'... loading ecco_latlon_grid_mappings_{dataset_dim}.xz')
                        grid_mappings_k = pickle.load(lzma.open(grid_mapping_fname_2D, 'rb'))
                    elif dataset_dim == '3D':
                        if extra_prints: print(f'... loading ecco_latlon_grid_mappings_{dataset_dim}_{k}.xz')
                        grid_mappings_k = pickle.load(lzma.open(grid_mapping_fname_3D, 'rb'))
            except:
                print(f'Unable to load grid mapping factors: {mapping_factors_dir}')
        else:
            print(f'Grid mapping factors have not been created or cannot be found: {mapping_factors_dir}')

    return (grid_mappings_all, grid_mappings_k)


def create_mapping_factors(ea, dataset_dim, mapping_factors_dir, debug_mode, source_grid_all, target_grid, target_grid_radius, source_grid_min_L, source_grid_max_L, source_grid_k, nk):
    print('\nCreating Grid Mappings')

    if not mapping_factors_dir.exists():
        try:
            mapping_factors_dir.mkdir()
        except:
            print ('cannot make %s ' % mapping_factors_dir)
    
    grid_mapping_fname_all = mapping_factors_dir / 'ecco_latlon_grid_mappings_all.xz'
    grid_mapping_fname_2D = mapping_factors_dir / 'ecco_latlon_grid_mappings_2D.xz'
    grid_mapping_fname_3D = mapping_factors_dir / '3D'

    if not grid_mapping_fname_3D.exists():
        try:
            grid_mapping_fname_3D.mkdir()
        except:
            print ('cannot make %s ' % grid_mapping_fname_3D)

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
        grid_mappings_all = []
        grid_mappings_k = []
    else:
        # first check to see if you have already calculated the grid mapping factors
        if dataset_dim == '3D':
            all_3D = True
            all_3D_fnames = [f'ecco_latlon_grid_mappings_3D_{i}.xz' for i in range(nk)]
            curr_3D_fnames = os.listdir(grid_mapping_fname_3D)
            for fname in all_3D_fnames:
                if fname not in curr_3D_fnames:  
                    all_3D = False
                    break

        if (dataset_dim == '2D' and grid_mapping_fname_2D.is_file()) or (dataset_dim == '3D' and all_3D):
            # Factors already made, continuing
            print('... mapping factors already created')

        else:
            # if not, make new grid mapping factors
            print('... no mapping factors found, recalculating')

            if ~(grid_mapping_fname_all.is_file()):
                # find the mapping between all points of the ECCO grid and the target grid.
                grid_mappings_all = \
                    ea.find_mappings_from_source_to_target(source_grid_all,
                                                            target_grid,
                                                            target_grid_radius,
                                                            source_grid_min_L,
                                                            source_grid_max_L)

                # Save grid_mappings_all
                try:
                    pickle.dump(grid_mappings_all, lzma.open(grid_mapping_fname_all, 'wb'))
                except:
                    print('cannot make %s ' % grid_mapping_fname_all)

            # If the dataset is 2D, only compute one level of the mapping factors
            if dataset_dim == '2D':
                nk = 1

            # Find the mapping factors between all wet points of the ECCO grid
            # at each vertical level and the target grid
            for k_i in range(nk):
                print(k_i)
                grid_mappings_k = \
                    ea.find_mappings_from_source_to_target(source_grid_k[k_i],
                                                            target_grid,
                                                            target_grid_radius,
                                                            source_grid_min_L,
                                                            source_grid_max_L)

                try:
                    if dataset_dim == '2D':
                        pickle.dump(grid_mappings_k, lzma.open(grid_mapping_fname_2D, 'wb'))
                    elif dataset_dim == '3D':
                        fname_3D = grid_mapping_fname_3D / f'ecco_latlon_grid_mappings_3D_{k_i}.xz'
                        pickle.dump(grid_mappings_k, lzma.open(fname_3D, 'wb'))
                except:
                    print('cannot make %s ' % mapping_factors_dir)
    return


def get_land_mask(mapping_factors_dir, k=0, debug_mode=False, extra_prints=False):
    if extra_prints: print('\nGetting Land Mask')

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING LAND MASK')
        land_mask_ll = []
    else:
        # check to see if you have already calculated the land mask
        land_mask_fname = mapping_factors_dir / 'land_mask' / f'ecco_latlon_land_mask_{k}.xz'

        if land_mask_fname.is_file():
            # if so, load
            if extra_prints: print('.... loading land_mask_ll')
            try:
                land_mask_ll = pickle.load(lzma.open(land_mask_fname, 'rb'))
            except:
                    print(f'Unable to load land mask: {land_mask_fname}')
        else:
            print(f'Land mask has not been created or cannot be found: {land_mask_fname}')

    return land_mask_ll


def create_land_mask(ea, mapping_factors_dir, debug_mode, nk, target_grid_shape, ecco_grid, dataset_dim):
    print('\nCreating Land Mask')

    ecco_land_mask_c = ecco_grid.maskC.copy(deep=True)
    ecco_land_mask_c.values = np.where(ecco_land_mask_c==True, 1, np.nan)

    if not mapping_factors_dir.exists():
        try:
            mapping_factors_dir.mkdir()
        except:
            print ('cannot make %s ' % mapping_factors_dir)

    land_mask_fname = mapping_factors_dir / 'land_mask'

    if not land_mask_fname.exists():
        try:
            land_mask_fname.mkdir()
        except:
            print ('cannot make %s ' % land_mask_fname)

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING LAND MASK')
        land_mask_ll = []

    else:
        # first check to see if you have already calculated the landmask
        all_mask = True
        all_mask_fnames = [f'ecco_latlon_land_mask_{i}.xz' for i in range(nk)]
        curr_mask_fnames = os.listdir(land_mask_fname)
        for fname in all_mask_fnames:
            if fname not in curr_mask_fnames:  
                all_mask = False
                break

        if all_mask:
            # Land mask already made, continuing
            print('... land mask already created')
            return

        else:
            # if not, recalculate.
            # print('.... making new land_mask_ll')
            # land_mask_ll = np.zeros((nk, target_grid_shape[0], target_grid_shape[1]))

            (source_indices_within_target_radius_i, \
            nearest_source_index_to_target_index_i), _ = get_mapping_factors(dataset_dim, mapping_factors_dir,
                                                                            'all', debug_mode)

            for k in range(nk):
                print(k)

                source_field = ecco_land_mask_c.values[k,:].ravel()

                land_mask_ll = ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                                            nearest_source_index_to_target_index_i,
                                                            source_field, target_grid_shape,
                                                            operation='nearest', 
                                                            allow_nearest_neighbor=True)

                try:
                    fname_mask = land_mask_fname / f'ecco_latlon_land_mask_{k}.xz'
                    pickle.dump(land_mask_ll.ravel(), lzma.open(fname_mask, 'wb'))
                except:
                    print ('cannot pickle dump %s ' % land_mask_fname)
    return


def get_latlon_grid(mapping_factors_dir, debug_mode=False):
    # print('\nGetting latlon grid')

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING latlon grid')
        latlon_grid = {}
    else:
        # check to see if you have already calculated the latlon_grid
        latlon_grid_name = mapping_factors_dir / f'latlon_grid.xz'

        if latlon_grid_name.is_file():
            # if so, load
            # print('.... loading latlon_grid')
            try:
                latlon_grid = pickle.load(lzma.open(latlon_grid_name, 'rb'))
            except:
                    print(f'Unable to load land mask: {latlon_grid_name}')
        else:
            print(f'Land mask has not been created or cannot be found: {latlon_grid_name}')

    return latlon_grid


def create_all_factors(ea, config_metadata, dataset_dim, debug_mode):
    mapping_factors_dir = config_metadata['mapping_factors_dir']
    nk = config_metadata['num_vertical_levels']
    ecco_grid = xr.open_dataset(config_metadata['ecco_grid_dir'] / config_metadata['ecco_grid_filename'])

    if not mapping_factors_dir.exists():
        try:
            mapping_factors_dir.mkdir()
        except:
            print ('cannot make %s ' % mapping_factors_dir)

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING LATLON GRID')

    else:
        wet_pts_k = {}
        xc_wet_k = {}
        yc_wet_k = {}

        # Dictionary of pyresample 'grids' for each level of the ECCO grid where
        # there are wet points.  Used for the bin-averaging.  We don't want to bin
        # average dry points.
        source_grid_k = {}
        # print('\nSwath Definitions')
        # print('... making swath definitions for latlon grid levels 1..nk')
        for k in range(nk):
            wet_pts_k[k] = np.where(ecco_grid.hFacC[k,:] > 0)
            xc_wet_k[k] = ecco_grid.XC.values[wet_pts_k[k]]
            yc_wet_k[k] = ecco_grid.YC.values[wet_pts_k[k]]

            source_grid_k[k] = pr.geometry.SwathDefinition(lons=xc_wet_k[k], lats=yc_wet_k[k])


        # The pyresample 'grid' information for the 'source' (ECCO grid) defined using
        # all XC and YC points, even land.  Used to create the land mask
        source_grid_all =  pr.geometry.SwathDefinition(lons=ecco_grid.XC.values.ravel(),
                                                        lats=ecco_grid.YC.values.ravel())

        # the largest and smallest length of grid cell size in the ECCO grid.  Used
        # to determine how big of a lookup table we need to do the bin-average interp.
        source_grid_min_L = np.min([float(ecco_grid.dyG.min().values), float(ecco_grid.dxG.min().values)])
        source_grid_max_L = np.max([float(ecco_grid.dyG.max().values), float(ecco_grid.dxG.max().values)])


        # Define the TARGET GRID -- a lat lon grid
        ## create target grid.
        product_name = ''

        data_res = config_metadata['data_res']
        data_max_lat = config_metadata['data_max_lat']
        area_extent = config_metadata['area_extent']
        dims = [int(d/data_res) for d in config_metadata['dims']]

        # Grid projection information
        proj_info = {'area_id':'longlat',
                        'area_name':'Plate Carree',
                        'proj_id':'EPSG:4326',
                        'proj4_args':'+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'}

        _, _, target_grid, target_grid_lons, target_grid_lats = ea.generalized_grid_product(product_name,
                                                                                            data_res,
                                                                                            data_max_lat,
                                                                                            area_extent,
                                                                                            dims,
                                                                                            proj_info)

        # pull out just the lats and lons (1D arrays)
        target_grid_lons_1D = target_grid_lons[0,:]
        target_grid_lats_1D = target_grid_lats[:,0]

        # calculate the areas of the lat-lon grid
        ea_area = ea.area_of_latlon_grid(-180, 180, -90, 90, data_res, data_res, less_output=True)
        lat_lon_grid_area = ea_area['area']
        target_grid_shape = lat_lon_grid_area.shape


        # calculate effective radius of each target grid cell.  required for the bin
        # averaging
        target_grid_radius = np.sqrt(lat_lon_grid_area / np.pi).ravel()


        # CALCULATE GRID-TO-GRID MAPPING FACTORS
        if not isinstance(dataset_dim, list):
            dataset_dim = [dataset_dim]
        for dim in dataset_dim:
            create_mapping_factors(ea, dim, mapping_factors_dir, debug_mode, 
                                    source_grid_all, target_grid, target_grid_radius, 
                                    source_grid_min_L, source_grid_max_L, source_grid_k, nk)

            # make a land mask in lat-lon using hfacC
            create_land_mask(ea, mapping_factors_dir, debug_mode, nk, 
                                target_grid_shape, ecco_grid, dim)

        ## MAKE LAT AND LON BOUNDS FOR NEW DATA ARRAYS
        lat_bounds = np.zeros((dims[1],2))
        for i in range(dims[1]):
            lat_bounds[i,0] = target_grid_lats[i,0] - data_res/2
            lat_bounds[i,1] = target_grid_lats[i,0] + data_res/2

        lon_bounds = np.zeros((dims[0],2))
        for i in range(dims[0]):
            lon_bounds[i,0] = target_grid_lons[0,i] - data_res/2
            lon_bounds[i,1] = target_grid_lons[0,i] + data_res/2

        # Make depth bounds
        depth_bounds = np.zeros((nk,2))
        tmp = np.cumsum(ecco_grid.drF.values)

        for k in range(nk):
            if k == 0:
                depth_bounds[k,0] = 0.0
            else:
                depth_bounds[k,0] = -tmp[k-1]
            depth_bounds[k,1] = -tmp[k]


        latlon_bounds = {'lat':lat_bounds, 'lon':lon_bounds}
        target_grid_dict = {'shape':target_grid_shape, 'lats_1D':target_grid_lats_1D, 'lons_1D':target_grid_lons_1D}
        latlon_grid = [latlon_bounds, depth_bounds, target_grid_dict, wet_pts_k]

        # print('\nCreating latlon grid')
        latlon_grid_name = mapping_factors_dir / f'latlon_grid.xz'
        if latlon_grid_name.is_file():
            # latlon grid already made, continuing
            print('... latlon grid already created')
            return
        else:
            # if not, recalculate.
            # print('.... making new latlon_grid')
            try:
                pickle.dump(latlon_grid, lzma.open(latlon_grid_name, 'wb'))
            except:
                print ('cannot pickle dump %s ' % latlon_grid_name)
    return


# ==========================================================================================================================
# VARIABLE LOADING UTILS
# ==========================================================================================================================
def latlon_load_2D(ea, ecco, wet_pts_k, target_grid, mds_var_dir, mds_file, record_end_time, mapping_factors_dir):
    if isinstance(mds_file, str):
        mds_file_name = mds_file.split('/')[-1]
    else:
        mds_file_name = mds_file.name
    F = ecco.read_llc_to_tiles(mds_var_dir, mds_file_name,
                                llc=90, skip=0,
                                nk=1, nl=1,
                                filetype='>f',
                                less_output=True,
                                use_xmitgcm=False)

    F_wet_native = F[wet_pts_k[0]]

    # get mapping factors for the the surface level
    _, grid_mappings_k = get_mapping_factors('2D', mapping_factors_dir, 'k', k=0)
    ll_land_mask = get_land_mask(mapping_factors_dir, k=0)

    source_indices_within_target_radius_i, \
    nearest_source_index_to_target_index_i = grid_mappings_k

    # transform to new grid
    F_ll =  \
        ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                    nearest_source_index_to_target_index_i,
                                    F_wet_native,
                                    target_grid['shape'],
                                    operation='mean',
                                    land_mask=ll_land_mask,
                                    allow_nearest_neighbor=True)

    # expand F_ll with time dimension
    F_ll = np.expand_dims(F_ll, 0)

    F_DA = xr.DataArray(F_ll,
                        coords=[[record_end_time],
                                target_grid['lats_1D'],
                                target_grid['lons_1D']],
                        dims=["time", "latitude","longitude"])
    
    return F_DA


def latlon_load_3D(ea, ecco, Z, wet_pts_k, target_grid, mds_var_dir, 
                    mds_file, record_end_time, nk, mapping_factors_dir):
    if isinstance(mds_file, str):
        mds_file_name = mds_file.split('/')[-1]
    else:
        mds_file_name = mds_file.name
    F = ecco.read_llc_to_tiles(mds_var_dir, mds_file_name,
                                llc=90, skip=0, nk=nk,
                                nl=1,
                                filetype='>f',
                                less_output=True,
                                use_xmitgcm=False)

    F_ll = np.zeros((nk,360,720))

    for k in range(nk):
        F_wet_native = F[k][wet_pts_k[k]]

        _, grid_mappings_k = get_mapping_factors('3D', mapping_factors_dir, 'k', k=k)
        ll_land_mask = get_land_mask(mapping_factors_dir, k=k)

        source_indices_within_target_radius_i, \
        nearest_source_index_to_target_index_i = grid_mappings_k

        F_ll[k,:] =  \
            ea.transform_to_target_grid(source_indices_within_target_radius_i,
                                        nearest_source_index_to_target_index_i,
                                        F_wet_native, target_grid['shape'], land_mask=ll_land_mask,
                                        operation='mean', allow_nearest_neighbor=True)


    # expand F_ll with time dimension
    F_ll = np.expand_dims(F_ll, 0)

    # Delete F from memory, not needed anymore
    del(F)

    F_DA = xr.DataArray(F_ll,
                        coords=[[record_end_time],
                                Z,
                                target_grid['lats_1D'],
                                target_grid['lons_1D']],
                        dims=["time", "Z", "latitude","longitude"])

    return F_DA


def latlon_load(ea, ecco, Z, wet_pts_k, target_grid, mds_var_dir, mds_file,
                record_end_time, nk, dataset_dim, var, output_freq_code, mapping_factors_dir):
    if dataset_dim == '2D':
        F_DA = latlon_load_2D(ea, ecco, wet_pts_k, target_grid, 
                                mds_var_dir, mds_file, record_end_time, mapping_factors_dir)
    elif dataset_dim == '3D':
        F_DA = latlon_load_3D(ea, ecco, Z, wet_pts_k, target_grid, 
                                mds_var_dir, mds_file, record_end_time, nk, mapping_factors_dir)

    # assign name to data array
    # print('... assigning name', var)
    F_DA.name = var

    F_DS = F_DA.to_dataset()

    #   add time bounds object
    if 'AVG' in output_freq_code:
        tb_ds, _ = \
            ecco.make_time_bounds_and_center_times_from_ecco_dataset(F_DS,
                                                                        output_freq_code)
        F_DS = xr.merge((F_DS, tb_ds))
        F_DS = F_DS.set_coords('time_bnds')

    return F_DS


def native_load(ecco, var, ecco_grid, ecco_grid_dir_mds, mds_var_dir, mds_file, output_freq_code, cur_ts):
    # land masks
    ecco_land_mask_c  = ecco_grid.maskC.copy(deep=True)
    ecco_land_mask_c.values = np.where(ecco_land_mask_c==True, 1, np.nan)
    ecco_land_mask_w  = ecco_grid.maskW.copy(deep=True)
    ecco_land_mask_w.values = np.where(ecco_land_mask_w==True, 1, np.nan)
    ecco_land_mask_s  = ecco_grid.maskS.copy(deep=True)
    ecco_land_mask_s.values = np.where(ecco_land_mask_s==True, 1, np.nan)

    short_mds_name = mds_file.name.split('.')[0]

    F_DS = ecco.load_ecco_vars_from_mds(mds_var_dir,
                                        mds_grid_dir = ecco_grid_dir_mds,
                                        mds_files = short_mds_name,
                                        vars_to_load = var,
                                        drop_unused_coords = True,
                                        grid_vars_to_coords = False,
                                        output_freq_code=output_freq_code,
                                        model_time_steps_to_load=cur_ts,
                                        less_output = True)

    vars_to_drop = set(F_DS.data_vars).difference(set([var]))
    F_DS.drop_vars(vars_to_drop)

    # determine all of the dimensions used by data variables
    all_var_dims = set([])
    for ecco_var in F_DS.data_vars:
        all_var_dims = set.union(all_var_dims, set(F_DS[ecco_var].dims))

    # mask the data variables
    for data_var in F_DS.data_vars:
        data_var_dims = set(F_DS[data_var].dims)
        if len(set.intersection(data_var_dims, set(['k','k_l','k_u','k_p1']))) > 0:
            data_var_3D = True
        else:
            data_var_3D = False

        # 'i, j = 'c' point
        if len(set.intersection(data_var_dims, set(['i','j']))) == 2 :
            if data_var_3D:
                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_c.values
                # print('... masking with 3D maskC ', data_var)
            else:
                # print('... masking with 2D maskC ', data_var)
                F_DS[data_var].values= F_DS[data_var].values * ecco_land_mask_c[0,:].values

        # i_g, j = 'u' point
        elif len(set.intersection(data_var_dims, set(['i_g','j']))) == 2 :
            if data_var_3D:
                # print('... masking with 3D maskW ', data_var)
                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_w.values
            else:
                # print('... masking with 2D maskW ', data_var)
                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_w[0,:].values

        # i, j_g = 's' point
        elif len(set.intersection(data_var_dims, set(['i','j_g']))) == 2 :
            if data_var_3D:
                # print('... masking with 3D maskS ', data_var)
                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_s.values
            else:
                # print('... masking with 2D maskS ', data_var)
                F_DS[data_var].values = F_DS[data_var].values * ecco_land_mask_s[0,:].values

        else:
            print('I cannot determine dimension of data variable ', data_var)
            sys.exit()
    
    return F_DS


# ==========================================================================================================================
# METADATA UTILS
# ==========================================================================================================================
def global_DS_changes(F_DS, output_freq_code, grouping, var, array_precision, ecco_grid,
                        depth_bounds, product_type, latlon_bounds, netcdf_fill_value, dataset_dim, record_times):
    if 'AVG' in output_freq_code:
        F_DS.time_bnds.values[0][0] = record_times['start']
        F_DS.time_bnds.values[0][1] = record_times['end']
        F_DS.time.values[0] = record_times['center']

    # Possibly rename variable if indicated
    if 'variable_rename' in grouping.keys():
        rename_pairs = grouping['variable_rename'].split(',')

        for rename_pair in rename_pairs:
            orig_var_name, new_var_name = rename_pair.split(':')

            if var == orig_var_name:
                F_DS = F_DS.rename({orig_var_name:new_var_name})
                # print('renaming from ', orig_var_name, new_var_name)
                # print(F_DS.data_vars)

    # cast data variable to desired precision
    for data_var in F_DS.data_vars:
        if F_DS[data_var].values.dtype != array_precision:
            F_DS[data_var].values = F_DS[data_var].astype(array_precision)


    # set valid min and max, and replace nan with fill values
    for data_var in F_DS.data_vars:
        F_DS[data_var].attrs['valid_min'] = np.nanmin(F_DS[data_var].values)
        F_DS[data_var].attrs['valid_max'] = np.nanmax(F_DS[data_var].values)

        # replace nan with fill value
        F_DS[data_var].values = np.where(np.isnan(F_DS[data_var].values),
                                            netcdf_fill_value, F_DS[data_var].values)

    # add bounds to spatial coordinates
    if product_type == 'latlon':
        #   assign lat and lon bounds
        F_DS=F_DS.assign_coords({"latitude_bnds": (("latitude","nv"), latlon_bounds['lat'])})
        F_DS=F_DS.assign_coords({"longitude_bnds": (("longitude","nv"), latlon_bounds['lon'])})

        #   if 3D assign depth bounds, use Z as index
        if dataset_dim == '3D' and 'Z' in list(F_DS.coords):
            F_DS = F_DS.assign_coords({"Z_bnds": (("Z","nv"), depth_bounds)})

    elif product_type == 'native':
        if 'XC_bnds' in ecco_grid.coords:
            F_DS = F_DS.assign_coords({"XC_bnds": (("tile","j","i","nb"), ecco_grid['XC_bnds'])})
        if 'YC_bnds' in ecco_grid.coords:
            F_DS = F_DS.assign_coords({"YC_bnds": (("tile","j","i","nb"), ecco_grid['YC_bnds'])})

        #   if 3D assign depth bounds, use k as index
        if dataset_dim == '3D' and 'Z' in list(F_DS.coords):
            F_DS = F_DS.assign_coords({"Z_bnds": (("k","nv"), depth_bounds)})
    
    return F_DS


def sort_attrs(attrs):
    od = OrderedDict()

    keys = sorted(list(attrs.keys()),key=str.casefold)

    for k in keys:
        od[k] = attrs[k]

    return od


def find_podaac_metadata(podaac_dataset_table, filename, debug=False):
    """Return revised file metadata based on an input ECCO `filename`.

    This should consistently parse a filename that conforms to the
    ECCO filename conventions and match it to a row in my metadata
    table.

    """

    # Use filename components to find metadata row from podaac_dataset_table.
    if "_snap_" in filename:
        head, tail = filename.split("_snap_")
        head = f"{head}_snap"
    elif "_day_mean_" in filename:
        head, tail = filename.split("_day_mean_")
        head = f"{head}_day_mean"
    elif "_mon_mean_" in filename:
        head, tail = filename.split("_mon_mean_")
        head = f"{head}_mon_mean"
    else:
        raise Exception("Error: filename may not conform to ECCO V4r4 convention.")

    if debug:
        print('split filename into ', head, tail)

    tail = tail.split("_ECCO_V4r4_")[1]

    if debug:
        print('further split tail into ',  tail)

    # Get the filenames column from my table as a list of strings.
    names = podaac_dataset_table['DATASET.FILENAME']

    # Find the index of the row with a filename with matching head, tail chars.
    index = names.apply(lambda x: all([x.startswith(head), x.endswith(tail)]))

    # Select that row from podaac_dataset_table table and make a copy of it.
    metadata = podaac_dataset_table[index].iloc[0].to_dict()

    podaac_metadata = {
        'id': metadata['DATASET.PERSISTENT_ID'].replace("PODAAC-","10.5067/"),
        'metadata_link': f"https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName={metadata['DATASET.SHORT_NAME']}",
        'title': metadata['DATASET.LONG_NAME'],
    }
    if debug:
        print('\n... podaac metadata:')
        pprint(podaac_metadata)

    return podaac_metadata


def apply_podaac_metadata(xrds, podaac_metadata):
    """Apply attributes podaac_metadata to ECCO dataset and its variables.

    Attributes that are commented `#` are retained with no modifications.
    Attributes that are assigned `None` are dropped.
    New attributes added to dataset.

    """
    # REPLACE GLOBAL ATTRIBUTES WITH NEW/UPDATED DICTIONARY.
    atts = xrds.attrs
    for name, modifier in podaac_metadata.items():
        if callable(modifier):
            atts.update({name: modifier(x=atts[name])})

        elif modifier is None:
            if name in atts:
                del atts[name]
        else:
            atts.update({name: modifier})

    xrds.attrs = atts

    # MODIFY VARIABLE ATTRIBUTES IN A LOOP.
    for v in xrds.variables:
        if "gmcd_keywords" in xrds.variables[v].attrs:
            del xrds.variables[v].attrs['gmcd_keywords']

    return xrds  # Return the updated xarray Dataset.


def set_metadata(ecco, G, product_type, all_metadata, dataset_dim, output_freq_code, netcdf_fill_value,
                    grouping, filename_tail, output_dir_freq, dataset_description, podaac_dir):

    # ADD VARIABLE SPECIFIC METADATA TO VARIABLE ATTRIBUTES (DATA ARRAYS)
    # print('\n... adding metadata specific to the variable')
    G, grouping_gcmd_keywords = \
        ecco.add_variable_metadata(all_metadata['var_native'], G, less_output=False)

    if product_type == 'latlon':
        # print('\n... using latlon dataseta metadata specific to the variable')
        G, grouping_gcmd_keywords = \
            ecco.add_variable_metadata(all_metadata['var_latlon'], G, less_output=False)

    # ADD COORDINATE METADATA
    if product_type == 'latlon':
        # print('\n... adding coordinate metadata for latlon dataset')
        G = ecco.add_coordinate_metadata(all_metadata['coord_latlon'],G)

    elif product_type == 'native':
        # print('\n... adding coordinate metadata for native dataset')
        G = ecco.add_coordinate_metadata(all_metadata['coord_native'],G)

    # ADD GLOBAL METADATA
    # print("\n... adding global metadata for all datasets")
    G = ecco.add_global_metadata(all_metadata['global_all'], G, dataset_dim)

    if product_type == 'latlon':
        # print('\n... adding global meta for latlon dataset')
        G = ecco.add_global_metadata(all_metadata['global_latlon'], G, dataset_dim)
    elif product_type == 'native':
        # print('\n... adding global metadata for native dataset')
        G = ecco.add_global_metadata(all_metadata['global_native'], G, dataset_dim)

    # ADD GLOBAL METADATA ASSOCIATED WITH TIME AND DATE
    # print('\n... adding time / data global attrs')
    if 'AVG' in output_freq_code:
        G.attrs['time_coverage_start'] = str(G.time_bnds.values[0][0])[0:19]
        G.attrs['time_coverage_end'] = str(G.time_bnds.values[0][1])[0:19]

    else:
        G.attrs['time_coverage_start'] = str(G.time.values[0])[0:19]
        G.attrs['time_coverage_end'] = str(G.time.values[0])[0:19]

    # current time and date
    current_time = datetime.datetime.now().isoformat()[0:19]
    G.attrs['date_created'] = current_time
    G.attrs['date_modified'] = current_time
    G.attrs['date_metadata_modified'] = current_time
    G.attrs['date_issued'] = current_time

    # add coordinate attributes to the variables
    dv_coordinate_attrs = {}

    for dv in list(G.data_vars):
        dv_coords_orig = set(list(G[dv].coords))

        # REMOVE TIME STEP FROM LIST OF COORDINATES (PODAAC REQUEST)
        set_intersect = dv_coords_orig.intersection(set(['XC','YC','XG','YG','Z','Zp1','Zu','Zl','time']))

        dv_coordinate_attrs[dv] = " ".join(set_intersect)


    # print('\n... creating variable encodings')
    # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH DATA VAR
    dv_encoding = {}
    for dv in G.data_vars:
        dv_encoding[dv] = {'zlib':True,
                            'complevel':5,
                            'shuffle':True,
                            '_FillValue':netcdf_fill_value}

        # overwrite default coordinats attribute (PODAAC REQUEST)
        G[dv].encoding['coordinates'] = dv_coordinate_attrs[dv]

    # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
    # print('\n... creating coordinate encodings')
    coord_encoding = {}

    for coord in G.coords:
        # default encoding: no fill value, float32
        coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}

        if (G[coord].values.dtype == np.int32) or (G[coord].values.dtype == np.int64):
            coord_encoding[coord]['dtype'] ='int32'

        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

            if 'units' in G[coord].attrs:
                # apply units as encoding for time
                coord_encoding[coord]['units'] = G[coord].attrs['units']
                # delete from the attributes list
                del G[coord].attrs['units']

        elif coord == 'time_step':
            coord_encoding[coord]['dtype'] ='int32'

    # MERGE ENCODINGS for coordinates and variables
    encoding = {**dv_encoding, **coord_encoding}

    # MERGE GCMD KEYWORDS
    # print('\n... merging GCMD keywords')
    common_gcmd_keywords = G.keywords.split(',')
    gcmd_keywords_list = set(grouping_gcmd_keywords + common_gcmd_keywords)

    gcmd_keyword_str = ''
    for gcmd_keyword in gcmd_keywords_list:
        if len(gcmd_keyword_str) == 0:
            gcmd_keyword_str = gcmd_keyword
        else:
            gcmd_keyword_str += ', ' + gcmd_keyword

    #print(gcmd_keyword_str)
    G.attrs['keywords'] = gcmd_keyword_str

    ## ADD FINISHING TOUCHES

    # uuic
    # print('\n... adding uuid')
    G.attrs['uuid'] = str(uuid.uuid1())

    # add any dataset grouping specific comments.
    if 'comment' in grouping:
        G.attrs['comment'] = G.attrs['comment'] + ' ' + grouping['comment']

    # set the long name of the time attribute
    if 'AVG' in output_freq_code:
        G.time.attrs['long_name'] = 'center time of averaging period'
    else:
        G.time.attrs['long_name'] = 'snapshot time'

    # set averaging period duration and resolution
    # print('\n... setting time coverage resolution')
    # --- AVG MON
    if output_freq_code == 'AVG_MON':
        G.attrs['time_coverage_duration'] = 'P1M'
        G.attrs['time_coverage_resolution'] = 'P1M'

        date_str = str(np.datetime64(G.time.values[0],'M'))
        ppp_tttt = 'mon_mean'

    # --- AVG DAY
    elif output_freq_code == 'AVG_DAY':
        G.attrs['time_coverage_duration'] = 'P1D'
        G.attrs['time_coverage_resolution'] = 'P1D'

        date_str = str(np.datetime64(G.time.values[0],'D'))
        ppp_tttt = 'day_mean'

    # --- SNAPSHOT
    elif output_freq_code == 'SNAPSHOT':
        G.attrs['time_coverage_duration'] = 'P0S'
        G.attrs['time_coverage_resolution'] = 'P0S'

        G.time.attrs.pop('bounds')

        # convert from original
        #   '1992-01-16T12:00:00.000000000'
        # to new format
        # '1992-01-16T120000'
        date_str = str(G.time.values[0])[0:19].replace(':','')
        ppp_tttt = 'snap'

    ## construct filename
    # print('\n... creating filename')

    filename = grouping['filename'] + '_' + ppp_tttt + '_' + date_str + filename_tail

    # make subdirectory for the grouping
    output_dir = output_dir_freq / grouping['filename']
    # print('\n... creating output_dir', output_dir)

    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except:
            print ('cannot make %s ' % output_dir)


    # create full pathname for netcdf file
    netcdf_output_filename = output_dir / filename

    # add product name attribute = filename
    G.attrs['product_name'] = filename

    # add summary attribute = description of dataset
    G.attrs['summary'] = dataset_description + ' ' + G.attrs['summary']

    # get podaac metadata based on filename
    # print('\n... getting PODAAC metadata')
    # podaac_dataset_table = read_csv(podaac_dir / 'datasets.csv')
    podaac_dataset_table = read_csv(podaac_dir)
    podaac_metadata = find_podaac_metadata(podaac_dataset_table, filename)

    # apply podaac metadata based on filename
    # print('\n... applying PODAAC metadata')
    G = apply_podaac_metadata(G, podaac_metadata)

    # sort comments alphabetically
    # print('\n... sorting global attributes')
    G.attrs = sort_attrs(G.attrs)

    # add one final comment (PODAAC request)
    G.attrs["coordinates_comment"] = \
        "Note: the global 'coordinates' attribute describes auxillary coordinates."

    return (G, netcdf_output_filename, encoding)