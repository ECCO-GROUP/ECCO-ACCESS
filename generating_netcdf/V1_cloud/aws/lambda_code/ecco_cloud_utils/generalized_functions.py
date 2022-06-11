# -*- coding: utf-8 -*-
import numpy as np
import pyresample as pr


# return (source_grid_min_L, source_grid_max_L, source_grid, data_grid_lons, data_grid_lats)
def generalized_grid_product(product_name,
                             data_res,
                             data_max_lat,
                             area_extent,
                             dims,
                             proj_info):

    # data_res: in degrees
    
    # minimum Length of data product grid cells (km)
    source_grid_min_L = np.cos(np.deg2rad(data_max_lat))*data_res*112e3

    # maximum length of data roduct grid cells (km)
    # data product at equator has grid spacing of data_res*112e3 m
    source_grid_max_L = data_res*112e3


    #area_extent: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
    areaExtent = (area_extent[0], area_extent[1],
                  area_extent[2], area_extent[3])

    # Corressponds to resolution of grid from data
    cols = dims[0]
    rows = dims[1]

    # USE PYRESAMPLE TO GENERATE THE LAT/LON GRIDS
    # -- note we do not have to use pyresample for this, we could
    # have created it manually using the np.meshgrid or some other method
    # if we wanted.
    tmp_data_grid = pr.area_config.get_area_def(proj_info['area_id'], proj_info['area_name'],
                                                proj_info['proj_id'], proj_info['proj4_args'],
                                                cols, rows, areaExtent)

    data_grid_lons, data_grid_lats = tmp_data_grid.get_lonlats()

    # Changes longitude bounds from 0-360 to -180-180, doesnt change if its already -180-180
    data_grid_lons, data_grid_lats = pr.utils.check_and_wrap(data_grid_lons,
                                                             data_grid_lats)

    # Define the 'swath' (in the terminology of the pyresample module)
    # as the lats/lon pairs of the source observation grid
    # The routine needs the lats and lons to be one-dimensional vectors.
    source_grid = pr.geometry.SwathDefinition(lons=data_grid_lons.ravel(),
                                              lats=data_grid_lats.ravel())

    return (source_grid_min_L, source_grid_max_L, source_grid, data_grid_lons, data_grid_lats)