# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:28:40 2020

@author: Ian
"""
import numpy as np


#%%


def area_of_latlon_grid_cell(lon0, lon1, lat0, lat1):
    #https://gis.stackexchange.com/questions/29734/how-to-calculate-area-of-1-x-1-degree-cells-in-a-raster
    

    #It is a consequence of a theorem of Archimedes (c. 287-212 BCE) that 
    #for a spherical model of the earth, the area of a cell spanning 
    #longitudes l0 to l1 (l1 > l0) and latitudes f0 to f1 (f1 > f0) equals
    
    #(sin(f1) - sin(f0)) * (l1 - l0) * R^2
    
    #where
    #
    #    l0 and l1 are expressed in radians (not degrees or whatever).
    #    l1 - l0 is calculated modulo 2*pi (e.g., -179 - 181 = 2 degrees, not -362 degrees).
    #
    #    R is the authalic Earth radius, almost exactly 6371 km.
    #    (sin(f1) - sin(f0)) * (l1 - l0) * R^2
        
    R_Earth = 6371.0088e3 # m
    
    A = (np.sin(np.deg2rad(lat1)) - np.sin(np.deg2rad(lat0))) * \
        (np.deg2rad(lon1) - np.deg2rad(lon0)) * \
        R_Earth **2

    return A


def area_of_latlon_grid(lon0, lon1, lat0, lat1, dx, dy, less_output=False):
    # Calculates area of a latlon grid with edges lon0 and lon1
    # lat0 and lat1 with grid spacing of dx and dy
    
    # lons and lats are in degrees
    # dx and y are in degrees
    
    # resulting array has columns of lon, rows of lat.
    
    # Using -180, 180, -90, 90 we get total area of 510065.88 x10^6 km^2
    # using     R_Earth = 6371.0088e3 # m

    num_grid_cells_x = int((lon1-lon0)/dx)
    num_grid_cells_y = int((lat1-lat0)/dy)
    
    lons_grid_cell_edges = np.linspace(lon0, lon1, num_grid_cells_x + 1)
    lats_grid_cell_edges = np.linspace(lat0, lat1, num_grid_cells_y + 1)

    A = np.zeros((num_grid_cells_y))
    
    if not less_output:
        print(lons_grid_cell_edges)
        print(lats_grid_cell_edges)
        
    for lat_i in range(num_grid_cells_y):
        A[lat_i] = area_of_latlon_grid_cell(lons_grid_cell_edges[0], lons_grid_cell_edges[1],\
         lats_grid_cell_edges[lat_i], lats_grid_cell_edges[lat_i+1])
        
    return np.tile(A, (num_grid_cells_x,1)).T


