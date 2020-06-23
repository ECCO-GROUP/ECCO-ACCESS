# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:28:40 2020

@author: Ian
"""
import numpy as np
import pyresample as pr

#%%
def find_mappings_from_source_to_target(source_grid, target_grid,\
                                        target_grid_radius, \
                                        source_grid_min_L, source_grid_max_L, \
                                        neighbours = 100):
                                    
    #%%
    # source grid, target_grid : area or grid defintion objects from pyresample
    
    # target_grid_radius       : a vector indicating the radius of each
    #                            target grid cell (m)
    
    # source_grid_min_l, source_grid_max_L : min and max distances 
    #                            between adjacent source grid cells (m)
    
    # neighbours     : Specifies number of neighbours to look for when getting
    #                  the neighbour info of a cell using pyresample.
    #                  Default is 100 to limit memory usage.
    #                  Value given must be a whole number greater than 0
    
    # # of element of the source and target grids
    len_source_grid = source_grid.size
    len_target_grid = target_grid.size
    
    # the maximum radius of the target grid
    max_target_grid_radius = np.nanmax(target_grid_radius)
    
    # the maximum number of neighbors to consider when doing the bin averaging
    # assuming that we have the largets target grid radius and the smallest
    # source grid length. (upper bound)
    # the ceiling is used to ensure the result is a whole number > 0
    neighbours_upper_bound = np.ceil((max_target_grid_radius*2/source_grid_min_L)**2)
    
    # compare provided and upper_bound value for neighbours.
    # limit neighbours to the upper_bound if the supplied neighbours value is larger
    # since you dont need more neighbours than exists within a cell.
    if neighbours > neighbours_upper_bound:
        print('using more neighbours than upper bound.  limiting to the upper bound ' \
              f'of {int(neighbours_upper_bound)} neighbours')
        neighbours = neighbours_upper_bound
        
    # make sure neighbours is an int for pyresample
    # neighbours_upper_bound is float, and user input can be float
    neighbours = int(neighbours)
    
    ## FIRST FIND THE SET OF SOURCE GRID CELLS THAT FALL WITHIN THE SERACH
    ## RADIUS OF EACH TARGET GRID CELL
    
    # "target_grid_radius" is the half of the distance between 
    # target grid cells.  No need to search for source
    # grid points more than halfway to the next target grid cell.
    
    # the get_neighbour_info returned from pyresample is quite useful.  
    # Ax[2] is the matrix of 
    # closest data grid points for each model grid point
    # Ax[3] is the actual distance in meters
    # also cool is that Ax[3] is sorted, first column is closest, last column
    # is furthest.
    # for some reason the radius of influence has to be in an int.
    
    Ax_max_target_grid_r = \
        pr.kd_tree.get_neighbour_info(source_grid, \
                                      target_grid, \
                                      radius_of_influence=int(max_target_grid_radius),\
                                      neighbours=neighbours)
    
    # define a dictionary, which will contain the list of SOURCE grid cells
    # that are within the search radius of each TARGET grid cell 
    source_indices_within_target_radius_i = dict()
    
    # define a vector which is a COUNT of the # of SOURCE grid cells
    # that are within the search radius of each TARGET grid cell 
    num_source_indices_within_target_radius_i =\
        np.zeros((target_grid_radius.shape))
    

    ## SECOND FIND THE SINGLE SOURCE GRID CELL THAT IS CLOSEST TO EACH 
    ## TARGET GRID CELL, BUT ONLY SEARCH AS FAR AS SOURCE_GRID_MAX_L
    
    # the kd_tree can also find the sigle nearest neighbor within the
    # radius 'source_grid_max_L'.  This second search is needed because sometimes
    # the source grid is finer than the target grid and therefore we may 
    # end up in a situation where none of the centers of the SOURCE grid
    # fall within the small centers of the TARGET grid.
    # we'll look for the nearest SOURCE grid cell within 'source_grid_max_L'       
    
    Ax_nearest_within_source_grid_max_L = \
        pr.kd_tree.get_neighbour_info(source_grid, target_grid, \
                                      radius_of_influence=int(source_grid_max_L), \
                                      neighbours=1)
    
    # define a vector that will store the index of the source grid closest to
    # the target grid within the search radius 'source_grid_max_L'
    nearest_source_index_to_target_index_i = dict()


    # >> a list of i's to print debug statements for
    debug_is =  np.linspace(0,len_target_grid,20).astype(int)
    
    # loop through every model grid cell, pull out the SOURCE grid cells
    # that fall within the target grid radius and stick into the 
    # 'source_indices_within_target_radius_i' dictionary
    # and then do the same for the nearest neighbour business.
    
    current_valid_target_i = 0
    
    for i in range(len_target_grid):
        
        if Ax_nearest_within_source_grid_max_L[1][i] == True:
            
            # Ax[2][i,:] are the closest source grid indices 
            #            for target grid cell i
            # data_within_search_radius[i,:] is the T/F array for which
            #            of the closest 'neighbours' source grid indices are within
            #            the radius of this target grid cell i
            # -- so we're pulling out just those source grid indices
            #    that fall within the target grid cell radius
            
            ## FIRST RECORD THE SOURCE POINTS THAT FALL WITHIN TARGET_GRID_RADIUS
            dist_from_src_to_target = Ax_max_target_grid_r[3][current_valid_target_i]
            
            dist_within_target_r = dist_from_src_to_target <= target_grid_radius[i]
           
            src_indicies_here = Ax_max_target_grid_r[2][current_valid_target_i,:]            
        
            source_indices_within_target_radius_i[i] = \
                src_indicies_here[dist_within_target_r == True]
               
            # count the # source indices here 
            num_source_indices_within_target_radius_i[i] = \
                int(len(source_indices_within_target_radius_i[i] ))
            
            
        
            ## NOW RECORD THE NEAREST NEIGHBOR POINT WIHTIN SOURCE_GRID_MAX_L
            # when there is no source index within the search radius then
            # the 'get neighbour info' routine returns a dummy value of 
            # the length of the source grid.  so we test to see if that's the
            # value that was returned.  If not, then we are good to go.
            if Ax_nearest_within_source_grid_max_L[2][current_valid_target_i]\
                < len_source_grid:
                nearest_source_index_to_target_index_i[i] =\
                    Ax_nearest_within_source_grid_max_L[2][current_valid_target_i]
        
            # increment this little bastard            
            current_valid_target_i += 1
        
        # print progress.  always nice 
        if i in debug_is:
            print(str(int(i/len_target_grid*100)) + ' %')

    #%%
    return source_indices_within_target_radius_i,\
           num_source_indices_within_target_radius_i,\
           nearest_source_index_to_target_index_i


