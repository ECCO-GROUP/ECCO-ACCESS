# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:10:13 2020

@author: Ian
"""

import numpy as np




# %%
def llc_tiles_to_compact(data_tiles, less_output = False):
    """

    Converts a numpy binary array in the 'compact' format of the 
    lat-lon-cap (LLC) grids and converts it to the '13 tiles' format
    of the LLC grids.  

    Parameters
    ----------
    data_tiles : ndarray
        a numpy array organized by, at most, 
        13 tiles x nl x nk x llc x llc
    
        where dimensions 'nl' and 'nk' are optional.
        
    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output
        
    Returns
    -------
    data_compact : ndarray
        a numpy array of dimension nl x nk x 13*llc x llc 

    Note
    ----
    If dimensions nl or nk are singular, they are not included 
    as dimensions in data_compact

    """
   
    data_faces   = llc_tiles_to_faces(data_tiles, less_output=less_output)
    data_compact = llc_faces_to_compact(data_faces, less_output=less_output)
        
    return data_compact



def llc_tiles_to_faces(data_tiles, less_output=False):
    """

    Converts an array of 13 'tiles' from the lat-lon-cap grid
    and rearranges them to 5 faces.  Faces 1,2,4, and 5 are approximately 
    lat-lon while face 3 is the Arctic 'cap' 

    Parameters
    ----------
    data_tiles : 
        An array of dimension 13 x nl x nk x llc x llc
        
    If dimensions nl or nk are singular, they are not included 
        as dimensions of data_tiles    

    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
    
    Returns
    -------
    F : dict
        a dictionary containing the five lat-lon-cap faces
        
        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F
        
        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc  
    
    """
    
    # ascertain how many dimensions are in the faces (minimum 3, maximum 5)
    dims = data_tiles.shape
    num_dims = len(dims)
    
    # the final dimension is always length llc
    llc = dims[-1]

    # tiles is always just before (y,x) dims
    num_tiles = dims[-3]

    if less_output == False:
        print('num tiles, ', num_tiles)

    if num_dims == 3: # we have a 13 2D slices (tile, y, x)
        f1 = np.zeros((3*llc, llc))
        f2 = np.zeros((3*llc, llc))
        f3 = np.zeros((llc, llc))
        f4 = np.zeros((llc, 3*llc))
        f5 = np.zeros((llc, 3*llc))

    elif num_dims == 4: # 13 3D slices (time or depth, tile, y, x)

        nk = dims[0]
        
        f1 = np.zeros((nk, 3*llc, llc))
        f2 = np.zeros((nk, 3*llc, llc))
        f3 = np.zeros((nk, llc, llc))
        f4 = np.zeros((nk, llc, 3*llc))
        f5 = np.zeros((nk, llc, 3*llc))

    elif num_dims == 5: # 4D slice (time or depth, time or depth, tile, y, x)
        nl = dims[0]
        nk = dims[1]

        f1 = np.zeros((nl,nk, 3*llc, llc))
        f2 = np.zeros((nl,nk, 3*llc, llc))
        f3 = np.zeros((nl,nk, llc, llc))
        f4 = np.zeros((nl,nk, llc, 3*llc))
        f5 = np.zeros((nl,nk, llc, 3*llc))

    else:
        print ('can only handle tiles that have 2, 3, or 4 dimensions!')
        return []

    # Map data on the tiles to the faces

    # 2D slices on 13 tiles
    if num_dims == 3: 
        
        f1[llc*0:llc*1,:] = data_tiles[0,:]

        f1[llc*1:llc*2,:] = data_tiles[1,:]
        f1[llc*2:,:]      = data_tiles[2,:]

        f2[llc*0:llc*1,:] = data_tiles[3,:]
        f2[llc*1:llc*2,:] = data_tiles[4,:]
        f2[llc*2:,:]      = data_tiles[5,:]
        
        f3                = data_tiles[6,:]

        f4[:,llc*0:llc*1] = data_tiles[7,:]
        f4[:,llc*1:llc*2] = data_tiles[8,:]
        f4[:,llc*2:]      = data_tiles[9,:]
        
        f5[:,llc*0:llc*1] = data_tiles[10,:]
        f5[:,llc*1:llc*2] = data_tiles[11,:]
        f5[:,llc*2:]      = data_tiles[12,:]

    # 3D slices on 13 tiles
    elif num_dims == 4: 

        for k in range(nk):
            f1[k,llc*0:llc*1,:] = data_tiles[k,0,:]

            f1[k,llc*1:llc*2,:] = data_tiles[k,1,:]
            f1[k,llc*2:,:]      = data_tiles[k,2,:]

            f2[k,llc*0:llc*1,:] = data_tiles[k,3,:]
            f2[k,llc*1:llc*2,:] = data_tiles[k,4,:]
            f2[k,llc*2:,:]      = data_tiles[k,5,:]
            
            f3[k,:]             = data_tiles[k,6,:]

            f4[k,:,llc*0:llc*1] = data_tiles[k,7,:]
            f4[k,:,llc*1:llc*2] = data_tiles[k,8,:]
            f4[k,:,llc*2:]      = data_tiles[k,9,:]
            
            f5[k,:,llc*0:llc*1] = data_tiles[k,10,:]
            f5[k,:,llc*1:llc*2] = data_tiles[k,11,:]
            f5[k,:,llc*2:]      = data_tiles[k,12,:]

    # 4D slices on 13 tiles
    elif num_dims == 5: 
        for l in range(nl):
            for k in range(nk):
                f1[l,k,llc*0:llc*1,:] = data_tiles[l,k,0,:]

                f1[l,k,llc*1:llc*2,:] = data_tiles[l,k,1,:]
                f1[l,k,llc*2:,:]      = data_tiles[l,k,2,:]

                f2[l,k,llc*0:llc*1,:] = data_tiles[l,k,3,:]
                f2[l,k,llc*1:llc*2,:] = data_tiles[l,k,4,:]
                f2[l,k,llc*2:,:]      = data_tiles[l,k,5,:]
                
                f3[l,k,:]             = data_tiles[l,k,6,:]

                f4[l,k,:,llc*0:llc*1] = data_tiles[l,k,7,:]
                f4[l,k,:,llc*1:llc*2] = data_tiles[l,k,8,:]
                f4[l,k,:,llc*2:]      = data_tiles[l,k,9,:]
                
                f5[l,k,:,llc*0:llc*1] = data_tiles[l,k,10,:]
                f5[l,k,:,llc*1:llc*2] = data_tiles[l,k,11,:]
                f5[l,k,:,llc*2:]      = data_tiles[l,k,12,:]

    # Build the F dictionary
    F = {}
    F[1] = f1
    F[2] = f2
    F[3] = f3
    F[4] = f4
    F[5] = f5
    
    return F


def llc_faces_to_compact(F, less_output=True):
    """
    
    Converts a dictionary containing five 'faces' of the lat-lon-cap grid
    and rearranges it to the 'compact' llc format.


    Parameters
    ----------
    F : dict
        a dictionary containing the five lat-lon-cap faces
        
        F[n] is a numpy array of face n, n in [1..5]

        dimensions of each 2D slice of F

        - f1,f2: 3*llc x llc
        -    f3: llc x llc
        - f4,f5: llc x 3*llc  
    
    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output
        
    Returns
    -------
    data_compact : ndarray
        an array of dimension nl x nk x nj x ni 
        
        F is in the llc compact format.

    Note
    ----
    If dimensions nl or nk are singular, they are not included 
    as dimensions of data_compact

    """

    # pull the individual faces out of the F dictionary
    f1 = F[1]
    f2 = F[2]
    f3 = F[3]
    f4 = F[4]
    f5 = F[5]

    # ascertain how many dimensions are in the faces (minimum 2, maximum 4)
    dims = f3.shape
    num_dims = len(dims)

    # the final dimension is always the llc # 
    llc = dims[-1]

    # initialize the 'data_compact' array
    if num_dims == 2: # we have a 2D slice (x,y)
        data_compact = np.zeros((13*llc, llc))

    elif num_dims == 3: # 3D slice (x, y, time or depth)
        nk = dims[0]
        data_compact = np.zeros((nk, 13*llc, llc))

    elif num_dims == 4: # 4D slice (x,y,time and depth)
        nl = dims[0]
        nk = dims[1]
        data_compact = np.zeros((nl, nk, 13*llc, llc))
    else:
        print ('can only handle faces that have 2, 3, or 4 dimensions!')
        return []

    if less_output == False:
        print ('shape of face 3 ', f3.shape)

    if num_dims == 2:
        
        data_compact[:3*llc,:]      = f1
        data_compact[3*llc:6*llc,:] = f2
        data_compact[6*llc:7*llc,:] = f3
        
        for f in range(8,11):
            i1 = np.arange(0, llc)+(f-8)*llc
            i2 = np.arange(0,3*llc,3) + 7*llc + f -8
            data_compact[i2,:] = f4[:,i1]

        for f in range(11,14):
            i1 = np.arange(0, llc)+(f-11)*llc
            i2 = np.arange(0,3*llc,3) + 10*llc + f -11
            data_compact[i2,:] = f5[:,i1]
    
    elif num_dims == 3:
        # loop through k indicies
        print ('size of data compact ', data_compact.shape)
       
        for k in range(nk):
            data_compact[k,:3*llc,:]      = f1[k,:]
            data_compact[k,3*llc:6*llc,:] = f2[k,:]
            data_compact[k,6*llc:7*llc,:] = f3[k,:]

            # if someone could explain why I have to transpose
            # f4 and f5 when num_dims =3 or 4 I would be so grateful.
            # Just could not figure this out.  Transposing works but why!?
            for f in range(8,11):
                i1 = np.arange(0, llc)+(f-8)*llc
                i2 = np.arange(0,3*llc,3) + 7*llc + f - 8

                data_compact[k,i2,:] = f4[k,0:llc,i1].T

            for f in range(11,14):
                i1 = np.arange(0, llc)+(f-11)*llc
                i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                data_compact[k,i2,:] = f5[k,:,i1].T

    elif num_dims == 4:
        # loop through l and k indices
        for l in range(nl):
            for k in range(nk):
                data_compact[l,k,:3*llc,:]      = f1[l,k,:]
                data_compact[l,k,3*llc:6*llc,:] = f2[l,k,:]
                data_compact[l,k,6*llc:7*llc,:] = f3[l,k,:]
                
                for f in range(8,11):
                    i1 = np.arange(0, llc)+(f-8)*llc
                    i2 = np.arange(0,3*llc,3) + 7*llc + f -8
                    data_compact[l,k,i2,:]      = f4[l,k,:,i1].T

                for f in range(11,14):
                    i1 = np.arange(0, llc)+(f-11)*llc
                    i2 = np.arange(0,3*llc,3) + 10*llc + f -11
                    data_compact[l,k,i2,:]      = f5[l,k,:,i1].T


    if less_output == False:
        print ('shape of data_compact ', data_compact.shape)

    return data_compact