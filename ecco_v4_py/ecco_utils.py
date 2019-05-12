#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ECCO v4 Python: Utililites

This module includes utility routines that operate on the Dataset or DataArray Objects 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""

from __future__ import division, print_function
import numpy as np
import xarray as xr
import datetime
import shapefile
import time
import dateutil
import glob
import os



#%%

def make_time_bounds_and_center_times_from_ecco_dataset(ecco_dataset, \
                                                        output_freq_code):
    """

    Given an ecco_dataset object (ecco_dataset) with time variables that 
    correspond with the 'end' of averaging time periods
    and an output frequency code (AVG_MON, AVG_DAY, AVG_WEEK, or AVG_YEAR), 
    create a time_bounds array of dimension 2xn, with 'n' being the number
    of time averaged records in ecco_dataset, each with two datetime64 
    variables, one for the averaging period start, one
    for the averaging period end.  
    
    The routine also creates an array of times corresponding to the 'middle'
    of each averaging period.
    
    Parameters
    ----------    
    ecco_dataset : xarray Dataset
        an xarray dataset with 'time' variables representing the times at 
        the 'end' of an averaging periods


    Returns
    -------
    time_bnds : np.array(dtype=np.datetime64)
        a datetime64 with the start and end time(s) of the averaging periods
    
    center_times :np.array(dtype=np.datetime64)
        a numpy array containing the 'center' time of the averaging period(s)
    

    """ 
 
    if ecco_dataset.time.size == 1:
        
        if isinstance(ecco_dataset.time.values, np.ndarray):
            time_tmp = ecco_dataset.time.values[0]
        else:
            time_tmp = ecco_dataset.time.values
       
        time_bnds, center_times = \
            make_time_bounds_from_ds64(time_tmp,\
                                       output_freq_code)

        time_bnds=np.expand_dims(time_bnds,0)

    else:
        time_start = []
        time_end  = []
        center_time = []
        for time_i in range(len(ecco_dataset.iter)):
             tb, ct = \
                 make_time_bounds_from_ds64(ecco_dataset.time.values[time_i], 
                                  output_freq_code)
             
             time_start.append(tb[0])
             time_end.append(tb[1])
             
             center_time.append(ct)
             
        # convert list to array
        center_times = np.array(center_time,dtype=np.datetime64)
        time_bnds    = np.array([time_start, time_end],dtype='datetime64')  
        time_bnds    = time_bnds.T
       
    # make time bounds dataset
    if 'time' not in ecco_dataset.dims.keys():
         ecco_dataset = ecco_dataset.expand_dims(dim='time')
         
    #print ('-- tb shape ', time_bnds.shape)
    #print ('-- tb type  ', type(time_bnds))
    time_bnds_ds = xr.Dataset({'time_bnds': (['time','nv'], time_bnds)},
                             coords={'time':ecco_dataset.time}) #,
                                     #'nv':range(2)})
    
    #print ('tbds ' , time_bnds_ds.time_bnds.shape)
    #print ('tbds ', time_bnds_ds.time_bnds.shape)
    
    return time_bnds_ds, center_times

#%%
def make_time_bounds_from_ds64(rec_avg_end, output_freq_code):
    """

    Given a datetime64 object (rec_avg_end) representing the 'end' of an 
    averaging time period (usually derived from the mitgcm file's timestep)
    and an output frequency code
    (AVG_MON, AVG_DAY, AVG_WEEK, or AVG_YEAR), create a time_bounds array
    with two datetime64 variables, one for the averaging period start, one
    for the averaging period end.  Also find the middle time between the
    two..
    
    Parameters
    ----------    
    rec_avg_end : numpy.datetime64 
        the time at the end of an averaging period

    output_freq_code : str 
        code indicating the time period of the averaging period
        - AVG_DAY, AVG_MON, AVG_WEEK, or AVG_YEAR


    Returns
    -------
    time_bnds : numpy.array(dtype=numpy.datetime64)
        a datetime64 array with the start and end time of the averaging periods
    
    center_times : numpy.datetime64
        the 'center' of the averaging period
    
    """ 
    
    if  output_freq_code in ('AVG_MON','AVG_DAY','AVG_WEEK','AVG_YEAR'):
        rec_year, rec_mon, rec_day, \
        rec_hour, rec_min, rec_sec = \
            extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(rec_avg_end)
        
        
        rec_avg_end_as_dt = datetime.datetime(rec_year, rec_mon, 
                                                  rec_day, rec_hour,
                                                  rec_min, rec_sec)
        
        if output_freq_code     == 'AVG_MON':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(months=1)    
        elif output_freq_code   == 'AVG_DAY':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(days=1)  
        elif output_freq_code   == 'AVG_WEEK':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(weeks=1)  
        elif output_freq_code   == 'AVG_YEAR':
            rec_avg_start =  rec_avg_end_as_dt - \
                dateutil.relativedelta.relativedelta(years=1)    

        rec_avg_start =  np.datetime64(rec_avg_start)
        
        rec_avg_delta = rec_avg_end - rec_avg_start
        rec_avg_middle = rec_avg_start + rec_avg_delta/2
        #print rec_avg_end, rec_avg_start, rec_avg_middle
        
        rec_time_bnds = np.array([rec_avg_start, rec_avg_end])
        
        return rec_time_bnds, rec_avg_middle
    
    else:
        print ('output_freq_code must be: AVG_MON, AVG_DAY, AVG_WEEK, OR AVG_YEAR')
        print ('you provided ' + str(output_freq_code))
        return [],[]   
    
#%%
def extract_yyyy_mm_dd_hh_mm_ss_from_datetime64(dt64):
    """

    Extract separate fields for year, monday, day, hour, min, sec from
    a datetime64 object
    
    Parameters
    ----------    
    dt64 : numpy.datetime64 
        a datetime64 object

    Returns 
    -------    
    year, mon, day, hh, mm, ss : int

    """
    
    s = str(dt64)
    year = int(s[0:4])
    mon = int(s[5:7])
    day = int(s[8:10])
    hh = int(s[11:13])
    mm = int(s[14:16])
    ss = int(s[17:18])
    
    #print year, mon, day, hh, mm, ss
    return year,mon,day,hh,mm,ss 

#%%
def createShapefileFromXY(outDir, outName, X,Y,subset):
    """

    This routine takes an X,Y array of grid points (e.g., XC, YC or XG, YG)) 
    and creates one of two types of shapefiles.
    1: polylines shapefile that trace the cell boundaries  (subset = 'boundary_points')
    2: point shapefile with a point in the cell centers    (subset = 'center points') 

    #Note This routine was originally written by Michael Wood. This version was modified by Fenty, 2/28/2019 to handle a newer version of pyshp (2.1.0)

    Parameters
    ----------
    outDir    : str
        directory into which shapefile and accessory files will be written
    
    outName   : str
        base of the filename (4 files will be created, 
    
    X,Y       : numpy.ndarray
        arrays of lat/lon locations
    
    subset    : str
        
        - 'boundary_points'to create polyline shapefile 
        - 'points' for to create point shapefile

    """


    if subset=='center_points':
        
        fname = outDir +'/' + outName + '_Grid_Center_Points/' + outName + '_Grid_Center_Points'
        
        w=shapefile.Writer(fname + '.shp')
        w.shapeType = shapefile.POINT
        w.field('id')

        counter=0
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                w.point(X[i,j],Y[i,j])
                w.record(counter)
                counter+=1
        w.close()

        f=open(fname + '.prj','w')
        f.write('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        f.close()

    elif subset=='boundary_points':
        
        fname = outDir + '/' + outName + '_Grid_Boundary_Points/' + outName + '_Grid_Boundary_Points'
        print(fname)
        w=shapefile.Writer(fname + '.shp')
        w.shapeType = shapefile.POLYLINE
        w.field('id')

        counter=0
        #create the vertical lines
        for i in range(np.shape(X)[0]):
            lines=[]
            for j in range(np.shape(X)[1]):
                lines.append( [ X[i,j], Y[i,j] ])

            w.line([lines])
            w.record(counter)
            counter+=1

        # create the horizontal lines
        XT = X.T
        YT = Y.T
        for i in range(np.shape(XT)[0]):
            lines=[]
            for j in range(np.shape(XT)[1]):
                lines.append( [ XT[i,j], YT[i,j] ])

            w.line([lines])
            w.record(counter)
            counter+=1

        w.close()

        f = open(fname + '.prj', 'w')
        f.write(
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]')
        f.close()
    else:
        print("subset must be either center_points or boundary_points")


#%%
def minimal_metadata(ds):
    """

    This routine removes some of the redundant metadata that is included with the ECCO v4 netcdf tiles from the Dataset object `ds`.  Specifically, metadata with the tags `A` through `Z` (those metadata records) that describe the origin of the ECCO v4 output.    

    Parameters
    ----------
    ds : xarray Dataset
        An `xarray` Dataset object that was created by loading an 
        ECCO v4 tile netcdf file
    
        
    """

    print('Removing Dataset Attributes A-Z\n')
    # generate a list of upper case letters in teh alphabet
    myDict= map(chr, range(65, 91))

    for key, value in ds.attrs.items():
        if key in myDict: 
            del ds.attrs[key]
         


#%%
def months2days(nmon=288, baseyear=1992, basemon=1):
    """ 

    This routine converts the mid-month time to days from January 1st of a particular year.
    
    Parameters
    ----------
    nmon : dtype=integer
        number of months 

    baseyear : dtype=integer
        year of time of origin
        
    basemon : dtype=integer
        month of time of origin 
    
    Returns
    -------
        time_days : ndarray
            the middle time of each month in days from Jan 1 baseyear (numpy array [nmon], dtype=double)
        
        time_days_bnds : ndarray
            time bounds (numpy array [nmon, 2], dtype=double)

        ansi_date : ndarray
            array of ANSI date strings

    """
    
    time_days_bnds = np.zeros([nmon,2])
    time_1stdayofmon = np.zeros([nmon+1])
    
    basetime = datetime.datetime(baseyear, basemon, 1, 0, 0, 0)

    for mon in range(nmon+1):
        #monfrombasemon is how many months fron basemon
        monfrombasemon=basemon+mon-1
        yrtmp = monfrombasemon//12+baseyear
        montmp = monfrombasemon%12+1
        tmpdate = datetime.datetime(yrtmp,montmp,1,0,0,0)-basetime
        time_1stdayofmon[mon] = tmpdate.days
    #time bounds are the 1st day of each month.
    time_days_bnds[:,0]= time_1stdayofmon[0:nmon]
    time_days_bnds[:,1]= time_1stdayofmon[1:nmon+1]
    #center time of each month is the mean of the time bounds.    
    time_days = np.mean(time_days_bnds,axis=1)

    ansi_datetmp = np.array([basetime + datetime.timedelta(days=time_days[i]) for i in xrange(nmon)])
    ansi_date = [str.replace(ansi_datetmp[i].isoformat(),'T',' ') for i in range(nmon)]

    return time_days, time_days_bnds, ansi_date

#%%

