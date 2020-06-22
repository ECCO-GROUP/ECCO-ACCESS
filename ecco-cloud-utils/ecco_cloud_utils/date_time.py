# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:55:22 2020

@author: Ian
"""


import dateutil as dateutil
import numpy as np
from datetime import datetime

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
        
        
        rec_avg_end_as_dt = datetime(rec_year, rec_mon, 
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