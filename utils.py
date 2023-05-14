import numpy as np, matplotlib.pyplot as plt, pandas as pd, xarray as xr

def wrf_to_datetime(ds):
    '''
        Set the time coordinates to datetime64 index
    args:
        ds        [obj]  : xarray.dataset object with input data
    returns:
        ds_fixed  [obj]  : xarray.dataset object with fixed time index
    '''
    time_strs = [str(i.values)[1:].replace("_"," ") for i in ds.Times]
    times = pd.to_datetime(time_strs)
    ds_fixed = ds.rename({'Time':'time'})
    ds_fixed = ds_fixed.drop('Times')
    
    return ds_fixed.assign(time=times)
    
    
def wrf_to_latlon(ds):
    ds = ds.assign_coords(
        lat = ds.coords['XLAT'],#.squeeze('time'),
        lon = ds.coords['XLONG'])#.squeeze('time'),
        #landmask = ds.LANDMASK.squeeze('time'),
        #lakemask = ds.LAKEMASK.squeeze('time'))
    ds = ds.rename({'south_north':'y', 'west_east':'x'})
    ds = ds.drop(['XLAT', 'XLONG'])
    return ds

def timeidx_from_datetime(data, datetime_object):
    '''Helper function for returning the model time index for a given time in datetime format'''
    timestep = data.xrData['XTIME'][1] - data.xrData['XTIME'][0]
    return int((datetime_object - data.date).total_seconds()/60/timestep)