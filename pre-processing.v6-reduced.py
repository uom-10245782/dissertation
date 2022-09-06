#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:21:02 2022

@author: oliviadavidson

VERSION 6:
    REDUCED COLUMNS
    DONT DROP NA 
"""

# conda install -c conda-forge metpy
# Import packages
# Metpy
import metpy.calc as mpcalc
from metpy.units import units
# Pandas
import pandas as pd 
from pandas import DataFrame
from pandas import concat
# Numpy
import numpy as np
# Scipy
import scipy.stats

# Extract wind direction feature #
# Speed, direction
def wind_spddir_to_uv(wspd,wdir):
    """
    calculated the u and v wind components from wind speed and direction
    Input:
        wspd: wind speed
        wdir: wind direction
    Output:
        u: u wind component
        v: v wind component
    """    
    u = pd.DataFrame()
    v = pd.DataFrame()
    # use radians
    rad = 4.0*np.arctan(1)/180.
    u = -wspd*np.sin(rad*wdir)
    v = -wspd*np.cos(rad*wdir)
    return u, v

# Encode hour as cyclical
def var_to_cyclical(var):
  
    u = pd.DataFrame()
    v = pd.DataFrame()
    u = np.sin(var)
    v = np.cos(var)

    return pd.DataFrame([u,v]).T

# Define series to supervised functions #

# Input #
# 1. Meteorology - 1 hour
def preprocess_inputs_MET(data, n_in=1):
    
    n_vars = data.shape[1]
    df = data
    cols, names = list(), list()
    

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
          
    # put it all together
    met1 = concat(cols, axis=1)
    met1.columns = names
        
    # Collect 1 hour for all, 
    # and 1, 2, 12 hour for temp
    met2 = pd.DataFrame()
    
    for var in names:
        if 'var1' in var:
            met2['msl(t-1)'] = met1['var1(t-1)'] 
        elif 'var2' in var:
            met2['temp(t-12)'] = met1['var2(t-12)']  
            met2['temp(t-2)'] = met1['var2(t-2)']             
            met2['temp(t-1)'] = met1['var2(t-1)'] 
        elif 'var3' in var: 
            met2['ws(t-1)'] = met1['var3(t-1)'] 
        elif 'var4' in var:
            met2['wd_u(t-1)'] = met1['var4(t-1)'] 
        else:
            met2['wd_v(t-1)'] = met1['var5(t-1)'] 
            
    return met2

# 2.1. Pollutants - 24 hour mean and 1 hour
def preprocess_inputs_POLL(data, n_in=1):
    
    n_vars = data.shape[1]
    df = data
    cols, names = list(), list()
    
    # input sequence (t-, ..., t-1)
    # traffic (t-24m) > compute mean of last 24hours
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    poll24 = concat(cols, axis=1)
    poll24.columns = names

    # collect t-1 and t-24mean for each var
    poll24m = pd.DataFrame()
    
    for var in poll24:
        if 'var1' in var:
            mask = poll24.columns.str.contains("var1")
            #chosen = poll24.loc
            poll24m['NO(t-24m)'] = poll24.loc[:,mask].mean(axis=1)
            poll24m['NO(t-1)'] = poll24['var1(t-1)']
        elif 'var2' in var:
            mask = poll24.columns.str.contains("var2")
            poll24m['NOx(t-24m)'] = poll24.loc[:,mask].mean(axis=1)
            poll24m['NOx(t-1)'] = poll24['var2(t-1)']
        elif 'var3' in var:
            mask = poll24.columns.str.contains("var3")
            poll24m['SO2(t-24m)'] = poll24.loc[:,mask].mean(axis=1)
            poll24m['SO2(t-1)'] = poll24['var3(t-1)']
        elif 'var4' in var:
            mask = poll24.columns.str.contains("var3")
            poll24m['PM10(t-24m)'] = poll24.loc[:,mask].mean(axis=1)
            poll24m['PM10(t-1)'] = poll24['var4(t-1)'] 
        else:
            mask = poll24.columns.str.contains("var3")
            poll24m['PM25(t-24m)'] = poll24.loc[:,mask].mean(axis=1)
            poll24m['PM25(t-1)'] = poll24['var5(t-1)']    
       
    return poll24m

# 2.2. Pollutants: NO2 - 24hour
def preprocess_inputs_NO2(data, n_in=1):
    
    n_vars = data.shape[1]
    df = data
    cols, names = list(), list()
    
    # input sequence (t-, ..., t-1)
    # traffic (t-24m) > compute mean of last 24hours
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # then change var1 to df[1] name etc
    replace_names = list()
    
    for var in names:
        replace_names += [var.replace("var1","NO2")] 

    # put it all together
    no2_24 = concat(cols, axis=1)
    no2_24.columns = replace_names
        
    return no2_24

# 3. Traffic - 24 hour mean and 1 hour
def preprocess_inputs_TV(data, n_in=1):
    
    n_vars = data.shape[1]
    df = data
    cols, names = list(), list()
    
    # input sequence (t-n, ..., t-1)
    # traffic (t-24m) > compute mean of last 24hours
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # then change var1 to df[1] name etc
    replace_names = list()
    
    for var in names:
        replace_names += [var.replace("var1","traffic")] 
    
    # put it all together
    tv24 = concat(cols, axis=1)
    tv24.columns = replace_names
    # find mean of each row 
    # equivalent to hist.24 hour mean(-24m)
    tv24m = DataFrame()
    tv24m['traffic(t-24m)'] = tv24.mean(axis=1)
    tv24m['traffic(t-1)'] = tv24['traffic(t-1)']
    
    return tv24m


# Output #
# NO2 - t, t+1 t+2, t+3 
# Forecasting 3 hours ahead
def preprocess_output_NO2(data, n_out=1):
    
    n_vars = data.shape[1]
    df = data
    cols, names = list(), list()
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):

        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
    # then change var1 to df[1] name etc
    replace_names = list()
    
    for var in names:
        replace_names += [var.replace("var1","NO2")] 
    
    # put it all together
    no2_output = concat(cols, axis=1)
    no2_output.columns = replace_names

    return no2_output 	


# Read data 
data1 = pd.read_csv('MAN_DATA_2020_simple.csv')

# Extract necessaary data into new dataframes
# Traffic data
df_T = data1[['Traffic']]
# Pollutants
df_P = data1[['NO','NOXasNO2','SO2','PM10','PM2.5']]
# NO2
df_NO2 = data1[['NO2']]
###############
# Meteorology #
# Divide msl by 100 to get hPa
msl = data1[['msl']]/100
# Wind direction > tranform into azimuth u, v
wd = data1['wd']
ws = data1['ws']
# wind direction, speed 
uv = pd.DataFrame(wind_spddir_to_uv(ws,wd)).T
uv.columns = ['u','v']
# Test correlation
# df = pd.concat([no2,uv], axis=1)
# df.dropna(inplace=True)
# r, p = scipy.stats.pearsonr(df[],df[])
# Combine all meteorological components
df_M = pd.concat([msl, data1[['temp','ws']], uv['u'], uv['v']], axis=1)

# Day of week > Dummy Encode 
# Boolean: 1 = Weekday, 0 = Weekend
df_isWeekday= pd.DataFrame()
df_isWeekday = pd.DataFrame(np.where(data1['dayofweek'] > 5,0,1))
df_isWeekday.columns = ['isWeekday']

# Extract time in datetime > encode 
# Datetime
df_date = pd.DataFrame()
df_date['date'] = pd.to_datetime(data1['Date1'] + ' ' + data1['Time'])
# Turn time variables into cyclical variables # 
# Extract hour as int
df_hour = pd.DataFrame()
df_hour['hour'] = pd.to_datetime(df_date['date']).dt.hour
cyc_hour = var_to_cyclical(df_hour['hour'])
cyc_hour.columns = ['sin_hour','cos_hour']      # change col names
# Month 
df_month = data1[['Month']]
cyc_month = var_to_cyclical(df_month['Month'])
cyc_month.columns = ['sin_month', 'cos_month']  # change col names

# Run function 
input_24m_poll = preprocess_inputs_POLL(df_P, 24)
input_24m_tv = preprocess_inputs_TV(df_T, 24)
input_1_met = preprocess_inputs_MET(df_M, 12)
input_24_no2 = preprocess_inputs_NO2(df_NO2, 24)
output_3_no2 = preprocess_output_NO2(df_NO2, 3)

# Merge
data2 = pd.concat([df_date, input_24m_poll, 
                   input_24m_tv, input_1_met,
                   input_24_no2,df_isWeekday, 
                   cyc_month, cyc_hour, 
                   output_3_no2], axis=1)

# Drop NA
data2.dropna(inplace=True) 
data3 = data2.set_index('date')     # set index to datetime

# Export as CSV
data3.to_csv("dmatrix-reducedVAR-omitNA-wswdUV.csv")