#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:07:18 2022

@author: oliviadavidson
"""

# Plot timeseries
import matplotlib.dates as mdates
for timestep in timeseriesplot:
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    # NO2(t)
    ax1.plot(y_train.index, y_train.iloc[:,0].values)
    ax1.plot(y_pred_df.index, y_pred_df.iloc[:,0].values)
    myFmt = mdates.DateFormatter('%Y-%m')
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.title.set_text("NO2(t)")
    ax1.set_ylabel("NO2 Concentration (ug/m3)")
    # NO2(t+1)
    ax1 = fig.add_subplot(3,1,2)
    ax1.plot(y_train.index, y_train.iloc[:,1].values)
    ax1.plot(y_pred_df.index, y_pred_df.iloc[:,1].values)
    myFmt = mdates.DateFormatter('%Y-%m')
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.title.set_text("NO2(t+1)")
    ax1.set_ylabel("NO2 Concentration (ug/m3)")
    # NO2(t+2)
    ax1 = fig.add_subplot(3,1,3)
    ax1.plot(y_train.index, y_train.iloc[:,2].values)
    ax1.plot(y_pred_df.index, y_pred_df.iloc[:,2].values)
    myFmt = mdates.DateFormatter('%Y-%m')
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.title.set_text("NO2(t+2)")
    ax1.set_ylabel("NO2 Concentration (ug/m3)")


# Plot original full then plot
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# NO2(t)
ax1.plot(y_train.index, y_train.iloc[:,0].values, label = 'True')
ax1.plot(y_pred_df.index, y_pred_df.iloc[:,0].values, label = 'Predicted')
myFmt = mdates.DateFormatter('%Y-%m')
plt.xticks(rotation=45)
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc = 'upper right')
ax1.title.set_text("NO2(t) Performance")
ax1.set_ylabel("NO2 Concentration (ug/m3)")


# Group by day-of-year > find best and worst performing days
# Group by month > find best and worst performing months 

y_pred_df = pd.DataFrame(y_pred)    # df
y_pred_df['NO2(t)'],y_pred_df['NO2(t+1)'],y_pred_df['NO2(t+2)'] = y_pred_df[0], y_pred_df[1], y_pred_df[2]
y_pred_df = y_pred_df.iloc[:,3:]    
y_pred_df.index = y_test.index      # reset index

# obtain dayofyear and month variable using index
# for prediction data
y_pred_df.index = pd.to_datetime(y_pred_df.index)
y_pred_df['datetime'] = y_pred_df.index
y_pred_df['month'] = y_pred_df['datetime'].dt.month
y_pred_df['dayofyear'] = y_pred_df['datetime'].dt.date
# for test data
y_test.index = pd.to_datetime(y_test.index)
y_test['datetime'] = y_test.index
y_test['month'] = y_test['datetime'].dt.month
y_test['dayofyear'] = y_test['datetime'].dt.date

# Reset index
y_test_rs=y_test.reset_index(drop=True)
y_pred_df_rs=y_pred_df.reset_index(drop=True)
# Obtain true and predicted dataframe for each timestep
no2_t_true_pred = pd.concat([y_test_rs['NO2(t)'],y_pred_df_rs['NO2(t)'],y_pred_df_rs['month'],y_pred_df_rs['dayofyear']], axis=1)
no2_t_true_pred.columns = ['true','pred','month','date']
no2_t1_true_pred = pd.concat([y_test_rs['NO2(t+1)'],y_pred_df_rs['NO2(t+1)'],y_pred_df_rs['month'],y_pred_df_rs['dayofyear']], axis=1)
no2_t1_true_pred.columns = ['true','pred','month','date']
no2_t2_true_pred = pd.concat([y_test_rs['NO2(t+2)'],y_pred_df_rs['NO2(t+2)'],y_pred_df_rs['month'],y_pred_df_rs['dayofyear']], axis=1)
no2_t2_true_pred.columns = ['true','pred','month','date']
# Calculate error and absolute error by dayofyear and month #
# NO2(t)
no2_t_true_pred['error'] = no2_t_true_predd['true'] - no2_t_true_pred['pred']
no2_t_true_pred['abs_error'] = no2_t_true_pred['error'].apply(np.abs)
error_by_day_t = no2_t_true_pred.groupby(['date']) \
    .mean()[['true','pred','error','abs_error']]
error_by_month_t = no2_t_true_pred.groupby(['month']) \
    .mean()[['true','pred','error','abs_error']]
# NO2(t+1)
no2_t1_true_pred['error'] = no2_t1_true_pred['true'] - no2_t1_true_pred['pred']
no2_t1_true_pred['abs_error'] = no2_t1_true_pred['error'].apply(np.abs)
error_by_day_t1 = no2_t1_true_pred.groupby(['date']) \
    .mean()[['true','pred','error','abs_error']]
error_by_month_t1 = no2_t1_true_pred.groupby(['month']) \
    .mean()[['true','pred','error','abs_error']]
# NO2(t+2)  
no2_t2_true_pred['error'] = no2_t2_true_pred['true'] - no2_t2_true_pred['pred']
no2_t2_true_pred['abs_error'] = no2_t2_true_pred['error'].apply(np.abs)
error_by_day_t2 = no2_t2_true_pred.groupby(['date']) \
    .mean()[['true','pred','error','abs_error']]
error_by_month_t2 = no2_t2_true_pred.groupby(['month']) \
    .mean()[['true','pred','error','abs_error']]
    
# EXPORT
error_by_month_t.to_csv("error_by_month_t.csv")
error_by_day_t.to_csv("error_by_day_t.csv")


# Find best and worst periods
# Plot worst period during 01/12/2020

y_test['dayofyear'] = pd.to_datetime(y_test['dayofyear'])
start_date = '2020-11-30'
end_date = '2020-12-02'
mask = (y_test['dayofyear'] >= start_date) & (y_test['dayofyear'] <= end_date)
mask_test = y_test.loc[mask]
y_pred_df['dayofyear'] = pd.to_datetime(y_pred_df['dayofyear'])
mask = (y_pred_df['dayofyear'] >= start_date) & (y_pred_df['dayofyear'] <= end_date)
mask_pred = y_pred_df.loc[mask]

# Plot the  period, using true and predicted values
fig, ax = plt.subplots()

ax.plot(mask_test['NO2(t+1)'], label = 'True')
ax.plot(mask_pred['NO2(t+1)'], label = 'Predicted')
ax.legend(loc = 'upper right')
ax.title.set_text("NO2(t+1) Performance")
ax.set_ylabel("NO2 Concentration (ug/m3)")
plt.xticks(rotation=45)
plt.show()
plt.savefig("NO2(t+1)_3011_02122020.png", dpi=300)

# Plot good period
# 17/12/2020

y_test['dayofyear'] = pd.to_datetime(y_test['dayofyear'])
start_date = '2020-12-15'
end_date = '2020-12-18'
mask = (y_test['dayofyear'] >= start_date) & (y_test['dayofyear'] <= end_date)
mask_test = y_test.loc[mask]
y_pred_df['dayofyear'] = pd.to_datetime(y_pred_df['dayofyear'])
mask = (y_pred_df['dayofyear'] >= start_date) & (y_pred_df['dayofyear'] <= end_date)
mask_pred = y_pred_df.loc[mask]
# Plot period, using true and predicted values
fig, ax = plt.subplots()
ax.plot(mask_test['NO2(t+2)'], label = 'True')
ax.plot(mask_pred['NO2(t+2)'], label = 'Predicted')
ax.legend(loc = 'upper right')
ax.title.set_text("NO2(t+2) Performance")
ax.set_ylabel("NO2 Concentration (ug/m3)")
plt.xticks(rotation=45)
plt.show()
plt.savefig("NO2(t)_1512_18122020.png", dpi=300)

# plot  good period in september
y_test['dayofyear'] = pd.to_datetime(y_test['dayofyear'])
start_date = '2020-09-18'
end_date = '2020-09-21'
mask = (y_test['dayofyear'] >= start_date) & (y_test['dayofyear'] <= end_date)
mask_test = y_test.loc[mask]
y_pred_df['dayofyear'] = pd.to_datetime(y_pred_df['dayofyear'])
mask = (y_pred_df['dayofyear'] >= start_date) & (y_pred_df['dayofyear'] <= end_date)
mask_pred = y_pred_df.loc[mask]
# Plot period, using true and predicted values
fig, ax = plt.subplots()
ax.plot(mask_test['NO2(t+2)'], label = 'True')
ax.plot(mask_pred['NO2(t+2)'], label = 'Predicted')
ax.legend(loc = 'upper right')
ax.title.set_text("NO2(t+2) Performance")
ax.set_ylabel("NO2 Concentration (ug/m3)")
plt.xticks(rotation=45)
plt.show()
plt.savefig("NO2(t+2)_1809_21092020.png", dpi=300)
