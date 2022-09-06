# Conda
# conda install -c conda-forge xgboost
# Import necessary packages
# XGBoost 
import xgboost as xgb
from xgboost import plot_importance
# Pandas
import pandas as pd 
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
# SKLearn
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
# Plotting
from matplotlib import pyplot as plt
# Datetime
import datetime
from datetime import datetime
# Numpy
import numpy as np
from numpy import sort
# Timer 
import time

# Set plot standards format
plt.rc('axes',labelsize=12)
plt.rc('xtick',labelsize=9)
plt.rc('ytick',labelsize=9)

# Load data 
series = read_csv('dmatrix-reducedVAR-omitNA-wswdUV.csv', header=0, index_col=0)
# Exclude the last 15% of observations, which is the holdout-test set that will not be used in the CV
len(series)
# 7488
## 10% of 7488 == 749
## 70% of 7488 == 5241

# https://github.com/at-tan/Forecasting_Air_Pollution/blob/main/Beijing_PM_Stack-CV.ipynb#

# Train
train = series.iloc[:5241,:]
x_train = train.iloc[:,:48]
y_train = train.iloc[:,48:]

# Test
test = series.iloc[5241:,:]
x_test = test.iloc[:,:48]
y_test = test.iloc[:,48:]
# Evaluation set for final model
eval_set = [(x_test, y_test)]

# TimeSeriesSplit
splits = TimeSeriesSplit(n_splits=10)


model = xgb.XGBRegressor(objective='reg:squarederror', 
                         gamma=0.1,
                         min_child_weight = 2.5, 
                         n_estimators = 100)
# Test eta grid search
eta_range = [0.0001, 0.001, 0.01, 0.1,0.15, 0.2, 0.3]
pg = {'eta': eta_range}
score = make_scorer(mean_squared_error)
grid_search = GridSearchCV(model, param_grid=pg,
                           scoring=score,
                           n_jobs=-1, cv=splits, verbose=2)
start = time.time()
grid_result = grid_search.fit(x_train, y_train)
end = time.time()
print(end - start)

# Collect results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Best: using {'eta': 0.1}
# Plot eta vs MSE score
fig2, ax2 = plt.subplots()
ax2.errorbar(eta_range, means, yerr=stds)
ax2.title.set_text("XGBoost 'eta' accuracy")
ax2.set_xlabel('eta')
ax2.set_ylabel('MSE')
ax2.set_xscale('log')
plt.show()

# Update model
# Add early stopping round to avoid 
model = xgb.XGBRegressor(objective='reg:squarederror', 
                         eta = 0.1, 
                         max_depth = 8, 
                         min_child_weight = 2.5, 
                         n_estimators = 100,
                         gamma = 0.1,
                         early_stopping_round = 10)

model.fit(x_train, y_train, eval_set=eval_set)     # fit
y_pred = model.predict(x_test)  # predict


# plot importance with built-in function
# gain
plot_importance(model, grid=False, show_values=True,
                importance_type='gain',
                title="Feature Importance: Gain")
plt.savefig("plot_importance_gain.png",dpi=300)
# weights
plot_importance(model, grid=False, show_values=True,
                importance_type='weight',
                title="Feature Importance: Weights")
plt.savefig("plot_importance_weight.png",dpi=300)


# Accuracy testing 
accuracy_MAE =  mean_absolute_error(y_test, y_pred)
# 7.789
accuracy_MAPE = mean_absolute_percentage_error(y_test, y_pred)
# 0.246
accuracy_R2 = r2_score(y_test, y_pred)
# 0.595
accuracy_RMSE = mean_squared_error(y_test, y_pred, squared=False)
# 10.673

# Create dataframe for predictions
y_pred_df = pd.DataFrame(y_pred)    # df
y_pred_df['NO2(t)'],y_pred_df['NO2(t+1)'],y_pred_df['NO2(t+2)'] = y_pred_df[0], y_pred_df[1], y_pred_df[2]
y_pred_df = y_pred_df.iloc[:,3:]    
y_pred_df.index = y_test.index      # reset index

# Accuracy score for each timestep
list_mae = list()
list_r2 = list()
list_mape = list()
list_rmse = list()        
for i in y_pred_df:
    for j in y_test:
        if i==j:
            i_cont = y_pred_df[i]
            j_cont = y_test[j]
            MAE =  mean_absolute_error(j_cont.values,i_cont.values)
            list_mae.append(MAE)
            MAPE = mean_absolute_percentage_error(j_cont.values,i_cont.values)*100
            list_mape.append(MAPE)
            R2 = r2_score(j_cont.values,i_cont.values)
            list_r2.append(R2)
            RMSE = mean_squared_error(j_cont.values,i_cont.values, squared=False)
            list_rmse.append(RMSE)

# Concatenate lists to export
df_accuracy_column = pd.DataFrame(list(zip(list_mae,list_mape,list_r2,list_rmse)),
                              columns = ['MAE','MAPE',
                                         'R2', 'RMSE'])
df_accuracy_column = df_accuracy_column.transpose()             # transpose
df_accuracy_column.columns = ['NO2(t)','NO2(t+1)','NO2(t+2)']   # change column names
df_accuracy_column.to_csv("model_accuracy_NO2timestep_ne100.csv")

# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
# collect data
thresh_val = list()
feat_num = list()
MAE_val = list()
R2_val = list()

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 
    select_X_train = selection.transform(x_train)
	# train model
    selection_model = xgb.XGBRegressor(objective='reg:squarederror', 
                             eta = 0.2, 
                             max_depth = 8, 
                             min_child_weight = 2.5, 
                             n_estimators = 100,
                             gamma = 0.1,
                             early_stopping_round = 10)
    selection_model.fit(select_X_train, y_train)
	# eval model
    select_X_test = selection.transform(x_test) 
    predictions = selection_model.predict(select_X_test)    # predict
    MAE_accuracy = mean_absolute_error(y_test, predictions) 
    r2_accuracy = mean_squared_error(y_test, predictions, squared=False)
    # add to df
    thresh_val.append(thresh)
    feat_num.append(select_X_train.shape[1]) 
    MAE_val.append(MAE_accuracy) 
    R2_val.append(r2_accuracy) 
    print("Thresh=%.3f, n=%d, R2: %.2f" % (thresh, select_X_train.shape[1], r2_accuracy))
    print("Thresh=%.3f, n=%d, MAE: %.2f" % (thresh, select_X_train.shape[1], MAE_accuracy))


df_thresh_test = pd.DataFrame(list(zip(thresh_val, feat_num, MAE_val, R2_val)),
                              columns = ['Thresh','Num_features',
                                         'MAE', 'R2'])
df_thresh_test.to_csv("threshold_test_MAE_R2.csv")
# plot MAE for number of features
plt.plot(df_thresh_test['Num_features'],df_thresh_test['MAE'])
plt.title('MAE score determined by number of features')
plt.xlabel('Number of features')
plt.ylabel('MAE score')
plt.show()
plt.savefig("mae_by_nofeature.png",dpi=300)

# Appears to be small accuracy increase at features = 42
# Therefore redo model > avoid overfitting
# slight accuracy improvement losing bottom 6 var

# Built-in XGBoost Feature Importance 
col_name = list()
f_score = list()
# Obtain scores
for col,score in zip(x_train.columns,model.feature_importances_):
    col_name.append(col)
    f_score.append(score)
feat_imp = [col_name,f_score]
# Create dataframe
df_feat_importance = pd.DataFrame(feat_imp).transpose()
df_feat_importance.columns = ['feature','importance']
df_feat_importance=df_feat_importance.sort_values(by='importance').reset_index(drop=True)
# Subset df where importance <1%
df_fi_less1perc = df_feat_importance[df_feat_importance['importance']<0.01]
# 32 features less than 1% feature importance each
# Plot - barh
fig, ax = plt.subplots()
ax.barh(df_fi_less1perc['feature'],df_fi_less1perc['importance'])
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
ax.set_title('XGBoost Built-in Feature Importance')
plt.savefig("feature_importance_lessthan1perc_xgboost_builtin.png",dpi=300)
# plot all
fig, ax = plt.subplots()
hbar = ax.barh(df_feat_importance['feature'],df_feat_importance['importance'])
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
ax.set_title('XGBoost Built-in Feature Importance')
ax.bar_label(hbar, fmt='%.3f', fontsize=9)
plt.savefig("feature_importance_xgboost_builtin.png",dpi=300)

