# Model > Analyse Quantiative
# conda install -c conda-forge mlxtend
from mlxtend.evaluate import bias_variance_decomp
from math import sqrt

# Accuracy testing 
accuracy_MAE =  mean_absolute_error(y_test, y_pred)

accuracy_MAPE = mean_absolute_percentage_error(y_test, y_pred)*100

accuracy_R2 = r2_score(y_test, y_pred)

accuracy_RMSE = sqrt(mean_squared_error(y_test, y_pred))


# Create dataframe for predictions
y_pred_df = pd.DataFrame(y_pred)    # df
y_pred_df['NO2(t)'],y_pred_df['NO2(t+1)'],y_pred_df['NO2(t+2)'] = y_pred_df[0], y_pred_df[1], y_pred_df[2]
y_pred_df = y_pred_df.iloc[:,3:]    
y_pred_df.index = y_test.index      # reset index

# Accuracy score and bias-variance for each output timestep
list_mae = list()
list_r2 = list()
list_mape = list()
list_rmse = list() 
list_bias = list()
list_var = list()       
for i in y_pred_df:
    for j in y_test:
        if i==j:
            i_cont = y_pred_df[i]
            j_cont = y_test[j]
            # MAE
            MAE =  mean_absolute_error(j_cont.values,i_cont.values)
            list_mae.append(MAE)
            # MAPE
            MAPE = mean_absolute_percentage_error(j_cont.values,i_cont.values)*100
            list_mape.append(MAPE)
            # R2
            R2 = r2_score(j_cont.values,i_cont.values)
            list_r2.append(R2)
            # RMSE
            RMSE = sqrt(mean_squared_error(j_cont.values,i_cont.values))
            list_rmse.append(RMSE)
            # MSE, BIAS, VAR
            var = np.var(j_cont.values - i_cont.values)
            bias = round(mean_squared_error(j_cont.values, i_cont.values), 2)
            list_bias.append(bias)
            list_var.append(var)
           # mse, bias, var = bias_variance_decomp(model, X_train, j_cont.values, X_test, i_cont.values, loss='mse', num_rounds=200, random_seed=1)
            #list_bias.append(bias)
            #list_var.append(var)

# Concatenate lists to export
df_accuracy_column = pd.DataFrame(list(zip(list_mae,list_mape,list_r2,
                                           list_rmse, list_bias, list_var)),
                              columns = ['MAE','MAPE','R2', 
                                         'RMSE', 'bias', 'variance'])
df_accuracy_column = df_accuracy_column.transpose()             # transpose
df_accuracy_column.columns = ['NO2(t)','NO2(t+1)','NO2(t+2)']   # change column names
df_accuracy_column.to_csv("model_accuracy_NO2timestep_afterFS_biasvar_ne100.csv")
