## for data
import pandas as pd
import numpy as np## for plotting
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns## for statistical tests
import scipy
from scipy import stats
from scipy.stats import probplot
import numpy as np
from scipy.stats import kstest
# Simple Data Exploration
# import library for visualization
from matplotlib.ticker import FuncFormatter
# calculate VIF scores
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif 
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as smf
import statsmodels.api as sm## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from catboost import CatBoostRegressor, Pool, cv
import os

#Pickle
import gzip
import pickle
#import time
from time import time

from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from yellowbrick.regressor import ResidualsPlot

#import dataset
with open('compressed_x_train.pkl', 'rb') as f:
    compressed_data = pickle.load(f)
# Decompress the data using gzip
x_train = pickle.loads(gzip.decompress(compressed_data))
y_train = pd.read_pickle('y_train.pkl')
x_test = pd.read_pickle('X_test.pkl')
y_test = pd.read_pickle('y_test.pkl')

#Cross Validation Catboost Libraries
def cross_val_cb(x,y,n_fold):
    start = time()
    X_pool = Pool(data=x, label=y)
    cv(X_pool, params={'random_state':42, 'loss_function':'RMSE'}, fold_count=n_fold, plot=True, verbose=False)
    scores = cv(X_pool, params={'random_state':42, 'loss_function':'RMSE'}, fold_count=n_fold, plot=True, verbose=False)
    print("Cross Validation Score : ", scores)
    end=time()
    print ('Time needed (at work) in minutes: {0}'.format((end-start)/60))

def train_catboost_regressor():
    global model
    model_catboost = CatBoostRegressor(random_state=42)

    model_catboost.fit(x_train,y_train, early_stopping_rounds=5, verbose=True, eval_set=[(x_train, y_train)])
    
    # # save the model to disk with a custom name
    # model_name = "catboost_regressor_v1.model"
    # if os.path.exists(model_name):
    #     i = 1
    #     while True:
    #         new_model_name = model_name.rsplit(".", 1)[0] + "_v{}".format(i) + "." + model_name.rsplit(".", 1)[1]
    #         if not os.path.exists(new_model_name):
    #             model_name = new_model_name
    #             break
    #         i += 1
    # #saving model train
    # model_catboost.save_model(model_name)

    #saving model in specific path
    model_name = "catboost_regressor_v1.model"
    model_path = os.path.join(model_name)

    if os.path.exists(model_path):
        i = 1
        while True:
            new_model_name = model_name.rsplit(".", 1)[0] + "_v{}".format(i) + "." + model_name.rsplit(".", 1)[1]
            new_model_path = os.path.join(new_model_name)
            if not os.path.exists(new_model_path):
                model_path = new_model_path
                break
            i += 1

    #saving model train
    model_catboost.save_model(model_path)

def evaluation_model_catboost(name_model):
    global model
    
    # initialize model
    start = time()

    model_name = name_model
    model_path = os.path.join(model_name)

    # loading model
    model_catboost = CatBoostRegressor()
    model =  model_catboost.load_model(model_path)

    ## train prediction
    train_predicted = model.predict(x_train)
    ## test prediction
    test_predicted = model.predict(x_test)
    
    ## Evaluation
    print('==================== Evaluation Model ====================')
    r2 = round(metrics.r2_score(y_test, test_predicted), 2)
    mape = round(np.mean(np.abs((y_test-test_predicted)/test_predicted)), 2)
    mae = metrics.mean_absolute_error(y_test, test_predicted)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, test_predicted))
    print("R2 (explained variance):", r2)
    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", mape)
    print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(mae))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(rmse))## residuals
    
    #Calculate Residuals
    residuals = y_test - test_predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
    max_true, max_pred = y_test[max_idx], test_predicted[max_idx]
    print("Max Error from Residuals : ", "{:,.0f}".format(max_error))
    
    print('==================== Evaluation Both in Train and Test ====================')
    # MAE
    mae_train = mean_absolute_error(y_train, train_predicted)
    mae_test = mean_absolute_error(y_test, test_predicted)
    print('MAE for training data is {}'.format(mae_train))
    print('MAE for testing data is {}'.format(mae_test))
    
    #RMSE 
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predicted))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predicted))
    print('RMSE for training data is {}'.format(rmse_train))
    print('RMSE for testing data is {}'.format(rmse_test))
    
    end=time()
    print ('Time needed (at work) in minutes: {0}'.format((end-start)/60))


### Check Statistical Assumption :
def check_statistical_assumption(name_model):
    #load model
     # initialize model
    start = time()

    model_name = name_model
    model_path = os.path.join(model_name)
    # loading model
    model_catboost = CatBoostRegressor()
    model =  model_catboost.load_model(model_path)
    
    ## test prediction
    test_predicted = model.predict(x_test)
    
    #Calculate Residuals
    residuals = y_test - test_predicted
    
    # Store the result
    df_results = pd.DataFrame(
        {"Actual":y_test, 
        "Predicted":test_predicted, 
        "Residuals":residuals
        }
    )
    
    #Linearity Check
    print('============================ Check Linearity ============================')
    r2 = round(metrics.r2_score(y_test, test_predicted), 2)
    
    # Check if the p-value is less than 0.05
    if r2 >= 0.7 :
        print("Linearity check is passed with R2 (explained variance) : ",r2)
    else:
        print("The linearity model is not passed.")
    
    # Use Jarque-Bera Test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html
    # If the requirement of Jarque-Bera test is having at least 2000 rows,
    # we can't use this test becase our data row is about 1000 only.
    
    print('============================ Check P-Value Test for Q-Q plot ============================')
    jarque_bera_test = stats.jarque_bera(df_results["Residuals"])

    print('JB : ', jarque_bera_test)

    print('JB Value : ', jarque_bera_test.statistic)

    print('P-Value: ', jarque_bera_test.pvalue)

    # Check if the p-value is less than 0.05
    if jarque_bera_test.pvalue < 0.05:
        print("The data is not normally distributed.")
    else:
        print("The data is normally distributed.")
    
    print('============================ Check Non Autocorrelation ============================')
    # =======================
    # Non Autocorrelation
    # =======================
    # Allowed DW Statistics:
    # 1.5 < DW < 2.5
    # =======================
    from statsmodels.stats.stattools import durbin_watson

    durbinWatson = durbin_watson(df_results["Residuals"])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')
    
    #Check VIF Value
    #Add Constant stats model
    X = add_constant(x_train)
    print('============================ Check Multi Colinearity ============================')
    # =======================
    # Non-Multicollinearity
    # =======================
    # Remove when VIF > 10
    # =======================
    vif_df = pd.DataFrame([vif(X.values, i) 
                for i in range(X.shape[1])], 
                index=X.columns).reset_index()
    vif_df.columns = ['feature','vif_score']
    vif_df = vif_df.loc[vif_df.feature!='const']
    print(vif_df)
    
    end=time()
    print ('Time needed (at work) in minutes: {0}'.format((end-start)/60))

### Check Statistical Assumption :
def check_statistical_plot(name_model):

    #load model
    start = time()
    model_name = name_model
    model_path = os.path.join(model_name)
    # loading model
    model_catboost = CatBoostRegressor()
    model =  model_catboost.load_model(model_path)   
    print('==================== Success Fitting the Model ====================')
    
    ## test prediction
    test_predicted = model.predict(x_test)
    
    #Calculate Residuals
    residuals = y_test - test_predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
    max_true, max_pred = y_test[max_idx], test_predicted[max_idx]
    
    # Store the result
    df_results = pd.DataFrame(
        {"Actual":y_test, 
        "Predicted":test_predicted, 
        "Residuals":residuals
        }
    )
    
    #Linearity Check
    print('============================ Check Linearity ============================')
    
    # =======================
    # Linearity
    # =======================
    # When prediction align with actual
    # =======================
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False)
            
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
         color='darkorange', linestyle='--')
    plt.savefig('Linearity Plot.png')
    
    
    print('============================ Check Q-Q Plot ============================')
    # 3. Use QQPlot (Quantile-quantile Plot)
    from statsmodels.graphics.gofplots import qqplot
    qqplot(df_results["Residuals"], line='s')
    plt.savefig('QQ Plot.png')

    # Create Q-Q plot
    # probplot(df_results["Residuals"], dist="norm", plot=plt) #dist = norm, expon, uniform
    # plt.show()

    print('============================ Check Residuals Distribution ============================')
    # 4. Use Distribution Plot
    plt.subplots(figsize=(12, 6))
    sns.distplot(df_results["Residuals"])
    plt.savefig('Distribution of Residuals.png')
    
    # =======================
    # Normality of Residual
    # =======================
    # When p-value > alpha
    # H0 = Normally Distributed
    # =======================
    fig, ax = plt.subplots(nrows=1, ncols=2)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(test_predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()
    
    ## Plot predicted vs residuals
    ax[1].scatter(test_predicted, residuals, color="green")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(test_predicted), xmax=np.max(test_predicted))
    ax[1].legend()
    plt.savefig('Plot Normality of Residual.png')
    
    
    # =======================
    # Non Heteroscedasticity
    # =======================
    # A linear horizontal pattern
    # =======================
    
    print('============================ Check Non Heteroscedasticity  ============================')
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results["Residuals"].index, y=df_results["Residuals"], alpha=0.5)
    plt.plot(np.repeat(0, df_results["Residuals"].max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.savefig('Checking Non Heteroscedasticity.png')
    
    end=time()
    print ('Time needed (at work) in minutes: {0}'.format((end-start)/60))

#pickle model
def create_pickle(model,model_name):
    pickle.dump(model,open(model_name,'wb'))
    print('Success Created Pickling the Model : ',model_name)

#Running the code for get new model if there is new dataset
#cross_val_cb(x_train,y_train,5)
#train_catboost_regressor()

#Run for evaluation
evaluation_model_catboost("catboost_regressor_v1.model")
check_statistical_assumption("catboost_regressor_v1.model")
check_statistical_plot("catboost_regressor_v1.model")

#"/Users/anwar_raif/anwar_raif/cse-cml-regressor-development/model_autotrain"