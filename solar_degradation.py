import numpy as np
from scipy import stats
import pandas as pd
import statsmodels as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import TheilSenRegressor

'''
Objectives:
1. Determine the rate of performance degradation of the SERF-EAST solar photovoltaic system up to the year 2023 using data collected by the National Laboratory of the Rockies NLR (formerly 
National Renewable Energy Laboratory, NREL) and any additional supplemental data.
2. Determine the effectiveness of a simple normalization scheme at removing the influence of temperature and irradiance variations.
'''

# specify the FILEPATH
# dataset can be downloaded from github: https://github.com/picoHacking/AE498_project1.git
FILEPATH = 'C:/Users/Amanda/AE498_homework/AE 498 project 1/pv_summary_2011_2023.csv'
y_cols = ['performance ratio']
x_cols = ['T_cell_avg','poa_irradiance']
colors = ["#98939e87","#ffa6009a","#559BEB85"]

def get_years(FILEPATH):
    df = pd.read_csv(FILEPATH, delimiter=',', usecols=['year'])
    df = df.to_numpy().T
    df = df[0]
    return df

def get_months(FILEPATH):
    df = pd.read_csv(FILEPATH, delimiter=',', usecols=['month'])
    df = df.to_numpy().T
    df = df[0]
    return df

def outlier_mask(df):
    Q1 = df.quantile(0.15)
    Q3 = df.quantile(0.85)
    IQR = Q3 - Q1
    
    return ((df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)).any(axis=1)

def csv_to_input(FILEPATH, x_cols, y_cols):
    '''
    convert raw csv to input data
    input: input array, n elements
    y: observation array, n elements 
    output: x, y, t arrays with outliers removed from x and y
    
    for this project, y = f(x, t) is approximated as y = a + bx + ct
    we will test the significance of b and c
    '''
    inp = pd.read_csv(FILEPATH, delimiter=',',usecols=x_cols)
    obs = pd.read_csv(FILEPATH, delimiter=',',usecols=y_cols)
    #print('inp',inp)
    #print('obs',obs)
    years = get_years(FILEPATH)
    months = get_months(FILEPATH)
    t = 12*(years-years[0])+months
    x = inp
    x['time'] = t
    x.insert(0,'ones',1)
    y = obs # observation (i.e. performance ratio)
    # print('x shape', x.shape)
    # print('y shape', y.shape)
    # Remove outliers (simple method)
    
# Remove outliers, assuming ~normally distributed data
    
    x_outliers = outlier_mask(x)
    y_outliers = outlier_mask(y)
    # Keep only the rows that fall within the IQR
    keep = ~(y_outliers)
    x = x[keep]
    y = y[keep]
    t = t[keep]
    # Remove PR > 1
    # keep2 = (y <= 1).all(axis=1)
    # x = x[keep2]
    # y = y[keep2]
    # t = t[keep2]
    x_columns = x.columns
    return x, y, x_columns

def rlm(FILEPATH,x_cols,y_cols,i):
    '''
        linear regression via RLM algorithm
        (Robust linear regression method)
            inputs: 
            file path name, 
            name of predictor column(s) (not time), 
            name of observation column,
            index of parameter         
            output: 
            x datapoints, predicted y datapoints
    '''
    x, y, x_columns = csv_to_input(FILEPATH,x_cols,y_cols)
    
    x = x.to_numpy()
    y = y.to_numpy()
    x_bar = np.mean(x,axis=0)
    model = smapi.RLM(y, smapi.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    coeffs = model.params
    y_hat = x_bar @ coeffs - coeffs[i]*x_bar[i] + x.T[i]*coeffs[i]
    return x, y_hat, coeffs, model
    
def mlr(FILEPATH,x_cols,y_cols): 
    '''
    linear regression
        inputs: 
        file path name, 
        name of predictor column(s) (not time), 
        name of output column
        
        output: x and y datapoints, model coefficients, x column names list, model (to summarize)
    '''
    x, y, x_columns = csv_to_input(FILEPATH,x_cols,y_cols)
    x = x.to_numpy()
    y = y.to_numpy()
    # Fit the data with ordinary least squares
    model = sm.regression.linear_model.OLS(y, x).fit() 
    # Return the estimated coefficients
    
    return x, y, model.params, x_columns, model

def mlr_line(x, x_bar, beta, i):
    """
    multiple linear regression line for the ith predictor. 
        All x's except the ith x are fixed at their mean value. 
        Resulting curve: y_hat = B1*x1_bar + B2*x2_bar + ... + Bi*xi + ... + Bp*xp_bar
        
        inputs:
        x: input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. 
        The first column should be all ones.
        x_bar: Means of x, taken along axis 1, array-like
        beta: Model coefficients, array-like
        i: index of predictor, int
        
        output: predicted y values, array-like
    """
    y_hat = x_bar @ beta - beta[i]*x_bar[i] + x.T[i]*beta[i]
    #print("yhat",y_hat)
    return y_hat

def scatterplot(FILEPATH,x_cols,y_cols,compare=False):
    '''
    scatterplot matrix 
        compares to prediction by OLS and RLM.
        inputs: 
        FILEPATH: str file name of the dataset
        x_cols: list-like strs referring to the columns used for predictors
        y_cols: str name of the column of observations
    '''
    
    x, y, beta, x_columns, model = mlr(FILEPATH,x_cols,y_cols)
    print('beta', beta)
    n = x.shape[1]    
    x_bar = np.mean(x, axis=0)
    
    fig, axes = plt.subplots(
        nrows=n-1,
        ncols=2,
        figsize=(10, 3 * (n-1))
    )
    # replace x_columns with more aesthetic text
    x_columns_print = [r"Average cell temperature, $^\circ$C", r"POA irradiance, W/m$^2$", "Year"]
    
    for i, col in enumerate(x.T):
        if i>0:
            j=i-1
            
            axes[j, 0].scatter(col, y, color=colors[j], label='Experiment')
            axes[j, 0].set_xlabel(x_columns_print[j])
            axes[j, 0].set_ylabel("Weather-corrected PR")
            
            if compare:
                axes[j, 0].plot(col, mlr_line(x,x_bar,beta,i), color='black', lw='2', label='OLS Prediction')
                x_ts, y_tsi, betas, model2 = rlm(FILEPATH,x_cols,y_cols,i)
                axes[j,0].plot(col, y_tsi, lw='2', color="#FF0000FF", label='RLM Prediction')
                axes[j,0].legend(loc='lower left')

            axes[j, 1].hist(col, bins=20, color=colors[j], rwidth=0.9)
            axes[j, 1].set_xlabel(x_columns_print[j])
            axes[j, 1].set_ylabel("Frequency")

            axes[j, 0].grid(visible=True)
            
            #axes[j,0].set_ylim(0,100) 
    years = get_years(FILEPATH).astype(int)
    # print(years)
    # change x time ticks to years
    year_ticks = np.arange(years.min(),years.max()+1)
    tick_pos = 12*(year_ticks-years[0])+1
    axes[2,0].set_xticks(tick_pos)
    axes[2,0].set_xticklabels(year_ticks, rotation=90)
    axes[2,1].set_xticks(tick_pos)
    axes[2,1].set_xticklabels(year_ticks, rotation=90)
    
    plt.tight_layout(pad=1.0, w_pad=1.0,h_pad=2.0)
    plt.show()

def anova(FILEPATH,x_cols,y_cols,anova_remove):
    # Analysis of variance - determine individual influence of predictors  
    # test_params = which model params to exclude (1=temperature, 2=time)
    x, y, x_columns = csv_to_input(FILEPATH,x_cols,y_cols)
    reduced = np.delete(x, anova_remove, axis=1) 
    model = sm.regression.linear_model.OLS(y,x, missing='raise')
    results = model.fit()
    reduced_model = sm.regression.linear_model.OLS(y,reduced, missing='raise')
    reduced_results = reduced_model.fit()
    f_stat, p_value, df_diff = results.compare_f_test(reduced_results)
    print(f"effect of {x_columns[anova_remove[0]]} and {x_columns[anova_remove[1]]} on performance ratio")
    print(f"f stat: {f_stat}\np value: {p_value}\ndifference: {df_diff}")

anova(FILEPATH,x_cols,y_cols,anova_remove=[1,2])
    
# Print additional information from the model
x, y, model_params, x_columns, model = mlr(FILEPATH, x_cols, y_cols)
print("OLS model coefficients: (offset, temperature, irradiance, time)", model_params)
degradation_rate = (model_params[3]*12)*100 # convert per month to per year
print("OLS degradation rate",degradation_rate,"%/year")


x2, y2, model2_params, model2 = rlm(FILEPATH, x_cols, y_cols,0)
print("RLM model coefficients:",model2_params)
degradation_rate2 = (model2_params[3]*12)*100
print("RLM degradation rate", degradation_rate2,"%/year")

print("\n",model.summary())

print("\n",model2.summary())

scatterplot(FILEPATH,x_cols,y_cols,compare=True) 

def mlr_res_plot():
    '''
    residual plot
    '''
    x, y, beta, x_columns, model = mlr(FILEPATH,x_cols,y_cols)
    x_bar = np.mean(x, axis=0)
    y_hat_1 = mlr_line(x, x_bar, beta, 1)
    y_hat_2 = mlr_line(x, x_bar, beta, 2)
    y_hat_3 = mlr_line(x, x_bar, beta, 3)
    residuals = model.resid
    plt.scatter(y_hat_1,residuals,color=colors[0],label='avg cell temperature, $^\circ$C')
    plt.scatter(y_hat_2,residuals,color=colors[1],label='POA irradiance, W/m$^2$')
    plt.scatter(y_hat_3,residuals,color=colors[2],label='time, months')
    plt.xlabel('Fitted value')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid()
    plt.show()


mlr_res_plot()

def rlm_res_plot():
    x, y1, beta, model2 = rlm(FILEPATH,x_cols,y_cols,1)
    x, y2, beta, model2 = rlm(FILEPATH,x_cols,y_cols,2)
    x, y3, beta, model2 = rlm(FILEPATH,x_cols,y_cols,3)
    plt.scatter(y1, model2.resid, color=colors[0],label='avg cell temperature, $^\circ$C')
    plt.scatter(y2, model2.resid, color=colors[1],label='POA irradiance, W/m$^2$')
    plt.scatter(y3, model2.resid, color=colors[2],label='time, months')
    plt.xlabel('Fitted value')
    plt.ylabel('Residual')
    plt.grid()
    plt.legend()
    plt.show()

rlm_res_plot() # plot is too similar to OLS residual plot for visual comparison