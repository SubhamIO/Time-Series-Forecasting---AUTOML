import warnings
warnings.filterwarnings("ignore")

import math
import sys
import time
import itertools
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import date

from tqdm.notebook import tqdm
tqdm.pandas()
import plotly.express as px
import os
import yaml
import argparse

from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from functools import reduce
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
import datetime
from dateutil.relativedelta import relativedelta
import holidays
from collections import defaultdict
import statsmodels.api as sm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

## Read data from config file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    mdm_file_path = config["path_params"]["mdm_file_path"]
    raw_data_ten_years = config["path_params"]["raw_data_ten_years"]
    feature_file_path = config["path_params"]["feature_file_path"]
    revenue_date_EFD_by_day_path = config["path_params"]["revenue_date_EFD_by_day_path"]
    daily_gallons_full_path = config["path_params"]["daily_gallons_full_path"]
    card_counts_full_path = config["path_params"]["card_counts_full_path"]
    
    window_size_lstm = config["lstm_params"]["window_size_lstm"]
    optimizer = config["lstm_params"]["optimizer"]
    loss = config["lstm_params"]["loss"]
    
    train_period_lstm_model = config["multivariate_lstm_params"]["train_period_lstm_model"]
    pred_period_lstm_model = config["multivariate_lstm_params"]["pred_period_lstm_model"]
    num_units_lstm_model = config["multivariate_lstm_params"]["num_units_lstm_model"]
    optimizer_lstm_model = config["multivariate_lstm_params"]["optimizer_lstm_model"]
    loss_function_lstm_model = config["multivariate_lstm_params"]["loss_function_lstm_model"]
    batch_size_lstm_model = config["multivariate_lstm_params"]["batch_size_lstm_model"]
    num_epochs_lstm_model = config["multivariate_lstm_params"]["num_epochs_lstm_model"]
    
    seasonal_period = config["hw_params"]["seasonal_period"]
    trend = config["hw_params"]["trend"]
    seasonal_hw = config["hw_params"]["seasonal"]
    alpha = config["hw_params"]["alpha"]
    beta = config["hw_params"]["beta"]
    gamma = config["hw_params"]["gamma"]
    
    information_criterion = config["arimax_params"]["information_criterion"]
    seasonal_arimax = config["arimax_params"]["seasonal"]
    n_jobs = config["arimax_params"]["n_jobs"]
    
    learning_rate_xgb = config["xgboost_params"]["learning_rate"]
    n_estimators_xgb = config["xgboost_params"]["n_estimators"]
    subsample_xgb = config["xgboost_params"]["subsample"]
    max_depth_xgb = config["xgboost_params"]["max_depth"]
    colsample_bytree_xgb = config["xgboost_params"]["colsample_bytree"]
    min_child_weight_xgb = config["xgboost_params"]["min_child_weight"]
    
    
    date_year = config["primary_params"]["date_year"]
    date_month = config["primary_params"]["date_month"]
    forecast_target_column = config["primary_params"]["forecast_target_column"]
    business_program_name = config["primary_params"]["business_program_name"]
    
    models = config["global_params"]["models"]
    model_for_exception_accounts = config["global_params"]["model_for_exception_accounts"]
    granularity_level = config["global_params"]["granularity_level"]
    current_month = config["global_params"]["current_month"]
    validation_window_length = config["global_params"]["evaluation_window_length"]
    future_prediction_months = config["global_params"]["future_prediction_months"]
    performance_assessment_window = config["global_params"]["performance_assessment_window"]
    external_feature_list = config["global_params"]["external_feature_list"]
    choice = config["global_params"]["choice"]
    
    lstm_train_window_length_EFD = config["efd_lstm_params"]["lstm_train_window_length"]
    num_units_EFD = config["efd_lstm_params"]["num_units"]
    activation_function_EFD = config["efd_lstm_params"]["activation_function"]
    optimizer_EFD = config["efd_lstm_params"]["optimizer"]
    loss_function_EFD = config["efd_lstm_params"]["loss_function"]
    batch_size_EFD = config["efd_lstm_params"]["batch_size"]
    num_epochs_EFD = config["efd_lstm_params"]["num_epochs"]
    MODELS_EFD = config["efd_lstm_params"]["MODELS"]
    ma_mapping_EFD = config["efd_lstm_params"]["ma_mapping"]
    
    return mdm_file_path,raw_data_ten_years,feature_file_path,revenue_date_EFD_by_day_path,daily_gallons_full_path,window_size_lstm,optimizer,loss,models,granularity_level,seasonal_period,trend,seasonal_hw,information_criterion,seasonal_arimax,n_jobs,current_month,validation_window_length,future_prediction_months,date_year,date_month,forecast_target_column,performance_assessment_window,lstm_train_window_length_EFD,num_units_EFD,activation_function_EFD,optimizer_EFD,loss_function_EFD,batch_size_EFD,num_epochs_EFD,MODELS_EFD,ma_mapping_EFD,external_feature_list,learning_rate_xgb,n_estimators_xgb,subsample_xgb,max_depth_xgb,colsample_bytree_xgb,min_child_weight_xgb,train_period_lstm_model,pred_period_lstm_model,num_units_lstm_model,optimizer_lstm_model,loss_function_lstm_model,batch_size_lstm_model,num_epochs_lstm_model,alpha,beta,gamma,card_counts_full_path,model_for_exception_accounts,choice,business_program_name

def find_len(df):
    row = df['gallons_list']
    l = len(row)
    return l

def state_length_check(grouped_id_level_ind):
    actual_length =len(grouped_id_level_ind['rev_date'])
    state_var = grouped_id_level_ind['state'][0]
    state_list = [state_var for i in range(actual_length)]
    return state_list

## Function for transforming data to Wex_id monthly level aggregated purchase_gallons_qty/revenue_amount
def data_transform_to_id_level(mdm_data,ten_yrs_data,date_year,date_month,forecast_target_column,granularity_level):
    # merging mdm_file and ten_years_raw_data to get the wex_ids
    combined_df = ten_yrs_data[['customer_account_id','customer_account_name','account_city','account_state_prov_code','customer_source_system_code','revenue_date',date_year,date_month,forecast_target_column]].merge(mdm_data[['accountnumber',granularity_level]],left_on = 'customer_account_id',right_on = 'accountnumber',how = 'left')
    # grouping at wex_id month level and calculating total_purchase_gallons_quantity/revenue
    grouped_id_level = combined_df.groupby([granularity_level,date_year,date_month],as_index=False).agg(total_values = (forecast_target_column,sum))
    
    # preprocessing the date
    DATE = []
    for y, m in zip(grouped_id_level[date_year], grouped_id_level[date_month]):
        DATE.append(date(y, m, 1))
    grouped_id_level['rev_date'] = DATE
    
    id_state = mdm_data.groupby(granularity_level).agg(state = ('location_state','last'))

    grouped_id_level=grouped_id_level.merge(id_state,on=granularity_level,how='left')
    
    grouped_id_level = grouped_id_level.groupby(granularity_level).agg(rev_date = ('rev_date',list),gallons_list = ('total_values',list),state = ('state',list))
    
    grouped_id_level['total_records'] = grouped_id_level.progress_apply(find_len,axis=1)

    grouped_id_level.sort_values('total_records',ascending=False,inplace = True)

    return grouped_id_level

def merge_cardcounts_feature_file(grouped_id_level_ind , card_data, granularity_level):
    df = pd.DataFrame()
    wid = grouped_id_level_ind[granularity_level]
    df['rev_date'] = grouped_id_level_ind['rev_date']
    df['gallons_list'] = grouped_id_level_ind['gallons_list']
    df['rev_date']=pd.to_datetime(df['rev_date'])

    # Creating unique column for joining tables data+state
    card_data = card_data[card_data[granularity_level]==wid]
    card_data['rev_date']=pd.to_datetime(card_data['rev_date'])
    card_data = card_data.sort_values('rev_date',ascending=True)    

    # Joining the tables to merge the extra features to our data
    df = df.merge(card_data[['rev_date','active_card_count','outstanding_cards_count']],on='rev_date',how='left')
    return list(df['active_card_count']),list(df['outstanding_cards_count'])

def merge_EFD_feature_file(grouped_id_level_ind , efd_features_monthly):
    df = pd.DataFrame()
    df['rev_date'] = grouped_id_level_ind['rev_date']
    df = df.merge(efd_features_monthly,on='rev_date',how='left')
    return list(df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd']),list(df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m']),list(df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m'])

def merge_CMD_feature_file(grouped_id_level_ind , cmd_features_monthly):
    df = pd.DataFrame()
    df['rev_date'] = grouped_id_level_ind['rev_date']
    df = df.merge(cmd_features_monthly,on='rev_date',how='left')
    return list(df['retail_and_recreation_percent_change_from_baseline'])

# def merge_feature_file_EFD(grouped_wex_id_level_ind , feature_file_path):
#     if feature_file_path!='':
#         df = pd.DataFrame()
#         df['rev_date'] = grouped_wex_id_level_ind['rev_date']

#         monthwise_efd = pd.read_csv(feature_file_path)
#         monthwise_efd['rev_date'] = pd.to_datetime(monthwise_efd['rev_date'])
#         df = df.merge(monthwise_efd,on = 'rev_date',how='left')
#         return list(df['EFD_per_month'])
    
# ## Function to merge external feature file with Wex_id level data
# def merge_feature_file(grouped_wex_id_level_ind , feature_file_path):
#     if feature_file_path!='':
#         df = pd.DataFrame()
#         df['rev_date'] = grouped_wex_id_level_ind['rev_date']
#         df['gallons_list'] = grouped_wex_id_level_ind['gallons_list']
#         df['state'] = grouped_wex_id_level_ind['state']
#         df['total_records_after_padding'] = grouped_wex_id_level_ind['total_records_after_padding']
        
#          # Creating unique column for joining tables data+state
#         df['rev_date_state_meta'] = df['rev_date'].astype('str') + ' ' + df['state'].astype('str')
        
#         statewise_ewd = pd.read_csv('D:/Users/W505723/WEX_Data_exploration/EDA_NewSanitisedData/effective_working_days_US.csv')
#         statewise_ewd['Month'] = statewise_ewd['Month'].astype('str')

#         DATE = []
#         for dt in zip(statewise_ewd['Month']):
#             y = int(dt[0].split('-')[0])
#             m = int(dt[0].split('-')[1])
#             DATE.append(date(y, m, 1))
#         statewise_ewd['rev_date'] = DATE
        
#         # Creating unique column for joining tables data+state
#         statewise_ewd['rev_date_state_meta'] = statewise_ewd['rev_date'].astype('str') + ' ' + statewise_ewd['State'].astype('str')
        
        
#         # Joining the tables to merge the extra features to our data
#         df = df.merge(statewise_ewd[['rev_date_state_meta','EWD','Holidays','Business Days']],on='rev_date_state_meta',how='left')
#         return list(df['EWD']),list(df['Holidays']),list(df['Business Days'])

## Function to check whether 2 years of data is present for training or not
# def dates_checker(df_dt,current_month):
#     row = df_dt['rev_date']
#     years_list = set(list(map(lambda x:x.year,row)))
#     last_date = row[-1]
#     if current_month.year -2 in(years_list) and current_month.year -1 in(years_list) and current_month.year in(years_list) and last_date >= pd.to_datetime(current_month):
#         return 1
#     else:
#         return 0
# def dates_checker(df_dt,current_month):
#     row = df_dt['rev_date']
#     years_list = set(list(map(lambda x:x.year,row)))
#     first_date = row[0]
#     last_date = row[-1]
#     if first_date <= pd.to_datetime('2016-01-01') and current_month.year -2 in(years_list) and current_month.year -1 in(years_list) and current_month.year in(years_list) and last_date >= pd.to_datetime(current_month):
#         return 1
#     else:
#         return 0    
def dates_checker(df_dt,current_month):
    row = df_dt['rev_date']
    years_list = set(list(map(lambda x:x.year,row)))
    first_date = row[0]
    last_date = row[-1]
    if first_date <= (current_month + pd.DateOffset(months=-36+1)) and current_month.year -2 in(years_list) and current_month.year -1 in(years_list) and current_month.year in(years_list) and last_date > pd.to_datetime(current_month):
        return 1
    elif first_date > (current_month + pd.DateOffset(months=-36+1)) and current_month.year in(years_list) and last_date > pd.to_datetime(current_month):
        return 2
    else:
        return 0
    
def dates_checker_2(df_dt,current_month):
    row = df_dt['rev_date']
    years_list = set(list(map(lambda x:x.year,row)))
    first_date = row[0]
    last_date = row[-1]
    if first_date <= (current_month + pd.DateOffset(months=-36+1)) and current_month.year -2 in(years_list) and current_month.year -1 in(years_list) and current_month.year in(years_list): #and last_date > pd.to_datetime(current_month):
        return 1
    elif first_date > (current_month + pd.DateOffset(months=-36+1)) and current_month.year in(years_list): #and last_date > pd.to_datetime(current_month):
        return 2
    else:
        return 0
    
## Missing value treatment
def padder(data,combined_date_list,combined_gallons_list,validation_window_length,current_month):
    dt = data['rev_date']
    gl = data['gallons_list']
    srs = pd.Series(dict(zip(dt,gl)))
    first_point_yr = dt[0].year
    first_point_month = dt[0].month
    #print(first_point_yr ,current_month.year-2)
#     if first_point_yr == current_month.year-2:
#         first_date = '01'+'-01-'+str(first_point_yr)
#     else:
        
#         first_date = str(first_point_month)+'-01-'+str(first_point_yr)
    first_date = str(first_point_month)+'-01-'+str(first_point_yr)
    idx = pd.date_range(first_date, current_month+pd.DateOffset(months = validation_window_length),freq='MS') #mm-dd-yyyy ##2020
    #idx = pd.date_range(first_date, '12-01-2019',freq='MS') #mm-dd-yyyy ##2020
    srs.index = pd.DatetimeIndex(dt)
    srs = srs.reindex(idx, fill_value=0)
    s_df = pd.DataFrame(srs).reset_index()
    s_df = s_df.rename(columns={'index':'dates',0:'gallons'})
    dates = list(s_df['dates'])
    gallons = list(s_df['gallons'])
    combined_date_list.append(dates)
    combined_gallons_list.append(gallons)

## Function for calculating symmetric MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/(y_true+1e-4))) *100


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Create LSTM model
def create_lstm(time_step,optimizer,loss):
    model = Sequential()
    # Input layer
    model.add(LSTM(100, return_sequences=True,input_shape=(time_step,1)))
    # Hidden layer
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer=optimizer,loss=loss)
    return model

# Create BiLSTM model
def create_bilstm(time_step):
    model = Sequential()
    # Input layer
    model.add(Bidirectional(
              LSTM(100, return_sequences=True), 
              input_shape=(time_step,1)))
    model.add(Dropout(0.2))
    # Hidden layer
    model.add(Bidirectional(LSTM(75,return_sequences=True)))
    model.add(Bidirectional(LSTM(64,return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer=optimizer,loss=loss)
    return model

# Create GRU model
def create_gru(time_step):
    model = Sequential()
    # Input layer
    model.add(GRU (100, return_sequences = True, input_shape = [time_step,1]))
    model.add(Dropout(0.2)) 
    # Hidden layer
    model.add(GRU(75,return_sequences=True))
    model.add(GRU(64,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(50)) 
    model.add(Dense(1)) 
    #Compile model
    model.compile(optimizer=optimizer,loss=loss)
    return model

def fit_model(model,X_train,y_train):
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
#                                                patience = 10)
#     history = model.fit(X_train, y_train ,validation_data=(X_test,ytest), epochs = 700,  
#                         batch_size = 16, 
#                         callbacks = [early_stop])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=1)
    history = model.fit(X_train, y_train , epochs = 5,  validation_split = 0.33, shuffle = True,callbacks = [early_stop])

    return history


class HoltWinters:
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    # series - initial time series
    # slen - length of a season
    # alpha, beta, gamma - Holt-Winters model coefficients
    # n_preds - predictions horizon
    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


def split_sequence_for_lstm(sequence,train_period,pred_period,input_columns,output_columns):
    X, y = [], []
    
    for i in range(len(sequence)+1):
        x_start = i
        x_end = i+train_period
        
        if x_end > len(sequence):
            break
        seqx = sequence.iloc[x_start:x_end][input_columns].values
        X.append(seqx)
        y.append(sequence.iloc[x_end-1][output_columns])
    return np.array(X), np.array(y)
def create_lstm_model(X_train,optimizer,loss):
    model = Sequential()
    # Input layer
    model.add(LSTM(20, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
    # Hidden layer
#     model.add(LSTM(64,return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(8))
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer=optimizer,loss=loss)
    return model

## Multivariate LSTM model
def create_and_train_data_for_lstm_multivariate(id_level_data,
                                                current_month,
                                                validation_window_length,
                                                external_feature_list,
                                                train_period,
                                                pred_period,
                                                optimizer,
                                                loss_function,
                                                batch_size,
                                                num_epochs,
                                                granularity_level):
    df = pd.DataFrame()
    ids = id_level_data[granularity_level]
    df['rev_date'] = id_level_data['rev_date']
    df['gallons_list'] = id_level_data['gallons_list']
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in external_feature_list:
        df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m'] = id_level_data['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m']
    if 'covid_mobility_feature' in external_feature_list:
        df['covid_mobility_feature'] = id_level_data['covid_mobility_feature']
            
    
    df=df.replace(0,1)

    #Train Test split
    df_idx = df.reset_index(drop=True)
    idx = df_idx[pd.to_datetime(df_idx['rev_date'])==pd.to_datetime(current_month)].index
    idx=idx[0]
    training_window_df = df_idx[:idx-validation_window_length+1] 
    performance_assessment_window_df = df_idx[idx-validation_window_length+1:idx+1]  ##only efds present

    training_window_df=training_window_df.set_index('rev_date')
    performance_assessment_window_df=performance_assessment_window_df.set_index('rev_date')

    training_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    training_window_df.fillna(1,inplace=True)

    performance_assessment_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    performance_assessment_window_df.fillna(1,inplace=True)

    external_feature_list = [element for element in external_feature_list if element in training_window_df.columns]
    external_feature_list = ['gallons_list']+external_feature_list

    training_window_df = training_window_df[external_feature_list]
    training_window_df_EFD = training_window_df.copy()

    '''Scaling/Normalisation'''
    scaler=MinMaxScaler(feature_range=(0,1))
    colss = training_window_df.columns
    training_window_df = scaler.fit_transform(training_window_df)
    training_window_df = pd.DataFrame(training_window_df, columns =colss)

    efds = []
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd' in training_window_df.columns:
        efds.append('REV_EQUIVALENT_FUEL_DAY_FACTOR_efd')
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in training_window_df.columns:
        efds.append('REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m')
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m' in training_window_df.columns:
        efds.append('REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m')
    if 'covid_mobility_feature' in training_window_df.columns:
        efds.append('covid_mobility_feature')
    train_EFD_features = training_window_df_EFD[efds]
    scaler_efds = MinMaxScaler(feature_range=(0,1)).fit(train_EFD_features.values)
    train_EFD_features = scaler_efds.transform(train_EFD_features.values)

    performance_assessment_window_EFD_features = performance_assessment_window_df[efds]
    performance_assessment_window_EFD_features = scaler_efds.transform(performance_assessment_window_EFD_features.values)
    performance_assessment_window_EFD_features = pd.DataFrame(performance_assessment_window_EFD_features, columns =efds)

#             forecasting_window_EFD_features = forecasting_window_df[efds]
#             forecasting_window_EFD_features = scaler_efds.transform(forecasting_window_EFD_features.values)
#             forecasting_window_EFD_features = pd.DataFrame(forecasting_window_EFD_features, columns =efds)


    n_features = len(external_feature_list)
    inp_features = external_feature_list
    out_features = 'gallons_list'

    X_train, y_train = split_sequence_for_lstm(training_window_df,train_period,pred_period,inp_features,out_features)
    X_train =X_train.reshape(X_train.shape[0],train_period,n_features)
    try:
        model_mv = create_lstm_model(X_train,optimizer,loss_function)
        history = model_mv.fit(X_train, y_train,batch_size = batch_size,epochs = num_epochs, verbose=False)

        if train_period<12:
            n_months_train_data_needed = training_window_df.tail(12)
            length_shift = len(n_months_train_data_needed) - train_period
        else:
            n_months_train_data_needed = training_window_df.tail(train_period)
        n_months_train_data_needed.reset_index(drop=True,inplace=True)


        import warnings
        warnings.filterwarnings('ignore')
        input_columns=external_feature_list
        for i in range(validation_window_length):#12
            df_for_pred = pd.DataFrame()
            X = []
            if train_period<12:
                x_start = i+length_shift
                x_end = i+train_period+length_shift
            else:
                x_start = i
                x_end = i+train_period
            seqx = n_months_train_data_needed.iloc[x_start:x_end][input_columns].values
            X.append(seqx)
            X = np.array(X)

            X = X.reshape(X.shape[0],train_period,n_features)
            prediction = model_mv.predict(X)
            df_for_pred['gallons_list'] = prediction[0]

            if 'lag_12' in n_months_train_data_needed.columns:   
                gvl12 = n_months_train_data_needed.iloc[(idx+i)-(12-1),n_months_train_data_needed.columns.get_loc("gallons_list")]
                df_for_pred['lag_12'] = [gvl12]
            if 'lag_6' in n_months_train_data_needed.columns:   
                gvl6 = n_months_train_data_needed.iloc[(idx+i)-(6-1),n_months_train_data_needed.columns.get_loc("gallons_list")]
                df_for_pred['lag_6'] = [gvl6]
            if 'MA12' in n_months_train_data_needed.columns:   
                gvma12 = n_months_train_data_needed.iloc[(idx+i)-(12-1):(idx+i)+1,n_months_train_data_needed.columns.get_loc("gallons_list")].mean()
                df_for_pred['MA12'] = [gvma12]
            if 'MA6' in n_months_train_data_needed.columns:   
                gvma6 = n_months_train_data_needed.iloc[(idx+i)-(6-1):(idx+i)+1,n_months_train_data_needed.columns.get_loc("gallons_list")].mean()
                df_for_pred['MA6'] = [gvma6]
            if 'MSD12' in n_months_train_data_needed.columns:   
                gvmsd12 = n_months_train_data_needed.iloc[(idx+i)-(12-1):(idx+i)+1,n_months_train_data_needed.columns.get_loc("gallons_list")].std()
                df_for_pred['MSD12'] = [gvmsd12]
            if 'MSD6' in n_months_train_data_needed.columns:   
                gvmsd6 = n_months_train_data_needed.iloc[(idx+i)-(6-1):(idx+i)+1,n_months_train_data_needed.columns.get_loc("gallons_list")].std()
                df_for_pred['MSD6'] = [gvmsd6]
            if 'expanding_mean' in n_months_train_data_needed.columns:   
                gvem = n_months_train_data_needed.iloc[(idx+i)-(12-1),n_months_train_data_needed.columns.get_loc("expanding_mean")]
                df_for_pred['expanding_mean'] = [gvem]
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd' in n_months_train_data_needed.columns:  
                efd = performance_assessment_window_EFD_features.iloc[i,performance_assessment_window_EFD_features.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd'] = [efd]
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in n_months_train_data_needed.columns:   
                efd3m = performance_assessment_window_EFD_features.iloc[i,performance_assessment_window_EFD_features.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m'] = [efd3m]
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m' in n_months_train_data_needed.columns:   
                efd6m = performance_assessment_window_EFD_features.iloc[i,performance_assessment_window_EFD_features.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m'] = [efd6m]
            if 'covid_mobility_feature' in n_months_train_data_needed.columns:   
                cmf = performance_assessment_window_EFD_features.iloc[i,performance_assessment_window_EFD_features.columns.get_loc("covid_mobility_feature")]
                df_for_pred['covid_mobility_feature'] = [cmf]

            n_months_train_data_needed = n_months_train_data_needed.append(df_for_pred)

        '''Inverse Scaling/Normalisation'''
        cols = n_months_train_data_needed.columns
        n_months_train_data_needed = scaler.inverse_transform(n_months_train_data_needed)
        n_months_train_data_needed = pd.DataFrame(n_months_train_data_needed, columns =cols)

    #             future_horizon_forecast = n_months_train_data_needed[['gallons_list']].tail(24)#24
    #             performance_assessment_forecast= future_horizon_forecast[['gallons_list']].head(12) 
        performance_assessment_forecast= n_months_train_data_needed[['gallons_list']].tail(validation_window_length)
        performance_df = pd.DataFrame(np.column_stack([performance_assessment_window_df['gallons_list'],performance_assessment_forecast]))
        performance_df.index = pd.date_range(current_month+pd.DateOffset(months = -(validation_window_length-1)),current_month,freq='MS')
#         performance_df.index = pd.date_range(current_month+pd.DateOffset(months = 1),current_month+pd.DateOffset(months = 12),freq='MS')

        req_idx = performance_df.index

        performance_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
        performance_df.fillna(1,inplace=True)
        mape_lstm = mean_absolute_percentage_error(performance_df[0], performance_df[1])
        rmse_lstm = math.sqrt(mean_squared_error(performance_df[0], performance_df[1]))
        return [ids,mape_lstm,rmse_lstm,'LSTM_Multivariate',list(req_idx),list(performance_df[0]),list(performance_df[1])]
    except Exception as e:
        print(f'Model LSTM_Multivariate failed for {ids} with error {e}')
        sys.stdout.flush()
        req_idx = pd.date_range(current_month+pd.DateOffset(months = -(validation_window_length-1)),current_month,freq='MS')
        return [ids,np.nan,np.nan,'ERROR',list(req_idx),e,np.nan]

## Function for Training LSTM Model
def create_and_train_data_for_lstm(wex_id_level_data,window_size_lstm,future_prediction_months,optimizer,loss,current_month,validation_window_length):
    df = pd.DataFrame()

    df['rev_date'] = wex_id_level_data['rev_date']
    df['gallons_list'] = wex_id_level_data['gallons_list']
    wex_id = wex_id_level_data['wex_id']
    
    idx = df[pd.to_datetime(df['rev_date'])==pd.to_datetime(current_month)].index
    # Split into train and test set
    training_window = df[:idx[0]-validation_window_length+1] #validation_window_length=12
    validation_window = df[idx[0]-validation_window_length+1:idx[0]+1] 
    #test = df_hw[idx[0]+1:]

    df = df[pd.to_datetime(df['rev_date'])<=pd.to_datetime('2018-12-01')]

    idx = df[pd.to_datetime(df['rev_date'])==pd.to_datetime('2018-12-01')].index #17

    df1 = df.reset_index()['gallons_list']
    
    '''Scaling/Normalisation'''
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    '''Train Test Split'''
    train_data,test_data=df1[0:idx[0]+1,:],df1[idx[0]+1:len(df1),:1]
    #train_data = df1
    
    '''reshape into X=t,t+1,t+2,t+3 and Y=t+4'''
    time_step = window_size_lstm
    X_train, y_train = create_dataset(train_data, time_step)
    #X_test, ytest = create_dataset(test_data, time_step)
    
    '''reshape input to be [samples, time steps, features] which is required for LSTM'''
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    #X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    '''creating the model'''
    model_lstm = create_lstm(time_step,optimizer,loss)
    # model_bilstm = create_bilstm(time_step,optimizer,loss)
    # model_gru = create_gru(time_step,optimizer,loss)
    
    history_lstm = fit_model(model_lstm,X_train,y_train)
    #history_bilstm = fit_model(model_bilstm)
    #history_gru = fit_model(model_gru)
    
    '''Plotting'''
    # plot_loss (history_gru, 'GRU')
    # plot_loss (history_bilstm, 'Bidirectional LSTM')
    # plot_loss (history_lstm, 'LSTM')
    
    model = model_lstm
    #model = model_bilstm
    #model = model_gru
    
    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    #test_predict=model.predict(X_test)

    ##Transformback to original form
    y_train = scaler.inverse_transform(y_train.reshape(-1,1))
    train_predict=scaler.inverse_transform(train_predict)
    #test_predict=scaler.inverse_transform(test_predict)
    
    x_input=train_data[len(train_data)-time_step:].reshape(1,-1)
    print(x_input.shape)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    # x_input=test_data[len(test_data)-look_back:].reshape(1,-1)
    # print(x_input.shape)

    # temp_input=list(x_input)
    # temp_input=temp_input[0].tolist()
    
    lst_output=[]
    n_steps=time_step
    i=0
    while(i<future_prediction_months-12):#how many next days/months output

        if(len(temp_input)>time_step):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
            
    pred_df = pd.DataFrame()
    pred_df['ground_truth'] = test_df['gallons_list']
    pred_df['prediction'] = scaler.inverse_transform(lst_output)
    
    mape_lstm = mean_absolute_percentage_error(pred_df['ground_truth'], pred_df['prediction'])
    rmse_lstm = math.sqrt(mean_squared_error(pred_df['ground_truth'],pred_df['prediction']))
    
    pred_df_idx = pred_df.reset_index()
    final = pd.DataFrame()
    idx = pd.date_range('01-01-2019', '12-01-2019',freq='MS') #mm-dd-yyyy
    final['date'] = pd.DatetimeIndex(idx)
    # df3=df1.tolist()
    # df3.extend(lst_output)
    # df3=scaler.inverse_transform(df3).tolist()
    final['gallons'] = pred_df_idx['ground_truth']
    final['prediction'] = scaler.inverse_transform(lst_output)
    
    '''to shorten the plotting window'''
    df.set_index('rev_date',inplace=True)
    final.set_index('date',inplace=True)
    df_2016_2018 = df[pd.to_datetime('2016-01-01'):] 
   
    df_2016_2018 = pd.DataFrame(np.array(df_2016_2018).reshape(36, 1))
    df_2016_2018 = df_2016_2018.rename(columns = {0:'gallons_list'})
    idx = pd.date_range('01-01-2016', '12-01-2018',freq='MS') #mm-dd-yyyy
    df_2016_2018['rev_date'] = pd.DatetimeIndex(idx)
    
    final = pd.DataFrame(np.array(final).reshape(12, 2))
    final = final.rename(columns = {0:'prediction',1:'gallons'})
    idx = pd.date_range('01-01-2019', '12-01-2019',freq='MS') #mm-dd-yyyy
    test_index = idx
    final['date'] = pd.DatetimeIndex(idx)
    
    
    df_2016_2018.set_index('rev_date',inplace=True)
    final.set_index('date',inplace=True)
    test_df.set_index('rev_date',inplace=True)
    
    '''Plotting'''
    plt.figure(figsize=(10,4))
    plt.plot(df_2016_2018['gallons_list'], 'b--',label = 'TRAIN')
    plt.plot(test_df['gallons_list'], 'g--',label='TEST')
    plt.plot(final['gallons'], 'r--',label= 'LSTM Prediction: '+str(round(mape_lstm,2)))
    plt.title('LSTM Predictions')
    plt.legend()
    plt.grid()
    plt.show()
    
    preds = list(itertools.chain(*scaler.inverse_transform(lst_output)))
    
#     return wex_id,mape_lstm,rmse_lstm,'LSTM',list(wex_id_level_data['rev_date']),list(wex_id_level_data['gallons_list']),preds
    return wex_id,mape_lstm,rmse_lstm,'LSTM',list(test_index),list(test_df['gallons_list']),preds

## Function for Training Holt's Winter Model
def create_and_train_data_for_hw(id_level_data,
current_month,
validation_window_length,
alpha,
beta,
gamma,
granularity_level):
    df = pd.DataFrame()
    ids = id_level_data[granularity_level]
    df['rev_date'] = id_level_data['rev_date']
    df['gallons_list'] = id_level_data['gallons_list']
    df=df.replace(0,1)
#         df.set_index('rev_date', inplace=True)
#         df.sort_index(inplace=True) # sort the data as per the index      
    #Train Test split
    df_idx = df.reset_index(drop=True)
    idx = df_idx[pd.to_datetime(df_idx['rev_date'])==pd.to_datetime(current_month)].index
    idx=idx[0]
    training_window_df = df_idx[:idx-validation_window_length+1] 
    performance_assessment_window_df = df_idx[idx-validation_window_length+1:idx+1]  ##only efds present
    
    training_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    training_window_df.fillna(1,inplace=True)
    performance_assessment_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    performance_assessment_window_df.fillna(1,inplace=True)

    training_window_df=training_window_df.set_index('rev_date')
    performance_assessment_window_df=performance_assessment_window_df.set_index('rev_date')

   # Fit the model
#         alpha = 0.3
#         beta = 0.16
#         gamma = 0.3

#         if len(training_window_df)>=24:
#             slen=12
#             model = HoltWinters(series=training_window_df['gallons_list'], slen=slen, 
#                             alpha=alpha, beta=beta, gamma=gamma, n_preds=len(performance_assessment_window_df))
#             model.triple_exponential_smoothing()
#         else:
    try:
        slen = len(training_window_df)//2
        model = HoltWinters(series=training_window_df['gallons_list'], slen=slen, 
                        alpha=alpha, beta=beta, gamma=gamma, n_preds=validation_window_length)
        model.triple_exponential_smoothing()
    #         print(model.result)
        #performance_assessment_window_predictions = model.result[-performance_assessment_window:]
        forecasting_window_predictions = model.result[-validation_window_length:]
        performance_assessment_window_predictions = forecasting_window_predictions[:validation_window_length]

        performance_df = pd.DataFrame()
        performance_df["ground_truth"] = performance_assessment_window_df["gallons_list"]
        performance_df["Forecast_HW"] = performance_assessment_window_predictions
        #print(performance_df)
        req_idx = performance_df.index

        performance_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
        performance_df.fillna(1,inplace=True)
        mape_hw = mean_absolute_percentage_error(performance_df.ground_truth, performance_df.Forecast_HW)
        rmse_hw = math.sqrt(mean_squared_error(performance_df.ground_truth, performance_df.Forecast_HW))

        future_df = pd.DataFrame()
        future_df["forecasts"] = forecasting_window_predictions
        future_df.fillna(1,inplace=True)
        return [ids,mape_hw,rmse_hw,'HW',list(req_idx),list(performance_df['ground_truth']),list(performance_df['Forecast_HW'])]
        
    except Exception as e:
        print(f'Model HW failed for {ids} with error {e}')
        sys.stdout.flush()
        req_idx = pd.date_range(current_month+pd.DateOffset(months = -11),current_month,freq='MS')
        return [ids,np.nan,np.nan,'ERROR',list(req_idx),np.nan,np.nan]


def create_and_train_data_for_hwwt(id_level_data,future_prediction_months,seasonal_period,trend,seasonal,current_month,performance_assessment_window,validation_window_length,alpha,beta,gamma,granularity_level):
    df = pd.DataFrame()

    df['rev_date'] = id_level_data['rev_date']
    df['gallons_list'] = id_level_data['gallons_list']
   
    ids = id_level_data[granularity_level]
    
    df_hw = df.copy()
    df_hw['gallons_list']=df_hw['gallons_list'].replace(0,1)
    df_hw.set_index('rev_date', inplace=True)
    df_hw.sort_index(inplace=True) # sort the data as per the index
    
    #idx = df[pd.to_datetime(df['rev_date'])==pd.to_datetime('2018-12-01')].index
    idx = df[pd.to_datetime(df['rev_date'])==pd.to_datetime(current_month)].index
    # Split into train and test set
    training_window = df_hw[:idx[0]-validation_window_length+1] #validation_window_length=12
    validation_window = df_hw[idx[0]-validation_window_length+1:idx[0]+1] 
    #test = df_hw[idx[0]+1:]
    training_window.replace([np.inf,-np.inf],np.NaN,inplace=True)
    training_window.fillna(1,inplace=True)
    validation_window.replace([np.inf,-np.inf],np.NaN,inplace=True)
    validation_window.fillna(1,inplace=True)
#     print(len(training_window),len(validation_window))
    #print(training_window.head())
    # Fit the model
#     alpha = 0.3
#     beta = 0.16
#     gamma = 0.3
    
    if len(training_window)>=24:
        slen=12
        model = HoltWinters(series=training_window['gallons_list'], slen=slen, 
                        alpha=alpha, beta=1, gamma=gamma, n_preds=len(validation_window))
        model.triple_exponential_smoothing()
    else:
        slen = len(training_window)//2
        model = HoltWinters(series=training_window['gallons_list'], slen=slen, 
                        alpha=alpha, beta=1, gamma=gamma, n_preds=len(validation_window))
        model.triple_exponential_smoothing()
#     if len(training_window)>=24:
#         fitted_model = ExponentialSmoothing(training_window['gallons_list'],
#                                             trend=trend,seasonal=seasonal,
#                                             seasonal_periods=seasonal_period).fit()
#     else:
#         fitted_model = ExponentialSmoothing(training_window['gallons_list'],
#                                             trend=trend).fit()
#     validation_window_predictions = fitted_model.forecast(validation_window_length) #evaluation_window_length=12, 12 months till Dec2018
    validation_window_predictions = model.result[-validation_window_length:]  
    pred_df_hw = pd.DataFrame()
    pred_df_hw['ground_truth'] = validation_window['gallons_list']
    pred_df_hw['predictions'] = validation_window_predictions
    req_idx = pred_df_hw.index
#     idx = pd.date_range('01-01-2018', '12-01-2018',freq='MS') #mm-dd-yyyy
#     req_idx = idx
#     pred_df_hw['rev_date'] = pd.DatetimeIndex(idx)
#     pred_df_hw.set_index('rev_date',inplace=True)
    pred_df_hw.replace([np.inf,-np.inf],np.NaN,inplace=True)
    pred_df_hw.fillna(1,inplace=True)
    mape_hw = mean_absolute_percentage_error(pred_df_hw['ground_truth'], pred_df_hw['predictions'])
    rmse_hw = math.sqrt(mean_squared_error(pred_df_hw['ground_truth'],pred_df_hw['predictions']))
    
#     '''to shorten the plotting window'''
#     df_2016_2018 = df_hw[pd.to_datetime('2016-01-01'):]
    
    '''Plotting'''
#     plt.figure(figsize=(10,4))
#     plt.plot(training_window['gallons_list'], 'b--',label = 'TRAINING WINDOW')
#     plt.plot(validation_window['gallons_list'], 'g--',label='VALIDATION WINDOW')
#     plt.plot(pred_df_hw['predictions'], 'y--',label='HW Prediction on VALIDATION WINDOW: '+str(round(mape_hw,2)))
#     plt.title('HW Predictions')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
    return ids,mape_hw,rmse_hw,'HWWT',list(req_idx),list(pred_df_hw['ground_truth']),list(pred_df_hw['predictions'])

def create_and_train_data_for_hwws(id_level_data,future_prediction_months,seasonal_period,trend,seasonal,current_month,performance_assessment_window,validation_window_length,alpha,beta,gamma,granularity_level):
    df = pd.DataFrame()

    df['rev_date'] = id_level_data['rev_date']
    df['gallons_list'] = id_level_data['gallons_list']
   
    ids = id_level_data[granularity_level]
    
    df_hw = df.copy()
    df_hw['gallons_list']=df_hw['gallons_list'].replace(0,1)
    df_hw.set_index('rev_date', inplace=True)
    df_hw.sort_index(inplace=True) # sort the data as per the index
    
    #idx = df[pd.to_datetime(df['rev_date'])==pd.to_datetime('2018-12-01')].index
    idx = df[pd.to_datetime(df['rev_date'])==pd.to_datetime(current_month)].index
    # Split into train and test set
    training_window = df_hw[:idx[0]-validation_window_length+1] #validation_window_length=12
    validation_window = df_hw[idx[0]-validation_window_length+1:idx[0]+1] 
    #test = df_hw[idx[0]+1:]
    training_window.replace([np.inf,-np.inf],np.NaN,inplace=True)
    training_window.fillna(1,inplace=True)
    validation_window.replace([np.inf,-np.inf],np.NaN,inplace=True)
    validation_window.fillna(1,inplace=True)
#     print(len(training_window),len(validation_window))
    #print(training_window.head())
    # Fit the model
#     alpha = 0.3
#     beta = 0.16
#     gamma = 0.3
    
#     if len(training_window)>=24:
#         slen=12
#         model = HoltWinters(series=training_window['gallons_list'], slen=slen, 
#                         alpha=alpha, beta=1, gamma=gamma, n_preds=len(validation_window))
#         model.triple_exponential_smoothing()
#     else:
#         slen = len(training_window)//2
#         model = HoltWinters(series=training_window['gallons_list'], slen=slen, 
#                         alpha=alpha, beta=1, gamma=gamma, n_preds=len(validation_window))
#         model.triple_exponential_smoothing()
    
    fitted_model = ExponentialSmoothing(training_window['gallons_list'],trend=trend).fit()
    validation_window_predictions = fitted_model.forecast(validation_window_length) #evaluation_window_length=12, 12 months till Dec2018
#     validation_window_predictions = model.result[-validation_window_length:]  
    pred_df_hw = pd.DataFrame()
    pred_df_hw['ground_truth'] = validation_window['gallons_list']
    pred_df_hw['predictions'] = validation_window_predictions
    req_idx = pred_df_hw.index
#     idx = pd.date_range('01-01-2018', '12-01-2018',freq='MS') #mm-dd-yyyy
#     req_idx = idx
#     pred_df_hw['rev_date'] = pd.DatetimeIndex(idx)
#     pred_df_hw.set_index('rev_date',inplace=True)
    pred_df_hw.replace([np.inf,-np.inf],np.NaN,inplace=True)
    pred_df_hw.fillna(1,inplace=True)
    mape_hw = mean_absolute_percentage_error(pred_df_hw['ground_truth'], pred_df_hw['predictions'])
    rmse_hw = math.sqrt(mean_squared_error(pred_df_hw['ground_truth'],pred_df_hw['predictions']))
    
#     '''to shorten the plotting window'''
#     df_2016_2018 = df_hw[pd.to_datetime('2016-01-01'):]
    
    '''Plotting'''
#     plt.figure(figsize=(10,4))
#     plt.plot(training_window['gallons_list'], 'b--',label = 'TRAINING WINDOW')
#     plt.plot(validation_window['gallons_list'], 'g--',label='VALIDATION WINDOW')
#     plt.plot(pred_df_hw['predictions'], 'y--',label='HW Prediction on VALIDATION WINDOW: '+str(round(mape_hw,2)))
#     plt.title('HW Predictions')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
    return ids,mape_hw,rmse_hw,'HWWS',list(req_idx),list(pred_df_hw['ground_truth']),list(pred_df_hw['predictions'])

## Function for Training ARIMAX Model
def create_and_train_data_for_arimax(id_level_data,
                                        current_month,
                                        validation_window_length,
                                        information_criterion,
                                        seasonal_arimax,
                                        external_feature_list,
                                        n_jobs,
                                        granularity_level):
    # read required data
    df = pd.DataFrame()
    ids = id_level_data[granularity_level]
    df['rev_date'] = id_level_data['rev_date']
    df['gallons_list'] = id_level_data['gallons_list']
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in external_feature_list:
        df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m'] = id_level_data['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m']
    if 'covid_mobility_feature' in external_feature_list:
        df['covid_mobility_feature'] = id_level_data['covid_mobility_feature']
    
        
    
    df=df.replace(0,1)
    # Introduce lag features
    external_feature_list = [element for element in external_feature_list if element in df.columns]
    exogenous_features = external_feature_list
    #Train Test split
    df_idx = df.reset_index(drop=True)
    idx = df_idx[pd.to_datetime(df_idx['rev_date'])==pd.to_datetime(current_month)].index
    idx=idx[0]
    training_window_df = df_idx[:idx-validation_window_length+1] 
    performance_assessment_window_df = df_idx[idx-validation_window_length+1:idx+1]  ##only efds present
    
    training_window_df=training_window_df.set_index('rev_date')
    performance_assessment_window_df=performance_assessment_window_df.set_index('rev_date')
#             forecasting_window_df=forecasting_window_df.set_index('rev_date')

    training_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    training_window_df.fillna(1,inplace=True)
    training_window_df=training_window_df.replace(0,1)
    performance_assessment_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    performance_assessment_window_df.fillna(1,inplace=True)
    performance_assessment_window_df=performance_assessment_window_df.replace(0,1)
#             forecasting_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
#             forecasting_window_df.fillna(1,inplace=True)
#             forecasting_window_df=forecasting_window_df.replace(0,1)
#         print(training_window_df)
#         print(performance_assessment_window_df[exogenous_features])
    
    try:
        model = auto_arima(training_window_df.gallons_list, exogenous=training_window_df[exogenous_features], 
                        error_action="ignore", suppress_warnings=True,seasonal=seasonal_arimax,
                        information_criterion=information_criterion,n_jobs=n_jobs)
        model_fit = model.fit(training_window_df.gallons_list, exogenous=training_window_df[exogenous_features])
        performance_assessment_forecast = []
        for i in range(validation_window_length): #12
            df_for_pred = pd.DataFrame()
            
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in exogenous_features:   
                efd3m = performance_assessment_window_df.iloc[i,performance_assessment_window_df.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m'] = [efd3m]
            if 'covid_mobility_feature' in exogenous_features:   
                cmf = performance_assessment_window_df.iloc[i,performance_assessment_window_df.columns.get_loc("covid_mobility_feature")]
                df_for_pred['covid_mobility_feature'] = [cmf]
                
            forecast = model_fit.predict(n_periods=1, exogenous=df_for_pred[exogenous_features])
            performance_assessment_forecast.append(forecast)
    #             model = auto_arima(training_window_df.gallons_list, exogenous=training_window_df[exogenous_features], 
    #                                error_action="ignore", suppress_warnings=True,seasonal=seasonal_arimax,
    #                                information_criterion=information_criterion,n_jobs=n_jobs)
    #             model_fit = model.fit(training_window_df.gallons_list, exogenous=training_window_df[exogenous_features])
    #         model=sm.tsa.statespace.SARIMAX(training_window_df.gallons_list,order=(6,0,2),
    #                                         seasonal_order=(6,0,2,12), exog=training_window_df[exogenous_features],
    #                                         initialization='approximate_diffuse')
    #         start_params = np.r_[[0] * (model.k_params - 1), 1]
    #         model_fit=model.fit(start_params=start_params)

    #         performance_assessment_forecast = model_fit.predict(start=len(training_window_df),
    #                                  end=len(training_window_df)+len(performance_assessment_window_df)-1,
    #                                  dynamic=True, 
    #                                  exog=performance_assessment_window_df[exogenous_features])
    #         future_horizon_forecast = model_fit.predict(start=len(training_window_df),
    #                                  end=len(training_window_df)+len(forecasting_window_df)-1,
    #                                  dynamic=True, 
    #                                  exog=forecasting_window_df[exogenous_features])

        performance_assessment_forecast= model_fit.predict(n_periods=validation_window_length,
                                                        exogenous=performance_assessment_window_df[exogenous_features])
    #             future_horizon_forecast= model_fit.predict(n_periods=future_prediction_months,
    #                                                                exogenous=forecasting_window_df[exogenous_features])

        performance_df = pd.DataFrame(np.column_stack([performance_assessment_window_df["gallons_list"],performance_assessment_forecast]))
        performance_df.index = pd.date_range(current_month+pd.DateOffset(months = -(validation_window_length-1)),current_month,freq='MS')
#         performance_df.index = pd.date_range(current_month+pd.DateOffset(months = 1),current_month+pd.DateOffset(months = 12),freq='MS')

        req_idx = performance_df.index

        performance_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
        performance_df.fillna(1,inplace=True)
        mape_arimax = mean_absolute_percentage_error(performance_df[0], performance_df[1])
        rmse_arimax = math.sqrt(mean_squared_error(performance_df[0], performance_df[1]))

    #         training_window_df = training_window_df.reset_index()
    #         performance_assessment_window_df = performance_assessment_window_df.reset_index()
    #         performance_df = performance_df.reset_index()
        return [ids,mape_arimax,rmse_arimax,'ARIMAX',list(req_idx),list(performance_df[0]),list(performance_df[1])]
    except Exception as e:
        print(f'Model ARIMAX failed for {ids} with error {e}')
        sys.stdout.flush()
        req_idx = pd.date_range(current_month+pd.DateOffset(months = -11),current_month,freq='MS')
        return [ids,np.nan,np.nan,'ERROR',list(req_idx),np.nan,np.nan]

def create_and_train_data_for_xgboost(id_level_data,
                                    current_month,
                                    validation_window_length,
                                    external_feature_list,
                                    learning_rate_xgb,
                                    n_estimators_xgb,
                                    subsample_xgb,
                                    max_depth_xgb,
                                    colsample_bytree_xgb,
                                    min_child_weight_xgb,
                                    granularity_level):

    import xgboost as xgb
    # read required data
    df = pd.DataFrame()
    ids = id_level_data[granularity_level]
    df['rev_date'] = id_level_data['rev_date']
    df['gallons_list'] = id_level_data['gallons_list']

    #df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd'] = id_level_data['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd']
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in external_feature_list:
        df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m'] = id_level_data['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m']
    #df['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m'] = id_level_data['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m']
    if 'covid_mobility_feature' in external_feature_list:
        df['covid_mobility_feature'] = id_level_data['covid_mobility_feature']
    ## Adding time based features
#         for i in [1,2,3,6,9,12]:
#         for i in [1,2,3,6,12]:
#             df['lag_{}'.format(i)] = df['gallons_list'].shift(i)
#             df['lag_{}'.format(i)].fillna(df['lag_{}'.format(i)].mean(),inplace=True)

#         ## Moving Average

#         for i in [6,12]:
#             df['MA{}'.format(i)] = df['gallons_list'].rolling(window=i).mean()
#             df['MA{}'.format(i)].fillna(df['MA{}'.format(i)].mean(),inplace=True)

#         ## Moving Std Dev
#         for i in [6,12]:
#             df['MSD{}'.format(i)] = df['gallons_list'].rolling(window=i).std()
#             df['MSD{}'.format(i)].fillna(df['MSD{}'.format(i)].mean(),inplace=True)

#         ## Expanding mean
#         df['expanding_mean'] = df['gallons_list'].expanding(2).mean()  
#         df['expanding_mean'].fillna(df['expanding_mean'].mean(),inplace=True)
    df=df.replace(0,1)
    
    
    #Train Test split
    df_idx = df.reset_index(drop=True)
    idx = df_idx[pd.to_datetime(df_idx['rev_date'])==pd.to_datetime(current_month)].index
    idx=idx[0]
    training_window_df = df_idx[:idx-validation_window_length+1] 
    performance_assessment_window_df = df_idx[idx-validation_window_length+1:idx+1]  ##only efds present
    
    for i in [1,2,3,6,9,12]:
        training_window_df['lag_{}'.format(i)] = training_window_df['gallons_list'].shift(i)
        training_window_df['lag_{}'.format(i)].fillna(training_window_df['lag_{}'.format(i)].mean(),inplace=True)
#                 performance_assessment_window_df['lag_{}'.format(i)] = performance_assessment_window_df['gallons_list'].shift(i)
#                 performance_assessment_window_df['lag_{}'.format(i)].fillna(performance_assessment_window_df['lag_{}'.format(i)].mean(),inplace=True)

    ## Moving Average
    for i in [6,12]:
        training_window_df['MA{}'.format(i)] = training_window_df['gallons_list'].rolling(window=i).mean()
        training_window_df['MA{}'.format(i)].fillna(training_window_df['MA{}'.format(i)].mean(),inplace=True)
#                 performance_assessment_window_df['MA{}'.format(i)] = performance_assessment_window_df['gallons_list'].rolling(window=i).mean()
#                 performance_assessment_window_df['MA{}'.format(i)].fillna(performance_assessment_window_df['MA{}'.format(i)].mean(),inplace=True)

    ## Moving Std Dev
    for i in [6,12]:
        training_window_df['MSD{}'.format(i)] = training_window_df['gallons_list'].rolling(window=i).std()
        training_window_df['MSD{}'.format(i)].fillna(training_window_df['MSD{}'.format(i)].mean(),inplace=True)
#                 performance_assessment_window_df['MSD{}'.format(i)] = performance_assessment_window_df['gallons_list'].rolling(window=i).std()
#                 performance_assessment_window_df['MSD{}'.format(i)].fillna(performance_assessment_window_df['MSD{}'.format(i)].mean(),inplace=True)

    ## Expanding mean
    training_window_df['expanding_mean'] = training_window_df['gallons_list'].expanding(2).mean()  
    training_window_df['expanding_mean'].fillna(training_window_df['expanding_mean'].mean(),inplace=True)
#             performance_assessment_window_df['expanding_mean'] = performance_assessment_window_df['gallons_list'].expanding(2).mean()  
#             performance_assessment_window_df['expanding_mean'].fillna(performance_assessment_window_df['expanding_mean'].mean(),inplace=True)
    


    training_window_df=training_window_df.set_index('rev_date')
#             forecasting_window_df=forecasting_window_df.set_index('rev_date') ##
    performance_assessment_window_df=performance_assessment_window_df.set_index('rev_date')

    training_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    training_window_df.fillna(1,inplace=True)
#             forecasting_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
#             forecasting_window_df.fillna(1,inplace=True)
    performance_assessment_window_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
    performance_assessment_window_df.fillna(1,inplace=True)

    

    external_feature_list = [element for element in external_feature_list if element in training_window_df.columns]
    exogeneous_features = external_feature_list
    X_train = training_window_df[exogeneous_features]
    y_train = training_window_df[['gallons_list']]
#     print(X_train.shape,y_train.shape)
#         print(X_train.columns)
#         X_cv = performance_assessment_window_df[exogeneous_features]
#         y_cv = performance_assessment_window_df[['gallons_list']]

#         X_test = forecasting_window_df[exogeneous_features]
    collms = []
    if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in external_feature_list:
        collms.append('REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m')
    if 'covid_mobility_feature' in external_feature_list:
        collms.append('covid_mobility_feature')
    X_cv = performance_assessment_window_df[collms]
    y_cv = performance_assessment_window_df[['gallons_list']]

#             X_test = forecasting_window_df[['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m']]
    if 'expanding_mean' in X_train.columns:
        gvem_sc = X_train['expanding_mean']
        gvem_df = pd.DataFrame()
        gvem_df['expanding_mean'] = gvem_sc
        gvem_df.reset_index(inplace=True,drop=True)
    ## Standardization
    cols = X_train.columns
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    X_train = scaler.fit_transform(X_train)
    #X_cv1 = scaler.transform(X_cv)
    #X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns =cols)
    #X_cv1 = pd.DataFrame(X_cv1, columns =cols)
    #X_test = pd.DataFrame(X_test, columns =cols)


    try:
        # initialize Our first XGBoost model...
        regr = xgb.XGBRegressor(random_state=15)

        # declare parameters for hyperparameter tuning
        parameters ={'learning_rate': learning_rate_xgb,
                                'n_estimators':n_estimators_xgb,
                                'subsample': subsample_xgb,
                                'max_depth': max_depth_xgb,
                                'colsample_bytree': colsample_bytree_xgb,
                                'min_child_weight': min_child_weight_xgb
                                }
        # Perform cross validation 
        clf = RandomizedSearchCV(regr,
                            param_distributions = parameters,
                            scoring="neg_mean_squared_error",
                            cv=2,
                            n_jobs = -1,
                            verbose = False)
        result = clf.fit(X_train, y_train)

        # Summarize results
        #print("Best: %f using %s" % (result.best_score_, result.best_params_))
        means = result.cv_results_['mean_test_score']
        stds = result.cv_results_['std_test_score']
        params = result.cv_results_['params']
        #for mean, stdev, param in zip(means, stds, params):
            #print("%f 1(%f) with: %r" % (mean, stdev, param))

        xgb = result.best_estimator_
        xgb.fit(X_train, y_train)

        #================
        #y_train = y_train.reset_index(drop=True,inplace=False)
        X_cv = X_cv.reset_index(drop=True,inplace=False)
        #print(gvem_df)
        X_cv_plus = X_cv.copy()
        
        for i in range(validation_window_length): #12
            df_for_pred = pd.DataFrame()
            if 'lag_12' in X_train.columns: 

                gvl12 = y_train.iloc[i-(12-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_12'] = [gvl12]
            if 'lag_11' in X_train.columns:   
                gvl11 = y_train.iloc[i-(11-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_11'] = [gvl11]
            if 'lag_10' in X_train.columns:   
                gvl10 = y_train.iloc[i-(10-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_10'] = [gvl10]
            if 'lag_9' in X_train.columns:   
                gvl9 = y_train.iloc[i-(9-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_9'] = [gvl9]
            if 'lag_8' in X_train.columns:   
                gvl8 = y_train.iloc[i-(8-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_8'] = [gvl8]
            if 'lag_7' in X_train.columns:   
                gvl7 = y_train.iloc[i-(7-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_7'] = [gvl7]

            if 'lag_6' in X_train.columns:   
                gvl6 = y_train.iloc[i-(6-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_6'] = [gvl6]
            if 'lag_5' in X_train.columns:   
                gvl5 = y_train.iloc[i-(5-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_5'] = [gvl5]
            if 'lag_4' in X_train.columns:   
                gvl4 = y_train.iloc[i-(4-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_4'] = [gvl4]

            if 'lag_3' in X_train.columns:   
                gvl3 = y_train.iloc[i-(3-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_3'] = [gvl3]
            if 'lag_2' in X_train.columns:   
                gvl2 = y_train.iloc[i-(2-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_2'] = [gvl2]                      
            if 'lag_1' in X_train.columns:   
                gvl1 = y_train.iloc[i-(1-1),y_train.columns.get_loc("gallons_list")]
                df_for_pred['lag_1'] = [gvl1]
            if 'MA12' in X_train.columns:   
                gvma12 = y_train.iloc[i-(12-1):(idx+i)+1,y_train.columns.get_loc("gallons_list")].mean()
                df_for_pred['MA12'] = [gvma12]
            if 'MA6' in X_train.columns:   
                gvma6 = y_train.iloc[i-(6-1):(idx+i)+1,y_train.columns.get_loc("gallons_list")].mean()
                df_for_pred['MA6'] = [gvma6]
            if 'MSD12' in X_train.columns:   
                gvmsd12 = y_train.iloc[i-(12-1):(idx+i)+1,y_train.columns.get_loc("gallons_list")].std()
                df_for_pred['MSD12'] = [gvmsd12]
            if 'MSD6' in X_train.columns:   
                gvmsd6 = y_train.iloc[i-(6-1):(idx+i)+1,y_train.columns.get_loc("gallons_list")].std()
                df_for_pred['MSD6'] = [gvmsd6]
            if 'expanding_mean' in X_train.columns:   
                gvem = gvem_df.iloc[(idx+i)-(12-1),gvem_df.columns.get_loc("expanding_mean")] #2=columns number for expanding mean
                df_for_pred['expanding_mean'] = [gvem]
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd' in X_train.columns:  
                efd = X_cv_plus.iloc[i,X_cv_plus.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd'] = [efd]
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m' in X_train.columns:   
                efd3m = X_cv_plus.iloc[i,X_cv_plus.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m'] = [efd3m]
            if 'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m' in X_train.columns:   
                efd6m = X_cv_plus.iloc[i,X_cv_plus.columns.get_loc("REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m")]
                df_for_pred['REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m'] = [efd6m]
            if 'covid_mobility_feature' in X_train.columns:   
                cmf = X_cv_plus.iloc[i,X_cv_plus.columns.get_loc("covid_mobility_feature")]
                df_for_pred['covid_mobility_feature'] = [cmf]
            
    #             df_for_pred = pd.DataFrame({'lag_12': [gvl12], 'lag_6': [gvl6],'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd':[efd],
    #                            'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma3m':[efd3m],'REV_EQUIVALENT_FUEL_DAY_FACTOR_efd_ma6m':[efd6m]})
            if 'expanding_mean' in X_train.columns:
                gvem_df = gvem_df.append(df_for_pred['expanding_mean'])

            colss = df_for_pred.columns
            df_for_pred = scaler.transform(df_for_pred)
            df_for_pred = pd.DataFrame(df_for_pred, columns =colss)

            performance_assessment_forecast_test= xgb.predict(df_for_pred)
            y_train_pred = pd.DataFrame()
            y_train_pred['gallons_list'] = performance_assessment_forecast_test


            X_train = X_train.append(df_for_pred)
            y_train = y_train.append(y_train_pred)

    #         print(y_train.iloc[idx-12+1:idx])
    #         print(X_train.tail(12))
    #         print(X_cv.tail(5))
        #================

    #             future_horizon_forecast = y_train[['gallons_list']].tail(24)#24
    #             performance_assessment_forecast= future_horizon_forecast[['gallons_list']].head(12)
        performance_assessment_forecast= y_train[['gallons_list']].tail(validation_window_length)
        performance_df = pd.DataFrame(np.column_stack([y_cv["gallons_list"],performance_assessment_forecast]))
        performance_df.index = pd.date_range(current_month+pd.DateOffset(months = -(validation_window_length-1)),current_month,freq='MS')
#         performance_df.index = pd.date_range(current_month+pd.DateOffset(months = 1),current_month+pd.DateOffset(months = 12),freq='MS')

        req_idx = performance_df.index

        performance_df.replace([np.inf,-np.inf],np.NaN,inplace=True)
        performance_df.fillna(1,inplace=True)
        mape_xgb = mean_absolute_percentage_error(performance_df[0], performance_df[1])
        rmse_xgb = math.sqrt(mean_squared_error(performance_df[0], performance_df[1]))
        return [ids,mape_xgb,rmse_xgb,'XGB',list(req_idx),list(performance_df[0]),list(performance_df[1])]
    except Exception as e:
        print(f'Model XGB failed for {ids} with error {e}')
        sys.stdout.flush()
        req_idx = pd.date_range(current_month+pd.DateOffset(months = -(validation_window_length-1)),current_month,freq='MS')
        return [ids,np.nan,np.nan,'ERROR',list(req_idx),np.nan,np.nan]

## Function for Comparing different trained models for each time series data
def compare_models(models_trained,granularity_level):
    if len(models_trained)>1:
        results = reduce(lambda df1,df2: pd.merge(df1,df2,on=granularity_level,how='outer'), models_trained)
    else:
        results = models_trained[0]
    filter_col = [col for col in results if col.startswith('mape')]
    results.fillna(float('inf'),inplace=True)
    best_algo_list = list(results[filter_col].idxmin(axis=1))
    best_algo = list(map(lambda x: x.split('_',1)[1] , best_algo_list))
    best_algo_uppercase =list(map(lambda x: x.upper() , best_algo))
#     print(best_algo_uppercase)
    results['best_algo'] = best_algo_uppercase
    return results[[granularity_level,'best_algo']]

## Function for Forecasting using the best model selected

def choose_model_and_forecast(id_level_data,
                                current_month,
                                validation_window_length,
                                information_criterion,
                                seasonal_arimax,
                                external_feature_list,
                                n_jobs,
                                learning_rate_xgb,
                                n_estimators_xgb,
                                subsample_xgb,
                                max_depth_xgb,
                                colsample_bytree_xgb,
                                min_child_weight_xgb,
                                train_period,
                                pred_period,
                                optimizer,
                                loss_function,
                                batch_size,
                                num_epochs,
                                alpha,
                                beta,
                                gamma,
                                granularity_level):
    
    best_algo = id_level_data['best_algo']
#     print(best_algo)
    ids = id_level_data[granularity_level]
    current_month  = current_month + pd.DateOffset(months=validation_window_length)
#     print(current_month)
    try:
        if best_algo == 'LSTM_MULTIVARIATE':
            print(1)
            return create_and_train_data_for_lstm_multivariate(id_level_data,
                                                                current_month,
                                                                validation_window_length,
                                                                external_feature_list,
                                                                train_period,
                                                                pred_period,
                                                                optimizer,
                                                                loss_function,
                                                                batch_size,
                                                                num_epochs,
                                                                granularity_level)
        if best_algo == 'XGB':
            return create_and_train_data_for_xgboost(id_level_data,
                                                        current_month,
                                                        validation_window_length,
                                                        external_feature_list,
                                                        learning_rate_xgb,
                                                        n_estimators_xgb,
                                                        subsample_xgb,
                                                        max_depth_xgb,
                                                        colsample_bytree_xgb,
                                                        min_child_weight_xgb,
                                                        granularity_level)
        if best_algo == 'ARIMAX':
            return create_and_train_data_for_arimax(id_level_data,
                                                        current_month,
                                                        validation_window_length,
                                                        information_criterion,
                                                        seasonal_arimax,
                                                        external_feature_list,
                                                        n_jobs,
                                                        granularity_level)
        elif best_algo == 'HW':
            return create_and_train_data_for_hw(id_level_data,
                                                    current_month,
                                                    validation_window_length,
                                                    alpha,
                                                    beta,
                                                    gamma,
                                                    granularity_level)
    except:
        mape = np.nan
        rmse = np.nan
#         print(sys.exc_info()[2].tb_lineno)
        return [ids,best_algo,np.nan,'ERROR',np.nan,np.nan,np.nan]
        

    
def twelve_months_mape_calc(df):
    try:
        actual = df['gallons_actual(performance_assessment_period)']
        forecast = df['gallons_forecasts(performance_assessment_period)']
        return mean_absolute_percentage_error(actual, forecast)
    except:
        return np.nan
def twelve_months_rmse_calc(df):
    try:
        actual = df['gallons_actual(performance_assessment_period)']
        forecast = df['gallons_forecasts(performance_assessment_period)']
        return math.sqrt(mean_squared_error(actual, forecast))
    except:
        return np.nan
## Function for calculating 3 months MAPE
def three_months_mape_calc(df):
    try:
        actual = df['gallons_actual(performance_assessment_period)']
        forecast = df['gallons_forecasts(performance_assessment_period)']
        actual = actual[:3]
        forecast = forecast[:3]
        return mean_absolute_percentage_error(actual, forecast)
    except:
        return np.nan

## Function for calculating weighted MAPE
def weighted_mape_calc(df):
    try:
        actual = np.array(df['gallons_actual(performance_assessment_period)'])
        forecast = np.array(df['gallons_forecasts(performance_assessment_period)'])
        
        #mape = np.abs((actual-forecast)/((actual+forecast)/2 +1e-4)) *100
        mape = np.abs((actual-forecast)/(actual +1e-4)) *100
        #mape = mean_absolute_percentage_error(actual, forecast)
        n=len(actual)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        
        return sum((weights*mape)/sum(weights))
    except:
        return np.nan

## External features - EFD
def smape(actual,pred):
    try:
        num = np.abs(actual-pred)
        den = np.abs((actual+pred+1e-5))/2
        return np.mean(num/den) * 100
    except:
        return np.nan

def mape(actual,pred):
    try:
        return np.mean(np.abs((actual-pred)/actual)) * 100
    except:
        return np.nan

def preprocess_data(ewd,holidays_me):
    ewd['WEEKEND'] = ewd['REV_CALENDAR_DATE'].apply(lambda x: x.weekday() in [5,6]).astype(int)
    ewd = ewd.merge(holidays_me[['REV_CALENDAR_DATE', 'HOLIDAY']], on=['REV_CALENDAR_DATE'], how='left')
    ewd.fillna(0, inplace=True)
    ewd.set_index('REV_CALENDAR_DATE', inplace=True)
    return ewd

def split_sequence(sequence, 
                   train_period=7, 
                   pred_period=1, 
                   input_columns=['WEEKEND', 'HOLIDAY'],
                   output_columns='REV_EQUIVALENT_FUEL_DAY_FACTOR'):
    X, y = [], []
    
    for i in range(len(sequence)+1):
        x_start = i
        x_end = i+train_period
        
        if x_end > len(sequence):
            break
        seqx = sequence.iloc[x_start:x_end][input_columns].values
        X.append(seqx)
        y.append(sequence.iloc[x_end-1][output_columns])
    return np.array(X), np.array(y)

def evaluate_model(model, X, y, plot=False):
    y_pred = model.predict(X)
    y_pred = np.where(np.abs(y_pred)  < 0.01, 0, y_pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    MAPE = smape(y, y_pred)
                   
    print(f'RMSE: {rmse:.3f}')
    print(f'MAPE: {MAPE:.3f}')
    if plot:
        plt.plot(y.reshape(1,-1)[0]);
        plt.plot(y_pred.reshape(1,-1)[0]);
    return rmse, MAPE

def efd_feature_generator(holidays_me,current_month,lstm_train_window_length,num_units,activation_function,optimizer,loss_function,batch_size,num_epochs,MODELS,ma_mapping,revenue_date_EFD_by_day_path,daily_gallons_full_path,validation_window_length,future_prediction_months):
    
    new_current_month = current_month
    if current_month > '2020-01-01':
        new_current_month = '2019-12-01'
    
    ## Calibrating dates as required
    val_end = datetime.datetime.strptime(new_current_month, '%Y-%m-%d') + relativedelta(day=31)
    val_start = datetime.datetime.strptime(new_current_month, '%Y-%m-%d') - relativedelta(months=validation_window_length-1)

    train_end = val_start - relativedelta(days=1)
    train_start = train_end - relativedelta(months=lstm_train_window_length-1, day=1)

    pred_start = datetime.datetime.strptime(new_current_month, '%Y-%m-%d') + relativedelta(day=31) + relativedelta(days=1)
    pred_end = datetime.datetime.strptime(current_month, '%Y-%m-%d') + relativedelta(months=validation_window_length, day=31)
    print(train_start,train_end,val_start,val_end,pred_start,pred_end)
    ## Modelling

    model_results = defaultdict(list)
    op_df = pd.DataFrame(index=pd.date_range('2010-07-01', pred_end.strftime('%Y-%m-%d'), freq='D'))

    for model in MODELS:
        if model == 'efd':
            data = pd.read_csv(revenue_date_EFD_by_day_path, sep='\t')
            data['REV_CALENDAR_DATE'] = pd.to_datetime(data['REV_CALENDAR_DATE'])
        elif model in ['efd_ma3m', 'efd_ma6m']:
            data = pd.read_csv(daily_gallons_full_path)
            data[model] = data['purchase_gallons_qty'].rolling(ma_mapping[model]).mean()
            data['REV_EQUIVALENT_FUEL_DAY_FACTOR'] = data['purchase_gallons_qty']/data[model]
            data.rename(columns={'rev_calendar_date':'REV_CALENDAR_DATE'}, inplace=True)
            data['REV_CALENDAR_DATE'] = pd.to_datetime(data['REV_CALENDAR_DATE'])
            data = (data.set_index('REV_CALENDAR_DATE').resample('D').asfreq().
                    REV_EQUIVALENT_FUEL_DAY_FACTOR.fillna(0).reset_index())
            
        data = preprocess_data(data,holidays_me)

        train_df = data[train_start - pd.to_timedelta(6, 'd'):train_end]
        val_df = data[val_start - pd.to_timedelta(6, 'd'):val_end]
        date_data = pd.DataFrame(index=pd.date_range((pred_start - pd.to_timedelta(6, 'd')).strftime('%Y-%m-%d'),
                                                     pred_end.strftime('%Y-%m-%d'), freq='D'))
        date_data['REV_EQUIVALENT_FUEL_DAY_FACTOR'] = 0
        date_data.index.name = 'REV_CALENDAR_DATE'
        date_data.reset_index(inplace=True)
        future_df = preprocess_data(date_data,holidays_me)
    
#         future_df = data[pred_start - pd.to_timedelta(6, 'd'):pred_end]
        print(len(future_df))
        # split into samples
        X_train, y_train = split_sequence(train_df)
        X_val, y_val = split_sequence(val_df)
        X_fut, y_fut = split_sequence(future_df)

        ## Multivariate TS LSTM Model
        model_mv = Sequential()
        model_mv.add(LSTM(units = num_units, activation = activation_function, input_shape=(None, 2)))
        model_mv.add(Dense(units = 1))
        model_mv.compile(optimizer = optimizer, loss = loss_function)
        model_mv.fit(X_train, y_train, batch_size = batch_size, epochs = num_epochs, verbose=False)
        
        y_train_pred = model_mv.predict(X_train).flatten()
        y_val_pred = model_mv.predict(X_val).flatten()

        y_train_pred = np.where(np.abs(y_train_pred) < 0.01, 0, y_train_pred)
        y_val_pred = np.where(np.abs(y_val_pred) < 0.01, 0, y_val_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        train_mape = smape(y_train, y_train_pred)
        val_mape = smape(y_val, y_val_pred)
    
        model_results[f'{model}_train'].extend([train_rmse, train_mape])
        model_results[f'{model}_validation'].extend([val_rmse, val_mape])

        train_eval_df = pd.DataFrame({'date': pd.date_range(train_start, train_end, freq='D'), 
                                      'train': y_train,
                                      'train_pred': y_train_pred})
        train_eval_df = train_eval_df.groupby([pd.Grouper(freq='MS',key='date')]).sum()
        train_rmse_M = np.sqrt(mean_squared_error(train_eval_df['train'], train_eval_df['train_pred']))
        train_mape_M = mape(train_eval_df['train'], train_eval_df['train_pred'])

        val_eval_df = pd.DataFrame({'date': pd.date_range(val_start, val_end, freq='D'), 
                                    'val': y_val,
                                    'val_pred': y_val_pred})
        val_eval_df = val_eval_df.groupby([pd.Grouper(freq='MS',key='date')]).sum()
        val_rmse_M = np.sqrt(mean_squared_error(val_eval_df['val'], val_eval_df['val_pred']))
        val_mape_M = mape(val_eval_df['val'], val_eval_df['val_pred'])

        model_results[f'{model}_train'].extend([train_rmse_M, train_mape_M])
        model_results[f'{model}_validation'].extend([val_rmse_M, val_mape_M])

        y_pred_fut = model_mv.predict(X_fut).flatten()
        y_pred = np.where(np.abs(y_pred_fut) < 0.01, 0, y_pred_fut)

        col = f'REV_EQUIVALENT_FUEL_DAY_FACTOR_{model}'
        pred_len = op_df.loc[:val_end].shape[0]
        op_df.loc[:val_end, col] = data.loc[:val_end, 'REV_EQUIVALENT_FUEL_DAY_FACTOR'].values[-pred_len:]
        op_df.loc[pred_start:pred_end, col] = y_pred

    op_df_m = op_df.resample('MS').sum()
    return op_df_m

def records_maintainer(final_res,df_dict,models_trained,granularity_level):
    wid = final_res[granularity_level]
    model_chosen = final_res['model_chosen']
    #model_chosen = 'XGB'
    if model_chosen == 'ARIMAX':
        results_arimax = models_trained[df_dict[model_chosen]]
        overall_mape = results_arimax[results_arimax[granularity_level] == wid]['mape_arimax'].iloc[0]

        evaluation_period_ground_truth = results_arimax[results_arimax[granularity_level] == wid]['ground_truth_arimax'].iloc[0]
        evaluation_period_predictions = results_arimax[results_arimax[granularity_level] == wid]['predictions_arimax'].iloc[0]

        evaluation_period_ground_truth_3m = evaluation_period_ground_truth[:3]
        evaluation_period_forecast_3m = evaluation_period_predictions[:3]
        mape_3m = mean_absolute_percentage_error(evaluation_period_ground_truth_3m, evaluation_period_forecast_3m)

        mape_initial = np.abs((np.array(evaluation_period_ground_truth)-np.array(evaluation_period_predictions))/(np.array(evaluation_period_ground_truth) +1e-4)) *100
        n=len(evaluation_period_ground_truth)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        wmape = sum((weights*mape_initial)/sum(weights))

        monthly_avg_vol_eval_period = np.mean(evaluation_period_ground_truth)
    if model_chosen == 'HW':
        results_hw = models_trained[df_dict[model_chosen]]
        overall_mape = results_hw[results_hw[granularity_level] == wid]['mape_hw'].iloc[0]

        evaluation_period_ground_truth = results_hw[results_hw[granularity_level] == wid]['ground_truth_hw'].iloc[0]
        evaluation_period_predictions = results_hw[results_hw[granularity_level] == wid]['predictions_hw'].iloc[0]

        evaluation_period_ground_truth_3m = evaluation_period_ground_truth[:3]
        evaluation_period_forecast_3m = evaluation_period_predictions[:3]
        mape_3m = mean_absolute_percentage_error(evaluation_period_ground_truth_3m, evaluation_period_forecast_3m)

        mape_initial = np.abs((np.array(evaluation_period_ground_truth)-np.array(evaluation_period_predictions))/(np.array(evaluation_period_ground_truth) +1e-4)) *100
        n=len(evaluation_period_ground_truth)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        wmape = sum((weights*mape_initial)/sum(weights))

        monthly_avg_vol_eval_period = np.mean(evaluation_period_ground_truth)
    if model_chosen == 'HWWT':
        results_hwwt = models_trained[df_dict[model_chosen]]
        overall_mape = results_hwwt[results_hwwt[granularity_level] == wid]['mape_hwwt'].iloc[0]

        evaluation_period_ground_truth = results_hwwt[results_hwwt[granularity_level] == wid]['ground_truth_hwwt'].iloc[0]
        evaluation_period_predictions = results_hwwt[results_hwwt[granularity_level] == wid]['predictions_hwwt'].iloc[0]

        evaluation_period_ground_truth_3m = evaluation_period_ground_truth[:3]
        evaluation_period_forecast_3m = evaluation_period_predictions[:3]
        mape_3m = mean_absolute_percentage_error(evaluation_period_ground_truth_3m, evaluation_period_forecast_3m)

        mape_initial = np.abs((np.array(evaluation_period_ground_truth)-np.array(evaluation_period_predictions))/(np.array(evaluation_period_ground_truth) +1e-4)) *100
        n=len(evaluation_period_ground_truth)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        wmape = sum((weights*mape_initial)/sum(weights))

        monthly_avg_vol_eval_period = np.mean(evaluation_period_ground_truth)
    
    if model_chosen == 'HWWS':
        results_hwws = models_trained[df_dict[model_chosen]]
        overall_mape = results_hwws[results_hwws[granularity_level] == wid]['mape_hwws'].iloc[0]

        evaluation_period_ground_truth = results_hwws[results_hwws[granularity_level] == wid]['ground_truth_hwws'].iloc[0]
        evaluation_period_predictions = results_hwws[results_hwws[granularity_level] == wid]['predictions_hwws'].iloc[0]

        evaluation_period_ground_truth_3m = evaluation_period_ground_truth[:3]
        evaluation_period_forecast_3m = evaluation_period_predictions[:3]
        mape_3m = mean_absolute_percentage_error(evaluation_period_ground_truth_3m, evaluation_period_forecast_3m)

        mape_initial = np.abs((np.array(evaluation_period_ground_truth)-np.array(evaluation_period_predictions))/(np.array(evaluation_period_ground_truth) +1e-4)) *100
        n=len(evaluation_period_ground_truth)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        wmape = sum((weights*mape_initial)/sum(weights))

        monthly_avg_vol_eval_period = np.mean(evaluation_period_ground_truth)
        
    if model_chosen == 'XGB':
        results_xgb = models_trained[df_dict[model_chosen]]
        overall_mape = results_xgb[results_xgb[granularity_level] == wid]['mape_xgb'].iloc[0]

        evaluation_period_ground_truth = results_xgb[results_xgb[granularity_level] == wid]['ground_truth_xgb'].iloc[0]
        evaluation_period_predictions = results_xgb[results_xgb[granularity_level] == wid]['predictions_xgb'].iloc[0]

        evaluation_period_ground_truth_3m = evaluation_period_ground_truth[:3]
        evaluation_period_forecast_3m = evaluation_period_predictions[:3]
        mape_3m = mean_absolute_percentage_error(evaluation_period_ground_truth_3m, evaluation_period_forecast_3m)

        mape_initial = np.abs((np.array(evaluation_period_ground_truth)-np.array(evaluation_period_predictions))/(np.array(evaluation_period_ground_truth) +1e-4)) *100
        n=len(evaluation_period_ground_truth)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        wmape = sum((weights*mape_initial)/sum(weights))

        monthly_avg_vol_eval_period = np.mean(evaluation_period_ground_truth)
    if model_chosen == 'LSTM_Multivariate':
        results_lstmmultivariate = models_trained[df_dict[model_chosen]]
        overall_mape = results_lstmmultivariate[results_lstmmultivariate[granularity_level] == wid]['mape_lstm_multivariate'].iloc[0]

        evaluation_period_ground_truth = results_lstmmultivariate[results_lstmmultivariate[granularity_level] == wid]['ground_truth_lstm_multivariate'].iloc[0]
        evaluation_period_predictions = results_lstmmultivariate[results_lstmmultivariate[granularity_level] == wid]['predictions_lstm_multivariate'].iloc[0]

        evaluation_period_ground_truth_3m = evaluation_period_ground_truth[:3]
        evaluation_period_forecast_3m = evaluation_period_predictions[:3]
        mape_3m = mean_absolute_percentage_error(evaluation_period_ground_truth_3m, evaluation_period_forecast_3m)

        mape_initial = np.abs((np.array(evaluation_period_ground_truth)-np.array(evaluation_period_predictions))/(np.array(evaluation_period_ground_truth) +1e-4)) *100
        n=len(evaluation_period_ground_truth)
        weights = [(n-i)/sum(range(n+1)) for i in range(n)]
        wmape = sum((weights*mape_initial)/sum(weights))

        monthly_avg_vol_eval_period = np.mean(evaluation_period_ground_truth)
    return overall_mape,mape_3m,wmape,monthly_avg_vol_eval_period

def metric_file_generator(final_res,df_dict,models_trained,models,granularity_level):
    
    wid = final_res[granularity_level]
#     pa_df = pd.DataFrame()
#     pa_df['month'] = final_res['month']
#     pa_df['gallons_forecast_using_bestmodel'] = final_res['gallons_forecasts(performance_assessment_period)']
#     pa_df = pa_df.apply(pd.Series.explode)


    overall_df = pd.DataFrame()  
    overall_df['month'] = final_res['rev_date']
    overall_df['gallons_actual'] = final_res['gallons_list']
    overall_df = overall_df.apply(pd.Series.explode)
    overall_df[granularity_level] = [wid]*len(overall_df)
    overall_df = overall_df[[granularity_level,'month','gallons_actual']]
    

    if 'XGB' in list(df_dict.keys()):
        results_xgb = models_trained[df_dict['XGB']]
        xgb_df = pd.DataFrame()
        xgb_df['month'] = results_xgb[results_xgb[granularity_level]==wid]['rev_date_xgb']
        xgb_df['gallons_evaluation_period_xgb'] = results_xgb[results_xgb[granularity_level]==wid]['predictions_xgb']
        xgb_df = xgb_df.apply(pd.Series.explode)
        overall_df = overall_df.merge(xgb_df,on='month',how='outer')
    if 'HW' in list(df_dict.keys()):
        results_hw = models_trained[df_dict['HW']]
        hw_df = pd.DataFrame()
        hw_df['month'] = results_hw[results_hw[granularity_level]==wid]['rev_date_hw']
        hw_df['gallons_evaluation_period_hw'] = results_hw[results_hw[granularity_level]==wid]['predictions_hw']
        hw_df = hw_df.apply(pd.Series.explode)
        overall_df = overall_df.merge(hw_df,on='month',how='outer')
    if 'HWWT' in list(df_dict.keys()):
        results_hwwt = models_trained[df_dict['HWWT']]
        hwwt_df = pd.DataFrame()
        hwwt_df['month'] = results_hwwt[results_hwwt[granularity_level]==wid]['rev_date_hwwt']
        hwwt_df['gallons_evaluation_period_hwwt'] = results_hwwt[results_hwwt[granularity_level]==wid]['predictions_hwwt']
        hwwt_df = hwwt_df.apply(pd.Series.explode)
        overall_df = overall_df.merge(hwwt_df,on='month',how='outer')
    if 'HWWS' in list(df_dict.keys()):
        results_hwws = models_trained[df_dict['HWWS']]
        hwws_df = pd.DataFrame()
        hwws_df['month'] = results_hwws[results_hwws[granularity_level]==wid]['rev_date_hwws']
        hwws_df['gallons_evaluation_period_hwws'] = results_hwws[results_hwws[granularity_level]==wid]['predictions_hwws']
        hwws_df = hwws_df.apply(pd.Series.explode)
        overall_df = overall_df.merge(hwws_df,on='month',how='outer')
    if 'ARIMAX' in list(df_dict.keys()):
        results_arimax = models_trained[df_dict['ARIMAX']]
        arimax_df = pd.DataFrame()
        arimax_df['month'] = results_arimax[results_arimax[granularity_level]==wid]['rev_date_arimax']
        arimax_df['gallons_evaluation_period_arimax'] = results_arimax[results_arimax[granularity_level]==wid]['predictions_arimax']
        arimax_df = arimax_df.apply(pd.Series.explode)
        overall_df = overall_df.merge(arimax_df,on='month',how='outer')
    if 'LSTM_Multivariate' in list(df_dict.keys()):
        results_lstmmultivariate = models_trained[df_dict['LSTM_Multivariate']]
        mul_lstm_df = pd.DataFrame()
        mul_lstm_df['month'] = results_lstmmultivariate[results_lstmmultivariate[granularity_level]==wid]['rev_date_lstm_multivariate']
        mul_lstm_df['gallons_evaluation_period_lstm_multivariate'] = results_lstmmultivariate[results_lstmmultivariate[granularity_level]==wid]['predictions_lstm_multivariate']
        mul_lstm_df = mul_lstm_df.apply(pd.Series.explode)
        overall_df = overall_df.merge(mul_lstm_df,on='month',how='outer')
        
    pa_df = pd.DataFrame()
    pa_df['month'] = final_res['month']
    pa_df['gallons_forecast_using_bestmodel'] = final_res['gallons_forecasts(performance_assessment_period)']
    pa_df = pa_df.apply(pd.Series.explode)
    overall_df = overall_df.merge(pa_df,on='month',how='outer')
    return overall_df

def metric_file_generator_for_exception_accounts(final_res_exp,granularity_level):
    
    wid = final_res_exp[granularity_level]

    overall_df = pd.DataFrame()  
    overall_df['month'] = final_res_exp['rev_date']
    overall_df['gallons_actual'] = final_res_exp['gallons_list']
    overall_df = overall_df.apply(pd.Series.explode)
    overall_df[granularity_level] = [wid]*len(overall_df)
    overall_df = overall_df[[granularity_level,'month','gallons_actual']]    
    
    df = pd.DataFrame()
    df['month'] = final_res_exp['month']
    df['gallons_forecast_using_bestmodel'] = final_res_exp['gallons_forecasts(performance_assessment_period)']
    df = df.apply(pd.Series.explode)
    overall_df = overall_df.merge(df,on='month',how='outer')
      
    return overall_df
def mean_calc(m,n):
    a,b = np.array(m), np.array(n)
    mean_ = (a + b)/2
    return mean_