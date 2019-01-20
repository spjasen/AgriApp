#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:47:13 2019
Model To Predict Price
@author: jasen
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

# Raw Data Import
raw = pd.read_csv('OninoMaharshtra.csv', sep=';')
agri = raw.dropna()
agri.index = pd.to_datetime(agri.Reported_Date)
agri.drop(columns=['State_Name','Group','Reported_Date'],axis=1,inplace=True)

# Data Cleaning
def cleanArrivales(data):
    return data.replace(",","")
agri['Arrivals'] = agri['Arrivals'].apply(cleanArrivales)
agri['Arrivals'] = agri['Arrivals'].astype(float)
agri['Arrivals'] = agri['Arrivals'].astype(int)
agri['Modal_Price'] = agri['Modal_Price'].astype(float)

# Data Export
agri.to_csv('onion.csv')

# Data Analysis
# -- Select Market
market = agri[agri['District_Name'] == 'Mumbai']
market.drop(columns=['District_Name', 'Market_Name'],axis=1,inplace=True)

# -- Plot Market Values
plt.plot(market.Arrivals,color='purple', label='Arrivals')
plt.plot(market.Min_Price,color='yellow', label='Min')
plt.plot(market.Max_Price,color='red', label='Max')
plt.plot(market.Modal_Price,color='blue', label='Modal')
plt.xlabel('Date')
plt.ylabel('Price in Qun')
plt.legend()

# Data Prepeartion for modal input
# -- LabelEncoder of Variety Column
labelEncoder = LabelEncoder()
market['Variety'] = labelEncoder.fit_transform(market['Variety'])
market = market[['Variety', 'Arrivals','Modal_Price', 'Min_Price','Max_Price']]

# --Declaration
num_features = 5
lag_steps = 1
label_feature = 'Max_Price'

def sequential_to_supervised(data, lag_steps = 1, n_out = 1, dropnan = True):
    features = 1 if type(data) is list else data.shape[1] 
    df = pd.DataFrame(data)
    cols = list()
    feature_names = list()
    
    for i in range(lag_steps, 0, -1):
        cols.append(df.shift(i))
        feature_names += [(str(df.columns[j])) + '(t-%d)' % (i) for j in range(features)] 
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            feature_names += [(str(df.columns[j])) + '(t)'  for j in range(features)] 
        else:
            feature_names += [(str(df.columns[j])) + '(t+%d)' % (i) for j in range(features)] 
    
    agg = pd.concat(cols, axis=1) 
    agg.columns = feature_names
    
    if dropnan:
        agg.dropna(inplace=True)
    return agg

supervised_dataset = sequential_to_supervised(market, lag_steps)
cols_at_end = [label_feature + '(t)']
supervised_dataset = supervised_dataset[[c for c in supervised_dataset if c not in cols_at_end] + 
                                        [c for c in cols_at_end if c in supervised_dataset]]

supervised_dataset.drop(supervised_dataset.columns[(num_features*lag_steps) : 
    (num_features*lag_steps + num_features -1)], axis=1, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
supervised_dataset_scaled = scaler.fit_transform(supervised_dataset)


split = int(supervised_dataset_scaled.shape[0]*0.8)
train = supervised_dataset_scaled[:split, :]
test = supervised_dataset_scaled[split:, :]

train_X, train_y = train[:, :-1], train[:, -1] 
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(85,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(85))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#model = load_model('newLSTM.h5')
history = model.fit(train_X, train_y, epochs=70, batch_size=175, validation_data=(test_X, test_y), 
                    verbose=2, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1) 
inv_yhat = scaler.inverse_transform(inv_yhat) 

inv_yhat = inv_yhat[:, num_features*lag_steps] 
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:, 0:], test_y), axis=1) 
inv_y = scaler.inverse_transform(inv_y) 
inv_y = inv_y[:, num_features*lag_steps] 

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat)) 
# print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y, label = 'Actual')
plt.plot(inv_yhat, label = 'Predicted')
plt.legend()

model.save('newLSTM.h5')
model.summary()
