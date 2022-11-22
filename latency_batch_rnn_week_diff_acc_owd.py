# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import pytz
import logging
import seaborn as sns
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pandas.io.json import json_normalize
from datetime import datetime
from time import gmtime, strftime
from os.path import exists

sns.set()

timestr = "%d %b %Y %H:%M:%S"
model_folder = "models/"
result_folder = "results/lstm_accumulated/owd/"
skip_existent_model = True

logging.basicConfig(filename="diff_lstm_week_acc_log.txt", level=logging.INFO)

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

def generate_model(data, model_name):

    logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Starting '+model_name+' processing.')

    dt = pd.read_parquet(data)
    dt['ord_date'] = dt['date'].dt.dayofyear
    dd = dt.groupby(['ord_date']).count()
    idx = dd.index.tolist()
    n = len(idx)

    for i in range(0,n-7):
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Starting '+model_name+str(i)+' processing.')

        if (skip_existent_model and exists(model_folder + model_name + str(i) +'.h5')):
            logging.info('['+str(strftime(timestr, gmtime()))+']: '+model_name+str(i)+' already exists. Skiping...')
            print('['+str(strftime(timestr, gmtime()))+']: '+model_name+str(i)+' already exists. Skiping...')
            continue
        if (exists(model_folder + model_name + str(i) +'.lock')):
            logging.info('['+str(strftime(timestr, gmtime()))+']: '+model_name+str(i)+' is processing. Skiping...')
            print('['+str(strftime(timestr, gmtime()))+']: '+model_name+str(i)+' is processing. Skiping...')
            continue
        open(model_folder + model_name + str(i) +'.lock', 'a').close()
        m = idx[0:i+8]
        df_train = dt[dt['ord_date'].between(m[0], m[len(m)-2])]
        logging.info('[' + str(strftime(timestr,gmtime())) + ']: Train dataset length: ' + str(len(m)-1))
        df_test = dt[dt['ord_date'] == m[len(m)-1]]
    
        dfc = dt[dt['ord_date'].between(m[0], m[len(m)-1])]

        df_train = df_train['diff'].values
        df_train = df_train.reshape(-1, 1)
        df_test = df_test['diff'].values
        df_test = df_test.reshape(-1, 1)

        dataset_train = np.array(df_train)
        dataset_test = np.array(df_test)

        scaler = MinMaxScaler(feature_range=(0,1))
        dataset_train = scaler.fit_transform(dataset_train)
        dataset_test = scaler.transform(dataset_test)

        if df_test.shape[0] < 50:
            logging.error('['+str(strftime(timestr, gmtime()))+']: '+'Model '+model_name+str(i)+' with no sufficient test tuples.')
            continue


        x_train, y_train = create_dataset(dataset_train)
        x_test, y_test = create_dataset(dataset_test)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        print('Starting '+model_name+' ' + str(i) + ' training...')
        start = time.time()
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Starting '+ model_name + str(i) +' training.')
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Starting time: '+ str(start))

        model.fit(x_train, y_train, epochs=50, batch_size=32)

        print('Ending training...')
        end = time.time()
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Ending '+ model_name + str(i) +' training.')
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'End time: '+ str(start))
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Total time: '+ str(end - start))

        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Saving model '+ model_name + str(i) + '.h5.')
        model.save(model_folder + model_name + str(i) +'.h5')
        os.remove(model_folder + model_name + str(i) +'.lock')

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        dfc = dfc[dfc.shape[0] - int(df_test.shape[0] - 50):]
        predictions = predictions.reshape(predictions.shape[0],)
        print(dfc.shape[0])
        print(len(predictions))
        result = pd.DataFrame(data={'datetime':dfc['date'],'latency':dfc['latency'],'diff':dfc['diff'],'predicted':predictions})
        
        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Saving results table ' + result_folder + model_name + str(i) +'-lstm.gzip\'.')
        result.to_parquet(result_folder + model_name + str(i) +'-lstm.gzip')

        logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Ending '+model_name+' '+str(i)+' processing.')

segments = [
    'AC-BA',
    'AC-DF',
    'AC-RJ',
    'AC-RO',
    'AC-SP',
    'BA-AC',
    'BA-DF',
    'BA-RJ',
    'BA-RO',
    'BA-SP',
    'DF-AC',
    'DF-BA',
    'DF-RJ',
    'DF-RO',
    'DF-SP',
    'RJ-AC',
    'RJ-BA',
    'RJ-DF',
    'RJ-RO',
    'RJ-SP',
    'RO-AC',
    'RO-BA',
    'RO-DF',
    'RO-RJ',
    'RO-SP',
    'SP-AC',
    'SP-BA',
    'SP-DF',
    'SP-RJ',
    'SP-RO' 
]

for seg in segments:
    generate_model('Datasets/owd_parquet/owd-'+seg+'_diff1pre.gzip', 'owd-diff_acc-'+seg)
    
