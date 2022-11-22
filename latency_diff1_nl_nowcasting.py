#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import time
from river import stream
from river import compose
from river import linear_model
from river import preprocessing
from river import metrics
from time import gmtime, strftime

segmentos = [
    'AC-BA',
    'BA-AC',
    'AC-DF',
    'DF-AC',
    'AC-RJ',
    'RJ-AC',
    'AC-RO',
    'RO-AC',
    'AC-SP',
    'SP-AC',
    'BA-DF',
    'DF-BA',
    'BA-RJ',
    'RJ-BA',
    'BA-RO',
    'RO-BA',
    'BA-SP',
    'SP-BA',
    'DF-RJ',
    'RJ-DF',
    'DF-RO',
    'RO-DF',
    'DF-SP',
    'SP-DF',
    'RJ-RO',
    'RO-RJ',
    'RJ-SP',
    'SP-RJ',
    'RO-SP',
    'SP-RO'
]

logging.basicConfig(filename="diff_nowcasting_log.txt", level=logging.INFO)
timestr = "%d %b %Y %H:%M:%S"
total_time = 0

def open_csv_stream(file):
    X_y = stream.iter_csv(
    file,
    converters={
        'diff': float,
        'diff-1': float, 
    },
    parse_dates={'date': '%Y-%m-%d %H:%M:%S'},
    drop={
        'latency', 
        'lat-1'
    },
    target='diff'
    )
    return X_y

# Get timestamp from date
def get_timestamp(x):
    return {'to_timestamp':x['date'].timestamp()}


extract_features = compose.TransformerUnion(get_timestamp)

scale = preprocessing.StandardScaler()

learn = linear_model.LinearRegression(intercept_lr=0)

learn_pa = linear_model.PARegressor(
        C=0.15,
        mode=1,
        eps=0.01,
        learn_intercept=True
    )


model = extract_features | scale | learn

model_pa = extract_features | scale | learn_pa


def evaluate_model(model, infile, outfile): 
    global total_time
    # Define errors metrics and windows
    mae = metrics.Rolling(metrics.MAE(), 10)    
    rmse = metrics.Rolling(metrics.RMSE(), 10)  
    smape = metrics.Rolling(metrics.SMAPE(), 10)

    latency = []
    predicted = []
    dates = []
    y_trues = []
    y_preds = []
    mae_hist = []
    rmse_hist = []
    smape_hist = []
    abse_hist = []
    
    df = pd.read_csv(infile)
    i = 0
    
    stream = open_csv_stream(infile)
    start = time.time()
    for x, y in stream:
        
        # Get previoous prediction and update model 
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        # Update error        
        mae.update(y + df.loc[i]['lat-1'], y_pred + df.loc[i]['lat-1'])
        rmse.update(y + df.loc[i]['lat-1'], y_pred + df.loc[i]['lat-1'])
        smape.update(y + df.loc[i]['lat-1'], y_pred + df.loc[i]['lat-1'])
        
        mae_hist.append(mae.get())
        rmse_val = rmse.get()
        if isinstance( rmse_val, complex ):
            rmse_hist.append(0.0)
        else:
            rmse_hist.append(rmse_val)
        smape_hist.append(smape.get())
        abse_hist.append(abs(y - y_pred ))

        # Store original value and predicted
        dates.append(x['date'])
        latency.append(y + df.loc[i]['lat-1'])
        predicted.append(y_pred + df.loc[i]['lat-1'])
        y_trues.append(y)
        y_preds.append(y_pred)
        
        i = i+1
    end = time.time()
    total_time += end - start
    logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Time: '+ str(end - start))

    result = pd.DataFrame(data={'datetime': dates,'latency':latency, 'predicted':predicted,
                                'diff': y_trues, 'diff_predicted':y_preds,'MAE':mae_hist,'RMSE':rmse_hist,'SMAPE':smape_hist,
                                'ABSE':abse_hist})
        
    result.to_parquet(outfile)

for seg in segmentos:
    logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Starting '+seg+' processing.')
    print('['+str(strftime(timestr, gmtime()))+']: '+'Starting '+seg+' processing.')
    
    # evaluate_model(model, 
    #               'Datasets/owd_csv/owd-'+seg+'_diff1pre.csv',
    #               'Resultados/Linear_Regression/diff1/pre/owd-'+seg+'_diff1pre_nl.gzip')
    
    # evaluate_model(model_pa, 
    #               'Datasets/owd_csv/owd-'+seg+'_diff1pre.csv',
    #               'Resultados/PA_Regressor/diff1/pre/owd-'+seg+'_diff1pre_nl.gzip')
    
    # evaluate_model(model, 
    #               'Datasets/rtt_csv/rtt-'+seg+'_diff1pre.csv',
    #               'Resultados/Linear_Regression/diff1/pre/rtt-'+seg+'_diff1pre_nl.gzip')
    
    evaluate_model(model_pa, 
                  'Datasets/rtt_csv/rtt-'+seg+'_diff1pre.csv',
                  'Resultados/PA_Regressor/diff1/pre/rtt-'+seg+'_diff1pre_nl.gzip')

    logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Ending '+seg+' processing.')
    print('['+str(strftime(timestr, gmtime()))+']: '+'Ending '+seg+' processing.')

logging.info('['+str(strftime(timestr, gmtime()))+']: '+'Total time: '+ str(total_time))
print('['+str(strftime(timestr, gmtime()))+']: '+'Total time: '+ str(total_time))
