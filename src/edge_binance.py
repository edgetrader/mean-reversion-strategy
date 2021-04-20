### CREDITS: Adapted from this post by Peter Nistrup. Thank you for sharing.
### https://medium.com/swlh/retrieving-full-historical-data-for-every-cryptocurrency-on-binance-bitmex-using-the-python-apis-27b47fd8137f

from binance.client import Client
from dateutil import parser
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

import time
import math
import os.path


class binanceClient:
    def __init__(self, key, secret, data_folder):
        self.binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
        self.key = key
        self.secret = secret
        self.client = Client(api_key=key, api_secret=secret)
        self.data_folder = data_folder

    ### FUNCTIONS
    def minutes_of_new_data(self, symbol, kline_size, data, source):
        if len(data) > 0:  
            old = parser.parse(data["timestamp"].iloc[-1])
        elif source == "binance": 
            old = datetime.strptime('1 Jan 2017', '%d %b %Y')

        if source == "binance": 
            new = pd.to_datetime(self.client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')

        return old, new


    def get_all_binance(self, symbol, kline_size, save = False):
        filename = self.data_folder + '%s-%s-data.csv' % (symbol, kline_size)
        
        if os.path.isfile(filename): 
            data_df = pd.read_csv(filename)
        else: 
            data_df = pd.DataFrame()
            
        oldest_point, newest_point = self.minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
        delta_min = (newest_point - oldest_point).total_seconds()/60
        available_data = math.ceil(delta_min/self.binsizes[kline_size])
        
        if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): 
            print('Downloading all available %s data for %s. Be patient..!' % 
                (kline_size, symbol))
        else: 
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % 
                (delta_min, symbol, available_data, kline_size))
            
        klines = self.client.get_historical_klines(symbol, kline_size, 
                                                    oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                    newest_point.strftime("%d %b %Y %H:%M:%S"))
        data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 
                                            'volume', 'close_time', 'quote_av', 'trades', 
                                            'tb_base_av', 'tb_quote_av', 'ignore' ])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
        else: 
            data_df = data
            
        data_df.set_index('timestamp', inplace=True)
        
        if save: 
            data_df.to_csv(filename)
            
        print('All caught up..!')
        return data_df

