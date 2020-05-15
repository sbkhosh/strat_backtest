#!/usr/bin/python3

import csv
import inspect
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ta
import time
import yaml

from functools import wraps

class Helper():
    def __init__(self, input_directory, input_prm_file):
        self.input_directory = input_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, input parameter file  = {}'.format(self.input_directory, self.input_prm_file))

    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
            
    @staticmethod
    def timing(f):
        """Decorator for timing functions
        Usage:
        @timing
        def function(a):
        pass
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()
            print('function:%r took: %2.2f sec' % (f.__name__,  end - start))
            return(result)
        return wrapper

    @staticmethod
    def get_delim(filename):
        with open(filename, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        return(dialect.delimiter)

    @staticmethod
    def get_class_membrs(clss):
        res = inspect.getmembers(clss, lambda a:not(inspect.isroutine(a)))
        return(res)

    @staticmethod
    def check_missing_data(data):
        print(data.isnull().sum().sort_values(ascending=False))
               
    @staticmethod
    def plot_chart(df,col_name,dir_out):
        # Filter number of observations to plot
        data = df
    
        # Create figure and set axes for subplots
        fig = plt.figure()
        fig.set_size_inches((20, 16))
        ax_price = fig.add_axes((0, 0.72, 1, 0.2))
        ax_macd = fig.add_axes((0, 0.48, 1, 0.2), sharex=ax_price)
        ax_rsi = fig.add_axes((0, 0.24, 1, 0.2), sharex=ax_price)
        ax_volume = fig.add_axes((0, 0, 1, 0.2), sharex=ax_price)
    
        # price
        ax_price.plot(data.index, ta.trend.ema_indicator(data[col_name]), label='price ' + col_name)
        ax_price.legend()
   
        # macd
        ax_macd.plot(data.index, ta.trend.macd(data[col_name]), label='macd')
        ax_macd.plot(data.index, ta.trend.macd_diff(data[col_name]), label='macd_hist')
        ax_macd.plot(data.index, ta.trend.macd_signal(data[col_name]), label='macd_signal')
        ax_macd.legend()
    
        # rsi - above 70% = overbought, below 30% = oversold
        ax_rsi.set_ylabel("(%)")
        ax_rsi.plot(data.index, ta.momentum.rsi(data[col_name]), label='rsi')
        ax_rsi.plot(data.index, [70] * len(data.index), label="overbought")
        ax_rsi.plot(data.index, [30] * len(data.index), label="oversold")
        ax_rsi.legend()

        ax_volume.fill_between(data.index.map(mdates.date2num), data['volume_'+col_name], 0, label='volume')
        ax_volume.legend()
                     
        if('=' in col_name):
            col_name = col_name.replace('=','_')
        fig.savefig(dir_out + '/' + col_name + '.pdf')
        plt.show()

        
