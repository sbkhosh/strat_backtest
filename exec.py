#!/usr/bin/python3

import csv
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import pandas as pd 
import warnings
import yaml

from datetime import datetime, timedelta
from dt_help import Helper
from dt_read import DataProcessor
from dt_model import MeanRevertStrat
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'

    obj_0 = DataProcessor('data_in','data_out','IWM.csv','conf_model.yml')
    obj_0.read_prm()
    obj_0.read_data()
    obj_0.process()

    obj_1 = DataProcessor('data_in','data_out','SPY.csv','conf_model.yml')
    obj_1.read_prm()
    obj_1.read_data()
    obj_1.process()

    yvar = obj_0.conf.get('yvar')
    df_0 = obj_0.data[[el for el in obj_0.data.columns if yvar in el]]
    df_1 = obj_1.data[[el for el in obj_1.data.columns if yvar in el]]
    df = pd.concat([df_0,df_1],axis=1,ignore_index=False)
    
    mr_strat = MeanRevertStrat(
        data=df,
        delta=0.01,
        z_entry_threshold=2.0,
        z_exit_threshold=1.0
    )

    mr_strat.calc_slope_kf()
    mr_strat.spread_zscore_kalman()
    mr_strat.long_short_market_signals()
    mr_strat.portfolio_returns()
    mr_strat.plot_eq_curve()
    
