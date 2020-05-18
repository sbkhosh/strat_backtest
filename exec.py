#!/usr/bin/python3

import matplotlib
import os
import pandas as pd 
import warnings

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

    # all_pairs = MeanRevertStrat.get_all_pairs()
    # Helper.plot_chart(df,obj_0.conf.get('pairs')[0],obj_0.output_directory)
    
    obj_0 = DataProcessor('data_in','data_out','conf_model.yml')
    obj_0.read_prm()
    obj_0.process()
    df = obj_0.data
    
    mr_strat = MeanRevertStrat(
        data=df,
        delta=0.01,
        z_entry_threshold=2.0,
        z_exit_threshold=1.0,
        periods=252
    )

    mr_strat.calc_slope_kf()
    mr_strat.draw_slope_intercept()
    mr_strat.spread_zscore_kalman()
    mr_strat.long_short_market_signals()
    mr_strat.portfolio_returns()
    mr_strat.plot_eq_curve()
    mr_strat.get_metrics()
    print(mr_strat.metrics)
