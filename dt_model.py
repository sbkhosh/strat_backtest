#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dt_help import Helper
from pykalman import KalmanFilter
from sklearn import linear_model

class MeanRevertStrat():
    def __init__(
        self, 
        data: pd.DataFrame, 
        delta: 0.01,
        z_entry_threshold: 2.0,
        z_exit_threshold: 1.0
    ):
        self.data = data 
        self.delta = delta
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        cols = self.data.columns.values
        self.col_0 = cols[0]
        self.col_1 = cols[1]

    @staticmethod
    def rolling_beta(X, y, idx, window):
        assert len(X)==len(y)
        
        out_dates = []
        out_beta = []

        model_ols = linear_model.LinearRegression()
        
        for iStart in range(0, len(X)-window):        
            iEnd = iStart+window

            model_ols.fit(X[iStart:iEnd].values.reshape(-1,1), y[iStart:iEnd].values.reshape(-1,1))

            #store output
            out_dates.append(idx[iEnd])
            out_beta.append(model_ols.coef_[0][0])

        return pd.DataFrame({'beta':out_beta}, index=out_dates)
            
    def calc_slope_kf(self):
        self.pair_0 = self.data[self.col_0]
        self.pair_1 = self.data[self.col_1]
        trans_cov = self.delta / (1 - self.delta) * np.eye(2)
        obs_mat = np.vstack(
            [self.pair_0, np.ones(self.pair_0.shape)]
            ).T[:, np.newaxis]
    
        kf = KalmanFilter(
            n_dim_obs=1, 
            n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
        transition_covariance=trans_cov
        )
        self.state_means, self.state_covs = kf.filter(self.pair_1.values)
        self.slope = self.state_means[:, 0]
        self.intercept = self.state_means[:, 1]
        
    def draw_slope_intercept(self):
        pairs_name = self.col_0+'_'+self.col_1

        df = pd.DataFrame(dict(slope=self.slope,intercept=self.intercept),index=self.data.index)
        df.plot(subplots=True,figsize=(32,20))
        plt.title('delta = '+str(self.delta)+' - '+str(pairs_name))
        plt.show()
    
    def spread_zscore_kalman(self):
        self.data['spread'] = self.data[self.col_0] - self.slope * self.data[self.col_1]
        self.data['zscore'] = (self.data['spread'] - np.mean(self.data['spread'])) / np.std(self.data['spread'])
       
    def spread_zscore_ols(self,lookback=100):
        # Use the pandas Ordinary Least Squares method to fit a rolling
        # linear regression between the two closing price time series
        print("Fitting the rolling Linear Regression...")
        rols = self.rolling_beta(self.data[self.col_0],self.data[self.col_1],self.data[self.col_1].index,lookback)

        # Construct the hedge ratio and eliminate the first 
        # lookback-length empty/NaN period
        self.data['hedge_ratio'] = rols['beta']
        print(self.data)
        self.data = self.data.dropna()
        print(self.data)

        # Create the spread and then a z-score of the spread
        print("Creating the spread/zscore columns...")
        self.data['spread'] = self.data[self.col_0] - self.data['hedge_ratio'] * self.data[self.col_1]
        self.data['zscore'] = (self.data['spread'] - np.mean(self.data['spread'])) / np.std(self.data['spread'])
        
    def long_short_market_signals(self):
        """Create the entry/exit signals based on the exceeding of 
        z_enter_threshold for entering a position and falling below
        z_exit_threshold for exiting a position."""

        # Calculate when to be long, short and when to exit
        self.data['longs'] = (self.data['zscore'] <= -self.z_entry_threshold)*1.0
        self.data['shorts'] = (self.data['zscore'] >= self.z_entry_threshold)*1.0
        self.data['exits'] = (np.abs(self.data['zscore']) <= self.z_exit_threshold)*1.0

        # These signals are needed because we need to propagate a
        # position forward, i.e. we need to stay long if the zscore
        # threshold is less than z_entry_threshold by still greater
        # than z_exit_threshold, and vice versa for shorts.
        self.data['long_market'] = 0.0
        self.data['short_market'] = 0.0

        # These variables track whether to be long or short while
        # iterating through the bars
        long_market = 0
        short_market = 0

        # Calculates when to actually be "in" the market, i.e. to have a
        # long or short position, as well as when not to be.
        print("Calculating when to be in the market (be long or short or not to be)..")
        for i, b in enumerate(self.data.iterrows()):
            # Calculate longs
            if b[1]['longs'] == 1.0:
                long_market = 1            
            # Calculate shorts
            if b[1]['shorts'] == 1.0:
                short_market = 1
            # Calculate exists
            if b[1]['exits'] == 1.0:
                long_market = 0
                short_market = 0
            # This directly assigns a 1 or 0 to the long_market/short_market
            # columns, such that the strategy knows when to actually stay in!
            self.data['long_market'].iloc[i] = long_market
            self.data['short_market'].iloc[i] = short_market

    def portfolio_returns(self):
        """Creates a portfolio pandas DataFrame which keeps track of
        the account equity and ultimately generates an equity curve.
        This can be used to generate drawdown and risk/reward ratios."""
    
        # Convenience variables for symbols
        sym1 = self.col_0
        sym2 = self.col_1

        # Construct the portfolio object with positions information
        # Note that minuses to keep track of shorts!
        print("Constructing a portfolio...")
        self.portfolio = pd.DataFrame(index=self.data.index)
        self.portfolio['positions'] = self.data['long_market'] - self.data['short_market']
        self.portfolio[sym1] = (-1.0) * self.data[sym1] * self.portfolio['positions']
        self.portfolio[sym2] = self.data[sym2] * self.portfolio['positions']
        self.portfolio['total'] = self.portfolio[sym1] + self.portfolio[sym2]

        # Construct a percentage returns stream and eliminate all 
        # of the NaN and -inf/+inf cells
        print("Constructing the equity curve...")
        self.portfolio['returns'] = self.portfolio['total'].pct_change()
        self.portfolio['returns'].fillna(0.0, inplace=True)
        self.portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
        self.portfolio['returns'].replace(-1.0, 0.0, inplace=True)

        # Calculate the full equity curve
        self.portfolio['returns'] = (self.portfolio['returns'] + 1.0).cumprod()

    def plot_eq_curve(self):
        pairs_name = self.col_0 +' - ' + self.col_1
        fig, ax = plt.subplots(figsize=(32,20))
        self.portfolio['returns'].plot()
        plt.title('Equity curve for ' + str(pairs_name))
        plt.show()
