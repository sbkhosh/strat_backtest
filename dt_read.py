#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import time
import yaml

from dt_help import Helper
from yahoofinancials import YahooFinancials

class DataProcessor():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)

    @Helper.timing
    def process(self):
        start_date = self.conf.get('start_date')
        end_date = self.conf.get('end_date')
        
        date_range = pd.bdate_range(start=start_date,end=end_date)
        values = pd.DataFrame({ 'Dates': date_range})
        values['Dates']= pd.to_datetime(values['Dates'])
        pairs = self.conf.get('pairs')
        
        for i in pairs:
            raw_data = YahooFinancials(i)
            raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
            df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose','volume']]
            df.columns = ['Dates1',i,'volume_'+i]
            df['Dates1']= pd.to_datetime(df['Dates1'])
            values = values.merge(df,how='left',left_on='Dates',right_on='Dates1')
            values = values.drop(labels='Dates1',axis=1)

        values = values.fillna(method="ffill",axis=0)
        values = values.fillna(method="bfill",axis=0)
        cols = values.columns.drop('Dates')
        values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=3)
        values.set_index('Dates',inplace=True)
        self.data = values

    def view_data(self):
        print(self.data.head())
        
    def drop_cols(self,col_names): 
        self.data.drop(col_names, axis=1, inplace=True)
        return(self)
               
    def write_to(self,name,flag):
        filename = os.path.join(self.output_directory,name)
        try:
            if('csv' in flag):
                self.data.to_csv(str(name)+'.csv')
            elif('xls' in flag):
                self.data.to_excel(str(name)+'xls')
        except:
            raise ValueError("not supported format")

    def save(self):
        pass
