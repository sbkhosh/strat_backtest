#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import time
import yaml

from dt_help import Helper

class DataProcessor():
    def __init__(self, input_directory, output_directory, input_file, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_file = input_file
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_file!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input file = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_file, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
    
    @Helper.timing
    def read_data(self):
        self.ext = self.input_file.split('.')[1]
        self.base = self.input_file.split('.')[0]
        filename = os.path.join(self.input_directory,self.input_file)

        try:
            if('csv' in self.ext):
                delim = Helper.get_delim(filename)
                self.data = pd.read_csv(filename,sep=delim)
                self.data.columns = [ ''.join(el for el in cl if el.isalnum()) for cl in self.data.columns.values ]
                self.data.columns = [ el if el != 'Date' else 'Dates' for el in self.data.columns ]
                self.cols = self.data.columns
                self.dims = self.data.shape
            elif('xls' in self.ext):
                self.data = pd.read_excel(filename)
                self.data.columns = [ ''.join(el for el in cl if el.isalnum()) for cl in self.data.columns.values ]
                self.data.columns = [ el if el != 'Date' else 'Dates' for el in self.data.columns ]
                self.cols = self.data.columns
                self.dims = self.data.shape
        except:
            raise ValueError("not supported format")
        
    @Helper.timing
    def process(self):
        self.data['Dates'] = pd.to_datetime(self.data['Dates'],format='%Y-%m-%d %H:%M:%S')
        self.data.sort_values('Dates',inplace=True)
        self.data.set_index('Dates',inplace=True)
        mask = (self.data.index >= self.conf.get('start_date')) & (self.data.index <= self.conf.get('end_date'))
        self.data = self.data.loc[mask]
        self.data.columns = self.data.columns + '_' + self.base 
        
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
