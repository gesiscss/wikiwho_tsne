#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os.path import join
import pandas as pd

class Sequencer:
    def __init__(self, df, params, path):
        self.df = df.copy()
        self.params = params
        self.path = path
        self.name = type(self).__name__ + '_' + '_'.join( [f'{k}_{v}' for k,v in self.params.items()])
        self.dirpath = join(self.path, self.name)        

    def save(self):
        self.df.to_pickle(f"{self.dirpath}.pkl")
        print(f'file saved in {self.dirpath}')      
 

