#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os.path import join
import pandas as pd
from sequencers import sequencer

class Vectorizer(sequencer.Sequencer):
    def add_features(self, features):
        """ add one column called features """
        self.df['features'] = features
 

