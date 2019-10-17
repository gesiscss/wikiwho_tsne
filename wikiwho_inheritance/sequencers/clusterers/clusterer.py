#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os.path import join
import pandas as pd
from sequencers import sequencer

class Clusterer(sequencer.Sequencer):
    def add_clusters(self, clusters):
        """ add one column called clusters """
        self.df['clusters'] = clusters
 

