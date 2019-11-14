#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sequencers.clusterers import clusterer

def common_tokens(a, b, c, d, INTERSECTION): 
    if (len(set(a).intersection(set(b))) > INTERSECTION) and (len(set(c).intersection(set(d))) > INTERSECTION): 
        return(cluster)
    return(np.nan)

class Token_Similarity(clusterer.Clusterer):    

    def get_clusters(self):
        global cluster
        cluster = 1
        self.df['clusters'] = np.nan
        if self.params['token_type'] == 'token id':
            for i in range(len(self.df)):
                if np.isnan(self.df.loc[i,'clusters']):
                    self.df.loc[i:len(self.df), 'clusters'] = self.df.apply(lambda token_id: common_tokens(self.df.loc[i,'right_token'], token_id['right_token'], self.df.loc[i,'left_token'], token_id['left_token'], self.params['intersection']), axis=1)
                    cluster += 1
                
        else:
            for i in range(len(self.df)):
                if np.isnan(self.df.loc[i,'clusters']):
                    self.df.loc[i:len(self.df), 'clusters'] = self.df.apply(lambda token_id: common_tokens(self.df.loc[i,'right_token_str_clean'],token_id['right_token_str_clean'], self.df.loc[i,'left_token_str_clean'], token_id['left_token_str_clean'], self.params['intersection']), axis=1)
                    cluster += 1
        for i, row in pd.DataFrame(self.df['clusters'].value_counts()).iterrows():
            if row.clusters<self.params['min_samples']:
                self.df.loc[self.df['clusters']==i, 'clusters'] = -1





