#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from os.path import join
from sequencers import sequencer
from sklearn.manifold import TSNE


    
class Tsne(sequencer.Sequencer):
    def transform_feat(self):
        features = pd.DataFrame()
        for i, row in self.df.iterrows():
            feat = pd.Series(row['features'])
            features = features.append(feat,ignore_index=True)
        return features
    def get_plot_data(self, df):
        X = TSNE(random_state=self.params['random_state']).fit_transform(self.transform_feat())
        plot_data = pd.concat([pd.DataFrame(X), df], axis=1)
        self.df = plot_data.rename(columns={0: "t-SNE-X", 1: "t-SNE-Y"})



