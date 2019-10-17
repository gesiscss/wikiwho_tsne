#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sequencers.clusterers import clusterer

class DBscan(clusterer.Clusterer):
    def transform_feat(self):
        features = pd.DataFrame()
        for i, row in self.df.iterrows():
            feat = pd.Series(row['features'])
            features = features.append(feat,ignore_index=True)
        return features
    def get_clusters(self):
        db = DBSCAN(eps=self.params['eps'], min_samples=self.params['min_samples']).fit(self.transform_feat())
        labels = db.labels_
        self.add_clusters(labels)





