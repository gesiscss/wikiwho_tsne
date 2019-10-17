#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sequencers.clusterers import clusterer

class Kmeans(clusterer.Clusterer):
    def transform_feat(self):
        features = pd.DataFrame()
        for i, row in self.df.iterrows():
            feat = pd.Series(row['features'])
            features = features.append(feat,ignore_index=True)
        return features
    def get_clusters(self):
        clusterer = KMeans(random_state=self.params['random_state'])
        clusters = clusterer.fit_predict(np.array(self.transform_feat()))
        self.add_clusters(clusters)





