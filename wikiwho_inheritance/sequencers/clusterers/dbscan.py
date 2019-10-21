#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sequencers.clusterers import clusterer
from sklearn.metrics import silhouette_samples, silhouette_score

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
        return labels
    def silhouette(self):
        X = self.transform_feat()
        cluster_labels = self.get_clusters()
        silhouette_avg = silhouette_score(X, cluster_labels)
        # Compute the silhouette scores for each sample
        self.df['silhouette_value'] = silhouette_samples(X, cluster_labels)
        return silhouette_avg





