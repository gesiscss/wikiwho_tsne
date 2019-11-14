#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sequencers.clusterers import clusterer
from sklearn.metrics import silhouette_samples, silhouette_score

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
        for i, row in pd.DataFrame(self.df['clusters'].value_counts()).iterrows():
            if row.clusters<self.params['min_samples']:
                self.df[self.df['clusters']==i, 'clusters'] = -1
        return clusters
    def silhouette(self):
        X = self.transform_feat()
        cluster_labels = self.get_clusters()
        silhouette_avg = silhouette_score(X, cluster_labels)
        # Compute the silhouette scores for each sample
        self.df['silhouette_value'] = silhouette_samples(X, cluster_labels)
        return silhouette_avg
    
    def evaluate_k(self):
        Sum_of_squared_distances = []
        K = range(4,10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(np.array(self.transform_feat()))
            Sum_of_squared_distances.append(km.inertia_)
        return(Sum_of_squared_distances)





