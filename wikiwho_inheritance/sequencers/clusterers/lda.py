#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sequencers.clusterers import clusterer
from sklearn.metrics import silhouette_samples, silhouette_score
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim 

stop = set(stopwords.words('english'))

class LDA(clusterer.Clusterer):
    def cleaning_tokens(tokenized_chobs):
        all_tokens = sum(tokenized_chobs, [])
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        cleaned_tokens = [[word for word in text if word not in tokens_once]for text in tokenized_chobs]
        return cleaned_tokens
    
    def get_corpus(self):
        df_cut = self.df[['left_token_str_clean', 'ins_tokens_str_clean', 'right_token_str_clean']]
        tokenized_chobs = []
        for i, row in df_cut.iterrows():
            one_sent = row[0] + ' ' + row[1] + ' ' + row[2]
            tokenized_chobs.append([i for i in one_sent.split() if i not in stop])
        self.tokenized_chobs = LDA.cleaning_tokens(tokenized_chobs)
        self.dictionary = corpora.Dictionary(self.tokenized_chobs)
        self.corpus = [self.dictionary.doc2bow(chobj) for chobj in self.tokenized_chobs]
        
    def choose_lda_model(self, limit, start, step):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, passes=10, chunksize=100)
            model_list.append(model)
            coherence_model_lda = CoherenceModel(model=model, texts=self.tokenized_chobs, dictionary=self.dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            coherence_values.append(coherence_lda)
        return model_list, coherence_values
    
    def get_clusters(self):
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,id2word = self.dictionary,num_topics = self.params['num_topics'], chunksize=100, passes=10)
        doc2bow_func = lambda x: self.dictionary.doc2bow(x) 
        doc_term_matrix = list(map(doc2bow_func, self.tokenized_chobs))
        get_topics = lambda x: self.lda_model.get_document_topics(x, minimum_probability=0)
        result = list(map(get_topics, doc_term_matrix))
        topic_to_chobj = []
        for chobj in result:
            topic_to_chobj.append(max(chobj,key=lambda item:item[1]))
        self.df['clusters'] = [int(i[0]) for i in topic_to_chobj]
        for i, row in pd.DataFrame(self.df['clusters'].value_counts()).iterrows():
            if row.clusters < self.params['min_samples']:
                self.df.loc[self.df['clusters']==i, 'clusters'] = -1
    
    def silhouette(self):
        X = self.transform_feat()
        cluster_labels = self.get_clusters()
        silhouette_avg = silhouette_score(X, cluster_labels)
        # Compute the silhouette scores for each sample
        self.df['silhouette_value'] = silhouette_samples(X, cluster_labels)
        return silhouette_avg





