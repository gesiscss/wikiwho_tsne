#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bert_serving.client import BertClient
import numpy as np
from sequencers.vectorizers import vectorizer

class Bert_LRG(vectorizer.Vectorizer):
    def get_features(self):
        bc = BertClient() #bertclient needs to be running on a server
        new_vector = []
        columns = ["left_token_str_clean", "right_token_str_clean", "ins_tokens_str_clean"]
        for i, row in self.df[columns].iterrows():
            vector=[]
            for token in row:
                if token.isspace() or token == '':
                    vector.extend(np.full((1, 768), 0))
                else:
                    vector.extend(bc.encode([token]))

            new_vector.append(np.concatenate((vector[0], vector[1], vector[2])))

        self.df['features'] = new_vector





