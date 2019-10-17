#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import numpy as np
from pytorch_transformers import (DistilBertConfig,
                                     DistilBertModel, 
                                     DistilBertTokenizer)
from pytorch_transformers.modeling_utils import SequenceSummary
from sequencers.vectorizers import vectorizer

class Distilbert_LR(vectorizer.Vectorizer):
    def get_features(self):
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                            output_hidden_states=True,
                                            output_attentions=True)
        sequence_summary = SequenceSummary(model.config)
        new_vector = []
        columns = ["left_token_str_clean", "right_token_str_clean"]
        for i, row in self.df[columns].iterrows():
            vector=[]
            for token in row:
                if token.isspace() or token == '':
                    vector.extend(np.full((768), 0))
                else:
                    sen = torch.tensor([tokenizer.encode(token)])
                    vector.extend(sequence_summary(model(sen)[0])[0].detach().numpy())
            new_vector.append(vector)
        self.add_features(new_vector)





