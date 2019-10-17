#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sequencers.vectorizers import vectorizer
import io
from nltk.corpus import stopwords
from gensim.sklearn_api import W2VTransformer
from gensim.models import KeyedVectors
from copy import deepcopy
import pdb

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

class Word_Embed(vectorizer.Vectorizer):
    embed = load_vectors('../../wiki-news-300d-1M-subword.vec')
    def transform(phrase : list, embedding):
        li_vecs = []
        for i in range(len(phrase)):
            if phrase[i] in embedding:
                li_vecs.append(list(deepcopy(embedding[phrase[i]])))
        if len(li_vecs) != 0:
            vecs = np.stack(li_vecs)

            return vecs            
        else:
            return None

    def filter_stopwords(phrase):
        important_words = []
        for word in phrase:
            if word not in stopwords.words('english'):
                important_words.append(word)
        return important_words

    def create_features(chobj, use_gap, context, word_embed_size):
        if context > 0:
            left_wordvecs = Word_Embed.transform(Word_Embed.filter_stopwords(list(chobj["left_token_str"][-context:])), Word_Embed.embed)
            if left_wordvecs is None:
                left_wordvecs = np.full(word_embed_size, 0)
            else:
                left_wordvecs = np.mean(left_wordvecs, axis=0)
            right_wordvecs = Word_Embed.transform(Word_Embed.filter_stopwords(list(chobj["right_token_str"][-context:])), Word_Embed.embed)  
            if right_wordvecs is None:
                right_wordvecs = np.full(word_embed_size, 0)
            else:
                right_wordvecs = np.mean(right_wordvecs, axis=0)
        if use_gap:
            ins_wordvecs = Word_Embed.transform(Word_Embed.filter_stopwords(list(chobj["ins_tokens_str"])), Word_Embed.embed)     
            del_wordvecs = Word_Embed.transform(Word_Embed.filter_stopwords(list(chobj["del_tokens_str"])), Word_Embed.embed)
            if ins_wordvecs is None:
                ins_wordvecs = np.full(word_embed_size, 0)
            else:
                ins_wordvecs = np.mean(ins_wordvecs, axis=0)
            if del_wordvecs is None:
                del_wordvecs = np.full(word_embed_size, 0)
            else:
                del_wordvecs = np.mean(del_wordvecs, axis=0)

        li = []
        for a in ["left_wordvecs", "right_wordvecs", "ins_wordvecs", "del_wordvecs"]:
            if a in vars():
                li.append(vars()[a])

        try:
            feat = np.nan_to_num(np.concatenate(li))
        except ValueError:
            pdb.set_trace()
        return feat
            
            
    def get_features(self):
        self.df['features'] = self.df.apply(lambda x: Word_Embed.create_features(x, use_gap = self.params['use_gap'], context = self.params['context'], word_embed_size = self.params['word_embed_size']), axis=1)

   





