#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from wikiwho_chobj import Chobjer
from os.path import join
from sequencers import sequencer
import sys
sys.path.insert(1, '../../wikiwho_tsne/utils/')
from merge import combine
import re

class Chobs(sequencer.Sequencer):
    def get_df(self, article_id, pickle_path, context, gap_length):
        co = Chobjer(article=article_id, pickles_path=pickle_path, lang='en', context=context)
        df = pd.DataFrame(co.iter_chobjs(), columns = next(co.iter_chobjs()).keys())
        self.df=df[(df['ins_tokens_str'].str.len() + df['del_tokens_str'].str.len()) <= gap_length]
    
    def merge(self, annotation_path):
        annotation = pd.read_csv(annotation_path)
        merged = self.df.apply(lambda x: combine(x, annotation), axis=1)
        # captures if we also want to use changeobjects that do not have tokens that are ground-truth labelled
        OUTER_JOIN = True
        merged = merged.dropna(how="all")
        if not OUTER_JOIN:
            merged = merged[(merged["birth_place"].isna() & merged["Bulk"].isna() & merged["nationality"].isna() & merged["Link"].isna())== False]
        self.df = merged.copy()
    
    def make_list(tokens):
        new_tokens = []
        for item in tokens:
            new_tokens.append(' '.join(word for word in item if re.match("^[a-zA-Z0-9_]*$", word)))

        return new_tokens

    def replace_tokens(self):
        all_tokens = []
        self.df = self.df.reset_index()
        self.df = self.df.reindex(columns=['index','Bulk','Link','action','birth_place','del_end_pos','del_start_pos','del_tokens','del_tokens_str','editor','from_rev','from_timestamp','ins_end_pos','ins_start_pos','ins_tokens','ins_tokens_str','left_neigh','left_token', 'left_token_str','nationality','page_id','right_neigh','right_token','right_token_str','text','to_rev','to_timestamp','token','cluster','left_token_str_clean', 'right_token_str_clean','ins_tokens_str_clean','del_tokens_str_clean'])
        for i, row in self.df.iterrows():
            all_tokens.append(Chobs.make_list([row['left_token_str'],row['right_token_str'],row['ins_tokens_str'],row['del_tokens_str']]))
            self.df.loc[i, ['left_token_str_clean','right_token_str_clean','ins_tokens_str_clean','del_tokens_str_clean']] = all_tokens[i]





