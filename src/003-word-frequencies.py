#!/Users/canderson/miniconda3/envs/generic-python/bin/python
#%%
import sqlite3
import os
from pathlib import Path as pth
import subprocess as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn  as sns
from typing import List, Tuple, Any, Dict
from itertools import chain
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
import unicodedata


#%% [markdown]
# ## Change working directory

#%%
os.chdir(pth(pth.home() / 'dev/sms-analysis'))


#%% [markdown]
# ## Load Data

#%% 
messages = pd.read_csv('processed-data/X-messages.csv')
messages.head()

#%%

# tokenize messages by sender

nlp = spacy.blank("en")

def tokenize(text):
    return [t.text.lower() for t in nlp(text) if t.is_alpha]

messages = (
    messages
        .dropna(subset=["text"])
        .assign(tokens=lambda df: df["text"].apply(tokenize))
)

my_tokens = (
    messages["tokens"][messages.from_me==1]
    .explode()
    .rename("token")
    .reset_index(drop=True)
)

their_tokens = (
    messages["tokens"][messages.from_me==0]
    .explode()
    .rename("token")
    .reset_index(drop=True)
)

#%%
# # Filter out stop words
# my_tokens = my_tokens.loc[~my_tokens.isin(STOP_WORDS)].reset_index(drop=True)
# their_tokens = their_tokens.loc[~their_tokens.isin(STOP_WORDS)].reset_index(drop=True)

# %%
my_tokens.value_counts().sort_values( ascending = False)
their_tokens.value_counts().sort_values( ascending = False)

# %%
# patt = re.compile(r'love')
their_token_total = their_tokens.shape[0]
my_token_total = my_tokens.shape[0]

def print_word_count(pattern: str):
    print(pattern)
    
    their_count = (their_tokens.str.contains(pattern, case=False)).sum()
    my_count = (my_tokens.str.contains(pattern, case=False)).sum()

    my_prop = round(100*my_count/my_token_total, 2)
    their_prop = round(100*their_count/their_token_total, 2)

    print("  Them: ", their_count, f'({their_prop}%)')
    print("  Me: ", my_count, f'({my_prop}%)')

# %%
patterns = ['love', 'like', 'happy', 'sad', 'miss', 'wish',  
            'hat', 'dog', 'mad', 'time', 'plan', 'you', 'me', 
            'how|why|where|when', 'sex']
for patt in patterns:
    print_word_count(patt)

# %% [markdown]
# ## use LWIC dictionary

#%% 

# Download dictionary
if not pth.exists(pth("raw-data/LIWC2007.dic")):
    sp.run([
        'curl',
        '-L',
        'https://raw.githubusercontent.com/Harsh-Panchal-1403/LIWC_PROJECT/master/LIWC2007_English100131.dic',
        '-o',
        'raw-data/LIWC2007.dic'
    ])

#%% [markdown]
# ### Read in dictionary 

#%%
def read_dic(path: str) -> Dict[re.Pattern, list[str]]:
    with open(path, 'r') as f:
        lines = f.readlines()
    # find lines with field names
    key_marker = [s.strip() == '%' for s in lines]
    key_marker_indx = np.where(key_marker)[0]

    # save keys as dict of key: refnum
    full_keys = lines[key_marker_indx[0] + 1 : key_marker_indx[1] ]
    values = lines[key_marker_indx[1]+1:]

    ref_num_dict = {}
    pat = re.compile(r'^(\d+)\t(.+)')
    for s in full_keys:
        m = pat.search(s)
        if m:
            ref_num_dict[int(m.group(1))] = m.group(2).strip()

    # save strings as dict string: refnum
    string_dict = {}
    # val = "sdlfkj   130294  13294   130459"
    for val in values:
        refs = [int(x) for x in re.findall(r'\d+', val)]
        string = val.split('\t', 1)[0].strip()
        string = re.sub(pattern=r'\s+',repl= '', string=string)
        string_dict.setdefault(string, []).extend(refs)

    # make one unified dict with key: strings
    full_dict = {}
    for string, ref_nums in string_dict.items():
        cats = []
        for ref_num in ref_nums:
            if ref_num in ref_num_dict:
                cats.append(ref_num_dict[ref_num])
        string = re.compile("^" + re.escape(string).replace(r"\*", ".*") + "$")
        full_dict[string] = cats

    return full_dict, [re.sub(r'\d+\t|\n', '', x) for x in full_keys]

#%% 

# Read dic
dic, categories = read_dic("raw-data/LIWC2007.dic")


#%% [markdown]
# ### Map Texts to Categories

#%%

# return categories for word
def get_categories(tokens) :
    res = []
    for tok in tokens:
        res.append([
            cat
            for patt, cats in dic.items()
            if patt.match(tok)
            for cat in cats
        ])

    # category x message series
    res = pd.Series(res).explode().value_counts()

    # add missing categories
    if not res.shape[0] == len(categories) :
        diff = set(categories).difference(set(res.index))
        add = pd.Series(0,  index = list(diff))
        res = pd.concat([res, add], axis = 0)

    res = res.sort_index()

    return res

#%%
# get_categories( messages.loc[1,'tokens'])

#%% 

# Get categories for each message as vector of category counts
category_counts = pd.DataFrame([get_categories(x) for x in messages.tokens])
messages = pd.concat([messages, category_counts], axis = 1)

#%% [markdown]
# ## Compare category counts

#%%
my_cat_counts = messages.loc[messages['from_me']==1,'achieve':].sum(axis = 0)
their_cat_counts = messages.loc[messages['from_me']==0,'achieve':].sum(axis = 0)
total_cat_counts = sum(my_cat_counts, their_cat_counts)

my_cat_freq = my_cat_counts.div(sum(my_cat_counts)).round(4)
their_cat_freq = their_cat_counts.div(sum(their_cat_counts)).round(4)

cat_freq_summary = pd.concat([my_cat_freq, their_cat_freq], axis = 1).rename(columns={0: "me", 1: "them"})

cat_freq_summary['diff'] = cat_freq_summary['me'].sub(cat_freq_summary['them'], axis = 0).round(4)
cat_freq_summary['norm_diff'] = cat_freq_summary['norm_diff']/(sum(cat_freq_summary['me'], cat_freq_summary['them']))

from collections import defaultdict

cat_to_patterns = defaultdict(list)
for patt, cats in dic.items():
    for cat in cats:
        cat_to_patterns[cat].append(patt.pattern)

cat_freq_summary["patterns"] = (
    cat_freq_summary.index.map(lambda c: cat_to_patterns.get(c, []))
)


(
    cat_freq_summary
    .sort_values(by="diff",key=lambda s: s.abs(), ascending = False)
    .to_csv('results/summary.txt', sep='\t')
)

# %% [markdown]
# ## Use ConvoKit tool
# Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2020. "ConvoKit: A Toolkit for the Analysis of Conversations". Proceedings of SIGDIAL.
#
#        1. Download the toolkit: pip3 install convokit
#        2. Download Spacy's English model: python3 -m spacy download en
#        3. Download NLTK's 'punkt' model: import nltk; nltk.download('punkt') (in Python interpreter)


# %% 

