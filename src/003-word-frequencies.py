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
import re
import unicodedata
from typing import List, Tuple, Any

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
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.blank("en")

def tokenize(text):
    return [t.text.lower() for t in nlp(text) if t.is_alpha]

my_tokens = (
    messages["text"][messages.from_me==1]
    .dropna()
    .apply(tokenize)
    .explode()
    .rename("token")
    .reset_index(drop=True)
)

their_tokens = (
    messages["text"][messages.from_me==0]
    .dropna()
    .apply(tokenize)
    .explode()
    .rename("token")
    .reset_index(drop=True)
)

#%%
my_tokens = my_tokens.loc[~my_tokens.isin(STOP_WORDS)].reset_index(drop=True)
their_tokens = their_tokens.loc[~their_tokens.isin(STOP_WORDS)].reset_index(drop=True)

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
            'how|why|where|when']
for patt in patterns:
    print_word_count(patt)

# %%
