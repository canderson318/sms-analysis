#!/Users/canderson/miniconda3/envs/generic-python/bin/python
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
import sqlite3
import os
from pathlib import Path as pth
import subprocess as sp
import warnings
from typing import List, Tuple, Any, Dict
from itertools import chain
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn  as sns
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re

# %% [markdown]
# ## Change working directory

# %%
os.chdir(pth(pth.home() / 'dev/sms-analysis'))


# %% [markdown]
# ## Load Data

# %%
messages = pd.read_csv('processed-data/X-messages.csv')
messages.head()

# %%

# tokenize messages by sender

nlp = spacy.blank("en")

def tokenize(text):
    return [t.text.lower() for t in nlp(text) if t.is_alpha]

messages = (
    messages
        .dropna(subset=["text"])
        .assign(tokens=lambda df: df["text"].apply(tokenize))
)

# # Filter out stop words
# stop_words = set(STOP_WORDS)
# messages["tokens"] = messages["tokens"].apply(
#     lambda toks: [t for t in toks if t not in stop_words]
# )


# filter for non_empty tokens
messages = messages[messages.tokens.apply(len) != 0]


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

# %%

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

# %% [markdown]
# ## use LWIC dictionary

# %%

# Download dictionary
if not pth.exists(pth("raw-data/LIWC2007.dic")):
    sp.run([
        'curl',
        '-L',
        'https://raw.githubusercontent.com/Harsh-Panchal-1403/LIWC_PROJECT/master/LIWC2007_English100131.dic',
        '-o',
        'raw-data/LIWC2007.dic'
    ])

# #%% [markdown]
# # ### Read in dictionary 

# #%%
# def read_dic(path: str) -> Dict[re.Pattern, list[str]]:
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     # find lines with field names
#     key_marker = [s.strip() == '%' for s in lines]
#     key_marker_indx = np.where(key_marker)[0]

#     # save keys as dict of key: refnum
#     full_keys = lines[key_marker_indx[0] + 1 : key_marker_indx[1] ]
#     values = lines[key_marker_indx[1]+1:]

#     ref_num_dict = {}
#     pat = re.compile(r'^(\d+)\t(.+)')
#     for s in full_keys:
#         m = pat.search(s)
#         if m:
#             ref_num_dict[int(m.group(1))] = m.group(2).strip()

#     # save strings as dict string: refnum
#     string_dict = {}
#     # val = "sdlfkj   130294  13294   130459"
#     for val in values:
#         refs = [int(x) for x in re.findall(r'\d+', val)]
#         string = val.split('\t', 1)[0].strip()
#         string = re.sub(pattern=r'\s+',repl= '', string=string)
#         string_dict.setdefault(string, []).extend(refs)

#     # make one unified dict with key: strings
#     full_dict = {}
#     for string, ref_nums in string_dict.items():
#         cats = []
#         for ref_num in ref_nums:
#             if ref_num in ref_num_dict:
#                 cats.append(ref_num_dict[ref_num])
#         string = re.compile("^" + re.escape(string).replace(r"\*", ".*") + "$")
#         full_dict[string] = cats

#     return full_dict, [re.sub(r'\d+\t|\n', '', x) for x in full_keys]

# #%% 

# # Read dic
# dic, categories = read_dic("raw-data/LIWC2007.dic")


# #%% [markdown]
# # ### Map Texts to Categories

# #%%

# # return categories for word
# def get_categories(tokens) :
#     res = []
#     for tok in tokens:
#         res.append([
#             cat
#             for patt, cats in dic.items()
#             if patt.match(tok)
#             for cat in cats
#         ])

#     # category x message series
#     res = pd.Series(res).explode().value_counts()

#     # add missing categories
#     if not res.shape[0] == len(categories) :
#         diff = set(categories).difference(set(res.index))
#         add = pd.Series(0,  index = list(diff))
#         res = pd.concat([res, add], axis = 0)

#     res = res.sort_index()

#     return res

# #%%
# # get_categories( messages.loc[1,'tokens'])

# #%% 

# # Get categories for each message as vector of category counts
# category_counts = pd.DataFrame([get_categories(x) for x in messages.tokens])
# category_counts.shape
# messages.shape

# messages.reset_index(drop = True, inplace = True)
# category_counts.reset_index(drop = True, inplace = True)

# messages.index
# messages.columns
# category_counts.index
# category_counts.columns

# x = pd.concat([messages, category_counts], axis = 1)

# if x.shape[0] == messages.shape[0]:
#     messages = x
# else:
#     warnings.warn(
#         "Row count mismatch: x does not match messages; assignment skipped.",
#         UserWarning
#     )

# #%% [markdown]
# # ## Compare category counts

# #%%
# my_cat_counts = messages.loc[messages['from_me']==1,'achieve':].sum(axis = 0)
# their_cat_counts = messages.loc[messages['from_me']==0,'achieve':].sum(axis = 0)

# # normalize counts
# # total_cat_counts = sum(my_cat_counts, their_cat_counts)
# my_cat_freq = my_cat_counts/my_cat_counts.sum(0)
# their_cat_freq = their_cat_counts/their_cat_counts.sum(0)

# cat_freq_summary = pd.concat([my_cat_freq, their_cat_freq], axis = 1).rename(columns={0: "me", 1: "them"})

# cat_freq_summary['me_over_them'] = cat_freq_summary['me'].div(cat_freq_summary['them'], axis = 0).round(4)
# cat_freq_summary['log_me_over_them'] = np.log2(cat_freq_summary['me_over_them'])

# # add patterns to df
# cat_to_patterns = defaultdict(list)
# for patt, cats in dic.items():
#     for cat in cats:
#         cat_to_patterns[cat].append(patt.pattern)

# cat_freq_summary["patterns"] = (
#     cat_freq_summary.index.map(lambda c: cat_to_patterns.get(c, []))
# )


# # print summary
# (
#     cat_freq_summary
#     .sort_values(by="log_me_over_them",key=lambda s: s.abs(), ascending = False)
#     .to_csv('results/summary.txt', sep='\t')
# )

# %% [markdown]
# ## Use ConvoKit tool
# Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2020. "ConvoKit: A Toolkit for the Analysis of Conversations". Proceedings of SIGDIAL.
#
#   1. Download the toolkit: pip3 install convokit
#   2. Download Spacy's English model: python3 -m spacy download en
#   3. Download NLTK's 'punkt' model: import nltk; nltk.download('punkt') (in Python interpreter)


# %%
from convokit import Corpus, Utterance, Speaker, TextParser, Coordination,PolitenessStrategies
import nltk; nltk.download('punkt')
# spacy.load('en_core_web_sm')


# %% [markdown]
# #### Construct corpus

# %%

messages.date_time = pd.to_datetime(messages.date_time)
df = messages.copy()


# %%


def speaker_id(row):
    return "me" if row["from_me"] == 1 else "them"
# make speakers
speakers = {
    "me": Speaker(id="me"),
    "them": Speaker(id="them")
}
utterances = []
conversation_id = "sms_conversation_1"
prev_utt_id = None

for i, row in df.iterrows():
    utt_id = f"utt_{i}"

    utt = Utterance(
        id=utt_id,
        speaker=speakers[speaker_id(row)],
        text=row["text"],
        reply_to=prev_utt_id,
        conversation_id=conversation_id,
        meta={
            "timestamp": row["date_time"].isoformat(),
            "from_me": row["from_me"],
            "sender": row["sender"]
            # "tokens": row["tokens"]
        }
    )

    utterances.append(utt)
    prev_utt_id = utt_id

corpus = Corpus(
    utterances=utterances
)

# %% [markdown]
# ### Tokenize

# %%

parser = TextParser('en_core_web_sm')
corpus = parser.transform(corpus)

# %% [markdown]
# ### Analyze Corpus

# %%
corpus.print_summary_stats()

[t["tok"] for sent in corpus.get_utterance("utt_0").meta["en_core_web_sm"] for t in sent["toks"]]

# %%
print("Speakers in corpus:", list(corpus.iter_speakers()))  
print(corpus.speaking_pairs(speaker_ids_only=True)  )

# %% [markdown]
# ### Speaker Coordination
# %%
# speaker coordination
coord = Coordination(target_thresh=3, speaker_thresh=5, utterances_thresh=5)  

coord.fit(corpus)  

coord.transform(corpus)

me_coord_scores = corpus.get_speaker("me").meta["coord"]['them']
them_coord_scores = corpus.get_speaker("them").meta["coord"]['me']
feature_freqs = pd.concat([pd.Series(me_coord_scores).rename("me_to_them"), pd.Series(them_coord_scores).rename("them_to_me")], axis = 1)

# %%
feature_freqs['diff'] = feature_freqs['me_to_them']-feature_freqs['them_to_me']

feature_freqs.sort_values('diff', key = lambda x: abs(x), ascending = False)

# %% [markdown]
# |Feature|me_to_them|them_to_me|Interpretation|
# |---|---|---|---|
# |auxverb|0.31|0.05|You strongly accommodate their auxiliary verbs; they barely adapt to yours|
# |pronoun|−0.02|0.21|You slightly diverge; they strongly accommodate|
# |article|0.00|0.00|No coordination either way|

# %% [markdown]
# ### Politeness

# %%
# Initialize politeness analyzer (requires parsed text)  
ps = PolitenessStrategies(parse_attribute_name="en_core_web_sm")  
corpus= ps.fit_transform(corpus)  


# %%
  
# Get politeness scores for each speaker's utterances  
me_utterances = list(corpus.iter_utterances(lambda x: x.speaker.id == "me"))  
them_utterances = list(corpus.iter_utterances(lambda x: x.speaker.id == "them"))  
  
# Calculate average politeness strategies per speaker  
me_strategies = pd.DataFrame([utt.meta["politeness_strategies"] for utt in me_utterances])
them_strategies = pd.DataFrame([utt.meta["politeness_strategies"] for utt in them_utterances])

out = pd.concat([me_strategies.sum(0), them_strategies.sum(0)],axis = 1)
out.columns = ['me', 'them']
out.index = [re.sub(pattern = r'feature_politeness_|==',repl = '', string= x) for x in out.index]

# normalize by total utterances spoken
utt_counts = Counter(
    utt.speaker.id
    for utt in corpus.iter_utterances()
)

out['me_self_normalized'] = out.me/utt_counts['me']
out['them_self_normalized'] = out.them/utt_counts['them']

out['diff'] = out.me_self_normalized-out.them_self_normalized

out.sort_values('diff', key = lambda x: abs(x), ascending = False)

# %% [markdown]
# Politeness features
#
# | Feature name              | What it measures | Typical interpretation in discourse analysis |
# |---------------------------|------------------|----------------------------------------------|
# | Please                    | Presence of the word “please” anywhere in the utterance | Politeness marker; mitigates imposition |
# | Please_start              | Utterance begins with “please” | High politeness or deference at turn entry |
# | HASHEDGE                  | Any hedge expression (aggregate indicator) | Linguistic uncertainty, softening, or non-commitment |
# | Indirect_(btw)            | Indirect discourse marker such as “by the way” | Topic shift or low-imposition insertion |
# | Hedges                    | Count or presence of hedging terms (e.g., “maybe”, “kind of”) | Reduced certainty; politeness or epistemic caution |
# | Factuality                | Use of factual/assertive language | Speaker presents information as objective or certain |
# | Deference                 | Deferential language (e.g., “if you don’t mind”) | Power asymmetry or respect toward interlocutor |
# | Gratitude                 | Expressions of thanks | Positive social signaling; rapport maintenance |
# | Apologizing               | Apologies or regret expressions | Face-saving, repair, or politeness strategy |
# | 1st_person_pl.            | First-person plural pronouns (“we”, “us”) | Inclusivity, shared responsibility, alignment |
# | 1st_person                | First-person singular pronouns (“I”, “me”) | Self-focus, agency, or ownership of stance |
# | 1st_person_start          | Utterance begins with a first-person pronoun | Self-initiated stance or framing |
# | 2nd_person                | Second-person pronouns (“you”) | Addressing, directing, or engaging the interlocutor |
# | 2nd_person_start          | Utterance begins with a second-person pronoun | Direct engagement; can signal instruction or confrontation |
# | Indirect_(greeting)       | Indirect greeting (e.g., “hey”, “hope you’re well”) | Social lubrication before substantive content |
# | Direct_question           | Explicit interrogative form | Information-seeking or directive questioning |
# | Direct_start              | Utterance begins with a direct request or statement | Low mitigation; task-oriented or assertive style |
# | HASPOSITIVE               | Presence of positive-affect words | Positive sentiment or encouragement |
# | HASNEGATIVE               | Presence of negative-affect words | Criticism, frustration, or negative sentiment |
# | SUBJUNCTIVE               | Subjunctive or hypothetical constructions (“would”, “could”) | Politeness, mitigation, or counterfactual framing |
# | INDICATIVE                | Indicative (statement-of-fact) constructions | Assertion, certainty, or declarative stance |

# %%
[x for x  in dir(corpus) if not bool(re.search(r'^_', x)) ]
