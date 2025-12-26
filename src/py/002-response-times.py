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
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn  as sns
import re
import unicodedata
from typing import List, Tuple, Any

# %% [markdown]
# ## Change working directory

# %%
os.chdir(pth(pth.home() / 'dev/sms-analysis'))


# %% [markdown]
# ## Load Data

# %%
messages = pd.read_csv('processed-data/X-messages.csv')
messages.head()

# %% [markdown]
# ## flag where they respond to me and where i respond to them

# %%
# flag their response 
messages = (
    messages
    .assign( their_response= lambda df: 
            (df["from_me"] == 0) & (df.from_me.shift(1)  == 1) # their response is when last is 1 and current 0 (shift(1) = lag(1))
        )
    .assign( my_response= lambda df: 
        (df["from_me"] == 1) & (df.from_me.shift(1)  == 0) # current me, previous them
    )
)

messages.head()

# %% [markdown]
# ## Calculate response time

# %%
messages['date_time'] = pd.to_datetime(messages.date_time)

messages = (
    messages 
    .assign(time_diff = lambda df: df.date_time.diff() )
    .assign(time_diff_sec = lambda df: (df.time_diff.dt.total_seconds().round().astype('Int64') ))
)

# %% [markdown]
# ## Analyze response times

# %%
print("Their median response time =",  round(np.median(messages['time_diff_sec'][messages['their_response']])/60), "minutes")
print("My median response time =",  round(np.median(messages['time_diff_sec'][messages['my_response']])/60), "minutes")

print("Their mean response time =",  round(np.mean(messages['time_diff_sec'][messages['their_response']])/60), "minutes")
print("My mean response time =",  round(np.mean(messages['time_diff_sec'][messages['my_response']])/60), "minutes")

#       min->hour->day->days
upper = 60*60*24*1
their_time = (
    messages.loc[messages["their_response"], "time_diff_sec"]
    .clip(lower=1, upper = upper)
    .dropna()
)

my_time = (
    messages.loc[messages["my_response"], "time_diff_sec"]
    .clip(lower=1, upper = upper)
    .dropna()
)

# plt.boxplot([np.log(their_time/60),np.log(my_time/60)], tick_labels=["their_response", "my_response"])
# plt.ylabel("ln(Time difference in minutes)")
# plt.show()

plt.figure(figsize=(10, 10))
sns.kdeplot(their_time/60, label="their_response", fill=True, clip = (0,None))
sns.kdeplot(my_time/60, label="my_response", fill=True, clip = (0,None))

plt.xlabel("Time difference in minutes")
plt.ylabel("Density")
plt.legend()
plt.show()

# %% [markdown]
# Hump around 500 minutes probably because they text late and i don't respond till i wake up, around 8 hours later.


# %%
