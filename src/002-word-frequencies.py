#!/Users/canderson/miniconda3/envs/generic-python/bin/python
import sqlite3
import os
from pathlib import Path as pth
import subprocess as sp
import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Tuple, Any

os.chdir(pth(pth.home() / 'dev/sms-analysis'))

# with open('processed-data/messages.txt','r') as f:
# 	msgs = f.readlines()

# print(msgs[:10])

messages = pd.read_csv('processed-data/X-messages.csv')

messages.head()

my_messages = messages.text[messages.from_me==1]
their_messages = messages.text[messages.from_me==0]



