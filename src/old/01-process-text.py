import os
from pathlib import Path as pth
import subprocess as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# set wd
os.chdir(pth("/Users/canderson/sms-analysis"))


# \\
# convert html of texts to text file
if not pth("processed-data/katrina-texts.txt").exists():
    cmd = [
        "pandoc",
        "raw-data/katrina-texts.html",
        "-t", "plain",
        "--wrap=none",
        "-o", "processed-data/katrina-texts.txt",
    ]

    sp.run(cmd, check=True)


#\\ 
# convert text file to list

path = pth("processed-data/katrina-texts.txt")

raw_lines = []

with path.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        raw_lines.append(line)

raw_lines = pd.DataFrame( raw_lines, columns = ["text"])

rm_text = ['\x04\x0bstream', '[]', '', 'EaseUS MobiMover', '', '[] +13608307613 Messages']

lines = raw_lines[~raw_lines.text.isin(rm_text)]

patt = re.compile('^202*-*')

lines = lines.copy()
lines["is_date"] = lines["text"].str.contains(patt, na=False)

# lines.loc[:, 'is_date_diff'] = 
x = lines['is_date'].values.astype(int)[:-1]
y = lines['is_date'].values.astype(int)[1:]

lines.loc[:, 'x'] = np.concatenate([x-y, [np.nan]])

lines[lines['x']==0]

