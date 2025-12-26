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
# %%
import sqlite3
import os
from pathlib import Path as pth
import subprocess as sp
import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Tuple, Any


# %%
os.chdir(pth(pth.home() / 'dev/sms-analysis'))

# %%
# copy chat db so no funky stuff
sp.run(["scp","/Users/canderson/Library/Messages/chat.db", "raw-data/copy-of-chat.db"])


# %%
# Connect to SQLite Database and create a cursor
db_path = pth('raw-data/copy-of-chat.db')
sqliteConnection = sqlite3.connect(db_path)


# %%

# %%
def query(sql: str) -> List[Tuple[Any, ...]]:
    """
    Return the result of a SQLite query using the existing connection.

    Parameters
    ----------
    sql : str
        SQL query formatted as a text block.
    """
    cursor = sqliteConnection.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    cursor.close()
    return res

# %%
query('SELECT sqlite_version();')

# %%
# Tables
query("""
SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;
""")


# %%
query_response = query("""
    SELECT
        datetime (message.date / 1000000000 + strftime ("%s", "2001-01-01"), "unixepoch", "localtime") AS message_date,
        message.text,
        message.is_from_me,
        chat.chat_identifier
    FROM
        chat
        JOIN chat_message_join ON chat. "ROWID" = chat_message_join.chat_id
        JOIN message ON chat_message_join.message_id = message. "ROWID"
    ORDER BY
        message_date ASC;
""")

# %%
sqliteConnection.close()

# %%
# make df
all_messages = pd.DataFrame(query_response, columns = ['date_time', 'text', 'from_me', 'sender'])

# %%
all_messages.shape

# %%
# filter for empty messages
s = all_messages.loc[8407, "text"]
for i, ch in enumerate(s):
    print(
        i,
        repr(ch),
        f"U+{ord(ch):04X}",
        unicodedata.name(ch, "UNKNOWN"),
        unicodedata.category(ch)
    )

# %%
messages = all_messages

# %%
messages['text'] = (
    messages['text']
    .str.replace('\uFFFC', '', regex = False)
)

# %%
# filter out empty cells
messages = messages[ messages.text.str.strip().ne("")]

# %%
# filter out None rows
messages = messages[messages.text.notna()]

# %%
# filter for katrina messages
X_messages = messages[messages.sender.str.contains('30761',na = False)]

# %%
# make date class
X_messages = X_messages.copy()
X_messages['date_time'] = pd.to_datetime(X_messages["date_time"])

# %%
X_messages = X_messages.copy()
X_messages=X_messages.sort_values("date_time").reset_index(drop=True)

# %%
print(X_messages.shape)
X_messages.tail(10)

# %%
X_messages.to_csv('processed-data/X-messages.csv',index = False)

# %%
with open("processed-data/messages.txt", "w", encoding="utf-8") as f:
    f.write("\n----------\n".join(
        X_messages["date_time"].astype(str) + ":: " + X_messages["from_me"].astype(str) + ":: " + X_messages["text"].astype(str)
        ))

# %%
# clean empty lines
with open("processed-data/messages.txt", encoding="utf-8") as f:
    lines = [l for l in f if l.strip()]

# %%
with open("processed-data/messages.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)
