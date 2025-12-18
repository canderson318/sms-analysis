#!/Users/canderson/miniconda3/envs/generic-python/bin/python

import sqlite3
import os
from pathlib import Path as pth
import subprocess as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
from typing import List, Tuple, Any


os.chdir(pth(pth.home() / 'sms-analysis'))

# copy chat db so no funky stuff
sp.run(["scp","/Users/canderson/Library/Messages/chat.db", "raw-data/copy-of-chat.db"])


# Connect to SQLite Database and create a cursor
db_path = pth('raw-data/copy-of-chat.db')
sqliteConnection = sqlite3.connect(db_path)



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

query('SELECT sqlite_version();')

# Tables
query("""
SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;
""")


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

sqliteConnection.close()

# make df
all_messages = pd.DataFrame(query_response, columns = ['date_time', 'text', 'from_me', 'sender'])

all_messages.shape

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

messages = all_messages

messages['text'] = (
    messages['text']
    .str.replace('\uFFFC', '', regex = False)
)

# filter out empty cells
messages = messages[ messages.text.str.strip().ne("")]

# filter out None rows
messages = messages[messages.text.notna()]

# filter for katrina messages
katrina_messages = messages[messages.sender.str.contains('30761',na = False)]

# make date class
katrina_messages = katrina_messages.copy()
katrina_messages['date_time'] = pd.to_datetime(katrina_messages["date_time"])

katrina_messages = katrina_messages.copy()
katrina_messages=katrina_messages.sort_values("date_time").reset_index(drop=True)

print(katrina_messages.shape)
katrina_messages.tail(10)

katrina_messages.to_csv('processed-data/katrina-messages.csv',index = False)
