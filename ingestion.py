#1. Read a CSV File
#2. Convert to a DataBase

import pandas as pd
import sqlite3
import os

#Step1: Convert CSV to Pandas

df=pd.read_csv('/Users/madhu/OneDrive/Desktop/cricket.csv')

#Step2: Setup a Connection to SQLITE

db = 'csvchatbot.db'
conn=sqlite3.connect(db)

#Step3: Store the DataFrame to SQLITE Database

df.to_sql('cricket',conn,if_exists='replace',index=False)

conn.close()



