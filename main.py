import sqlite3
import pandas as pd
import os
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

#Step1: User Question

question = "Which Player has the highest strike rate"
db_path = "/Users/madhu/OneDrive/Desktop/nlsql/csvchatbot.db"
table_name = "cricket"

#Step2: Read column names and 5 sample rows from the SQLITE Database

conn=sqlite3.connect('csvchatbot.db')
sample_df=pd.read_sql_query(f"Select * from {table_name} limit 10",conn)
column_names=sample_df.columns.to_list()

#print(column_names)

#Step3: LLM call 1: Schema Generation

llm=ChatOpenAI(temperature=0,model="gpt-4")

prompt_schema=PromptTemplate(
    input_variables=["column","sample_rows"],
    template=(
        """You are a Skilled Data Analyst who is an expert in documenting database information.
        Below are the column names and some sample rows from my sqlite table.

        columns: {column}
        sample data: {sample_rows}

        For each column, write a short description based on your knowkledge and context in the table:
        Strictly use below format

        Table Name: cricket

        1. Column name: Type. Clear Description

        Only retrun the numbered list. Do not give any explanation, extra spaces or formatting

        """  
    )
)

chain=prompt_schema | llm

response_schema=chain.invoke(
    {
        "column": column_names,
        "sample_rows":sample_df
    }
)

schema_text=response_schema.content

#Step4: LLM Call 2: Generate SQL Query

prompt_sql=PromptTemplate(
    input_variables=["table","schema","question"],
    template=(
        """You are a Professional Data Analyst who can convert Natural Language to SQL Queries
        You will be give:
        1. A Table Name
        2. Schema: All the columns in the sql table with type and description
        3. A User Question (Natural Language)

        Your Job is to write a valid sql query to answer the question:

        Table Name: {table}

        Schema:{schema}

        User Question:{question}

        Strictly Respond with only single valid sql query
        Do not include any extra text, markdown, explanation or formatting

        """
    )
)

formatted_sql_prompt=prompt_sql.format(
    table=table_name,
    schema=schema_text,
    question=question
)

response_sql=llm.invoke(formatted_sql_prompt)
sql_query=response_sql.content
print(sql_query)

#Step5: Execute the SQL Query

result=pd.read_sql_query(sql_query,conn)
print(f"Below is the Answer to Your Question: {result}")

#Step6: Response

prompt_answer=PromptTemplate(
    input_variables=["question","answer"],
    template=(
        """You are a Professional Cricket Analyst who can answer user question and the best tone and context
        Below is the user question and answer. Kindly frame a Very professional answer

        User Question:{question}
        Answer:{answer}

        Answer in 100 words or less. Don't just tell the number. Tell interesting story along the line as well

        """
    )
)

chain2=prompt_answer | llm

answer_schema=chain2.invoke(
    {
        "question":question,
        "answer":result
    }
)

print(answer_schema.content)