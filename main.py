import sqlite3
import pandas as pd
import os
import json
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Step1: User Question
question = "Find the team with the most batsmen having higher strike rate."
db_path = "/Users/madhu/OneDrive/Desktop/nlsql/csvchatbot.db"

# Step2: Read and Load two CSV files into SQLite
csv_path_1 = "/Users/madhu/OneDrive/Desktop/player.csv"
csv_path_2 = "/Users/madhu/OneDrive/Desktop/teams.csv"

df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

conn = sqlite3.connect(db_path)
df1.to_sql("player", conn, if_exists="replace", index=False)
df2.to_sql("teams", conn, if_exists="replace", index=False)

# Step3: Read column names and 5 sample rows from both tables
sample_df1 = pd.read_sql_query("SELECT * FROM player LIMIT 5", conn)
sample_df2 = pd.read_sql_query("SELECT * FROM teams LIMIT 5", conn)

columns1 = sample_df1.columns.to_list()
columns2 = sample_df2.columns.to_list()

# Step4: LLM call 1: Schema Generation for both tables
llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt_schema = PromptTemplate(
    input_variables=["table", "column", "sample_rows"],
    template=(
        """You are a Skilled Data Analyst who is an expert in documenting database information.
        Below are the column names and some sample rows from my sqlite table.

        Table Name: {table}

        columns: {column}
        sample data: {sample_rows}

        For each column, write a short description based on your knowledge and context in the table:
        Strictly use below format

        Table Name: {table}

        1. Column name: Type. Clear Description

        Only return the numbered list. Do not give any explanation, extra spaces or formatting
        """
    )
)

chain = prompt_schema | llm

schema1 = chain.invoke({"table": "player", "column": columns1, "sample_rows": sample_df1}).content
schema2 = chain.invoke({"table": "teams", "column": columns2, "sample_rows": sample_df2}).content

combined_schema = schema1 + "\n" + schema2

# Step5: LLM Call 2: Generate SQL Query using JOIN
prompt_sql = PromptTemplate(
    input_variables=["schema", "question"],
    template=(
        """You are a Professional Data Analyst who can convert Natural Language to SQL Queries.
        You will be given:
        1. Schema: Columns from multiple SQL tables with their type and description.
        2. A User Question (Natural Language).

        Your job is to write a valid SQL query to answer the question using joins if required.

        Schema:
        {schema}

        User Question: {question}

        Strictly respond with only a single valid SQL query.
        Do not include any extra text, markdown, explanation, or formatting.
        """
    )
)

formatted_sql_prompt = prompt_sql.format(
    schema=combined_schema,
    question=question
)

response_sql = llm.invoke(formatted_sql_prompt)
sql_query = response_sql.content
print("Generated SQL Query:\n", sql_query)

# Step6: Execute the SQL Query
result = pd.read_sql_query(sql_query, conn)
print(f"\nBelow is the Answer to Your Question:\n{result}")

# Step7: Final LLM Answer
prompt_answer = PromptTemplate(
    input_variables=["question", "answer"],
    template=(
        """You are a Professional Cricket Analyst who can answer user question in the best tone and context.
        Below is the user question and the answer. Kindly frame a very professional answer.

        User Question: {question}
        Answer: {answer}

        Answer in 100 words or less. Don't just tell the number. Tell an interesting story along the way as well.
        """
    )
)

chain2 = prompt_answer | llm

answer_schema = chain2.invoke(
    {
        "question": question,
        "answer": result
    }
)

print("\nFinal Answer:\n", answer_schema.content)