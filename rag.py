import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from google.protobuf.struct_pb2 import Struct


load_dotenv()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "y11.xlsx"
    db_path = base_dir / "mydatabase.db"

    dataframe = pd.read_excel(excel_path, index_col=0)

    with sqlite3.connect(db_path) as connection:
        dataframe.to_sql("mytable", connection, if_exists="replace")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file first.")

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel("gemini-1.5-flash")

    def sql_query(query: str):
        """Run a SQL SELECT query on the SQLite database and return the results."""
        return pd.read_sql_query(query, connection).to_dict(orient="records")

    system_prompt = """
You are an expert SQL analyst. When appropriate, generate SQL queries based on the user question and the database schema.
When you generate a query, use the 'sql_query' function to execute the query on the database and get the results.
Then, use the results to answer the user's question.

database_schema: [
    {
        table: 'mytable',
        columns: [
            { name: 'first_name', type: 'string' },
            { name: 'last_name', type: 'string' },
            { name: 'Age', type: 'int' },
            { name: 'Gender', type: literal['Male', 'Female'] },
            { name: 'Country', type: 'string' },
            { name: 'Date', type: 'datetime' },
            { name: 'Id', type: 'int' }
        ]
    }
]
""".strip()

    sql_gemini = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[sql_query],
        system_instruction=system_prompt,
    )

    question = " ".join(sys.argv[1:]).strip() or "What country has the highest number of employees?"
    messages = [{"role": "user", "parts": [question]}]

    response = sql_gemini.generate_content(messages)
    query = response.parts[0].function_call.args["query"]
    results = sql_query(query)

    s = Struct()
    s.update({"result": results})

    function_response = genai.protos.Part(
        function_response=genai.protos.FunctionResponse(name="sql_query", response=s)
    )

    messages.extend(
        [
            {"role": "model", "parts": response.parts},
            {"role": "user", "parts": [function_response]},
        ]
    )

    print(sql_gemini.generate_content(messages).text)


if __name__ == "__main__":
    main()
