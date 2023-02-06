import os
import argparse
import json
import pandas as pd
from langchain import PromptTemplate, OpenAI
from langchain import PromptTemplate, OpenAI
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from df_utils import get_columns_from_df, get_category_values_for_column

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"
llm = OpenAI(model_name = "text-davinci-002")

def main() -> None:    
    parser = argparse.ArgumentParser(
        prog = 'llm-pandas',
        description = "explore pandas dataframe with natural language"        
    )  
    parser.add_argument("--csv", type=str, help="name of csv file")    
    args = parser.parse_args()        
    df = pd.read_csv(args.csv)
    df_columns = get_columns_from_df(df)
    print(f"Loaded {args.csv} into dataframe...")    

    while True:
        query = input(f"Ask your question about: {args.csv}: ")        

        # get relevant columns for this query
        relevant_columns_prompt = PromptTemplate(input_variables=["query", "columns"], template = "Given this query: {query} and these dataframe columns: {columns}, find all relevant columns to satisfy the query. Return the result as an array. Example 1: ['column_value1', 'columns_value2'] Example 2: ['column_value3', 'column_value4']")
        relevant_columns_response = llm(relevant_columns_prompt.format(query=query, columns=df_columns))
        
        # deserialize + format quotes to make json compatible
        r = relevant_columns_response.replace('\'', "\"")
        print(r)
        columns_deser = json.loads(r)

        print("----RELEVANT COLUMNS----")
        print(columns_deser)

        col_to_examples = {}

        for col in columns_deser:
            col_values = get_category_values_for_column(df,col)
            if len(col_values) < 50:
                col_to_examples[col] = list(col_values)

        print("---EXAMPLE VALUES FOR COLUMNS:---")
        print(col_to_examples)        

        pandas_query_prompt = PromptTemplate(input_variables=["query", "columns_dict"], template = "Given this query: {query}, and this dictionary representing dataframe columns and example values: {columns_dict}, write the pandas query using a pandas df. Example: df[df['arrival_year'] == 2018]['booking_status'].value_counts()['Not_Canceled'] Output: ")
        pandas_query_response = llm(pandas_query_prompt.format(query=query, columns_dict=str(col_to_examples))).strip()

        print(f"LLM GENERATED QUERY: {pandas_query_response}")

        print("=================")

        print(f"Question: {query}")
        print(f"Answer: {eval(pandas_query_response)}")

        print("=================")            

        

        

        
    


if __name__ == "__main__":
    main()