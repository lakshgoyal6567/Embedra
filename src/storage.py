import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import os
import json

def save_to_parquet(data: Union[List[Dict], pd.DataFrame], output_path: str):
    """
    Saves the processed data to a Parquet file with Zstd compression.
    Expects data to be a list of dicts or a DataFrame.
    """
    
    def serialize_dict_column(df, col_name):
        if col_name in df.columns:
            # Force conversion for all rows if ANY is a dict, or just blindly apply
            # safer to just apply check on every element to be robust against mixed types
            df[col_name] = df[col_name].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    
    if isinstance(data, list):
        if not data:
            print("No data to save.")
            return

        df_rows = []
        for entry in data:
            row = entry.copy()
            # Convert embedding numpy array to list for storage compatibility
            if isinstance(row.get('embedding'), np.ndarray):
                row['embedding'] = row['embedding'].tolist()
                
            # Ensure complex fields are serializable
            for field in ['metadata', 'semantic_meta', 'fact_meta']:
                if field in row and isinstance(row[field], dict):
                     row[field] = json.dumps(row[field])
                 
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if df.empty:
            print("No data to save.")
            return
            
        # Ensure metadata columns are json string
        for col in ['metadata', 'semantic_meta', 'fact_meta']:
            serialize_dict_column(df, col)
                 
        # Ensure embedding is list (not numpy array object in cell) for pyarrow
        if 'embedding' in df.columns:
            # If cells contain numpy arrays, convert to list
            # We can't rely on just the first element if the Series has mixed types (some lists, some arrays)
            # Safe approach:
            df['embedding'] = df['embedding'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    else:
        print("Unsupported data format for save_to_parquet")
        return

    try:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='zstd')
        print(f"Successfully saved {len(df)} records to {output_path}")
    except Exception as e:
        print(f"Error saving to Parquet: {e}")