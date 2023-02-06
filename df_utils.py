import pandas as pd
from typing import Any, Dict, List

def get_columns_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    return df.dtypes.to_dict()

def get_uniques_for_column(df: pd.DataFrame) -> Dict[str, Any]:
    return {col: df[col].nunique() for col in df.columns}

def get_category_values_for_column(df, col: str) -> List[Any]:
    return df[col].unique()
