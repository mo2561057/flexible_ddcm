import numpy as np
import pandas as pd


def pandas_dot(df, series):
    return df[list(series.index.values)].dot(series)


def build_covariates(df, covariates):
    df = df.copy()
    for col, definition in covariates.items():
        if col in df:
            continue
        df[col] = df.eval(definition)
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def get_scalar_from_pandas_object(pd_container, key):
    out = pd_container.loc[key]
    return out.iloc[0] if type(out) in [pd.DataFrame, pd.Series] else out
