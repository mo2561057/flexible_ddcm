import numpy as np
import pandas as pd


def pandas_dot(df, series):
    return df[list(series.index.values)].dot(series)


def build_covariates(df, model_options):
    for col, definition in model_options.get("covariates", {}).items():
        df[col] = df.eval(definition)
    return df
