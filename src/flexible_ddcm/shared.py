import numpy as np
import pandas as pd
from scipy.special import softmax


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


def get_required_covariates_sampled_variables(params, model_options):
    locs = params.index.map(lambda x: x[0].startswith("observable_"))
    covariates = params[locs].index.get_level_values(1).unique()

    covriate_description = (
        model_options["first_period_covariates"]
        if "first_period_covariates" in model_options
        else model_options.get("covariates", {})
    )
    return {
        key: value for key, value in covriate_description.items() if key in covariates
    }


def sample_characteristics(df, params, name, values, seed):
    level_dict = {
        value: params.loc[f"observable_{name}_{value}"] for value in values[1:]
    }

    level_dict[values[0]] = pd.Series(data=[0], index=["constant"])
    z = ()
    for coefs in level_dict.values():
        x_beta = pandas_dot(df, coefs)
        z += (x_beta,)
    probabilities = softmax(np.column_stack(z), axis=1)

    choices = list(level_dict.keys())
    characteristic = _random_choice(choices, probabilities, seed)
    return characteristic


def _random_choice(choices, probabilities, seed, decimals=5):
    np.random.seed(seed)
    cumulative_distribution = probabilities.cumsum(axis=1)
    # Probabilities often do not sum to one but 0.99999999999999999.
    cumulative_distribution[:, -1] = np.round(cumulative_distribution[:, -1], decimals)
    u = np.random.rand(cumulative_distribution.shape[0], 1)

    # Note that :func:`np.argmax` returns the first index for multiple maximum values.
    indices = (u < cumulative_distribution).argmax(axis=1)

    return np.take(choices, indices)
