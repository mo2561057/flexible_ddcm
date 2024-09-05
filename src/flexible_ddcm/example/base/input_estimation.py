"""Tools for model estimation."""
"""Maps simulation output into a form that can be used to generate moments"""
import numpy as np
import pandas as pd
from flexible_ddcm.shared import pandas_dot

from src.estimation.config_estimation import RENAME_DEGREE_COMBINATIONS


def process_simulation_dict(
    simulate_dict, params, wage_periods, additional_cols, schooling_levels
):
    """Map simulate dict into dfs for moment creation."""
    wide_df = _create_wide_df(simulate_dict, schooling_levels)

    long_df = _create_long_df(
        simulate_dict, wide_df, params, wage_periods, additional_cols
    )

    return {"wide": wide_df, "long": long_df}


def _create_long_df(simulate_dict, wide_df, params, wage_periods, additional_cols):
    terminal_df = simulate_dict[max(simulate_dict.keys())]
    terminal_df["period"] = terminal_df.age - 16
    wages = {}
    # Wages. Only one wage df
    for period in wage_periods:
        # Check all people
        current_df = terminal_df[terminal_df.period <= period].copy()
        current_df["exp"] = period - current_df.period
        wages[period] = pd.concat(
            [
                _get_wages_sector(current_df, params, choice)
                for choice in ["academic", "vocational"]
            ]
        )

    # Order index correctly
    out = pd.concat(wages)
    out.columns = ["log_wage"]
    out.index.names = ["period", "Identifier"]
    out = out.join(wide_df[additional_cols]).reorder_levels(["Identifier", "period"])
    out["wage"] = np.exp(out.log_wage)
    out["exp"] = (period - (out["age"] - 16)).astype(float)
    return out


def _get_wages_sector(current_df, params, choice):

    if ~(current_df.choice == f"{choice}_work").any():
        return pd.DataFrame()

    draws = _create_wage_draws(
        current_df[current_df.choice == f"{choice}_work"],
        params.loc[(f"wage_shock_{choice}", "std"), "value"],
    )

    log_wages = pd.Series(
        data=
            pandas_dot(
                current_df[current_df.choice == f"{choice}_work"],
                params.loc[f"wage_{choice}"],
            )
            .values.reshape(len(draws))
            .astype(np.float64)
            + draws,
        index=current_df[current_df.choice == f"{choice}_work"].index,
    )
    return log_wages


def _create_wage_draws(current_period_df, std):
    return np.random.normal(0, std, size=current_period_df.shape[0])


def _create_wide_df(simulate_dict, schooling_levels):
    out = simulate_dict[max(simulate_dict.keys())]

    # Get degree combinations
    degree_df = pd.concat([df["schooling"] for df in simulate_dict.values()], axis=1)

    level_dict = dict()
    # Where to put these levels?
    for level in schooling_levels:
        level_dict[level] = (degree_df == level).any(axis=1)

    grouper = pd.concat(level_dict, axis=1).groupby(schooling_levels)

    rename_dict = {n: group for n, group in enumerate(grouper.groups)}
    rename_dict = {
        key: "_".join([col for n, col in enumerate(schooling_levels) if value[n]])
        for key, value in rename_dict.items()
    }
    out["degree_combination"] = (
        grouper.ngroup().replace(rename_dict).map(lambda x: x.replace("vmbo_", "").replace(
        "hbo","bachelor"))
    )

    # Get ever enrolled:
    choice_df = pd.concat([df["choice"] for df in simulate_dict.values()], axis=1)

    for level in schooling_levels:
        out[f"enrolled_{level}"] = np.array(choice_df == level).any(axis=1)

    # Get duration bachelor
    bachelor_ids = out.schooling == "hbo"

    bachelor_df = pd.concat(
        [
            df[(df.choice == "hbo") & (df.index.isin(bachelor_ids))]
            for df in simulate_dict.values()
        ]
    ).rename(columns={"age": "age_bachelor"})

    out = out.join(bachelor_df[["age_bachelor"]])
    out["duration_bachelor"] = (out.age - out.age_bachelor)
    out["age"] = out.age - 16
    return out
