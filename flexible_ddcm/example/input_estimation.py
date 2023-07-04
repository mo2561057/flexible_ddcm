"""Maps simulation output into a form that can be used to generate moments"""
import itertools

import numpy as np
import pandas as pd

from flexible_ddcm.shared import build_covariates
from flexible_ddcm.shared import pandas_dot
from src.estimation.config import SCHOOLING_LEVELS


def process_simulation_dict(
        simulation_dict,
        wage_periods,
        params,
        additional_cols
        ):

    long_df = _create_long_df(
        simulation_dict,
        wage_periods,
        params,
        additional_cols
        )
    wide_df = _create_wide_df()

    return {
        "wide": wide_df,
        "long":long_df
        }


def _create_long_df(simulation_dict, wage_periods, params, additional_cols):
    terminal_df = simulation_dict[-1]
    wages = {}
    # Wages.
    for period in wage_periods:
        # Check all people
        current_period_df = terminal_df[terminal_df.period <= period]
        # Calculate raw wages in this period
        current_period_df["exp"] = current_period_df["period"] - period
        # Now we need to diferentiate between academic and vocational
        vocational_draws = _create_wage_draws(
            current_period_df[current_period_df.choice == "vocational work"],
            params["wage_shocks_vocational"],
        )

        academic_draws = _create_wage_draws(
            current_period_df[current_period_df.choice == "vocational work"],
            params["wage_shocks_academic"],
        )

        vocational_wages = pd.Series(
            data=np.exp(
                pandas_dot(
                    current_period_df[current_period_df.choice == "vocational work"],
                    params.loc["wage vocational"],
                )
                + vocational_draws
            ),
            index=current_period_df[
                current_period_df.choice == "vocational work"
            ].index,
        )

        academic_wages = pd.Series(
            data=np.exp(
                pandas_dot(
                    current_period_df[current_period_df.choice == "academic work"],
                    params.loc["wage academic"],
                )
                + academic_draws
            ),
            index=current_period_df[current_period_df.choice == "academic work"].index,
        )
        wages[period] = pd.concat([academic_wages, vocational_wages])
    # Order index correctly
    out = pd.concat(wages)
    out.index.names = ["period", "personal_id"]
    out = out.join(simulation_dict[-1][additional_cols]).reorder_levels(
        ["personal_id", "period"])
    return out

    # Create wage dict


def _create_wage_draws(current_period_df, params):
    std = pandas_dot(current_period_df, params)
    return np.random.normal(0, std)


def _create_wide_df(simulation_dict):
    out = simulation_dict[-1]

    # Get degree combinations
    degree_df = pd.concat(
        [df["schooling"] for df in simulation_dict.values()],axis=0)
    
    level_dict = {}
    # Where to put these levels?
    for level in SCHOOLING_LEVELS:
        level_dict[level] = (
            degree_df == level).any(axis=1)
    
    grouper = pd.concat(
        level_dict,axis=0).groupby(SCHOOLING_LEVELS)
    
    rename_dict = {n:group for n, group in enumerate(grouper.groups)}
    rename_dict = {
        key:"_".join([col for n, col in SCHOOLING_LEVELS if value[pos]]\
                      for key, value in rename_dict.items())}
    out["degree_combination"] = grouper.ngroup(
            ).replace(rename_dict)
    
    # Get ever enrolled:
    choice_df = pd.concat(
        [df["choice"] for df in simulation_dict.values()],axis=0)
    for level in SCHOOLING_LEVELS:
        out[f"enrolled_{level}"] = (choice_df==level).any(axis=0)
    return out 
    

    
    