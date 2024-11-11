import functools

import numpy as np
import pandas as pd
from flexible_ddcm.shared import build_covariates
from flexible_ddcm.solve import solve
from flexible_ddcm.state_space import create_state_space
from flexible_ddcm.transitions import build_transition_func_from_params


def get_simulate_func(
    model_options,
    transition_function,
    reward_function,
    shock_function,
    map_transition_to_state_choice_entries,
    initial_states,
    auxiliary_function=lambda x, y, z: (y, z),
):
    state_space = create_state_space(model_options)

    transition_function, reward_function = auxiliary_function(
        state_space, transition_function, reward_function
    )

    return functools.partial(
        simulate,
        state_space=state_space,
        model_options=model_options,
        transition_function=transition_function,
        reward_function=reward_function,
        shock_function=shock_function,
        map_transition_to_state_choice_entries=map_transition_to_state_choice_entries,
        initial_states=initial_states,
    )


def simulate(
    params,
    state_space,
    model_options,
    transition_function,
    reward_function,
    shock_function,
    map_transition_to_state_choice_entries,
    initial_states,
):
    """For now only iid ev shocks are supported. Shock function can only change things in the
    first period.
    Solves and simulates a prespecified model

    Args:
     params: pd.Series
         Has to be a series otherwise will not work for now.
    """

    _, choice_specific_value_functions, transitions = solve(
        params,
        model_options,
        state_space,
        transition_function.get("subjective")
        if type(transition_function) == dict
        else transition_function,
        reward_function,
        map_transition_to_state_choice_entries,
    )

    if model_options.get("subjective", False):
        transitions = build_transition_func_from_params(
            params, model_options, state_space, transition_function["objective"]
        )

    model_options["first_period_covariates"] = {
        **model_options["covariates"],
        **model_options.get("covariates_simulation", {}),
    }

    simulation_df = _create_simulation_df(
        model_options, state_space, initial_states, params
    )

    simulation_data = {0: simulation_df}
    while True:
        period = max(simulation_data.keys())

        current_period_df = simulation_data[period].copy()[
            (simulation_data[period].choice.isna())
        ]
        # All states with a terminal choice are carried over to the next period.
        # Thus their choice col is not nan.
        terminal_df = simulation_data[period].copy()[
            (~simulation_data[period].choice.isna())
        ]

        # Add a flag for terminal individuals.
        terminal_df["terminal"] = 1

        choice_groups = {
            choice_key: current_period_df.loc[locs]
            for choice_key, locs in current_period_df.groupby(
                "choice_key"
            ).groups.items()
        }
        choice_objects = pd.concat(
            [
                get_choices(
                    df,
                    choice_specific_value_functions[choice_key],
                    params,
                    shock_function,
                    period,
                    model_options["seed"],
                )
                for choice_key, df in choice_groups.items()
            ]
        )

        # current_period_df = current_period_df.join(choice_objects.drop(columns=["choice"]))
        current_period_df.loc[
            choice_objects.index, choice_objects.columns
        ] = choice_objects.values

        simulation_data[period] = pd.concat([current_period_df, terminal_df])

        next_period_df = create_next_period_df(
            current_period_df,
            transitions,
            state_space,
            model_options,
            model_options["seed"] + period + 200,
        )

        next_period_df = pd.concat([next_period_df, terminal_df])
        simulation_data[period + 1] = next_period_df
        if (~next_period_df["choice"].isna()).all():
            break

    return simulation_data


def _create_simulation_df(model_options, state_space, initial_states, params):
    out = initial_states(model_options, params)
    out.index.name = "Identifier"
    out = _attach_information_to_simulated_df(
        out,
        state_space,
        model_options["state_space"],
        model_options["first_period_covariates"],
    )
    out = out.astype(model_options.get("dtypes", {}))
    out["choice"] = np.nan
    out["choice"] = out["choice"].astype(object)
    return out


def get_choices(
    simulation_df, choice_specific_value_function, params, shock_function, period, seed
):
    # First map values to each
    value_function_simulation = pd.DataFrame(
        data=choice_specific_value_function.loc[simulation_df["state_key"]].values,
        columns=choice_specific_value_function.columns,
        index=simulation_df.index,
    )

    taste_shocks, information = shock_function(
        value_function_simulation, simulation_df, params, period, seed
    )
    value_function_simulation = value_function_simulation + taste_shocks

    # Find the max column for each choice.
    choice = value_function_simulation.astype(float).idxmax(axis=1)
    out = taste_shocks.rename(columns={col: f"shock_{col}" for col in taste_shocks})
    out["choice"] = choice
    return out if information is None else out.join(information)


def create_next_period_df(current_df, transitions, state_space, model_options, seed):
    """
    Create dataframe for next period simulation.
    We carry on all terminal individuals to the next period dataframe.

    Args:
      current_df: pd.DataFrame
        current period data.
      transitions: dict of pd.DataFrame
        transition matrix for all admissible choice and
        variable_state combinations.

    """
    np.random.seed(seed)
    transition_grouper = current_df.groupby(["variable_key", "choice"]).groups
    arrival_states = pd.Series(index=current_df.index, dtype=object)
    for (variable_key, choice), locs in transition_grouper.items():
        probabilities = transitions[(choice, variable_key)].loc[
            current_df.loc[locs, "state_key"]
        ]

        cdf = np.array(probabilities.cumsum(axis=1))
        u = np.random.rand(len(cdf), 1)
        indices = (u < cdf).argmax(axis=1)

        variable_arrival_keys = pd.Series(indices, index=locs).map(
            lambda x: probabilities.columns[x]
        )
        subset_arrival_states = variable_arrival_keys.index.map(
            lambda x: state_space.state_and_next_variable_key_to_next_state[
                (current_df.loc[x, "state_key"], variable_arrival_keys.loc[x])
            ]
            if variable_arrival_keys.loc[x] != "terminal"
            else variable_arrival_keys.loc[x]
        )

        arrival_states[variable_arrival_keys.index] = subset_arrival_states.values

    # Which columns do we want to keep? Potentially need to curb this.
    next_df = pd.DataFrame(
        data=state_space.state_space.loc[
            arrival_states[arrival_states != "terminal"]
        ].values,
        columns=state_space.state_space.columns,
        index=arrival_states[arrival_states != "terminal"].index,
    )
    next_df["state_key"] = (
        arrival_states[arrival_states != "terminal"].astype(int).values
    )

    next_df = (
        pd.concat(
            [
                current_df.loc[arrival_states[arrival_states == "terminal"].index],
                next_df,
            ]
        )
        if (arrival_states == "terminal").any()
        else next_df
    )

    next_df = next_df.astype(model_options.get("dtypes", {}))
    next_df = build_covariates(next_df, model_options.get("covariates", {}))

    return next_df


def _attach_information_to_simulated_df(df, state_space, state_space_info, covariates):
    # Get variable key for each row
    df["state_key"] = df[list(state_space_info.keys())].apply(
        lambda x: state_space.state_space_indexer[tuple(x)], axis=1
    )
    df["variable_key"] = state_space.state_space.loc[
        df.state_key, "variable_key"
    ].values
    df["choice_key"] = state_space.state_space.loc[df.state_key, "choice_key"].values
    return build_covariates(df, covariates)
