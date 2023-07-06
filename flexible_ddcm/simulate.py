import functools

import numpy as np
import pandas as pd

from flexible_ddcm.shared import build_covariates
from flexible_ddcm.solve import solve
from flexible_ddcm.state_space import create_state_space


def get_simulate_func(
    model_options,
    transition_function,
    reward_function,
    external_probabilities,
    map_transition_to_state_choice_entries,
):
    state_space = create_state_space(model_options)
    return functools.partial(
        simulate,
        state_space=state_space,
        model_options=model_options,
        transition_function=transition_function,
        reward_function=reward_function,
        external_probabilities=external_probabilities,
        map_transition_to_state_choice_entries=map_transition_to_state_choice_entries,
    )


def simulate(
    params,
    state_space,
    model_options,
    transition_function,
    reward_function,
    external_probabilities,
    map_transition_to_state_choice_entries,
):

    _, choice_specific_value_functions, transitions = solve(
        params,
        model_options,
        transition_function,
        reward_function,
        map_transition_to_state_choice_entries,
    )

    simulation_df = _create_simulation_df(
        model_options, state_space, external_probabilities
    )

    simulation_data = {0: simulation_df}
    while True:
        # How should this be structured?
        period = max(simulation_data.keys())

        current_period_df = simulation_data[period].copy()[
            (simulation_data[period].choice.isna())
        ]

        terminal_df = simulation_data[period].copy()[
            (~simulation_data[period].choice.isna())
        ]

        choice_groups = {
            choice_key: current_period_df.loc[locs]
            for choice_key, locs in current_period_df.groupby(
                "choice_key"
            ).groups.items()
        }

        choices = pd.concat(
            [
                get_choices(df, choice_specific_value_functions[choice_key], params)
                for choice_key, df in choice_groups.items()
            ]
        )

        current_period_df.loc[choices.index, "choice"] = choices.values

        simulation_data[period] = pd.concat([current_period_df, terminal_df])

        next_period_df = create_next_period_df(
            current_period_df, transitions, state_space, model_options
        )

        next_period_df = pd.concat([next_period_df, terminal_df])
        simulation_data[period + 1] = next_period_df
        if (~next_period_df["choice"].isna()).all():
            break

    return simulation_data


def _create_simulation_df(model_options, state_space, external_probabilities):
    """External states must contain joint
    probability per combination of unobservables."""

    # Assign a start state to each individual.
    out = pd.DataFrame(
        index=range(model_options["n_simulation_agents"]),
        columns=model_options["state_space"].keys(),
    )

    # Assign all fixed states.
    for state, specs in model_options["state_space"].items():
        if specs["start"] not in ["random_external", "random_internal"]:
            out[state] = specs["start"]

    # Assign stochastic states
    locs_external = np.random.choice(
        external_probabilities.index,
        p=external_probabilities["probability"],
        size=len(out),
        replace=True,
    )

    states_external = [
        col for col in external_probabilities.columns if col != "probability"
    ]

    out[states_external] = external_probabilities.loc[
        locs_external, states_external
    ].values

    # Add estimated probabilities

    out.index.name = "Identifier"
    out = _attach_information_to_simulated_df(out, state_space, model_options)
    out["choice"] = np.nan
    return out


def get_choices(
    simulation_df,
    choice_specific_value_function,
    params,
):
    # First map values to each
    value_function_simulation = pd.DataFrame(
        data=choice_specific_value_function.loc[simulation_df["state_key"]].values,
        columns=choice_specific_value_function.columns,
        index=simulation_df.index,
    )

    taste_shocks = create_taste_shocks(value_function_simulation, params)
    value_function_simulation = value_function_simulation + taste_shocks
    # Find the max column for each choice.
    choice = value_function_simulation.astype(float).idxmax(axis=1)
    return choice


def create_taste_shocks(choice_value_func, params):
    shocks = pd.DataFrame(
        index=choice_value_func.index, columns=choice_value_func.columns
    )
    shocks[:] = np.random.gumbel(
        0,
        params.loc[("ev_shocks", "scale"), "value"],
        size=shocks.shape[0] * shocks.shape[1],
    ).reshape(shocks.shape)
    return shocks


def create_next_period_df(current_df, transitions, state_space, model_options):
    transition_grouper = current_df.groupby(["variable_key", "choice"]).groups
    arrival_states = pd.Series(index=current_df.index)
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
    next_df = pd.concat(
        [current_df.loc[arrival_states[arrival_states == "terminal"].index], next_df]
    )
    next_df = build_covariates(next_df, model_options)

    return next_df


def _attach_information_to_simulated_df(df, state_space, model_options):
    # Get variable key for each row
    df["state_key"] = df[list(model_options["state_space"].keys())].apply(
        lambda x: state_space.state_space_indexer[tuple(x)], axis=1
    )

    df["variable_key"] = state_space.state_space.loc[
        df.state_key, "variable_key"
    ].values
    df["choice_key"] = state_space.state_space.loc[df.state_key, "choice_key"].values
    return build_covariates(df, model_options)
