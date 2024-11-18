"""Solve the model."""
import functools
import itertools

import numpy as np
import pandas as pd
import scipy
from flexible_ddcm.rewards import calculate_rewards_state_choice_space
from flexible_ddcm.shared import get_scalar_from_pandas_object
from flexible_ddcm.state_space import create_state_space
from flexible_ddcm.transitions import build_transition_func_from_params


def solve(
    params,
    model_options,
    state_space,
    transition_function,
    reward_function,
    map_transition_to_state_choice_entries,
):
    # Need to put into options.
    segmentation_column = "age"
    transitions = build_transition_func_from_params(
        params, model_options, state_space, transition_function
    )
    rewards = calculate_rewards_state_choice_space(
        state_space.state_choice_space, params, reward_function
    )[["value"]]

    # Segment state space into chunks that we iterate over.
    state_grouper = state_space.state_space.groupby(segmentation_column).groups

    # Initiate Continuation values
    continuation_values = np.full(
        state_space.state_space.index.max+1, np.nan)

    # Initiate array to keep all entries from
    choice_specific_value_function = {
        choice_key: []
        for choice_key, choice_set in state_space.choice_key_to_choice_set.items()
        if choice_set
    }

    # Settle all states that are continuation values only.
    for _, locs in reversed(state_grouper.items()):
        # This could be paralellized
        for variable_key, locs_variable in (
            state_space.state_space.loc[locs].groupby("variable_key").groups.items()
        ):
            choices = state_space.variable_key_to_choice_set[variable_key]
            choice_key = state_space.choice_set_to_choice_key[tuple(choices)]

            if choices:
                (
                    choice_specific_value_function_key,
                    continuation_values_key,
                ) = get_choice_specific_values(
                    variable_key,
                    transitions,
                    rewards,
                    continuation_values,
                    choices,
                    params,
                    state_space,
                    map_transition_to_state_choice_entries,
                )
                continuation_values[
                    locs_variable
                ] = continuation_values_key.loc[locs_variable].values
 
                choice_specific_value_function[choice_key].append(
                    choice_specific_value_function_key
                )

            else:
                continuation_values.loc[locs_variable, "continuation_value"] = np.nan

    choice_specific_value_function = {
        key: pd.concat(value) for key, value in choice_specific_value_function.items()
    }
    continuation_values = pd.DataFrame(
        continuation_values, columns=["continuation_value"]).dropna(axis=0)
    return continuation_values, choice_specific_value_function, transitions


def get_choice_specific_values(
    variable_point,
    transitions,
    rewards,
    continuation_values,
    choices,
    params,
    state_space,
    map_transition_to_state_choice_entries,
):
    out = pd.DataFrame(
        index=transitions[(choices[0], variable_point)].index, columns=choices
    )

    for choice in choices:
        transition = transitions[(choice, variable_point)]
        # Get continuation values.
        out[choice] = get_continuation_value_for_transitions(
            transition,
            choice,
            rewards,
            params[("discount", "discount")],
            continuation_values,
            state_space,
            map_transition_to_state_choice_entries,
        ).sum(axis=1)

    return out, get_expected_value_ev_shocks(out, params[("ev_shocks", "scale")])


def get_continuation_value_for_transitions(
    transitions,
    choice,
    rewards,
    discount,
    continuation_values,
    state_space,
    map_transition_to_state_choice_entries,
):
    # This is the only reason we need all the chunks?
    fixed_keys = state_space.state_to_fixed_key[transitions.index]

    if "terminal" in transitions.columns:
        continuation_values = np.zeros(transitions.shape)
    else:
        positions_continuation = state_space.variable_and_fixed_key_to_state[
            np.ix_(fixed_keys.astype(np.int32), transitions.columns)
        ]
        continuation_values = continuation_values[positions_continuation]

    # Need to differentiate between different scenarios:
    # Accomodate the new sceanrio as well.
    rewards = rewards.loc[
        state_space.state_and_choice_to_state_choice[choice][transitions.index.values]]

    return transitions.values * (rewards + continuation_values * discount)


def _map_continuation_to_transition(
    initial,
    choice,
    variable_key,
    rewards,
    continuation_values,
    discount,
    state_space,
    map_transition_to_state_choice_entries,
):
    """Rewards are potentially stochastic."""

    arrival = (
        None
        if variable_key == "terminal"
        else state_space.variable_and_fixed_key_to_state[
            (variable_key, state_space.state_to_fixed_key[initial])
        ]
    )

    locs_rewards = map_transition_to_state_choice_entries(
        initial, choice, arrival, state_space
    )

    continuation_value = (
        continuation_values.loc[arrival].iloc[0] * (discount ** (len(locs_rewards)))
        if arrival
        else 0
    )

    return continuation_value + functools.reduce(
        lambda x, y: x + y,
        [x * (discount**n) for n, x in enumerate(rewards.loc[locs_rewards].values)],
    )


def get_expected_value_ev_shocks(
    choice_specific_values,
    scale_parameter,
):

    return pd.Series(
        data=scale_parameter
        * scipy.special.logsumexp(
            choice_specific_values.astype(float) / scale_parameter, axis=1
        ),
        index=choice_specific_values.index,
    )
