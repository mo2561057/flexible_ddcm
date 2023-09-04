"""Utilities. Gneral model spec utilities."""
import functools

import numpy as np
import pandas as pd
import scipy

from flexible_ddcm.shared import get_scalar_from_pandas_object
from flexible_ddcm.shared import pandas_dot


def map_transition_to_state_choice_entries(
    initial, choice, arrival, state_space, get_between_states
):

    position_stop = len(state_space.states)
    # Add n states attribute to state space.
    arrival_state = (
        tuple(state_space.state_space.loc[arrival])[:position_stop]
        if arrival
        else arrival
    )
    initial_state = tuple(state_space.state_space.loc[initial])[:position_stop]

    if arrival_state is None:
        return [state_space.state_choice_space_indexer[(*initial_state, choice)]]
    else:

        state_tuples = get_between_states(initial_state, arrival_state, choice)
    return [state_space.state_choice_space_indexer[tuple_] for tuple_ in state_tuples]


def between_states_age_variable(initial_state, arrival_state, choice):
    age_initial = initial_state[0]
    age_arrival = arrival_state[0]
    return [(x, *initial_state[1:], choice) for x in range(age_initial, age_arrival)]


def reward_function(state_choice_space, params, choice_reward_functions):
    """Map state choice to reward."""
    grouper = state_choice_space.groupby(["choice"]).groups
    list_dfs = [
        choice_reward_functions[choice](state_choice_space.loc[locs], params)
        for choice, locs in grouper.items()
    ]
    out = pd.concat(list_dfs)
    out.columns = ["value"]
    return out


def transition_function(
    states, choice, params, variable_state, choice_transition_functions
):
    """
    Maps an old state into a probability distribution of new states.
        Input:
            states: DatFrame
                (edu, age, parental_income, academic_ability)
            params: dict or pd.DataFrame
        Returns:
            dict:
              keys are state tuples and values are probabilities.
    """
    function_ = choice_transition_functions[choice]
    kwargs = {
        "states": states,
        "choice": choice,
        "params": params,
        "variable_state": variable_state,
    }
    arg_names = (
        function_.func.__code__.co_varnames
        if type(function_) == functools.partial
        else function_.__code__.co_varnames
    )

    return function_(
        **{key: value for key, value in kwargs.items() if key in arg_names}
    )


def work_transition(states):
    out = pd.DataFrame(index=states.index)
    out["terminal"] = 1
    return out


def nonstandard_academic_risk(states, params, choice, variable_state, suffix=""):
    age, initial_schooling, _ = variable_state
    dropout = _logit(params.loc[f"transition_risk_{choice}{suffix}"], states)

    length = _poisson_length(
        params.loc[f"transition_length_{choice}{suffix}"],
        states,
        get_scalar_from_pandas_object(params, ("transition_max", choice)),
        get_scalar_from_pandas_object(params, ("transition_min", choice)),
    )

    dropout_length = _assign_probabilities(
        params.loc[f"transition_length_dropout_{choice}{suffix}"], states
    )

    out = pd.DataFrame(index=states.index)
    out[[(col + age, choice, choice) for col in length]] = np.einsum(
        "ij,i->ij", length, dropout
    )

    out[[(col + age, initial_schooling, choice) for col in dropout_length]] = np.einsum(
        "ij,i->ij", dropout_length, (1 - dropout)
    )
    return out


def fixed_length_nonstandard(
    states, params, choice, variable_state, length, length_dropout, suffix=""
):
    age, initial_schooling, _ = variable_state
    dropout = _logit(params.loc[f"transition_risk_{choice}{suffix}"], states)
    out = pd.DataFrame(index=states.index)
    out[(age + length, choice, choice)] = dropout
    out[(age + length_dropout, initial_schooling, choice)] = 1 - dropout
    return out


def poisson_length(states, params, choice, variable_state):
    age, _, _ = variable_state

    length = _poisson_length(
        params.loc[f"transition_length_{choice}"],
        states,
        params.loc[("transition_max", choice)],
        params.loc[("transition_min", choice)],
    )

    out = pd.DataFrame(index=states.index)
    out[[(col + age, choice, choice) for col in length]] = length
    return out


def _logit(params, states):
    exp = np.exp(pandas_dot(states, params))
    return exp / (1 + exp)


def _assign_probabilities(params, states):
    out = pd.DataFrame(index=states.index, columns=params.index)
    for col in out.columns:
        out[col] = params.loc[col]
    out.columns = [int(x) for x in out.columns]
    return out


def _poisson_length(params, states, max, min):
    locs = pandas_dot(states, params)
    locs[locs < 0] = 0
    length = scipy.stats.poisson(pandas_dot(states, params), min).pmf
    out = {value: length(value) for value in range(int(min), int(max) + 1)}
    norm = sum(list(out.values()))
    return pd.DataFrame({key: value / norm for key, value in out.items()})


def nonpecuniary_reward(state_choice_df, input_params, subset):
    # Just a simple dot product over the state choice space
    params = input_params.loc[subset]

    return pandas_dot(state_choice_df, params)


def work_reward(state_choice_df, subset, input_params):
    # Just a simple dot product over the state choice space
    params = input_params[subset]
    return pandas_dot(state_choice_df, params)


def lifetime_wages(
    state_choice_space, params, wage_key, nonpec_key, discount_key, shock_std_key
):
    """Generate wages until the age of 50."""
    wage_params = params.loc[wage_key]
    nonpec_params = params.loc[nonpec_key]
    discount = get_scalar_from_pandas_object(params, discount_key)
    std = get_scalar_from_pandas_object(params, shock_std_key)
    age_auxiliary = range(16, 40)

    # Calculate relevant values:
    final_wage_dict = {}
    for age in age_auxiliary:
        im = state_choice_space.copy()
        im = im.rename(columns={"age": "age_start"})
        im["age"] = age
        im["exp"] = im["age"] - im["age_start"]
        im["exp**2"] = (im["exp"] ** 2) / 100
        final_wage_dict[age] = pd.Series(0, index=state_choice_space.index)
        if (im.exp >= 0).any():
            im = im[im.exp >= 0]
            log_wage = pandas_dot(im, wage_params).astype(float)
            log_wage[log_wage > 4] = 4
            work_utility = pandas_dot(im, nonpec_params)
            final_wage_dict[age].loc[work_utility.index] = (
                np.exp(log_wage + (std) ** 2 / 2) * (1500) + work_utility
            ) * (im.exp.map(lambda x: discount**x))
    # Sum up lifetime wages
    out = functools.reduce(lambda x, y: x + y, list(final_wage_dict.values()))
    return pd.DataFrame(out)


def extreme_value_shocks(choice_value_func, df, params, period):
    shocks = pd.DataFrame(
        index=choice_value_func.index, columns=choice_value_func.columns
    )
    shocks[:] = np.random.gumbel(
        0,
        params.loc[("ev_shocks", "scale")],
        size=shocks.shape[0] * shocks.shape[1],
    ).reshape(shocks.shape)
    return shocks, None
