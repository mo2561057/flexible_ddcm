"""Utilities. Gneral model spec utilities."""
import functools

import numpy as np
import pandas as pd
import scipy

from flexible_ddcm.shared import build_covariates
from flexible_ddcm.shared import get_required_covariates_sampled_variables
from flexible_ddcm.shared import get_scalar_from_pandas_object
from flexible_ddcm.shared import pandas_dot
from flexible_ddcm.shared import sample_characteristics


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
        Input:mbox
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
    state_choice_space,
    params,
    wage_key,
    nonpec_key,
    discount_key,
    shock_std_key,
    age_auxiliary=range(16, 40),
    age_continuation=None,
):
    """Generate wages until the age of 50."""
    wage_params = params.loc[wage_key]
    nonpec_params = params.loc[nonpec_key]
    discount_factor = get_scalar_from_pandas_object(params, discount_key)
    shock_std = get_scalar_from_pandas_object(params, shock_std_key)

    # Initialize the dictionary to store calculated wages for each age
    wage_results = {}
    for age in age_auxiliary:
        # Prepare intermediate DataFrame
        modified_df = state_choice_space.copy()
        modified_df = modified_df.rename(columns={"age": "age_start"})
        modified_df["age"] = age
        modified_df["exp"] = modified_df["age"] - modified_df["age_start"]
        modified_df["exp**2"] = (modified_df["exp"] ** 2) / 100

        # Initialize wages for the current age
        wage_results[age] = pd.Series(0, index=state_choice_space.index)

        if (modified_df.exp >= 0).any():
            valid_entries = modified_df[modified_df.exp >= 0]
            log_wage = pandas_dot(valid_entries, wage_params).astype(float)
            log_wage[log_wage > 4] = 4  # Cap log wage at 4
            work_utility = pandas_dot(valid_entries, nonpec_params)
            wage_results[age].loc[work_utility.index] = (
                np.exp(log_wage + (shock_std) ** 2 / 2) * 1500 + work_utility
            ) * (valid_entries.exp.map(lambda x: discount_factor**x))

    # Apply continuation wages if specified
    if age_continuation:
        for age in age_continuation:
            wage_results[age] = wage_results[max(age_auxiliary)]

    # Sum up all calculated wages to a single DataFrame
    cumulative_wages = functools.reduce(lambda x, y: x + y, wage_results.values())
    return pd.DataFrame(cumulative_wages)


def extreme_value_shocks(choice_value_func, df, params, period, seed):
    np.random.seed(seed + period)
    shocks = pd.DataFrame(
        index=choice_value_func.index, columns=choice_value_func.columns
    )

    base_draws = np.random.uniform(
        0,
        1,
        size=shocks.shape[0] * shocks.shape[1],
    )

    shocks[:] = (
        -params.loc[("ev_shocks", "scale")] * np.log(-np.log(base_draws))
    ).reshape(shocks.shape)
    return shocks, None


def initial_states_external_and_logit_probs(
    model_options, params, external_probabilities
):
    """External states must contain joint probability per combination of observables."""
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
    np.random.seed(model_options["seed"] + 2_000_000)
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

    # Build covariates required for type creation:
    covariates_type = get_required_covariates_sampled_variables(params, model_options)

    out = build_covariates(out, covariates_type)
    # Add estimated probabilities
    for col, specs in model_options["state_space"].items():
        if specs["start"] == "random_internal":
            out[col] = sample_characteristics(
                out, params, col, specs["list"], model_options["seed"] + 2_000_001
            )
    return out
