import copy
import functools

import numpy as np
import pandas as pd
import scipy

from src.model.shared import pandas_dot

# scaling_options = bounds


TRANSITION_FUNCTION = {
    "havo": "combined_logit_length",
    "mbo4": "combined_logit_length",
    "mbo3": "poisson_length",
    "mbo2": "poisson_length",
    "hbo": "combined_logit_length",
    "vocational_work": "work_transition",
    "academic_work": "work_transition",
}


def reward_function(state_choice_space, params):
    """Simply map state choice to reward."""
    grouper = state_choice_space.groupby(["choice"]).groups
    list_dfs = [
        reward_functions[choice](state_choice_space.loc[locs], params)
        for choice, locs in grouper.items()
    ]
    return pd.concat(list_dfs)


def transition_function(states, choice, params, variable_state):
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
    function_ = TRANSITION_FUNCTION[choice]

    if function_ == "combined_logit_length":
        return combined_logit_length(states, params, choice, variable_state)
    elif function_ == "poisson_length":
        return poisson_length(states, params, choice, variable_state)
    elif function_ == "work_transition":
        return work_transition(states)


def work_transition(states):
    out = pd.DataFrame(index=states.index)
    out["terminal"] = 1
    return out


def combined_logit_length(states, params, choice, variable_state):
    age, initial_schooling, _ = variable_state
    dropout = _probit(params.loc[f"transition_risk_{choice}", "value"], states).reshape(
        states.shape[0], 1
    )

    length = _poisson_length(
        params.loc[f"transition_length_{choice}", "value"],
        states,
        int(params.loc[("transition_max", choice), "value"]),
        int(params.loc[("transition_min", choice), "value"]),
    )

    dropout_length = _poisson_length(
        params.loc[f"transition_length_dropout_{choice}", "value"],
        states,
        int(params.loc[("transition_max", f"{choice}_dropout"), "value"]),
        int(params.loc[("transition_min", f"{choice}_dropout"), "value"]),
    )
    out = pd.DataFrame(index=states.index)
    out[[(col + age, choice, choice) for col in length]] = (length * dropout).values
    out[[(col + age, initial_schooling, choice) for col in dropout_length]] = (
        dropout_length * (1 - dropout)
    ).values
    return out


def poisson_length(states, params, choice, variable_state):
    age, _, _ = variable_state

    length = _poisson_length(
        params.loc[f"transition_length_{choice}", "value"],
        states,
        int(params.loc[("transition_max", choice), "value"]),
        int(params.loc[("transition_min", choice), "value"]),
    )

    out = pd.DataFrame(index=states.index)
    out[[(col + age, choice, choice) for col in length]] = length
    return out


def _probit(params, states):
    return scipy.special.softmax(pandas_dot(states, params))


def _poisson_length(params, states, max, min):
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


def lifetime_wages(state_choice_space, params, wage_key, nonpec_key, discount_key):
    """Generate wages until the age of 50."""
    wage_params = params.loc[wage_key, "value"]
    nonpec_params = params.loc[nonpec_key, "value"]
    discount = float(params.loc[discount_key, "value"])
    age_auxiliary = range(15, 55)

    # Calculate relevant values:
    final_wage_dict = {}
    for age in age_auxiliary:
        im = state_choice_space.copy()
        im = im.rename(columns={"age": "age_start"})
        im["age"] = age
        im["exp"] = im["age"] - im["age_start"]
        im = im[im.exp >= 0]
        log_wage = pandas_dot(im, wage_params).astype(float)
        work_utility = pandas_dot(im, nonpec_params)
        final_wage_dict[age] = pd.Series(0, index=state_choice_space.index)
        final_wage_dict[age].loc[work_utility.index] = (
            np.exp(log_wage) + work_utility
        ) * (im.exp.map(lambda x: discount**x))

    # Sum up lifetime wages
    out = functools.reduce(lambda x, y: x + y, list(final_wage_dict.values()))
    out.name = "value"
    return pd.DataFrame(out)


def map_transition_to_state_choice_entries(initial, choice, arrival, state_space):
    # Add n states attribute to state space.
    arrival_state = (
        tuple(state_space.state_space.loc[arrival])[:6] if arrival else arrival
    )
    initial_state = tuple(state_space.state_space.loc[initial])[:6]

    """In this case only the initial state"""
    if arrival_state is None:
        return [state_space.state_choice_space_indexer[(*initial_state, choice)]]
    else:
        age_initial = initial_state[0]
        age_arrival = arrival_state[0]
        state_tuples = [
            (x, initial_state[1], *initial_state[2:6], choice)
            for x in range(age_initial, age_arrival)
        ]
        return [
            state_space.state_choice_space_indexer[tuple_] for tuple_ in state_tuples
        ]


reward_functions = {
    "mbo4": functools.partial(nonpecuniary_reward, subset="nonpec_mbo4"),
    "havo": functools.partial(nonpecuniary_reward, subset="nonpec_havo"),
    "mbo3": functools.partial(nonpecuniary_reward, subset="nonpec_mbo3"),
    "hbo": functools.partial(nonpecuniary_reward, subset="nonpec_hbo"),
    "vocational_work": functools.partial(
        lifetime_wages,
        nonpec_key="nonpec_vocational",
        wage_key="wage_vocational",
        discount_key=("discount", "discount"),
    ),
    "academic_work": functools.partial(
        lifetime_wages,
        nonpec_key="nonpec_academic",
        wage_key="wage_academic",
        discount_key=("discount", "discount"),
    ),
}
