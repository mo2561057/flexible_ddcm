"""Build state space objects required to solve the model."""
import functools
from collections import namedtuple

import numpy as np
import pandas as pd
from flexible_ddcm.shared import build_covariates
from flexible_ddcm.shared import build_states


def create_state_space(model_options):
    """Build the full state space we need to loop through later."""

    states = build_states(model_options)

    fixed_states = [
        col
        for col in model_options["state_space"].keys()
        if model_options["state_space"][col]["fixed"]
    ]

    stochastic_states = [
        col
        for col in model_options["state_space"].keys()
        if model_options["state_space"][col]["stochastic"]
    ]

    state_space = _create_state_space_array(states, model_options)

    # Can also keep as a dict of arrays?
    fixed_states_group = state_space.groupby(fixed_states)
    variable_states_group = state_space.groupby(
        [
            col
            for col in state_space
            if (
                (col not in fixed_states) & (col in model_options["state_space"].keys())
            )
        ]
    )
    stochastic_states_group = state_space.groupby(stochastic_states)

    state_space["variable_key"] = variable_states_group.ngroup()
    state_space["fixed_key"] = fixed_states_group.ngroup()
    state_space["stochastic_key"] = stochastic_states_group.ngroup()

    # Now get state choice space.
    (
        state_choice_space,
        choice_key_df,
        choice_key_to_choice_set,
    ) = _create_choice_objects(state_space, model_options)

    state_space["choice_key"] = choice_key_df.values

    # Get state space indexer.
    state_space_indexer = {
        tuple(state_space.loc[ix, list(states.keys())]): ix for ix in state_space.index
    }

    state_choice_space_indexer = {
        tuple(state_choice_space.loc[ix, list(states.keys()) + ["choice"]]): ix
        for ix in state_choice_space.index
    }

    # Need this only for indexer
    variable_state_space = state_space.loc[
        ~state_space.variable_key.duplicated(),
        [col for col in states if col not in fixed_states] + ["variable_key"],
    ].set_index("variable_key")

    variable_state_space_indexer = {
        tuple(
            variable_state_space.loc[
                ix, [col for col in states if col not in fixed_states]
            ]
        ): ix
        for ix in variable_state_space.index
    }

    stochastic_state_space = state_space.loc[
        ~state_space.stochastic_key.duplicated(),
        [col for col in states if col in stochastic_states] + ["stochastic_key"],
    ].set_index("stochastic_key")

    (
        state_to_fixed_key,
        variable_and_fixed_key_to_state,
        variable_key_to_choice_set,
        choice_set_to_choice_key,
        state_and_choice_to_state_choice,
    ) = create_derived_opjects(state_space, choice_key_to_choice_set, state_choice_space)

    state_space_container = namedtuple(
        "state_space",
        [
            "state_space",
            "state_choice_space",
            "variable_state_space",
            "stochastic_state_space",
            "state_space_indexer",
            "state_choice_space_indexer",
            "variable_state_space_indexer",
            "state_to_fixed_key",
            "variable_and_fixed_key_to_state",
            "variable_key_to_choice_set",
            "choice_key_to_choice_set",
            "choice_set_to_choice_key",
            "states",
            "state_and_choice_to_state_choice",
        ],
    )

    return state_space_container(
        state_space,
        state_choice_space,
        variable_state_space,
        stochastic_state_space,
        state_space_indexer,
        state_choice_space_indexer,
        variable_state_space_indexer,
        state_to_fixed_key,
        variable_and_fixed_key_to_state,
        variable_key_to_choice_set,
        choice_key_to_choice_set,
        choice_set_to_choice_key,
        list(states.keys()),
        state_and_choice_to_state_choice
    )


def _create_state_space_array(states, model_options):
    state_space = _create_product_array(states)
    # Filter state space
    for definition in model_options.get("state_space_filter", []):
        state_space = state_space[~state_space.eval(definition)]
    # Build covariats
    state_space = build_covariates(state_space, model_options.get("covariates", {}))
    return state_space


def _create_product_array(array_dict):
    num_states = functools.reduce(
        lambda x, y: x * y, [len(x) for x in array_dict.values()]
    )
    column_names = list(array_dict.keys())
    return pd.DataFrame(
        {
            column_names[pos]: grid.reshape(num_states)
            for pos, grid in enumerate(
                np.meshgrid(*(np.array(x) for x in array_dict.values()))
            )
        }
    )


def create_derived_opjects(state_space, choice_key_to_choice_set, state_choice_space):
    variable_and_fixed_key_to_state = np.zeros(
        (state_space.fixed_key.max() + 1, state_space.variable_key.max() + 1)
    )

    variable_and_fixed_key_to_state[
        state_space.fixed_key, state_space.variable_key
    ] = state_space.index
    variable_and_fixed_key_to_state = variable_and_fixed_key_to_state.astype(int)

    state_to_fixed_key = np.zeros(
        (state_space.index.max() + 1)
    )

    state_to_fixed_key[state_space.index.values
                       ] = state_space.fixed_key.values

    state_to_fixed_key = state_to_fixed_key.astype(int)

    state_and_choice_to_state_choice = {
        choice: np.zeros(state_choice_space.state_key.max() + 1) for choice in state_choice_space.choice.unique()
    }
    for key in state_and_choice_to_state_choice:
        state_and_choice_to_state_choice[key][state_choice_space.state_key.values] = state_choice_space.index.values
    state_and_choice_to_state_choice = state_and_choice_to_state_choice.astype(int)

    # Map variable key to choice set:
    variable_key_to_choice_set = {
        variable_key: choice_key_to_choice_set[choice_key]
        for variable_key, choice_key in state_space.loc[
            ~state_space.variable_key.duplicated(), ["variable_key", "choice_key"]
        ].values
    }

    choice_set_to_choice_key = {
        tuple(value): key for key, value in choice_key_to_choice_set.items()
    }

    return (
        state_to_fixed_key,
        variable_and_fixed_key_to_state,
        variable_key_to_choice_set,
        choice_set_to_choice_key,
        state_and_choice_to_state_choice
    )


def _create_choice_objects(state_space, model_options):
    choices = pd.Series(list(model_options["choices"]["choice_sets"].keys()))
    choice_sets = model_options["choices"]["choice_sets"]
    choice_sets_solution = model_options["choices"].get("choice_sets_solution", None)

    choice_set_df = pd.DataFrame(index=state_space.index)
    for choice, condition_list in choice_sets.items():
        filter_ = [state_space.eval(condition) for condition in condition_list]
        choice_set_df[choice] = functools.reduce(lambda x, y: x | y, filter_)

    # Bit ugly with the switch argument.
    if choice_sets_solution:
        choice_set_solution_df = pd.DataFrame(index=state_space.index)
        for choice, condition_list in choice_sets_solution.items():
            filter_ = [state_space.eval(condition) for condition in condition_list]
            choice_set_solution_df[choice] = functools.reduce(
                lambda x, y: x | y, filter_
            )

    choice_grouper = (
        choice_set_solution_df.groupby(list(choice_sets.keys()))
        if choice_sets_solution
        else choice_set_df.groupby(list(choice_sets.keys()))
    )
    choice_key_df = choice_grouper.ngroup()

    choice_key_to_choice_set = {
        key: choices[list(choice_set)].to_list()
        for key, choice_set in enumerate(choice_grouper.groups.keys())
    }

    sc_space_chunks = []
    for col in choice_set_df:
        im_df = state_space.copy()[choice_set_df[col]]
        im_df["choice"] = col
        sc_space_chunks.append(im_df)

    state_choice_space = pd.concat(sc_space_chunks)
    state_choice_space["state_index"] = state_choice_space.index
    state_choice_space.index = range(state_choice_space.shape[0])
    return state_choice_space, choice_key_df, choice_key_to_choice_set
