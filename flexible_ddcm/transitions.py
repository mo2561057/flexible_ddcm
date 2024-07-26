"""Manages transitions."""
import numpy as np


def build_transition_func_from_params(params, state_space, transition_function):
    grouper = state_space.state_space.groupby(["variable_key"]).groups
    out = {}
    for variable_key, locs in grouper.items():
        for choice in state_space.variable_key_to_choice_set[variable_key]:
            variable_state = tuple(state_space.variable_state_space.loc[variable_key])
            out[(choice, variable_key)] = transition_function(
                state_space.state_space.loc[locs], choice, params, variable_state
            )
    return _process_transitions(out, state_space)


def _process_transitions(transition_dict, state_space):
    out = dict()
    for key, trans_df in transition_dict.items():
        im = trans_df.copy().rename(
            columns={
                col: state_space.variable_state_space_indexer[col]
                if col != "terminal"
                else col
                for col in trans_df
            }
        )
        out[key] = im
    return out
