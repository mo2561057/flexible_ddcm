"""Manages transitions."""
import numpy as np


def build_transition_func_from_params(
    params, model_options, state_space, transition_function
):
    out = {}
    for choice in model_options["choices"]["choice_sets"]:
        out = {**out, **transition_function(state_space, choice, params)}
    return _process_transitions(out, state_space)


def _process_transitions(transition_dict, state_space):
    out = dict()
    for key, trans_df in transition_dict.items():
        trans_df.columns = [
            state_space.variable_state_space_indexer[tuple(col)]
            if col != "terminal"
            else col
            for col in trans_df.columns
        ]

        # Check whether probabilities are close to one.
        assert np.allclose(trans_df.sum(axis=1), np.repeat(1, trans_df.shape[0]))

        out[key] = trans_df

    return out
