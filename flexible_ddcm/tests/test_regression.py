"""Entry point script"""
import pickle as pkl

import numpy as np
import pandas as pd
import yaml

from flexible_ddcm.estimation_utils import process_simulation_dict
from flexible_ddcm.example.base.input_functions import (
    map_transition_to_state_choice_entries_nonstandard,
)
from flexible_ddcm.example.base.input_functions import reward_function_nonstandard
from flexible_ddcm.example.base.input_functions import transition_function_nonstandard
from flexible_ddcm.model_spec_utils import extreme_value_shocks
from flexible_ddcm.simulate import get_simulate_func
from flexible_ddcm.solve import solve
from flexible_ddcm.state_space import create_state_space


def test_regression_base():
    simulate_dict, params, model_options, external_probabilities, seed = pd.read_pickle(
        "flexible_ddcm/tests/resources/reg_vault.pkl"
    )
    np.random.seed(seed)
    simulate = get_simulate_func(
        model_options,
        transition_function_nonstandard,
        reward_function_nonstandard,
        extreme_value_shocks,
        external_probabilities,
        map_transition_to_state_choice_entries_nonstandard,
    )

    simulate_dict_actual = simulate(params)

    # Compare vault top new thing:
    key_ = max(simulate_dict.keys())

    assert simulate_dict[key_].age.mean() == simulate_dict_actual[key_].age.mean()
