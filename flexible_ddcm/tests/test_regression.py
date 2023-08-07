"""Entry point script"""
import numpy as np
import pandas as pd
import yaml

from flexible_ddcm.estimation_utils import process_simulation_dict
from flexible_ddcm.example.types.input_functions import (
    map_transition_to_state_choice_entries_nonstandard,
)
from flexible_ddcm.example.types.input_functions import reward_function_nonstandard
from flexible_ddcm.example.types.input_functions import transition_function_nonstandard
from flexible_ddcm.simulate import get_simulate_func
from flexible_ddcm.solve import solve
from flexible_ddcm.state_space import create_state_space


def create_reg_vault(params, model_options, external_probabilities, seed):

    np.random.seed(seed)
    simulate = get_simulate_func(
        model_options,
        transition_function_nonstandard,
        reward_function_nonstandard,
        external_probabilities,
        map_transition_to_state_choice_entries_nonstandard,
    )

    simulate_dict = simulate(params)
    vault = (simulate_dict, params, model_options, external_probabilities)

    # with open("")


def test_regression():
    pass
