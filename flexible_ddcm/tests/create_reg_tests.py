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


def create_reg_vault(params, model_options, external_probabilities, seed):

    model_options["seed"] = seed
    simulate = get_simulate_func(
        model_options,
        transition_function_nonstandard,
        reward_function_nonstandard,
        extreme_value_shocks,
        external_probabilities,
        map_transition_to_state_choice_entries_nonstandard,
    )

    simulate_dict = simulate(params)

    vault = (simulate_dict, params, model_options, external_probabilities, seed)

    with open("flexible_ddcm/tests/resources/reg_vault.pkl", "wb") as writer:
        pkl.dump(vault, writer)


if __name__ == "__main__":
    params = pd.read_csv("flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )["value"]
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml")
    )
    external_probabilities = pd.read_csv(
        "flexible_ddcm/tests/resources/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])

    seed_base = 897
    create_reg_vault(params, model_options, external_probabilities, seed_base)
