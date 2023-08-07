"""Entry point script"""
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

params = pd.read_csv("flexible_ddcm/example/base/params.csv").set_index(
    ["category", "name"]
)[["value"]]
model_options = yaml.safe_load(open("flexible_ddcm/example/base/specification.yaml"))
external_probabilities = pd.read_csv(
    "flexible_ddcm/example/base/external_probabilities.csv"
).drop(columns=["Unnamed: 0"])

cont = solve(
    params,
    model_options,
    transition_function_nonstandard,
    reward_function_nonstandard,
    map_transition_to_state_choice_entries_nonstandard,
)

simulate = get_simulate_func(
    model_options,
    transition_function_nonstandard,
    reward_function_nonstandard,
    extreme_value_shocks,
    external_probabilities,
    map_transition_to_state_choice_entries_nonstandard,
)

simulate_dict = simulate(params)

wage_periods = range(3, 16)
additional_cols = ["ability", "parental_income"]
schooling_levels = ["vmbo", "mbo3", "mbo4", "havo", "hbo"]

simulate_processed_dict = process_simulation_dict(
    simulate_dict, params, wage_periods, additional_cols, schooling_levels
)
