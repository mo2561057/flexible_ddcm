"""Entry point script"""
import numpy as np
import pandas as pd
import yaml

from flexible_ddcm.estimation_utils import process_simulation_dict
from flexible_ddcm.example.types.input_functions import ev_shocks_and_transition_costs
from flexible_ddcm.example.types.input_functions import initial_states
from flexible_ddcm.example.types.input_functions import (
    map_transition_to_state_choice_entries_nonstandard,
)
from flexible_ddcm.example.types.input_functions import reward_function_nonstandard
from flexible_ddcm.example.types.input_functions import transition_function_nonstandard
from flexible_ddcm.simulate import get_simulate_func
from flexible_ddcm.solve import solve
from flexible_ddcm.state_space import create_state_space

params = (
    pd.read_csv("flexible_ddcm/example/types/params.csv")
    .set_index(["category", "name"])["value"]
    .sort_index()
)
model_options = yaml.safe_load(open("flexible_ddcm/example/types/specification.yaml"))
external_probabilities = pd.read_csv(
    "flexible_ddcm/example/types/external_probabilities.csv"
).drop(columns=["Unnamed: 0"])

simulate = get_simulate_func(
    model_options,
    transition_function_nonstandard,
    reward_function_nonstandard,
    ev_shocks_and_transition_costs,
    map_transition_to_state_choice_entries_nonstandard,
    initial_states,
)

simulate_dict = simulate(params)

# wage_periods = range(3, 16)
# additional_cols = ["ability", "parental_income"]
# schooling_levels = ["vmbo", "mbo3", "mbo4", "havo", "hbo"]
#
# simulate_processed_dict = process_simulation_dict(
#    simulate_dict, params, wage_periods, additional_cols, schooling_levels
# )
