"""Entry point script"""
import os

import numpy as np
import pandas as pd
import yaml
import time

from src.model.example.input_estimation import map_simulation_dict_in_long_df
from src.model.example.input_functions import reward_function
from src.model.example.input_functions import transition_function
from src.model.simulate import get_simulate_func
from src.model.state_space import create_state_space

params = pd.read_csv("src/model/example/params.csv").set_index(["category", "name"])
model_options = yaml.safe_load(open("src/model/example/specification.yaml"))
external_probabilities = pd.read_csv(
    "src/model/example/external_probabilities.csv"
).drop(columns=["Unnamed: 0"])

state_space = create_state_space(
    model_options
)

simulate = get_simulate_func(
    model_options, transition_function, reward_function, external_probabilities
)

start = time.time()
simulate_dict = simulate(params)
stop = time.time()

dur = stop-start