"""
Calculate rewards based on model specification.
How should I think about these rewards. How much 
do I want to hardcode?
"""
import functools
import itertools

import numpy as np
import pandas as pd


def calculate_rewards_state_choice_space(state_choice_df, params, reward_function):
    return reward_function(state_choice_df, params)
