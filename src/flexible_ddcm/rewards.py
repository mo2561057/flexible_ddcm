"""
Calculate rewards based on model specification.
"""
import numpy as np


def calculate_rewards_state_choice_space(state_choice_df, params, reward_function):
    rewards = reward_function(state_choice_df, params)["value"]
    out = np.zeros(state_choice_df.index.max() + 1)
    out[state_choice_df.index.values] = rewards.loc[
        state_choice_df.index.values].values
    return out
