"""
Calculate rewards based on model specification.
How should I think about these rewards.
"""
def calculate_rewards_state_choice_space(
        state_choice_df,
        params,
        reward_function):
    return reward_function(state_choice_df, params)
