import functools

import numpy as np
import pandas as pd
import scipy.stats as stats
from flexible_ddcm.model_spec_utils import bundle_choice_transition_functions
from flexible_ddcm.shared import build_covariates, pandas_dot, sample_characteristics


def transition_graduation(state_space, params, choice):
    FILT_ = state_space.state_space.age < 2
    out = pd.DataFrame(index=state_space.state_space.index, columns=[0, 1])
    out.loc[FILT_, 0] = 1
    out.loc[FILT_, 1] = 0
    out.loc[~FILT_, 1] = params[("probability_degree", choice)]
    out.loc[~FILT_, 0] = 1 - params[("probability_degree", choice)]
    return out.values.astype(np.float32)


def perceived_transition_attendance_vec(
    state_space, params, choice, transition_graduation
):
    # Only 1 stochastic thing changes
    transition_graduation = pd.DataFrame(
        transition_graduation(state_space, params, choice)
    )
    transition_graduation.index = state_space.state_space.index
    # transition_graduation.columns = state_space.variable_state_space.index

    out = {}
    for variable_state, locs in state_space.state_space.groupby(
        "variable_key"
    ).groups.items():
        if choice not in state_space.variable_key_to_choice_set[variable_state]:
            continue
        # Rename axis
        out[(choice, variable_state)] = transition_graduation.loc[locs, :]
        out[(choice, variable_state)] = out[(choice, variable_state)].loc[
            :, ~(out[(choice, variable_state)].sum(axis=0) < 1e-10)
        ]

        # Update the state
        initial_state = tuple(state_space.variable_state_space.loc[variable_state])

        index_transition = [
            state_space.variable_state_space_indexer[
                (initial_state[0] + 1, *initial_state[1:-2], choice, ix)
            ]
            for ix in out[(choice, variable_state)].columns
        ]

        out[(choice, variable_state)].columns = index_transition
        out[(choice, variable_state)] = (
            out[(choice, variable_state)]
            .loc[:, ~(out[(choice, variable_state)] == 0).all(axis=0)]
            .dropna(axis=1)
        )
    return out


def work_transition_vectorized(state_space, params):
    out = {}
    for variable_state, locs in state_space.state_space.groupby(
        "variable_key"
    ).groups.items():
        if "work" not in state_space.variable_key_to_choice_set[variable_state]:
            continue
        # Rename axis
        out[("work", variable_state)] = pd.DataFrame(
            index=locs, columns=["terminal"], data=1
        )
    return out


def auxiliary_function(
    state_space,
    model_options,
    params,
    transition_function,
    reward_function,
    transition_graduation,
    perceived_transition_attendance_vec,
):
    """Function to modify and precalculate important values."""
    # Deal with the choice later

    subjective_transition = {
        choice: functools.partial(
            func_,
            transition_full=perceived_transition_attendance_vec(
                state_space, params, choice, transition_graduation
            ),
        )
        for choice, func_ in transition_function["subjective"].items()
        if choice
        in [
            "very_low_attendance",
            "low_attendance",
            "medium_attendance",
            "high_attendance",
        ]
    }

    subjective_transition["work"] = transition_function["subjective"]["work"]

    return (
        functools.partial(
            bundle_choice_transition_functions,
            choice_transition_functions=subjective_transition,
        ),
        reward_function,
    )


def transition_function_actual_macedonia(
    state_space,
    params,
    choice,
    transition_full,
):
    return transition_full


def approximate_normal_vec(mean, sigma, grid):
    """
    Approximates the distribution of 'mean + shock' over a discrete grid.

    Parameters:
    mean : float
        The deterministic mean value.
    sigma : float
        The standard deviation of the shock (normal distribution).
    grid : numpy.ndarray
        The grid of points over which to approximate the distribution.

    Returns:
    probabilities : numpy.ndarray
        The probabilities of 'mean + shock' falling within the intervals defined by the grid.
    """
    # Calculate the midpoint between each consecutive pair of grid points for binning
    midpoints = (grid[:-1] + grid[1:]) / 2
    # Extend the first and last midpoints to cover all possible values
    extended_midpoints = np.concatenate(([-np.inf], midpoints, [np.inf]))

    # Calculate the probabilities using the CDF of the normal distribution
    probabilities = np.diff(
        stats.norm.cdf(extended_midpoints[:, np.newaxis], loc=mean, scale=sigma), axis=0
    )
    return pd.DataFrame(index=grid, data=probabilities)


def map_to_closest_grid_point(values, grid):
    # Find the index of the closest grid point for each point in the series
    # np.abs(grid - point) gives an array of absolute differences between the grid points and the point
    # np.argmin finds the index of the minimum value in this array
    closest_indices = np.array([np.argmin(np.abs(grid - point)) for point in values])

    # Use these indices to return the closest grid points
    closest_points = grid[closest_indices]

    # Convert the resulting array of closest points back to a pandas Series
    return closest_points


def initial_expectation(model_options, params, external_probabilities, probability_df):
    # Assign fixed states.
    out = pd.DataFrame(
        index=range(model_options["n_simulation_agents"]),
        dtype="object",
    )

    # Assign all fixed states.
    for state, specs in model_options["state_space"].items():
        if specs["start"] not in ["random_external", "random_internal", "custom"]:
            out[state] = specs["start"]

    # Assign stochastic states
    np.random.seed(model_options["seed"] + 2_000_000)
    locs_external = np.random.choice(
        external_probabilities.index,
        p=external_probabilities["probability"],
        size=len(out),
        replace=True,
    )
    states_external = [
        col for col in external_probabilities.columns if col != "probability"
    ]

    out[states_external] = external_probabilities.loc[
        locs_external, states_external
    ].values

    out = build_covariates(out, model_options["covariates"])

    # Add estimated probabilities
    for col, specs in model_options["state_space"].items():
        if specs["start"] == "random_internal":
            out[col] = sample_characteristics(
                out, params, col, specs["list"], model_options["seed"] + 2_000_001
            )
    out = build_covariates(out, model_options["covariates"])

    unemployment_grid = np.array(
        model_options["state_space"]["unemployment_rate"]["list"]
    )
    expected_return_grid = np.array(
        model_options["state_space"]["expected_wage_primary"]["list"]
    )
    expected_return_secondary_grid = np.array(
        model_options["state_space"]["expected_wage_secondary"]["list"]
    )

    initial_unemployment = params[("unemployment_rate", "mean")] + np.random.normal(
        0, params.loc[("unemployment_rate", "std")], len(out)
    )

    out["unemployment_rate"] = map_to_closest_grid_point(
        initial_unemployment, unemployment_grid
    )

    # Draw initial values for lagged expectations
    # Put type probabilities here

    type_contribution_primary = (
        out["type_1"] * params[("initial_primary", "type_1")]
        + out["type_2"] * params[("initial_primary", "type_2")]
        + out["type_3"] * params[("initial_primary", "type_3")]
    )

    type_contribution_secondary = (
        out["type_1"] * params[("initial_secondary", "type_1")]
        + out["type_2"] * params[("initial_secondary", "type_2")]
        + out["type_3"] * params[("initial_secondary", "type_3")]
    )

    initial_expectations_primary = pd.Series(
        map_to_closest_grid_point(
            params[("initial_primary", "mean")]
            + type_contribution_primary
            + np.random.normal(0, params.loc[("initial_primary", "std")], len(out)),
            expected_return_grid,
        )
    )

    initial_expectations_secondary = pd.Series(
        map_to_closest_grid_point(
            params[("initial_secondary", "mean")]
            + type_contribution_secondary
            + np.random.normal(0, params.loc[("initial_secondary", "std")], len(out)),
            expected_return_secondary_grid,
        )
    )

    tuples_ = list(
        zip(
            initial_expectations_primary,
            initial_expectations_secondary,
            out["unemployment_rate"],
        )
    )
    probability_states = probability_df.loc[tuples_, :]

    # Sample one column for each row based on probabilities
    sampled_columns = pd.Series(
        [
            np.random.choice(probability_states.columns, p=row)
            for row in probability_states.values
        ],
        index=probability_states.index,
    )
    out["expected_wage_primary"] = sampled_columns.map(lambda x: x[0]).values
    out["expected_wage_secondary"] = sampled_columns.map(lambda x: x[1]).values
    return out


def initial_expectation_variance(
    model_options, params, external_probabilities, probability_df
):
    # Assign fixed states.
    out = pd.DataFrame(
        index=range(model_options["n_simulation_agents"]),
        dtype="object",
    )

    # Assign all fixed states.
    for state, specs in model_options["state_space"].items():
        if specs["start"] not in ["random_external", "random_internal", "custom"]:
            out[state] = specs["start"]

    # Assign stochastic states
    np.random.seed(model_options["seed"] + 2_000_000)
    locs_external = np.random.choice(
        external_probabilities.index,
        p=external_probabilities["probability"],
        size=len(out),
        replace=True,
    )
    states_external = [
        col for col in external_probabilities.columns if col != "probability"
    ]

    out[states_external] = external_probabilities.loc[
        locs_external, states_external
    ].values

    out = build_covariates(out, model_options["covariates"])

    # Add estimated probabilities
    for col, specs in model_options["state_space"].items():
        if specs["start"] == "random_internal":
            out[col] = sample_characteristics(
                out, params, col, specs["list"], model_options["seed"] + 2_000_001
            )
    out = build_covariates(out, model_options["covariates"])

    unemployment_grid = np.array(
        model_options["state_space"]["unemployment_rate"]["list"]
    )
    expected_return_grid = np.array(
        model_options["state_space"]["expected_wage_primary"]["list"]
    )
    expected_return_secondary_grid = np.array(
        model_options["state_space"]["expected_wage_secondary"]["list"]
    )

    initial_unemployment = params[("unemployment_rate", "mean")] + np.random.normal(
        0, params.loc[("unemployment_rate", "std")], len(out)
    )

    out["unemployment_rate"] = map_to_closest_grid_point(
        initial_unemployment, unemployment_grid
    )

    # Draw initial values for lagged expectations
    # Put type probabilities here

    type_contribution_primary = (
        out["type_1"] * params[("initial_primary", "type_1")]
        + out["type_2"] * params[("initial_primary", "type_2")]
        + out["type_3"] * params[("initial_primary", "type_3")]
    )

    type_contribution_secondary = (
        out["type_1"] * params[("initial_secondary", "type_1")]
        + out["type_2"] * params[("initial_secondary", "type_2")]
        + out["type_3"] * params[("initial_secondary", "type_3")]
    )

    initial_expectations_primary = pd.Series(
        map_to_closest_grid_point(
            params[("initial_primary", "mean")]
            + type_contribution_primary
            + np.random.normal(0, params.loc[("initial_primary", "std")], len(out)),
            expected_return_grid,
        )
    )

    initial_expectations_secondary = pd.Series(
        map_to_closest_grid_point(
            params[("initial_secondary", "mean")]
            + type_contribution_secondary
            + np.random.normal(0, params.loc[("initial_secondary", "std")], len(out)),
            expected_return_secondary_grid,
        )
    )

    tuples_ = list(
        zip(
            initial_expectations_primary,
            initial_expectations_secondary,
            out["unemployment_rate"],
        )
    )
    probability_states = probability_df.loc[tuples_, :]

    # Sample one column for each row based on probabilities
    sampled_columns = pd.Series(
        [
            np.random.choice(probability_states.columns, p=row)
            for row in probability_states.values
        ],
        index=probability_states.index,
    )
    out["expected_wage_primary"] = sampled_columns.map(lambda x: x[0]).values
    out["expected_wage_secondary"] = sampled_columns.map(lambda x: x[1]).values
    out["variance_primary"] = sampled_columns.map(lambda x: x[2]).values
    out["variance_secondary"] = sampled_columns.map(lambda x: x[3]).values
    return out


def create_next_period_df(
    simulation_df, transition, state_space, model_options, params, seed, probability_df
):
    """Next period df for simulations. transitions is only a placeholder here to comply with flexible_ddcm api """
    # CHeck for all states that are already terminal
    np.random.seed(seed)
    age = simulation_df.age.max()
    terminal_df = simulation_df[simulation_df.age < age]
    current_df = simulation_df[simulation_df.age == age]

    list_dfs = []
    for choice, locs in current_df.groupby("choice").groups.items():
        if choice == "work":
            list_dfs.append(simulation_df.loc[locs])
        else:
            list_dfs.append(
                next_period_state(
                    simulation_df.loc[locs],
                    probability_df,
                    state_space,
                    params,
                    model_options,
                    choice,
                )
            )
    out = pd.concat([terminal_df, *list_dfs])
    return out


# This name could be changed
def next_period_state(
    state_df,
    probability_df,
    state_space,
    params,
    model_options,
    choice,
):
    grid_unemployment = state_space.state_space.unemployment_rate.unique()

    out = state_df.copy()
    unemployment = params[("unemployment_rate", "mean")] + np.random.normal(
        0, params.loc[("unemployment_rate", "std")], len(out)
    )

    out["unemployment_rate"] = map_to_closest_grid_point(
        unemployment, grid_unemployment
    )

    tuples_ = list(
        zip(
            state_df["expected_wage_primary"],
            state_df["expected_wage_secondary"],
            out["unemployment_rate"])
    )

    probability_df = probability_df.loc[tuples_, :]

    # Sample one column for each row based on probabilities
    sampled_columns = pd.Series(
        [
            np.random.choice(probability_df.columns, p=row)
            for row in probability_df.values
        ],
        index=probability_df.index,
    )

    # Get the required columns. Maybe this can be modelled with model_options
    cols = probability_df.columns.names
    for i, name in enumerate(cols):
        out[name] = sampled_columns.map(lambda x: x[i]).values

    out["lagged_choice"] = choice

    prob_degree = state_df.age.map(
        lambda x: 0 if x < 2 else params[("probability_degree", choice)]
    )
    out["degree"] = np.random.binomial(1, prob_degree)
    out["age"] = state_df["age"] + 1
    state_keys = [
        state_space.state_space_indexer[tuple(arr)]
        for arr in out[list(model_options["state_space"].keys())].values
    ]
    out = state_space.state_space.loc[state_keys]
    out.index = state_df.index
    out["state_key"] = state_keys
    out = build_covariates(out, model_options["covariates"])
    return out


def perceived_lifetime_wages(state_space, params, discount_key, horizon=range(0, 40)):
    """Generate wages until the age of 50."""
    discount_factor = params[discount_key]
    # Initialize the dictionary to store calculated wages for each age
    wage_results = {}
    for age in horizon:
        # Prepare intermediate DataFrame
        modified_df = state_space.copy()
        modified_df = modified_df.rename(columns={"age": "age_start"})
        modified_df["age"] = age
        modified_df["exp"] = modified_df["age"] - modified_df["age_start"]
        # Initialize wages for the current age
        wage_results[age] = pd.Series(0, index=state_space.index).astype(float)
        if (modified_df.exp >= 0).any():
            valid_entries = modified_df[modified_df.exp >= 0]
            expected_wage = expected_wages(valid_entries, params)
            wage_results[age].loc[valid_entries.index] = expected_wage * (
                valid_entries.exp.map(lambda x: discount_factor**x)
            )
    # Sum up all calculated wages to a single DataFrame
    cumulative_wages = functools.reduce(lambda x, y: x + y, wage_results.values())
    return pd.DataFrame(cumulative_wages, columns=[0])


def perceived_lifetime_wage_utility(
    state_space, params, discount_key, horizon=range(0, 40)
):
    discount_factor = params[discount_key]
    # Initialize the dictionary to store calculated wages for each age
    wage_results = {}
    for age in horizon:
        # Prepare intermediate DataFrame
        modified_df = state_space.copy()
        modified_df = modified_df.rename(columns={"age": "age_start"})
        modified_df["age"] = age
        modified_df["exp"] = modified_df["age"] - modified_df["age_start"]
        # Initialize wages for the current age
        wage_results[age] = pd.Series(0, index=state_space.index).astype(float)
        if (modified_df.exp >= 0).any():
            valid_entries = modified_df[modified_df.exp >= 0]
            expected_wage = expected_utility_log_wages(valid_entries, params)
            wage_results[age].loc[valid_entries.index] = expected_wage * (
                valid_entries.exp.map(lambda x: discount_factor**x)
            )
    # Sum up all calculated wages to a single DataFrame
    cumulative_wages = functools.reduce(lambda x, y: x + y, wage_results.values())
    return pd.DataFrame(cumulative_wages, columns=[0])


# Check this alternative implementation
def work_rewards(state_space, params, discount_key, horizon=range(0, 40)):
    lt_wages = perceived_lifetime_wages(state_space, params, discount_key, horizon)

    filt = (state_space["age"] == 3) & (state_space["degree"] == 1)

    lt_wages.loc[filt, 0] = (
        lt_wages.loc[filt, 0].values * params[discount_key]
        + reward_school_attendance(
            state_space[filt], params, "medium_attendance"
        ).values
    )
    return lt_wages


# Check this alternative implementation
def work_rewards_log_utility(
        state_space, params, discount_key, horizon=range(0, 40)):
    lt_wages = perceived_lifetime_wage_utility(state_space, params, discount_key, horizon)

    filt = (state_space["age"] == 3) & (state_space["degree"] == 1)

    lt_wages.loc[filt, 0] = (
        lt_wages.loc[filt, 0].values * params[discount_key]
        + reward_school_attendance(
            state_space[filt], params, "medium_attendance"
        ).values
    )
    return lt_wages



def expected_wages(df, params):
    # Intoduces a lifecylce profile of expected wages.
    out = pd.Series(0, index=df.index, dtype=float)


    out[df.secondary == 1] = (
        np.exp(df.loc[df.primary == 0, "expected_wage_secondary"]) * 12
    )

    out[df.primary == 1] = np.exp(df.loc[df.primary == 1, "expected_wage_primary"]) * 12

    return out


def expected_utility_log_wages(df, params):
    out = pd.Series(0, index=df.index, dtype=float)

    secondary_variance_correction = (np.exp(df.loc[df.secondary == 1, "variance_secondary"] ))/(
        np.exp(df.loc[df.secondary == 1, "expected_wage_seconday"])**2
    )
    #secondary = 
    out[df.secondary == 1] = 12 + df.loc[
        df.secondary == 1, "expected_wage_secondary"] - secondary_variance_correction

    primary_variance_correction = (np.exp(df.loc[df.primary == 1, "variance_primary"] ))/(
        np.exp(df.loc[df.primary == 1, "expected_wage_primary"])**2
    )

    out[df.primary == 1] = df.loc[
        df.primary == 1, "expected_wage_primary"] + 12 - primary_variance_correction
    return out


def reward_school_attendance(state_space, params, choice):
    if f"nonpec_{choice}" in params.index.get_level_values(0):
        nonpec_school = pandas_dot(
            state_space, params.loc["nonpec_school"]
        ) + pandas_dot(state_space, params.loc[f"nonpec_{choice}"])
    else:
        nonpec_school = pandas_dot(state_space, params.loc["nonpec_school"])
    outside_option = (
        np.exp(state_space["expected_wage_primary"])
        * 12
        * (params.loc["informal_work_penalty", choice])
    )

    time_share = params.loc["time_share", choice]
    return time_share * nonpec_school + (1 - time_share) * outside_option
