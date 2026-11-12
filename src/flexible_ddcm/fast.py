"""Vectorized solution and simulation.

Same model objects as solve.py / simulate.py (state space, reward functions,
shock functions, initial states), but transitions are stored per choice as flat
arrays ``(state, next_state, prob)`` instead of one DataFrame per
(choice, variable_key). Backward induction is a handful of numpy operations per
age and the simulation samples next states for all agents at once.

Transition functions may return either
  * the legacy format: dict {(choice, variable_key): DataFrame} with rows =
    states and columns = arrival variable keys (or "terminal"), or
  * the long format: DataFrame with columns ["state", "next_variable_key",
    "prob"], where next_variable_key == -1 means terminal.

Restrictions compared to solve.py: one period per transition (rewards are
collected at the origin state only), which is what all yearly models use.
"""
import functools
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from flexible_ddcm.rewards import calculate_rewards_state_choice_space
from flexible_ddcm.shared import build_covariates
from flexible_ddcm.state_space import create_state_space

ChoiceTransitions = namedtuple(
    "ChoiceTransitions", ["state", "next_state", "prob", "indptr"]
)
FastSolution = namedtuple(
    "FastSolution",
    ["values", "continuation_values", "choices", "available", "transitions"],
)


# --------------------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------------------
def build_transitions_fast(
    params, model_options, state_space, transition_function, states_subset=None
):
    """Evaluate transition_function for every choice and store flat arrays.

    Args:
        states_subset: optional boolean array over states; rows of other origin
            states are dropped (used when solving on representative states only).
    """
    n_states = len(state_space.state_space)
    fixed_keys = state_space.state_to_fixed_key
    to_state = state_space.variable_and_fixed_key_to_state

    out = {}
    for choice in model_options["choices"]["choice_sets"]:
        raw = transition_function(state_space, choice, params)
        state, next_variable_key, prob = _to_long(raw)

        if states_subset is not None:
            keep = states_subset[state]
            state, next_variable_key, prob = (
                state[keep], next_variable_key[keep], prob[keep]
            )

        terminal = next_variable_key < 0
        next_state = np.full(len(state), -1, dtype=np.int64)
        next_state[~terminal] = to_state[
            fixed_keys[state[~terminal]], next_variable_key[~terminal]
        ]
        invalid = (~terminal) & (next_state < 0) & (prob > 0)
        if invalid.any():
            raise ValueError(
                f"Choice {choice}: {invalid.sum()} transitions with positive "
                "probability lead to states removed by the state space filter, "
                f"e.g. from state {state[invalid][0]}."
            )

        order = np.argsort(state, kind="stable")
        state, next_state, prob = state[order], next_state[order], prob[order]
        indptr = np.zeros(n_states + 1, dtype=np.int64)
        np.cumsum(np.bincount(state, minlength=n_states), out=indptr[1:])

        row_sums = np.bincount(state, weights=prob, minlength=n_states)
        has_rows = np.diff(indptr) > 0
        if not np.allclose(row_sums[has_rows], 1):
            bad = np.flatnonzero(has_rows & ~np.isclose(row_sums, 1))[0]
            raise ValueError(
                f"Choice {choice}: transition probabilities of state {bad} sum "
                f"to {row_sums[bad]}."
            )
        out[choice] = ChoiceTransitions(state, next_state, prob, indptr)
    return out


def _to_long(raw):
    """Convert either transition format to flat arrays."""
    if isinstance(raw, pd.DataFrame):
        return (
            raw["state"].to_numpy(dtype=np.int64),
            raw["next_variable_key"].to_numpy(dtype=np.int64),
            raw["prob"].to_numpy(dtype=float),
        )

    states, next_keys, probs = [], [], []
    for trans in raw.values():
        values = trans.to_numpy(dtype=float)
        columns = np.array(
            [-1 if col == "terminal" else col for col in trans.columns],
            dtype=np.int64,
        )
        states.append(np.repeat(trans.index.to_numpy(dtype=np.int64), len(columns)))
        next_keys.append(np.tile(columns, len(trans)))
        probs.append(values.reshape(-1))
    if not states:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=float)
    return np.concatenate(states), np.concatenate(next_keys), np.concatenate(probs)


# --------------------------------------------------------------------------------------
# Solution
# --------------------------------------------------------------------------------------
def representative_states(state_space, invariant_states):
    """Map every state to the state with invariant_states set to their first level.

    States that only differ in variables the agent does not observe (and that do
    not enter subjective transitions or rewards) have identical value functions,
    so it suffices to solve the model for one representative.
    """
    if state_space.state_space_indexer is None:
        return _representative_states_lean(state_space, invariant_states)
    df = state_space.state_space[state_space.states].copy()
    for col in invariant_states:
        df[col] = sorted(state_space.state_space[col].unique())[0]
    keys = df.itertuples(index=False, name=None)
    return np.fromiter(
        (state_space.state_space_indexer[key] for key in keys),
        dtype=np.int64,
        count=len(df),
    )


_REPRESENTATIVE_CACHE = {}


def _representative_states_lean(state_space, invariant_states):
    """representative_states via (fixed key, variable key), for fixed invariant
    states. Does not depend on params, so it is cached per state space."""
    key = (id(state_space), tuple(invariant_states))
    if key in _REPRESENTATIVE_CACHE:
        return _REPRESENTATIVE_CACHE[key]
    if not set(invariant_states) <= set(state_space.fixed_states):
        raise ValueError("lean state spaces need invariant states that are fixed")
    fixed = state_space.fixed_states
    first_level = {
        col: min(value[fixed.index(col)] for value in state_space.fixed_state_space_indexer)
        for col in invariant_states
    }
    representative_fixed = np.empty(len(state_space.fixed_state_space_indexer), dtype=np.int64)
    for fixed_tuple, fixed_key in state_space.fixed_state_space_indexer.items():
        values = dict(zip(fixed, fixed_tuple))
        values.update(first_level)
        representative_fixed[fixed_key] = state_space.fixed_state_space_indexer[
            tuple(values[col] for col in fixed)
        ]
    ss = state_space.state_space
    out = state_space.variable_and_fixed_key_to_state[
        representative_fixed[ss["fixed_key"].to_numpy()], ss["variable_key"].to_numpy()
    ].astype(np.int64)
    _REPRESENTATIVE_CACHE[key] = out
    return out


def solve_fast(
    params,
    model_options,
    state_space,
    transition_function,
    reward_function,
    invariant_states=None,
    segmentation_column="age",
):
    """Backward induction over ages.

    Returns:
        FastSolution with
          values: (n_states, n_choices) choice specific values (-inf if the
            choice is not available),
          continuation_values: (n_states,) expected value before shocks,
          choices: list of choice names (column order of values),
          available: (n_states, n_choices) bool,
          transitions: dict choice -> ChoiceTransitions used in the solution.
    """
    ss = state_space.state_space
    n_states = len(ss)
    choices = list(model_options["choices"]["choice_sets"])
    choice_pos = {choice: pos for pos, choice in enumerate(choices)}
    discount = params[("discount", "discount")]
    scale = params[("ev_shocks", "scale")]

    if invariant_states:
        representative = representative_states(state_space, invariant_states)
        solve_mask = representative == np.arange(n_states)
    else:
        representative = np.arange(n_states)
        solve_mask = np.ones(n_states, dtype=bool)

    # State choice positions.
    sc_space = state_space.state_choice_space
    sc_state = sc_space["state_key"].to_numpy(dtype=np.int64)
    sc_choice = sc_space["choice"].map(choice_pos).to_numpy(dtype=np.int64)
    available = np.zeros((n_states, len(choices)), dtype=bool)
    available[sc_state, sc_choice] = True
    sc_position = np.full((n_states, len(choices)), -1, dtype=np.int64)
    sc_position[sc_state, sc_choice] = sc_space.index.to_numpy()

    rewards = calculate_rewards_state_choice_space(
        sc_space[solve_mask[sc_state]], params, reward_function
    )

    transitions = build_transitions_fast(
        params, model_options, state_space, transition_function, solve_mask
    )

    ages = ss[segmentation_column].to_numpy()
    values = np.full((n_states, len(choices)), -np.inf)
    continuation = np.full(n_states, np.nan)

    # Pre-split each choice's transition rows by the age of the origin state.
    rows_by_age = {}
    for choice, trans in transitions.items():
        origin_age = ages[trans.state]
        order = np.argsort(origin_age, kind="stable")
        rows_by_age[choice] = (order, origin_age[order])

    for age in np.sort(np.unique(ages))[::-1]:
        at_age = np.flatnonzero((ages == age) & solve_mask)
        for choice, trans in transitions.items():
            pos = choice_pos[choice]
            order, sorted_age = rows_by_age[choice]
            lo = np.searchsorted(sorted_age, age, side="left")
            hi = np.searchsorted(sorted_age, age, side="right")
            rows = order[lo:hi]
            states_c = at_age[available[at_age, pos]]
            if len(states_c) == 0:
                continue
            next_state = trans.next_state[rows]
            prob = trans.prob[rows]
            next_value = np.zeros(len(rows))
            nonterminal = (next_state >= 0) & (prob > 0)
            next_value[nonterminal] = continuation[next_state[nonterminal]]
            if np.isnan(next_value).any():
                bad = trans.state[rows][np.isnan(next_value)][0]
                raise ValueError(
                    f"State {bad}, choice {choice}: next state has no value. Does "
                    f"it lie in a later {segmentation_column} with an empty choice set?"
                )
            expected = np.bincount(
                trans.state[rows], weights=prob * next_value, minlength=n_states
            )
            values[states_c, pos] = (
                rewards[sc_position[states_c, pos]] + discount * expected[states_c]
            )

        has_choice = available[at_age].any(axis=1)
        continuation[at_age[has_choice]] = scale * logsumexp(
            values[at_age[has_choice]] / scale, axis=1
        )

    if invariant_states:
        values = values[representative]
        continuation = continuation[representative]

    return FastSolution(values, continuation, choices, available, transitions)


def solution_to_legacy(solution, state_space):
    """Choice specific value functions in the format of solve.solve."""
    ss = state_space.state_space
    out = {}
    for choice_key, choice_set in state_space.choice_key_to_choice_set.items():
        if not choice_set:
            continue
        locs = ss.index[ss.choice_key == choice_key]
        cols = [solution.choices.index(c) for c in choice_set]
        out[choice_key] = pd.DataFrame(
            solution.values[np.ix_(locs, cols)], index=locs, columns=choice_set
        )
    continuation = pd.DataFrame(
        solution.continuation_values, columns=["continuation_value"]
    ).dropna()
    return continuation, out


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
def get_simulate_func_fast(
    model_options,
    transition_function,
    reward_function,
    shock_function,
    initial_states,
    invariant_states=None,
    state_space=None,
    simulation_only_params=None,
):
    """Build the state space once and return simulate(params).

    simulation_only_params: params categories (first index level) that only enter
    the simulation (e.g. costs added to the shocks of the initial choice). If two
    consecutive calls differ only in these, the solution and the simulation
    transitions of the previous call are reused.
    """
    state_space = create_state_space(model_options) if state_space is None else state_space
    return functools.partial(
        simulate_fast,
        state_space=state_space,
        model_options=model_options,
        transition_function=transition_function,
        reward_function=reward_function,
        shock_function=shock_function,
        initial_states=initial_states,
        invariant_states=invariant_states,
        simulation_only_params=tuple(simulation_only_params or ()),
        solution_cache={},
    )


def simulate_fast(
    params,
    state_space,
    model_options,
    transition_function,
    reward_function,
    shock_function,
    initial_states,
    invariant_states=None,
    return_solution=False,
    simulation_only_params=(),
    solution_cache=None,
):
    """Solve and simulate.

    transition_function: callable, or dict with keys "subjective" (used in the
    solution) and "objective" (used to draw next states in the simulation).

    Returns dict period -> DataFrame in the same format as simulate.simulate:
    every agent appears in every period; agents that made a terminal choice
    are carried over with their terminal row. Columns of the initial states that
    are not state space columns (e.g. school type) are kept in every period.
    """
    subjective = (
        transition_function["subjective"]
        if isinstance(transition_function, dict)
        else transition_function
    )
    cache_key = _solution_cache_key(params, simulation_only_params)
    if solution_cache is not None and cache_key in solution_cache:
        solution, sim_transitions = solution_cache[cache_key]
    else:
        if solution_cache is not None:
            solution_cache.clear()  # never hold two solutions in memory
        solution = solve_fast(
            params, model_options, state_space, subjective, reward_function,
            invariant_states,
        )
        sim_transitions = (
            build_transitions_fast(
                params, model_options, state_space, transition_function["objective"]
            )
            if isinstance(transition_function, dict)
            else solution.transitions
        )
        if solution_cache is not None:
            solution_cache[cache_key] = (solution, sim_transitions)
    terminal_choices = set(model_options["choices"].get("terminal_choices", []))

    model_options = {
        **model_options,
        "first_period_covariates": {
            **model_options.get("covariates", {}),
            **model_options.get("covariates_simulation", {}),
        },
    }
    ss = state_space.state_space
    current = initial_states(model_options, params).copy()
    current.index.name = "Identifier"
    current["state_key"] = _state_keys(current, state_space)
    current = build_covariates(current, model_options["first_period_covariates"])
    persistent_cols = [c for c in current if c not in ss.columns and c != "state_key"]
    persistent = current[persistent_cols]
    current = _state_rows(ss, current["state_key"].to_numpy(), current.index, persistent)
    current["choice"] = np.nan
    current["choice"] = current["choice"].astype(object)

    seed = model_options["seed"]
    simulation_data = {}
    finished = pd.DataFrame()
    period = 0
    while len(current):
        current = _draw_choices(
            current, solution, state_space, params, shock_function, period, seed
        )
        simulation_data[period] = pd.concat([current, finished]).sort_index()

        is_terminal = current["choice"].isin(terminal_choices)
        finished = pd.concat([finished, current[is_terminal]])
        moving = current[~is_terminal]
        next_state = _draw_next_states(
            moving, sim_transitions, np.random.default_rng(seed + period + 200)
        )
        current = _state_rows(ss, next_state, moving.index, persistent)
        current["choice"] = np.nan
        current["choice"] = current["choice"].astype(object)
        period += 1

    if return_solution:
        return simulation_data, solution
    return simulation_data


def _solution_cache_key(params, simulation_only_params):
    """Hashable key of all params that enter the solution."""
    relevant = params
    if simulation_only_params and isinstance(params.index, pd.MultiIndex):
        relevant = params.drop(index=list(simulation_only_params), level=0, errors="ignore")
    return tuple(relevant.index), tuple(relevant.to_numpy().tolist())


def _state_keys(df, state_space):
    if state_space.state_space_indexer is None:
        variable = [col for col in state_space.states if col not in state_space.fixed_states]
        variable_keys = np.fromiter(
            (state_space.variable_state_space_indexer[key]
             for key in df[variable].itertuples(index=False, name=None)),
            dtype=np.int64, count=len(df),
        )
        fixed_keys = np.fromiter(
            (state_space.fixed_state_space_indexer[key]
             for key in df[state_space.fixed_states].itertuples(index=False, name=None)),
            dtype=np.int64, count=len(df),
        )
        return state_space.variable_and_fixed_key_to_state[fixed_keys, variable_keys]
    keys = df[state_space.states].itertuples(index=False, name=None)
    return np.fromiter(
        (state_space.state_space_indexer[key] for key in keys),
        dtype=np.int64,
        count=len(df),
    )


def _state_rows(ss, state_keys, index, persistent):
    out = ss.iloc[state_keys].copy()
    out.index = index
    out["state_key"] = state_keys
    return out.join(persistent.loc[index]) if len(persistent.columns) else out


def _draw_choices(df, solution, state_space, params, shock_function, period, seed):
    """Add choice and shock_<choice> columns, grouped by choice set as in simulate."""
    parts = []
    for choice_key, group in df.groupby("choice_key"):
        choice_set = state_space.choice_key_to_choice_set[choice_key]
        cols = [solution.choices.index(c) for c in choice_set]
        value = pd.DataFrame(
            solution.values[np.ix_(group["state_key"].to_numpy(), cols)],
            index=group.index,
            columns=choice_set,
        )
        # Offset the seed by choice key: with a common seed, agents at the same
        # position in different choice-set groups would get identical shocks.
        shocks, information = shock_function(
            value, group, params, period, seed + 7919 * int(choice_key)
        )
        total = value.to_numpy(dtype=float) + shocks.to_numpy(dtype=float)
        part = shocks.astype(float).rename(columns=lambda c: f"shock_{c}")
        part["choice"] = np.array(choice_set, dtype=object)[np.argmax(total, axis=1)]
        if information is not None:
            part = part.join(information)
        parts.append(part)
    drawn = pd.concat(parts)
    out = df.drop(columns=["choice"]).join(drawn)
    return out


def _draw_next_states(df, transitions, rng):
    next_state = np.empty(len(df), dtype=np.int64)
    states = df["state_key"].to_numpy()
    choice = df["choice"].to_numpy()
    for name, trans in transitions.items():
        mask = choice == name
        if not mask.any():
            continue
        start = trans.indptr[states[mask]]
        stop = trans.indptr[states[mask] + 1]
        if (stop <= start).any():
            raise ValueError(f"Choice {name} chosen in a state without transitions.")
        cumulative = np.cumsum(trans.prob)
        before = np.where(start > 0, cumulative[np.maximum(start - 1, 0)], 0.0)
        total = cumulative[stop - 1] - before
        target = before + rng.random(mask.sum()) * total
        pos = np.searchsorted(cumulative, target, side="right")
        pos = np.clip(pos, start, stop - 1)
        next_state[mask] = trans.next_state[pos]
    if (next_state < 0).any():
        raise ValueError("Non terminal choice drew a terminal transition.")
    return next_state
