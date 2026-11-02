"""Vectorized drop-in replacement for ``flexible_ddcm.solve.solve``.

``solve`` loops over every (age, variable_key) cell and does a handful of small
pandas operations per cell, and re-checks on every call that each transition row
sums to one. ``solve_fast`` does the backward induction one age at a time for all
states of that age at once:

* The transition structure is flattened once into numpy arrays (one entry per
  (state, arrival) pair: row position, arrival state or -1 for terminal,
  probability, reward index). The flattened "plan" is cached on the identity of
  the transition DataFrames, so models that build their transitions once (e.g.
  in an auxiliary function) only pay for it on the first call. Row sums are
  checked when the plan is built.
* Rewards and continuation values are recomputed on every call.

Same assumptions as ``solve``: rewards are collected at the origin state only,
and every transition moves to the next age (or to "terminal").

The output has the same content as that of ``solve``: the continuation values
are identical, and the choice-specific value frames contain the same values but
with rows in a different order (they are only ever accessed with ``.loc``).
Use it via ``get_simulate_func(..., fast_solve=True)`` or call it directly.
"""
import numpy as np
import pandas as pd
import scipy.special
from flexible_ddcm.rewards import calculate_rewards_state_choice_space

_PLAN_CACHE = {}


def solve_fast(
    params,
    model_options,
    state_space,
    transition_function,
    reward_function,
    map_transition_to_state_choice_entries=None,
):
    """Solve the model; same signature and return values as ``solve``.

    ``map_transition_to_state_choice_entries`` is accepted for compatibility and
    not used (``solve`` does not use it either).
    """
    transitions = {}
    for choice in model_options["choices"]["choice_sets"]:
        transitions.update(transition_function(state_space, choice, params))

    rewards = calculate_rewards_state_choice_space(
        state_space.state_choice_space, params, reward_function
    )
    plan = _get_plan(transitions, state_space)

    discount = params[("discount", "discount")]
    scale = params[("ev_shocks", "scale")]

    continuation_values = np.full(state_space.state_space.index.max() + 1, np.nan)
    value_parts = {
        choice_key: []
        for choice_key, choice_set in state_space.choice_key_to_choice_set.items()
        if choice_set
    }

    for age_groups in plan:
        for choice_key, choices, states, entries in age_groups:
            values = np.empty((len(states), len(choices)))
            for j, (row_pos, arrival, prob, reward_index) in enumerate(entries):
                cont = np.zeros(len(arrival))
                not_terminal = arrival >= 0
                cont[not_terminal] = continuation_values[arrival[not_terminal]]
                values[:, j] = np.bincount(
                    row_pos,
                    weights=prob * (rewards[reward_index] + cont * discount),
                    minlength=len(states),
                )
            continuation_values[states] = scale * scipy.special.logsumexp(
                values / scale, axis=1
            )
            value_parts[choice_key].append(
                pd.DataFrame(values, index=states, columns=choices)
            )

    choice_specific_value_function = {
        key: pd.concat(value) for key, value in value_parts.items() if value
    }
    continuation_values = pd.DataFrame(
        continuation_values, columns=["continuation_value"]
    ).dropna(axis=0)
    return continuation_values, choice_specific_value_function, transitions


def _get_plan(transitions, state_space):
    key = (id(state_space), tuple(id(df) for df in transitions.values()))
    if key not in _PLAN_CACHE:
        _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = _build_plan(transitions, state_space)
    return _PLAN_CACHE[key]


def _build_plan(transitions, state_space, segmentation_column="age"):
    """Flatten transitions into arrays, grouped by age (descending) and choice set."""
    ss = state_space.state_space
    vk_to_states = ss.groupby("variable_key").groups
    plan = []
    for _, locs_age in sorted(
        ss.groupby(segmentation_column).groups.items(), reverse=True
    ):
        age_groups = []
        ss_age = ss.loc[locs_age]
        for choice_key, locs_ck in ss_age.groupby("choice_key").groups.items():
            choices = list(state_space.choice_key_to_choice_set[choice_key])
            if not choices:
                continue
            variable_keys = ss_age.loc[locs_ck, "variable_key"].unique()
            states = np.concatenate(
                [np.asarray(vk_to_states[vk], dtype=int) for vk in variable_keys]
            )
            position = pd.Series(np.arange(len(states)), index=states)
            entries = []
            for choice in choices:
                rows, arrivals, probs = [], [], []
                for vk in variable_keys:
                    trans = transitions[(choice, vk)]
                    row_states = trans.index.to_numpy(dtype=int)
                    p = trans.to_numpy(dtype=float)
                    if not np.allclose(p.sum(axis=1), 1):
                        raise ValueError(
                            f"Transition probabilities for {(choice, vk)} "
                            "do not sum to one."
                        )
                    if "terminal" in trans.columns:
                        arr = np.full(p.shape, -1, dtype=int)
                    else:
                        fixed = state_space.state_to_fixed_key[row_states]
                        arr = state_space.variable_and_fixed_key_to_state[
                            np.ix_(fixed, trans.columns.to_numpy(dtype=int))
                        ]
                    rows.append(np.repeat(row_states, p.shape[1]))
                    arrivals.append(arr.ravel())
                    probs.append(p.ravel())
                rows = np.concatenate(rows)
                reward_index = state_space.state_and_choice_to_state_choice[choice][
                    rows
                ].astype(int)
                entries.append(
                    (
                        position.loc[rows].to_numpy(),
                        np.concatenate(arrivals),
                        np.concatenate(probs),
                        reward_index,
                    )
                )
            age_groups.append((choice_key, choices, states, entries))
        plan.append(age_groups)
    return plan
