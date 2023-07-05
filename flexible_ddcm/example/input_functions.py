import functools

from flexible_ddcm.model_spec_utils import nonstandard_academic_risk
from flexible_ddcm.model_spec_utils import poisson_length
from flexible_ddcm.model_spec_utils import work_transition
from flexible_ddcm.model_spec_utils import nonpecuniary_reward
from flexible_ddcm.model_spec_utils import lifetime_wages
from flexible_ddcm.model_spec_utils import transition_function
from flexible_ddcm.model_spec_utils import reward_function
from flexible_ddcm.model_spec_utils import map_transition_to_state_choice_entries
from flexible_ddcm.model_spec_utils import between_states_age_variable



choice_transition_functions_nonstandard = {
    "havo": nonstandard_academic_risk,
    "mbo4": nonstandard_academic_risk,
    "mbo3": poisson_length,
    "hbo": nonstandard_academic_risk,
    "vocational_work": work_transition,
    "academic_work": work_transition,
}

choice_reward_functions_nonstandard = {
    "mbo4": functools.partial(nonpecuniary_reward, subset="nonpec_mbo4"),
    "havo": functools.partial(nonpecuniary_reward, subset="nonpec_havo"),
    "mbo3": functools.partial(nonpecuniary_reward, subset="nonpec_mbo3"),
    "hbo": functools.partial(nonpecuniary_reward, subset="nonpec_hbo"),
    "vocational_work": functools.partial(
        lifetime_wages,
        nonpec_key="nonpec_vocational",
        wage_key="wage_vocational",
        discount_key=("discount", "discount"),
    ),
    "academic_work": functools.partial(
        lifetime_wages,
        nonpec_key="nonpec_academic",
        wage_key="wage_academic",
        discount_key=("discount", "discount"),
    ),
}


transition_function_nonstandard = functools.partial(
    transition_function,
    choice_transition_functions=choice_transition_functions_nonstandard
    )

reward_function_nonstandard = functools.partial(
    reward_function,
    choice_reward_functions=choice_reward_functions_nonstandard
    )

map_transition_to_state_choice_entries_nonstandard = functools.partial(
    map_transition_to_state_choice_entries,
    get_between_states=between_states_age_variable)
