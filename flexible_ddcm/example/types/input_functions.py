import functools

import numpy as np

from flexible_ddcm.model_spec_utils import between_states_age_variable
from flexible_ddcm.model_spec_utils import lifetime_wages
from flexible_ddcm.model_spec_utils import map_transition_to_state_choice_entries
from flexible_ddcm.model_spec_utils import nonpecuniary_reward
from flexible_ddcm.model_spec_utils import nonstandard_academic_risk
from flexible_ddcm.model_spec_utils import fixed_length_nonstandard
from flexible_ddcm.model_spec_utils import poisson_length
from flexible_ddcm.model_spec_utils import reward_function
from flexible_ddcm.model_spec_utils import transition_function
from flexible_ddcm.model_spec_utils import work_transition
from flexible_ddcm.model_spec_utils import extreme_value_shocks



def transition_mbo4(
        states,
        params,
        choice,
        variable_state):
    _,schooling,_ = variable_state
    if schooling=="mbo3":
        return fixed_length_nonstandard(
            states,
            params,
            choice,
            variable_state,
            length=1,
            length_dropout=1,
            suffix="_mbo3")
    
    elif schooling=="vmbo":
        return nonstandard_academic_risk(
            states,
            params,
            choice,
            variable_state
            )

    

choice_transition_functions_nonstandard = {
    "havo": functools.partial(
        fixed_length_nonstandard,
        length=2,
        length_dropout=1),
    "mbo4": transition_mbo4,
    "mbo3": nonstandard_academic_risk,
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
        shock_std_key=("wage_shock_vocational","std")
        
    ),
    "academic_work": functools.partial(
        lifetime_wages,
        nonpec_key="nonpec_academic",
        wage_key="wage_academic",
        discount_key=("discount", "discount"),
        shock_std_key=("wage_shock_academic","std")
    ),
}


transition_function_nonstandard = functools.partial(
    transition_function,
    choice_transition_functions=choice_transition_functions_nonstandard,
)

reward_function_nonstandard = functools.partial(
    reward_function, choice_reward_functions=choice_reward_functions_nonstandard
)

map_transition_to_state_choice_entries_nonstandard = functools.partial(
    map_transition_to_state_choice_entries,
    get_between_states=between_states_age_variable,
)

#def ev_shocks_and_transition_costs(choice_value_func, params, period):
#    if period==0:
#        shocks = extreme_value_shocks(choice_value_func, params, period)
#        transition_costs = np.random.normal(
#            params.loc[("transition_shock_havo", "mean")],
#            params.loc[("transition_shock_havo", "std")],
#            size=shocks.shape[0])
#        pos = list(choice_value_func.columns).index("havo")
#        
#        # Havo position,
#        shocks[pos,:] = shocks[pos,:] + transition_costs
#        return shocks, transition_costs 
#    else:
#        return extreme_value_shocks(choice_value_func, params, period)
#