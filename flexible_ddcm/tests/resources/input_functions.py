import functools

from flexible_ddcm.model_spec_utils import combined_logit_length
from flexible_ddcm.model_spec_utils import poisson_length
from flexible_ddcm.model_spec_utils import work_transition
from flexible_ddcm.model_spec_utils import nonpecuniary_reward
from flexible_ddcm.model_spec_utils import lifetime_wages


transition_functions = {
    "havo": combined_logit_length,
    "mbo4": combined_logit_length,
    "mbo3": poisson_length,
    "hbo": combined_logit_length,
    "vocational_work": work_transition,
    "academic_work": work_transition,
}

reward_functions = {
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
