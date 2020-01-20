# -*- coding: utf-8 -*-
"""

"""

from gym.envs.registration import register

# register(
#     id='LL-v0',
#     entry_point='utils.lunar_lander:LL',
# )

register(
    id='LL-v0',
    entry_point='utils.lunar_lander:LL',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
)
register(
        id='RacecarBulletDiscreteEnvLocal-v0',
    entry_point='utils.racecar:RacecarGymEnv',
    # timestep_limit=1000,
    reward_threshold=5.0,
        kwargs={'isDiscrete': True } 

)
register(
	id='RacecarBulletDiscreteEnv-v0',
    entry_point='utils.racecarGymEnv:RacecarGymEnv',
    # timestep_limit=5000,
    reward_threshold=5.0,
	kwargs={'isDiscrete': True } 
)
register(
    id='KukaDiscreteEnv-v0',
    entry_point='utils.kukaGymEnv:KukaGymEnv',
    # timestep_limit=1000,
    reward_threshold=5.0,
    kwargs={'isDiscrete': True} 
)

