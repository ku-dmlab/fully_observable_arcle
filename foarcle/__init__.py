from .foo2arcenv import FOO2ARCv2Env
from gymnasium.envs.registration import register

register(
    id='ARCLE/FOO2ARCv2Env-v0',
    entry_point='foarcle.foo2arcenv:FOO2ARCv2Env',
    max_episode_steps=1000000,
)