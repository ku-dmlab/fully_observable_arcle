import numpy as np
import gymnasium as gym
import pygame as pg
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from typing import Dict, Tuple, SupportsFloat, SupportsInt
from .actions.o2actions import act

class FOO2ARCv2Env(gym.Env):
    
    def __init__(self,
                 data_loader,
                 max_grid_size: Tuple[SupportsInt, SupportsInt],
                 colors: SupportsInt):
        self.loader = data_loader
        self.H, self.W = max_grid_size
        self.colors = colors

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "grid_dim": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
                "selected": spaces.MultiBinary((self.H,self.W)),
                "clip": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "clip_dim": spaces.Tuple((spaces.Discrete(self.H,start=0),spaces.Discrete(self.W,start=0))),
                "terminated": spaces.Box(0, 1, dtype=np.uint8),
                "objsel_active": spaces.Box(0, 1, dtype=np.uint8),
                "objsel": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "objsel_area": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "objsel_bg": spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                "objsel_coord": spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),
                "objsel_rot": spaces.Box(0, 1, dtype=np.uint8)
            }
        )
        self.action_space = spaces.Dict(
            {
                "selection": spaces.MultiBinary((self.H,self.W)), # selection Mask
                "operation": spaces.Discrete(35)
            }
        )

    def get_problem(self, options={}):
    
        ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=options.get('prob_index'))

        adaptation = bool(options.get('adaptation', True))
        inp = ex_in if adaptation else tt_in
        ans = ex_out if adaptation else tt_out

        subprob_index = options.get('subprob_index', np.random.randint(0, len(inp)))
        input_dim = inp[subprob_index].shape
        inp = np.pad(
            inp[subprob_index], [(0, self.H - input_dim[0]), (0, self.W - input_dim[1])], constant_values=0).astype(np.uint8)

        return {'input': inp, 'input_dim': input_dim, 'answer': ans[subprob_index]}

    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)
        info = self.get_problem(options)

        grid_dim = info['input_dim']
        grid = info['input'].copy()

        obs = {
            'grid': grid,
            'grid_dim': grid_dim,
            'selected': np.zeros((self.H, self.W), dtype=np.uint8),
            'clip': np.zeros((self.H, self.W), dtype=np.uint8),
            'clip_dim': (0, 0),
            'terminated': 0,
            "objsel_active": 0,
            "objsel": np.zeros((self.H, self.W), dtype=np.uint8),
            "objsel_area": np.zeros((self.H, self.W), dtype=np.uint8),
            "objsel_bg": np.zeros((self.H, self.W), dtype=np.uint8),
            "objsel_coord": (0, 0),
            "objsel_rot": 0
        }
        self.obs, self.info = obs, info
        return obs, info
    
    def step(self, action):
        ret = self._step(self.obs, action, self.info)
        self.obs = ret[0]
        return ret

    def _step(self, state: ObsType, action: ActType, info: Dict):
        next_state = {}
        (next_state["grid"], next_state["grid_dim"], next_state["selected"],
         next_state["clip"], next_state["clip_dim"], next_state["terminated"],
         next_state["objsel_active"], next_state["objsel"], next_state["objsel_area"],
         next_state["objsel_bg"], next_state["objsel_coord"], next_state["objsel_rot"]) = act(
                info["input"],
                info["input_dim"],
                state["grid"],
                state["grid_dim"],
                state["selected"],
                state["clip"],
                state["clip_dim"],
                state["terminated"],
                state["objsel_active"],
                state["objsel"],
                state["objsel_area"],
                state["objsel_bg"],
                state["objsel_coord"],
                state["objsel_rot"],
                action["selection"].astype(np.bool_),
                action["operation"])
        reward = self._reward(next_state, info)
        
        return next_state, reward, next_state['terminated'], False, info

    def _reward(self, state, info) -> SupportsFloat:
        if state['terminated']:
            if state['grid_dim'] == info['answer'].shape:
                h, w = info['answer'].shape
                if np.all(state["grid"][0:h, 0:w] == self.info['answer']):
                    return 1
        return 0