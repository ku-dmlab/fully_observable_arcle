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
                 colors: SupportsInt,
                 max_trial: SupportsInt=-1):
        self.loader = data_loader
        self.H, self.W = max_grid_size
        self.colors = colors
        self.max_trial = max_trial

        self.observation_space = spaces.Dict(
            {
                'trials_remain': spaces.Discrete(self.max_trial+2, start=-1),
                'terminated': spaces.Box(0, 1, dtype=np.uint8),
                
                'input': spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                'input_dim': spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),

                'grid': spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                'grid_dim': spaces.Tuple((spaces.Discrete(self.H,start=1),spaces.Discrete(self.W,start=1))),

                'selected': spaces.MultiBinary((self.H,self.W)),
                'clip': spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                'clip_dim': spaces.Tuple((spaces.Discrete(self.H+1,start=0),spaces.Discrete(self.W+1,start=0))),

                'object_states':spaces.Dict({
                    'active': spaces.Discrete(2),
                    'object': spaces.Box(0,self.colors,(self.H,self.W),dtype=np.uint8),
                    'object_sel': spaces.MultiBinary((self.H,self.W)),
                    'object_dim': spaces.Tuple((spaces.Discrete(self.H+1,start=0),spaces.Discrete(self.W+1,start=0))),
                    'object_pos': spaces.Tuple((spaces.Discrete(200,start=-100),spaces.Discrete(200,start=-100))),
                    'background': spaces.Box(0, self.colors, (self.H,self.W),dtype=np.uint8),
                    'rotation_parity': spaces.Discrete(2),
                })
            }
        )
        self.action_space = spaces.Dict(
            {
                'selection': spaces.MultiBinary((self.H,self.W)), # selection Mask
                'operation': spaces.Discrete(35)
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
            'trials_remain': self.max_trial,
            'terminated': 0,
            'input': info['input'],
            'input_dim': info['input_dim'],
            'grid': grid,
            'grid_dim': grid_dim,
            'selected': np.zeros((self.H, self.W), dtype=np.uint8),
            'clip': np.zeros((self.H, self.W), dtype=np.uint8),
            'clip_dim': (0, 0),
            'object_states': {
                'active': 0,
                'object': np.zeros((self.H, self.W), dtype=np.uint8),
                'object_sel': np.zeros((self.H, self.W), dtype=np.uint8),
                'object_dim': (0, 0),
                'object_pos': (0, 0),
                'background': np.zeros((self.H, self.W), dtype=np.uint8),
                'rotation_parity': 0
            }
        }
        self.obs, self.info = obs, info
        return obs, info
    
    def step(self, action):
        ret = self._step(self.obs, action, self.info)
        self.obs = ret[0]
        return ret

    def _step(self, state: ObsType, action: ActType, info: Dict):
        object_states = {}
        next_state = {
            'object_states': object_states,
            'input': info['input'],
            'input_dim': info['input_dim']}

        (next_state['grid'], next_state['grid_dim'], next_state['selected'],
         next_state['clip'], next_state['clip_dim'], next_state['terminated'],
         next_state['trials_remain'],
         object_states['active'], object_states['object'], object_states['object_sel'],
         object_states['object_dim'], object_states['object_pos'], 
         object_states['background'], object_states['rotation_parity'], reward) = act(
                info['input'],
                info['input_dim'],
                info['answer'],
                state['grid'],
                state['grid_dim'],
                state['selected'],
                state['clip'],
                state['clip_dim'],
                state['terminated'],
                state['trials_remain'],
                state['object_states']['active'],
                state['object_states']['object'],
                state['object_states']['object_sel'],
                state['object_states']['object_dim'],
                state['object_states']['object_pos'],
                state['object_states']['background'],
                state['object_states']['rotation_parity'],
                action['selection'].astype(np.bool_),
                action['operation'])
        
        return next_state, reward, next_state['terminated'], False, info
