import copy
import numpy as np
from arcle.actions.o2actions import act
import gymnasium as gym
from tqdm import trange
from arcle.loaders import Loader

import time

SIZE = 5
class TestLoader(Loader):
    def get_path(self, **kwargs):
        return ['']

    def parse(self, **kwargs):
        ti= np.zeros((SIZE,SIZE), dtype=np.uint8)
        to = np.zeros((SIZE,SIZE), dtype=np.uint8)
        ei = np.zeros((SIZE,SIZE), dtype=np.uint8)
        eo = np.zeros((SIZE,SIZE), dtype=np.uint8)

        ti[0:10, 0:10] = np.random.randint(0,10, size=[SIZE,SIZE])
        return [([ti],[to],[ei],[eo], {'desc': "just for test"})]

def test_equivalence():
    env = gym.make(
        'ARCLE/O2ARCv2Env-v0',
        data_loader=TestLoader(),
        max_grid_size=(SIZE,SIZE),
        colors=10)

    for _ in trange(10):
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            action["selection"] = action["selection"].astype(np.bool_)
            if 9 < action["operation"] <= 19:
                action["selection"] = np.zeros_like(action["selection"])
                action["selection"][3, 3] = True

            inp = np.zeros_like(obs["grid"])
            inp_dim = env.unwrapped.input_.shape
            inp[:inp_dim[0], :inp_dim[1]] = env.unwrapped.input_

            (grid,
            grid_dim,
            selected,
            clip,
            clip_dim,
            terminated,
            objsel_active,
            objsel,
            objsel_area,
            objsel_bg,
            objsel_coord,
            objsel_rot) = act(
                inp.copy(),
                inp_dim,
                obs["grid"].copy(),
                obs["grid_dim"],
                obs["selected"].copy(),
                obs["clip"].copy(),
                obs["clip_dim"],
                env.unwrapped.terminated,
                env.unwrapped.objsel_active,
                env.unwrapped.objsel.copy(),
                env.unwrapped.objsel_area.copy(),
                env.unwrapped.objsel_bg.copy(),
                env.unwrapped.objsel_coord,
                env.unwrapped.objsel_rot,
                action["selection"].copy(),
                action["operation"].copy()
            )

            next_obs, reward, term, trunc, info = env.step(action)

            assert (grid == next_obs["grid"]).all(), f'\n{obs["grid"]}\n{grid}\n{next_obs["grid"]}, \n{action["operation"]}, \n{action["selection"]}'
            assert grid_dim == next_obs["grid_dim"], action["operation"]
            assert (selected == next_obs["selected"]).all(), f'\n{obs["selected"]}\n{selected}\n{next_obs["selected"]}, \n{action["operation"]}, \n{action["selection"]}'
            assert (clip == next_obs["clip"]).all()
            assert clip_dim == next_obs["clip_dim"]
            assert terminated == env.unwrapped.terminated
            assert objsel_active == env.unwrapped.objsel_active
            assert (objsel == env.unwrapped.objsel).all()
            assert (objsel_area == env.unwrapped.objsel_area).all()
            assert (objsel_bg == env.unwrapped.objsel_bg).all()
            assert objsel_coord == env.unwrapped.objsel_coord
            assert objsel_rot == env.unwrapped.objsel_rot

            obs = next_obs

def benchmark(act_no, repeat=10000):
    env = gym.make(
        'ARCLE/O2ARCv2Env-v0',
        data_loader=TestLoader(),
        max_grid_size=(SIZE,SIZE),
        colors=10)
    obs, info = env.reset()

    inp = np.zeros_like(obs["grid"])
    inp_dim = env.unwrapped.input_.shape
    inp[:inp_dim[0], :inp_dim[1]] = env.unwrapped.input_

    action = env.action_space.sample()
    action["operation"] = act_no
    action["selection"] = action["selection"].astype(np.bool_)
    if 9 < action["operation"] <= 19:
        action["selection"] = np.zeros_like(action["selection"])
        action["selection"][3, 3] = True
        env.grid = np.zeros_like(env.grid)
    start = time.time()
    for _ in range(repeat):

        (grid,
        grid_dim,
        selected,
        clip,
        clip_dim,
        terminated,
        objsel_active,
        objsel,
        objsel_area,
        objsel_bg,
        objsel_coord,
        objsel_rot) = act(
            inp,
            inp_dim,
            obs["grid"],
            obs["grid_dim"],
            obs["selected"],
            obs["clip"],
            obs["clip_dim"],
            env.unwrapped.terminated,
            env.unwrapped.objsel_active,
            env.unwrapped.objsel,
            env.unwrapped.objsel_area,
            env.unwrapped.objsel_bg,
            env.unwrapped.objsel_coord,
            env.unwrapped.objsel_rot,
            action["selection"],
            action["operation"]
        )
    cython_time = time.time()-start
    start = time.time()
    for _ in range(repeat):
        next_obs, reward, term, trunc, info = env.step(action)
    python_time = time.time()-start
    print("act no: ", act_no)
    print(f"cython: {cython_time:.4f} vs python: {python_time:.4f} sec")
    print(f"{cython_time / python_time}")

    return cython_time, python_time

if __name__ == "__main__":
    test_equivalence()
    cs, ps = [], []
    for i in range(35):
        c, p = benchmark(i)
        cs.append(c)
        ps.append(p)
    print("on average: ")
    print(f"cython: {np.mean(cs):.4f} sec, python: {np.mean(ps):.4f} sec")
