import foarcle
import arcle
import gymnasium as gym
import time
import numpy as np
from arcle.loaders import Loader
from tqdm import trange


SIZE = 30

class TestLoader(Loader):

    def __init__(self, **kwargs):
        self.rng = np.random.default_rng(12345)
        super().__init__(**kwargs)

    def get_path(self, **kwargs):
        return ['']

    def parse(self, **kwargs):
        ti= np.zeros((SIZE,SIZE), dtype=np.uint8)
        to = np.zeros((SIZE,SIZE), dtype=np.uint8)
        ei = np.zeros((SIZE,SIZE), dtype=np.uint8)
        eo = np.zeros((SIZE,SIZE), dtype=np.uint8)

        ti[0:SIZE, 0:SIZE] = self.rng.integers(0,10, size=[SIZE,SIZE])
        return [([ti],[to],[ei],[eo], {'desc': "just for test"})]

def test_equality():
    env = gym.make('ARCLE/FOO2ARCv2Env-v0', data_loader = TestLoader(), max_grid_size=(SIZE, SIZE), colors=10)
    base_env = gym.make('ARCLE/O2ARCv2Env-v0', data_loader = TestLoader(), max_grid_size=(SIZE, SIZE), colors=10)

    obs, info = env.reset()
    base_obs, base_info = base_env.reset()

    for key in base_obs:
        assert np.allclose(obs[key], base_obs[key]), key

    for _ in trange(1000):
        
        action = env.action_space.sample()
        other_selection = np.random.randint(3)
        if other_selection != 2:
            action["selection"] = np.zeros_like(action["selection"])
            if other_selection == 1:
                action["selection"][np.random.randint(SIZE), np.random.randint(SIZE)] = True
        action["selection"] = action["selection"].astype(np.bool_)
        obs, reward, term, trunc, info = env.step(action)

        base_obs, base_reward, base_term, base_trunc, base_info = base_env.step(action)

        try:
            for key in base_obs:
                assert np.allclose(obs[key], base_obs[key]), f"{key}\n{obs[key]}\n{base_obs[key]}"
        except AssertionError as e:
            if action["operation"] >= 10 and action["operation"] <= 19 and ~np.any(action["selection"]):
                obs, info = env.reset()
                base_obs, base_info = base_env.reset()
                for key in base_obs:
                    assert np.allclose(obs[key], base_obs[key])
            else:
                print(obs)
                print(base_obs)
                raise e
                
        assert reward == base_reward
        assert term == base_term
        assert trunc == base_trunc

        if term or trunc:
            obs, info = env.reset()
            base_obs, base_info = base_env.reset()
            for key in base_obs:
                assert np.allclose(obs[key], base_obs[key])

    env.close()
    base_env.close()

def benchmark():
    env = gym.make('ARCLE/FOO2ARCv2Env-v0', data_loader = TestLoader(), max_grid_size=(SIZE, SIZE), colors=10)
    base_env = gym.make('ARCLE/O2ARCv2Env-v0', data_loader = TestLoader(), max_grid_size=(SIZE, SIZE), colors=10)

    start_time = time.time()
    for _ in range(100):
        obs, info = env.reset()
        for _ in trange(1000):
            action = env.action_space.sample()
            other_selection = np.random.randint(3)
            if other_selection != 2:
                action["selection"] = np.zeros_like(action["selection"])
                if other_selection == 1:
                    action["selection"][np.random.randint(SIZE), np.random.randint(SIZE)] = True
            action["selection"] = action["selection"].astype(np.bool_)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                obs, info = env.reset()
    cython_time = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        base_obs, base_info = base_env.reset()

        for _ in trange(1000):
            
            action = env.action_space.sample()
            other_selection = np.random.randint(3)
            if other_selection != 2:
                action["selection"] = np.zeros_like(action["selection"])
                if other_selection == 1:
                    action["selection"][np.random.randint(SIZE), np.random.randint(SIZE)] = True
            action["selection"] = action["selection"].astype(np.bool_)

            base_obs, base_reward, base_term, base_trunc, base_info = base_env.step(action)

            if term or trunc:
                base_obs, base_info = base_env.reset()
    python_time = time.time() - start_time
    print(f"cython: {cython_time:.4f} vs python: {python_time:.4f} sec")
    print(f"{cython_time / python_time}")

    env.close()
    base_env.close()


if __name__ == "__main__":
    test_equality()
    benchmark()