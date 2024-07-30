import gym
import numpy as np

class f64Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        obs = self.env.reset()
        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        return obs

    def step(self, act):
        (obs, reward, done, info) = self.env.step(act)
        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        return (obs, reward, done, info)

def make_f64(env_name):
    env = gym.make(env_name)
    return f64Wrapper(env)
