import numpy as np
import torch


class LunarLanderAgent:
    def __init__(self, qfunc, env):
        self.qfunc = qfunc
        self.env = env

    @torch.no_grad()
    def get_action(self, observations, validation=False, eps=0.1):
        if not validation and np.random.random() <= eps:
            return self.env.action_space.sample()
        tensor = torch.tensor(observations).unsqueeze(0).to(self.qfunc.device)
        action = torch.argmax(self.qfunc(tensor), dim=1).item()
        return action

    @torch.no_grad()
    def play_episode(self, buffer, validation=False, eps=0.):
        total_reward = 0
        curr_observations, _ = self.env.reset()
        while True:
            action = self.get_action(curr_observations, validation=validation, eps=eps)
            next_observations, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            buffer.append((curr_observations.copy(), action, reward, next_observations.copy(), terminated))
            if done: break
            curr_observations = next_observations
        return total_reward
