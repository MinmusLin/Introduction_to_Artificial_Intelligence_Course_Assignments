import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import ReplayBufferDataset


class Trainer:
    def __init__(self,
                 policy_net,
                 target_net,
                 batch_size,
                 optimizer,
                 sync_rate):
        self.policy_net = policy_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.sync_rate = sync_rate

    def run(self, replay_buffer):
        dataset = ReplayBufferDataset(replay_buffer, device=self.policy_net.device)
        xloader = DataLoader(dataset, self.batch_size, shuffle=True)
        avgloss = []
        for step, (s, a, r, sp, done) in enumerate(xloader):
            if step % self.sync_rate: self.update_target()
            self.optimizer.zero_grad()
            loss = self.dqn_loss(s, a, r, sp, done)
            loss.backward()
            self.optimizer.step()
            avgloss.append(loss.item())
        return np.mean(avgloss)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def dqn_loss(self, curr_s, curr_a, curr_r, next_s, curr_done, gamma=0.98):
        left_Q = self.policy_net(curr_s)[torch.arange(curr_s.shape[0]), curr_a]
        with torch.no_grad():
            qvalues = self.target_net(next_s).max(dim=1)[0].detach()
            qvalues[curr_done] = 0.
            right_Q = curr_r + gamma * qvalues
        return torch.nn.functional.mse_loss(left_Q, right_Q)
