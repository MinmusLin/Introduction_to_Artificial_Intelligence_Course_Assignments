import os
import argparse
import torch
import gymnasium as gym
from src.dqn import QFunc
from src.agent import LunarLanderAgent

parser = argparse.ArgumentParser()
parser.add_argument('--model-ckpt', type=str, default='models/model.ckpt')
parser.add_argument('--episodes', type=int, default=1)
args = parser.parse_args()

episodes = args.episodes
model_checkpoints = args.model_ckpt

assert os.path.exists(model_checkpoints), 'Invalid model checkpoints'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('LunarLander-v2', render_mode='human')

qfunc = QFunc(4, 8, 512, 512).to(device)
qfunc.load_state_dict(torch.load(model_checkpoints, map_location=device))
agent = LunarLanderAgent(qfunc, env)

for _ in range(episodes):
    agent.play_episode([], True)
