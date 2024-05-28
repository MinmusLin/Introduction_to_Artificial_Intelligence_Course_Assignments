import torch
from torch.utils.data import Dataset


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer, device='cuda'):
        self.buffer = buffer
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        s, a, r, s_p, done = self.buffer[index]
        return (
            torch.tensor(s, dtype=torch.float32).to(self.device),
            torch.tensor(a, dtype=torch.long).to(self.device),
            torch.tensor(r, dtype=torch.float32).to(self.device),
            torch.tensor(s_p, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.bool).to(self.device)
        )
