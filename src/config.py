import torch

gamma = 0.99
batch_size = 64
lr = 0.001
replay_memory_capacity = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 8
burn_in_length = 4