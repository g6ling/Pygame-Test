import torch

gamma = 0.99
batch_size = 64
lr = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
