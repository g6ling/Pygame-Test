import random
from collections import namedtuple, deque
import torch
import numpy as np

from .config import sequence_length, device

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'rnn_state'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = []
        self.local_memory = []
        self.capacity = capacity
        self.memory_idx = 0

    def push(self, state, next_state, action, reward, mask, rnn_state):
        self.local_memory.append(Transition(state, next_state, action, reward, mask, torch.stack(rnn_state).view(2, -1)))
        if mask == 0:
            if len(self.memory) < self.capacity:
                self.memory.append(self.local_memory)
            else:
                self.memory[self.memory_idx] = self.local_memory
            self.memory_idx = (self.memory_idx + 1) % self.capacity
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state = [], [], [], [], [], []
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size)
        indexes = []
        
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - sequence_length)
            indexes.append([batch_idx, start])

            transitions = episode[start:start + sequence_length]
            batch = Transition(*zip(*transitions))

            batch_state.append(torch.stack(list(batch.state)).to(device))
            batch_next_state.append(torch.stack(list(batch.next_state)).to(device))
            batch_action.append(torch.Tensor(list(batch.action)).to(device))
            batch_reward.append(torch.Tensor(list(batch.reward)).to(device))
            batch_mask.append(torch.Tensor(list(batch.mask)).to(device))
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)).to(device))
        
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state), indexes
    
    def rnn_state_update(self, indexes, rnn_state, update_on):
        if update_on is True:
            # rnn_state [sequence_length, 2, batch_size, hidden]
            rnn_state = rnn_state.transpose(2 ,0).transpose(1 ,2)
            # rnn_state [batch_size, sequence_length, 2, hidden]
            for rnn_state_batch_idx, idx in enumerate(indexes):
                [batch_idx, start] = idx
                rnn_state_sequence_idx = 0
                for transition_idx in range(start, start + sequence_length):
                    transition = self.memory[batch_idx][transition_idx]
                    self.memory[batch_idx][transition_idx] = Transition(transition.state, transition.next_state, transition.action, transition.reward, transition.mask, rnn_state[rnn_state_batch_idx][rnn_state_sequence_idx])
                    rnn_state_sequence_idx += 1
                
            

    def __len__(self):
        return len(self.memory)