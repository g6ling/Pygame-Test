import numpy as np
import torch
import torch.optim as optim

from .memory import Memory
from .model import DRQN
from .config import lr, batch_size, device
class Agent:
    def __init__(self, num_inputs, action_set, update_on, max_episode, hidden_size, sequence_length, repaly_memory_length):
        self.num_inputs = num_inputs
        self.action_set = action_set
        self.num_actions = len(action_set)
        self.update_on = update_on
        
        self.epsilon = 0.2
        self.lr = lr

        self.build_network(hidden_size)
        self.memory = Memory(repaly_memory_length, sequence_length)
        self.sequence_length = sequence_length
        if max_episode is not None:
            self.max_episode = max_episode
        else:
            self.max_episode = 5000
    
    def build_network(self, hidden_size):
        self.online_net = DRQN(self.num_inputs, self.num_actions, hidden_size).to(device)
        self.target_net = DRQN(self.num_inputs, self.num_actions, hidden_size).to(device)
        self.online_net.train()
        self.target_net.train()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.update_target_model()
    

    def reset(self):
        self.hidden = (torch.Tensor().new_zeros(1, 32), torch.Tensor().new_zeros(1, 32))

    def update_target_model(self):
        # Target <- Net
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_action(self, state):
        action, hidden = self.target_net.get_action(state, self.hidden)
        
        self.used_hidden = self.hidden
        self.hidden = hidden
        if np.random.random() <= self.epsilon:
            action_num = np.random.randint(self.num_actions)
        else:
            action_num = action
        return action_num, self.action_set[action_num]
    
    def push_replay(self, state, next_state, action, reward, mask):
        self.memory.push(state, next_state, action, reward, mask, self.used_hidden)

    def adjust_lr(self):
        self.lr = max(self.lr * 0.995, 0.00005)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def train(self):
        loss_sum, q_discrepancy_sum = 0, 0
        if len(self.memory) > batch_size:
            self.epsilon -= (1.5 / self.max_episode)
            self.epsilon = max(self.epsilon, 0.001)
            for _ in range(10):
                batch, indexes = self.memory.sample(batch_size)
                loss, q_discrepancy, new_rnn_state = DRQN.train_model(self.online_net, self.target_net, self.optimizer, batch, self.sequence_length)
                loss_sum += loss
                q_discrepancy_sum += abs(q_discrepancy)

                self.memory.rnn_state_update(indexes, new_rnn_state, self.update_on)
            self.update_target_model()
        
            self.adjust_lr()

        return loss_sum, q_discrepancy_sum
