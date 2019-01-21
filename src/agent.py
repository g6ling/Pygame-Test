import numpy as np
import torch.optim as optim

from .memory import Memory
from .model import DRQN
from .config import lr, batch_size, replay_memory_capacity, device
class Agent:
    def __init__(self, num_inputs, action_set, update_on, max_episode, hidden_size):
        self.num_inputs = num_inputs
        self.action_set = action_set
        self.num_actions = len(action_set)
        self.update_on = update_on
        self.reset()
        self.build_network(hidden_size)
        self.memory = Memory(replay_memory_capacity)
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
        self.hidden = None
        self.epsilon = 0.2
        self.lr = lr

    def update_target_model(self):
        # Target <- Net
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_action(self, state):
        action, hidden = self.target_net.get_action(state, self.hidden)
        
        self.hidden = hidden
        if np.random.random() <= self.epsilon:
            action_num = np.random.randint(self.num_actions)
        else:
            action_num = action
        return action_num, self.action_set[action_num]
    
    def push_replay(self, state, next_state, action, reward, mask):
        self.memory.push(state, next_state, action, reward, mask, self.hidden)

    def adjust_lr(self):
        self.lr = max(self.lr * 0.995, 0.0005)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def train(self):
        loss, q_discrepancy = None, None
        if len(self.memory) > batch_size:
            self.epsilon -= (1.5 / self.max_episode)
            self.epsilon = max(self.epsilon, 0.001)
            for _ in range(10):
                batch, indexes = self.memory.sample(batch_size)
                loss, q_discrepancy, new_rnn_state = DRQN.train_model(self.online_net, self.target_net, self.optimizer, batch)

                self.memory.rnn_state_update(indexes, new_rnn_state, self.update_on)
            self.update_target_model()
        
            self.adjust_lr()
        return loss, q_discrepancy
