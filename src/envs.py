import os
import numpy as np

import gym
from ple import PLE
from ple.games.catcher import Catcher
from ple.games.pixelcopter import Pixelcopter
from ple.games.snake import Snake
from ple.games.pong import Pong
from ple.games.flappybird import FlappyBird


class EnvWrapper():
    def __init__(self, name, env, goal_score, max_score, max_episode):
        self.name = name
        self.env = env
        self.goal_score = goal_score
        self.max_episode = max_episode
        self.max_score = max_score

def get_envs():
    envs = [
        EnvWrapper('cartpole', gym.make('CartPole-v1'), 500, 550, 4000),
        EnvWrapper('catcher', PLE(Catcher(init_lives=1), display_screen=False, reward_values={
            "positive": 1,
            "negative": -1,
            "loss": -1,
        }), 100, 110, 3000),
        EnvWrapper('snake', PLE(Snake(height=256, width=256), display_screen=False, reward_values={
            "tick": -0.01,
            "positive": 5,
            "loss": -1,
        }), 100, 110, 3000),
        EnvWrapper('flappybird', PLE(FlappyBird(), display_screen=False, reward_values={
            "positive": 1,
            "tick": 0.1,
            "loss": -1,
        }), 100, 110, 3000),
    ]
    return envs


def make_log_file_name(env_name, seed_num, update_on, sequence_length, repaly_memory_length):
    if update_on:
        log_path = 'logs/%d/%d/%s/update_on' % (sequence_length, repaly_memory_length ,env_name)
    else:
        log_path = 'logs/%d/%d/%s/update_off' % (sequence_length, repaly_memory_length ,env_name)
    log_path = os.path.join(os.getcwd(), log_path)
    
    if os.path.exists(log_path) is False:
        os.makedirs(log_path)
    loss_path = log_path + '/%d_loss.npy' % (seed_num)
    score_path = log_path + '/%d_score.npy'  % (seed_num)
    q_discrepancy_path = log_path + '/%d_q_discrepancy.npy' % (seed_num)

    return loss_path, score_path, q_discrepancy_path


class LogMaker():
    def __init__(self, env_name, seed_num, update_on, sequence_length, repaly_memory_length):
        self.loss_logs = []
        self.score_logs = []
        self.q_discrepancy_logs = []
        self.loss_path, self.score_path, self.q_discrepancy_path = make_log_file_name(env_name, seed_num, update_on, sequence_length, repaly_memory_length)
    
    def log(self, e, loss, score, q_discrepancy):
        if loss is not None and score is not None and q_discrepancy is not None:
            self.loss_logs.append([e, loss])
            self.score_logs.append([e, score])
            self.q_discrepancy_logs.extend([e, q_discrepancy])
            np.save(self.loss_path, np.array(self.loss_logs))
            np.save(self.score_path, np.array(self.score_logs))
            np.save(self.q_discrepancy_path, np.array(self.q_discrepancy_logs))