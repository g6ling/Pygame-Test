import os
import numpy as np

from ple import PLE
from ple.games.catcher import Catcher
from ple.games.pixelcopter import Pixelcopter
from ple.games.pong import Pong
from ple.games.snake import Snake
from ple.games.waterworld import WaterWorld


class EnvWrapper():
    def __init__(self, name, env, goal_score, max_episode):
        self.name = name
        self.env = env
        self.goal_score = goal_score
        self.max_episode = max_episode

envs_name = ['catcher', 'flappybird', 'pong', 'snake']

def get_envs():
    envs = [
        EnvWrapper('catcher', PLE(Catcher(init_lives=1), fps=30, display_screen=False), 200, 1000),
        EnvWrapper('pong', PLE(Pong(cpu_speed_ratio=0.25), display_screen=False), 100, 1000),
        EnvWrapper('pixelcopter', PLE(Pixelcopter(), fps=30, display_screen=False), 200, 1000),
        EnvWrapper('snake', PLE(Snake(), fps=30, display_screen=False), 30, 1000),
        # EnvWrapper('waterworld', PLE(WaterWorld(), fps=30, display_screen=False), 100, 10000),
    ]
    return envs


def make_log_file_name(env_name, seed_num, update_on):
    if update_on:
        log_path = 'logs/%s/update_on' % (env_name)
    else:
        log_path = 'logs/%s/update_off' % (env_name)
    log_path = os.path.join(os.getcwd(), log_path)
    
    if os.path.exists(log_path) is False:
        os.makedirs(log_path)
    loss_path = log_path + '/%d_loss.npy' % (seed_num)
    score_path = log_path + '/%d_score.npy'  % (seed_num)
    q_discrepancy_path = log_path + '/%d_q_discrepancy.npy' % (seed_num)

    return loss_path, score_path, q_discrepancy_path


class LogMaker():
    def __init__(self, env_name, seed_num, update_on):
        self.loss_logs = []
        self.score_logs = []
        self.q_discrepancy_logs = []
        self.loss_path, self.score_path, self.q_discrepancy_path = make_log_file_name(env_name, seed_num, update_on)
    
    def log(self, e, loss, score, q_discrepancy):
        if loss is not None and score is not None and q_discrepancy is not None:
            self.loss_logs.append([e, loss])
            self.score_logs.append([e, score])
            self.q_discrepancy_logs.extend([e, q_discrepancy])
            np.save(self.loss_path, np.array(self.loss_logs))
            np.save(self.score_path, np.array(self.score_logs))
            np.save(self.q_discrepancy_path, np.array(self.q_discrepancy_logs))