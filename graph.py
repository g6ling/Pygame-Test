import argparse
import os
import argparse
import numpy as np
from visdom import Visdom
from src.envs import make_log_file_name
graph_env_name = ['cartpole', 'catcher', 'flappybird', 'snake']
# graph_env_name = ['flappybird', 'flappybird-12', 'flappybird-0']

import matplotlib.pyplot as plt

def push_numpy(rewards, path):
    loss_path, score_path, q_discrepan_path = path

    _, loss = zip(*np.load(loss_path))
    _, reward = zip(*np.load(score_path))
    q_discrepance = np.load(q_discrepan_path)
    # _, q_discrepance = zip(*np.load(q_discrepan_path))

    rewards.append(list(reward))

def append_lack(rewards):
    max_length = max([len(reward) for reward in rewards])

    for reward in rewards:
        while len(reward) < max_length:
            reward.append(reward[-1])

def moving_avg(rewards):
    window = 10
    weights = np.repeat(1.0, window) / window

    moving = []
    for reward in rewards:
        moving.append(np.convolve(np.array(reward), weights, 'valid'))

    return np.array(moving)

def main():
    viz = Visdom(env='main')

    sequence_length=8
    replay_length = 100

    for env_num in [0,1,2,3]:
        update_off_rewards = []
        update_on_rewards = []

        
        for i in range(1, 4+1):
            seed_num = 100 * i
            push_numpy(update_off_rewards, make_log_file_name(graph_env_name[env_num], seed_num, False, sequence_length, replay_length))
            push_numpy(update_on_rewards, make_log_file_name(graph_env_name[env_num], seed_num, True, sequence_length, replay_length))

        append_lack(update_off_rewards + update_on_rewards)


        update_off_rewards = moving_avg(update_off_rewards)
        update_on_rewards = moving_avg(update_on_rewards)

        update_off_rewards = update_off_rewards.transpose()
        update_on_rewards = update_on_rewards.transpose()

        update_off_reward_avg = update_off_rewards.mean(axis=1)
        update_on_reward_avg = update_on_rewards.mean(axis=1)

        viz.line(
           X=np.array(range(len(update_off_reward_avg))).reshape(-1, 1).repeat(2, 1),
            Y=np.column_stack([update_off_reward_avg, update_on_reward_avg]),
            opts=dict(
                legend=['non-update-stored-state', 'upate-stored-state'], title='%s-%d-%d'%(graph_env_name[env_num], sequence_length, replay_length)
            )
        )


if __name__ == '__main__':
    main()
