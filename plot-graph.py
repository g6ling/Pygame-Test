# %%
import argparse
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.envs import make_log_file_name
graph_env_name = ['cartpole', 'catcher', 'flappybird', 'snake']


def push_numpy(rewards, path):
    loss_path, score_path, q_discrepan_path = path

    _, loss = zip(*np.load(loss_path))
    _, reward = zip(*np.load(score_path))
    q_discrepance = np.load(q_discrepan_path)

    rewards.append(list(reward))

def append_lack(rewards):
    max_length = max([len(reward) for reward in rewards])

    for reward in rewards:
        while len(reward) < max_length:
            reward.append(reward[-1])

def moving_avg(rewards):
    window = 30
    weights = np.repeat(1.0, window) / window

    moving = []
    for reward in rewards:
        moving.append(np.convolve(np.array(reward), weights, 'valid'))

    return np.array(moving)

# %%

# sequence_length = 16
sequence_lengths = [8, 16]

# replay_length = 100
replay_lengths = [100, 1000]

for sequence_length in sequence_lengths:
    for replay_length in replay_lengths:
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
            update_off_reward_std = update_off_rewards.std(axis=1) / 2
            update_on_reward_avg = update_on_rewards.mean(axis=1)
            update_on_reward_std = update_on_rewards.std(axis=1) / 2

            fig, ax = plt.subplots(1)
            plt.title('%s(replay length %d, burn-in length %d)'%(graph_env_name[env_num], replay_length, sequence_length-4))

            plt.xlabel('episode')
            plt.ylabel('score')
            

            t = np.array(range(len(update_off_reward_avg)))

            ax.plot(
                t,
                update_off_reward_avg,
                lw=2,
                label='non-update-stored-state', color='blue'
            )
            ax.fill_between(
                t,
                update_off_reward_avg + update_off_reward_std, update_off_reward_avg - update_off_reward_std, facecolor='blue', alpha=0.5
            )

            ax.plot(
                t,
                update_on_reward_avg,
                lw=2,
                label='update-stored-state', color='orange'
            )
            ax.fill_between(
                t,
                update_on_reward_avg + update_on_reward_std, update_on_reward_avg - update_on_reward_std, facecolor='orange', alpha=0.5
            )
            plt.legend(loc=2)
            plt.savefig(fname='graphs/%d-%d-%s.png' % (sequence_length, replay_length, graph_env_name[env_num]))

# %%