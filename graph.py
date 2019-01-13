import argparse
import os
import argparse
import numpy as np
from visdom import Visdom


seed_max = 9
def main():
    update_off_path = ['./logs/CartPole-v1/update_off/' + str(seed*100) for seed in range(1, seed_max+1)]
    update_on_path = ['./logs/CartPole-v1/update_on/' + str(seed*100) for seed in range(1, seed_max+1)]
    
    update_off_rewards = []
    update_on_rewards = []

    update_off_losses = []
    update_on_losses = []
    
    for path in update_off_path:
        episode, reward = zip(*np.load(os.path.join(path, 'score.npy')))
        _, loss = zip(*np.load(os.path.join(path, 'loss.npy')))
        update_off_rewards.append(reward)
        update_off_losses.append(loss)
    
    for path in update_on_path:
        episode, reward = zip(*np.load(os.path.join(path, 'score.npy')))
        _, loss = zip(*np.load(os.path.join(path, 'loss.npy')))
        update_on_rewards.append(reward)
        update_on_losses.append(loss)
    
    update_off_reward_avg = np.array(update_off_rewards).transpose().mean(axis=1)
    update_off_losses_avg = np.array(update_off_losses).transpose().mean(axis=1)

    update_on_reward_avg = np.array(update_on_rewards).transpose().mean(axis=1)
    update_on_losses_avg = np.array(update_on_losses).transpose().mean(axis=1)
    
    viz = Visdom(env='main')

    # Reward
    viz.line(
        X=np.array(episode).reshape(-1, 1).repeat(2, 1),
        Y=np.column_stack([update_off_reward_avg, update_on_reward_avg]),
        opts=dict(
            legend=['off_reward', 'on_reward']
        )
    )

    reward_legend = []
    for i in range(1, seed_max+1):
        reward_legend.append('off_reward_' + str(i))

    for i in range(1, seed_max+1):
        reward_legend.append('on_reward_' + str(i))
        
    viz.line(
        X=np.array(episode).reshape(-1, 1).repeat(seed_max * 2, 1),
        Y=np.column_stack(np.concatenate([update_off_rewards, update_on_rewards], axis=0)),
        opts=dict(
            legend=reward_legend
        )  
    )

    # Loss
    viz.line(
        X=np.array(episode).reshape(-1, 1).repeat(2, 1),
        Y=np.column_stack([update_off_losses_avg, update_on_losses_avg]),
        opts=dict(
            legend=['off_loss', 'on_loss']
        )
    )



if __name__ == '__main__':
    main()