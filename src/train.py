import torch
import random
import numpy as np

from .agent import Agent
from .config import device
from .envs import LogMaker, get_envs


def case_env_state(env_name, state):
    if env_name == 'snake':
        return torch.Tensor(list(state.values())[:4]).to(device)
    return torch.Tensor(list(state.values())).to(device)

def case_state_len(env_name, env):
    if env_name == 'snake':
        return 4
    return len(env.getGameState().values())

def run(env_wrapper, seed_num, update_on):
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)

    process_name = '{} - {} - {}'.format(env_wrapper.name, seed_num, str(update_on))
    
    log_maker = LogMaker(env_wrapper.name, seed_num, update_on)

    env = env_wrapper.env
    goal_score = env_wrapper.goal_score

    agent = Agent(case_state_len(env_wrapper.name, env), env.getActionSet(), update_on, env_wrapper.max_episode, 256)

    running_score = 0
    for e in range(env_wrapper.max_episode+1):
        env.reset_game()
        done = False
        state = env.getGameState()
        state = case_env_state(env_wrapper.name, state)
        score = 0

        while not done:
            action, real_action = agent.get_action(state)    
            reward = env.act(real_action)

            next_state = env.getGameState()
            next_state = case_env_state(env_wrapper.name, next_state)

            done = env.game_over()

            mask = 0 if done else 1

            agent.push_replay(state, next_state, action, reward, mask)

            state = next_state
            score += reward

            if goal_score is not None and score >= goal_score:
                done = True
        
        loss, q_discrepancy = agent.train()
            
        if running_score == 0:
            running_score = score
        else:
            running_score = 0.99 * running_score + 0.01 * score
        if e % 10 == 0:
            print('{} ||| {} episode | score: {:.2f} | epsilon: {:.2f}'.format(
               process_name, e, score, agent.epsilon))
        log_maker.log(e, loss, score, q_discrepancy)


if __name__=="__main__":
    envs = get_envs()
    run(envs[0], 500, True)
    # main(envs[0], 500, False)
