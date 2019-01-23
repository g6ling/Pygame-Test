import torch
import random
import numpy as np
from collections import deque
from .agent import Agent
from .config import device
from .envs import LogMaker, get_envs


def case_env_state(env_name, state, width, height):
    if env_name == 'cartpole':
        return torch.Tensor(state[[0, 2]]).to(device)
    if env_name == 'catcher':
        return torch.Tensor([state['player_x'] / width, state['fruit_x'] / width, state['fruit_y'] / height]).to(device)
    if env_name == 'snake':
        return torch.Tensor([state['snake_head_x'] / width, state['snake_head_y'] / height, state['food_x'] / width, state['food_y'] / height]).to(device) 

    if env_name == 'flappybird':
        return torch.Tensor([
            state['player_y'] / height,

            state['next_pipe_dist_to_player'] / width,
            state['next_pipe_top_y'] / height,
            state['next_pipe_bottom_y'] / height,

            state['next_next_pipe_dist_to_player'] / width,
            state['next_next_pipe_top_y'] / height,
            state['next_next_pipe_bottom_y'] / height,
        ]).to(device)
    return torch.Tensor(list(state.values())).to(device)

def case_state_len(env_name, env):
    if env_name == 'cartpole':
        return 2
    if env_name == 'catcher':
        return 3
    if env_name == 'snake':
        return 4
    if env_name == 'flappybird':
        return 7
    return len(env.getGameState().values())

def run(env_wrapper, seed_num, update_on, sequence_length, repaly_memory_length):
    env_name = env_wrapper.name
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)

    process_name = '{} - {} - {}'.format(env_wrapper.name, seed_num, str(update_on))
    
    log_maker = LogMaker(env_wrapper.name, seed_num, update_on, sequence_length, repaly_memory_length)

    env = env_wrapper.env
    goal_score = env_wrapper.goal_score
    max_score = env_wrapper.max_score

    if env_name == 'cartpole':
        agent = Agent(2, [0, 1], update_on, env_wrapper.max_episode, 256, sequence_length, repaly_memory_length)
        width, height = 0, 0
    else:
        agent = Agent(case_state_len(env_wrapper.name, env), env.getActionSet(), update_on, env_wrapper.max_episode, 256, sequence_length, repaly_memory_length)
        width, height = env.getScreenDims()

    recent_socres = deque(maxlen=10)
    for e in range(env_wrapper.max_episode+1):
        
        if env_name == 'cartpole':
            state = env.reset()
        else:
            env.reset_game()
            state = env.getGameState()

        done = False
        state = case_env_state(env_wrapper.name, state, width, height)
        
        score = 0
        agent.reset()

        while not done:
            action, real_action = agent.get_action(state)
            if env_name == 'cartpole':
                next_state, reward, done, _ = env.step(action)
            else:
                reward = env.act(real_action)
                
                next_state = env.getGameState()
                done = env.game_over()
            next_state = case_env_state(env_wrapper.name, next_state, width, height)

            if (max_score is not None and score >= max_score) or score < -3:
                done = True

            mask = 0 if done else 1

            agent.push_replay(state, next_state, action, reward, mask)

            state = next_state
            score += reward

        
        loss, q_discrepancy = agent.train()
        
        recent_socres.append(score)
        if e % 10 == 0:
            print('{} ||| {} episode | score: {:.2f} | epsilon: {:.4f} | min-score: {:.2f} | lr: {:.5f}'.format(
               process_name, e, score, agent.epsilon, np.array(recent_socres).min(), agent.lr))
        log_maker.log(e, loss, score, q_discrepancy)

        if len(recent_socres) == 10 and np.array(recent_socres).min() >= goal_score:
            break


