import os
import argparse

from src.train import run
from src.envs import envs

if __name__ == '__main__':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    parser = argparse.ArgumentParser()
    parser.add_argument('--update_on', default=False, action="store_true")
    parser.add_argument('--env_num', default=0)
    parser.add_argument('--seed_num', default=0)

    args = parser.parse_args()
    
    print(envs[int(args.env_num)].name, envs[int(args.env_num)].max_episode)
    
    run(envs[int(args.env_num)], int(args.seed_num), args.update_on)
    # for i in range(1, 11):
    #     seed_num = i * 100
    #     run(envs[int(args.env_num)], seed_num, args.update_on)