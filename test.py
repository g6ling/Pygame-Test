import os
import argparse
from send_slack import send

from src.train import run
from src.envs import get_envs

if __name__ == '__main__':
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    envs = get_envs()

    parser = argparse.ArgumentParser()
    parser.add_argument('--update_on', default=False, action="store_true")
    parser.add_argument('--env_num', default=0)
    parser.add_argument('--seed_num', default=0)

    args = parser.parse_args()
    
    print(envs[int(args.env_num)].name, envs[int(args.env_num)].max_episode)
    
    run(envs[int(args.env_num)], int(args.seed_num), args.update_on)
    
    send('Complete : ' + envs[int(args.env_num)].name + '\n seed: ' + args.seed_num + '\n update-on' + str(args.update_on))