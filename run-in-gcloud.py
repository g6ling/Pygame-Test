from functools import partial
import multiprocessing
from multiprocessing.dummy import Pool
from subprocess import call

from src.envs import make_log_file_name
from send_slack import send
import timeit
import os

# gcloud compute instances start pytorch-1-vm
# gcloud compute ssh --project pytorch-ml-228505 --zone us-west1-b pytorch-1-vm -- -L 8080:localhost:8080

# 시작 slack
# 종료시 걸린 시간, 실행했던 commands 들 slack
# 종료시 종료.

if __name__ == '__main__':
    send('Start Commands')

    envs_name = ['cartpole', 'catcher', 'snake', 'flappybird']
    pool = Pool(4)
    commands = []
    def push_command(env_num, seed_num, update_on, sequence_length, replay_memory):
        
        make_log_file_name(envs_name[env_num], seed_num, update_on, sequence_length, replay_memory)
        if update_on:
            commands.append('python test.py --seed_num=%d --env_num=%d --sequence_length=%d --replay_memory=%d --update_on' % (seed_num, env_num, sequence_length, replay_memory))
        else:
            commands.append('python test.py --seed_num=%d --env_num=%d --sequence_length=%d --replay_memory=%d' % (seed_num, env_num, seed_num, replay_memory))

    for env in [0,1,2,3]:
        for i in range(1, 4+1):
            seed = i * 100
            push_command(env, seed, True, 8, 100)
            push_command(env, seed, True, 8, 1000)
            push_command(env, seed, False, 8, 100)
            push_command(env, seed, False, 8, 1000)
    
    # for command in commands:
    #     call(command, shell=True)
    for i, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
        if return_code != 0:
            print('%d command fail: %d, %s' % (i, return_code, commands[i]))    





# Stop Instance If all python task is done
# !sudo shutdown -h now