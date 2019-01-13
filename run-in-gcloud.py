from functools import partial
import multiprocessing
from multiprocessing.dummy import Pool
from subprocess import call

import timeit

# gcloud compute instances start pytorch-1-vm
# gcloud compute ssh --project pytorch-ml-228505 --zone us-west1-b pytorch-1-vm -- -L 8080:localhost:8080

if __name__ == '__main__':
    start = timeit.default_timer()

    pool = Pool(multiprocessing.cpu_count() - 1)
    commands = []
    for env_num in [1]:
        for i in range(1, 11):
            seed_num = i * 100
            commands.append('python test.py --seed_num=%d --env_num=%d --update_on' % (seed_num, env_num))
            commands.append('python test.py --seed_num=%d --env_num=%d' % (seed_num, env_num))
    
    for i, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
        if return_code != 0:
            print('%d command fail: %d, %s' % (i, return_code, commands[i]))

    
    stop = timeit.default_timer()

    print('Time: ', stop - start)




# Stop Instance If all python task is done
# !sudo poweroff