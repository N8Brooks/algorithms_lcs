# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:01:04 2019

@author: DSU
"""

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from stopwatch import stopwatch
from longest_common_substring import algorithms

from string import ascii_lowercase as alpha
from random import choices, choice
from tqdm import trange

TRIALS = 100
MAX_LEN = 1000
MAX_TIME = 3
DATA_TYPE = 'random'

# generate strings a and b for i tests based on the type of data you want
def create_data(i, version=DATA_TYPE):
    if version == 'random':
        return [(''.join(choices(alpha, k=i)), ''.join(choices(alpha, k=i)))\
                for _ in range(TRIALS)]
    elif version == 'worst':
        return [[choice(alpha)*i]*2 for _ in range(TRIALS)]
    elif version == 'text':
        print("I'm working on it ...")
        raise NotImplementedError
    else:
        print(f'{version} creation data type does not exist.')
        raise ValueError

# test algo for each item in data return time in seconds it takes
# best of 3, corrected for time
def test_algo(algo, data, record):
    # timing class and times data structure
    clock = stopwatch()
    clock.start()
    times = list()
    
    # test algorithm for data
    for a, b in data:
        clock.start()
        algo(a, b)
        times.append(clock.time())
    
    # find best of 3, find average, correct for nanoseconds, add to record
    times = sorted(times)[:len(data) // 3]
    average = sum(times) / len(times) / 1e9
    record[algo.__name__] = average

# spawn and watch the algorithm
def task_algo(algo, data, record, skip, bar):
    if skip.get(algo.__name__, False):
        return
    
    p = mp.Process(target=test_algo, args=(algo, data, record))
    p.start()
    p.join(timeout=MAX_TIME)
    p.terminate()
    
    if p.exitcode is None:
        bar.write(f'Skipping {algo.__name__} on {i} due to timeout.')
        skip[algo.__name__] = True


# the experiment
if __name__ == '__main__':
    manager = mp.Manager()
    record = manager.dict()
    skip = manager.dict()
    df = pd.DataFrame()
    
    bar = trange(MAX_LEN)
    for i in bar:
        
        strings = create_data(i)
        record.clear()
        
        # test each algorithm with timeout
        for algo in algorithms:
            task_algo(algo, strings, record, skip, bar)
        
        # no algos finished
        if len(record) is 0:
            break
        
        df = df.append(pd.Series(dict(record), name=i))
    
    # plot data
    df.plot()
    plt.show()
    
    # record data
    df.to_csv(f'{DATA_TYPE}_data.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    