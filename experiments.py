# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:01:04 2019

@author: DSU
"""

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import re
import sys

from stopwatch import stopwatch
from longest_common_substring import algorithms

from string import ascii_lowercase as alpha
from random import choices, choice, randrange
from tqdm import trange

TRIALS = 1                              # how many trials of each n
TAKE = 1                                  # takes TAKE experiments out of TRIALS
MAX_LEN = 10000                          # max n that should be used
MAX_TIME = 10                            # how long before algo timeout
INCREMENT = 10                          # how much to increase n by
TEXT_DATA = 'text_files/moby_dick.txt'  # your text file for text data
DATA_TYPE = 'random'                    # default data type

# helper function to read in utf-8 text file
def read_text(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return re.sub('\s+', ' ', file.read()).strip()

book = read_text(TEXT_DATA)

# generate strings a and b for i tests based on the type of data you want
def create_data(i, version=DATA_TYPE):
    if version == 'random':
        data = [(''.join(choices(alpha, k=i)), ''.join(choices(alpha, k=i)))\
                for _ in range(TRIALS)]
    elif version == 'sided':
        data = [(''.join(choices(alpha, k=1000)), ''.join(choices(alpha, k=i)))\
                for _ in range(TRIALS)]
    elif version == 'worst':
        data = [[choice(alpha)*i]*2 for _ in range(TRIALS)]
    elif version == 'text':
        data = list()
        for _ in range(TRIALS):
            x = randrange(len(book) - i + 1)
            y = randrange(len(book) - i + 1)
            data.append((book[x:x+i], book[y:y+i]))
    else:
        print(f'{version} creation data type does not exist.')
        raise ValueError
    
    return data

# test algo for each item in data return time in seconds it takes
# best of TAKE, corrected for time
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
    
    # find best of TAKE, find average, correct for nanoseconds, add to record
    times = sorted(times)[:TAKE // len(data)]
    average = sum(times) / len(times) / 1e9
    record[algo.__name__] = average

# spawn and watch the algorithm
def task_algo(algo, data, record, skip, bar):
    if skip.get(algo.__name__, False):
        return
    
    # run the lcs function - if it excepts or times out, skip it from now on
    try:
        p = mp.Process(target=test_algo, args=(algo, data, record))
        p.start()
        p.join(timeout=MAX_TIME)
        p.terminate()
        
        if p.exitcode is None:
            bar.write(f'Skipping {algo.__name__} on {i} due to timeout.')
            skip[algo.__name__] = True
    except:
        bar.write(f'Skipping {algo.__name__} on {i} due to exception.')
        skip[algo.__name__] = True

# the experiment
if __name__ == '__main__':
    # make sure that you have a file type
    assert len(sys.argv) is 2
    
    # make sure the argument makes sense
    DATA_TYPE = str(sys.argv[1])
    assert DATA_TYPE in ['random', 'sided', 'worst', 'text']
    
    manager = mp.Manager()
    record = manager.dict()
    skip = manager.dict()
    df = pd.DataFrame()
    
    # perform the experiments increasing n each time
    bar = trange(0, MAX_LEN, INCREMENT)
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
        
        # record the data every 100 iterations
        if i % 100 is 0:
            df.to_csv(f'{DATA_TYPE}_data.csv')
    
    # record the final
    df.to_csv(f'{DATA_TYPE}_data.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    