# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:54:36 2019

@author: DSU
"""

import pandas as pd
import numpy as np
from longest_common_substring import algorithms

DATA_TYPE = 'random'

algorithms = [algo.__name__ for algo in algorithms]
df = pd.read_csv(f'{DATA_TYPE}_data.csv', index_col='Unnamed: 0')

for algo in algorithms:
    if algo not in df.columns: continue
    data = df[algo].dropna()
    params = np.polyfit(data.index, data, 2)
    f = np.poly1d(params)
    df[algo+'_predict'] = pd.Series(data.index.map(f), index=data.index)
    df[[algo, algo+'_predict']].dropna().plot()
