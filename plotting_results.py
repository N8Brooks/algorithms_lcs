# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:54:36 2019

@author: DSU
"""

import sys
import pandas as pd
import numpy as np
import longest_common_substring as lcs

POWER = 1

# specify or default the file you want to plot
if len(sys.argv) > 1:
    DATA_TYPE = sys.argv[1]
    assert DATA_TYPE in ['random', 'sided', 'worst', 'text']
else:
    DATA_TYPE = 'text'    # ['random', 'sided', 'worst', 'text']

# plots single algorithm for each data type
def plot_algo(name):
    algorithms = [algo.__name__ for algo in lcs.algorithms]
    for DATA_TYPE in ['worst', 'random', 'text', 'sided']:
        df = pd.read_csv(f'{DATA_TYPE}_data.csv', index_col='Unnamed: 0')
        
        for algo in algorithms:
            if algo not in df.columns: continue
            if algo is not name: continue
            try:
                data = df[algo].dropna()
                params = np.polyfit(data.index, data, POWER)
                f = np.poly1d(params)
                df[algo+'_predict'] = pd.Series(data.index.map(f), index=data.index)
                df[[algo, algo+'_predict']].dropna().plot(title=DATA_TYPE)
            except:
                pass

def table_algo(name):
    algorithms = [algo.__name__ for algo in lcs.algorithms]
    data = pd.DataFrame()
    for DATA_TYPE in ['worst', 'random', 'text']:
        df = pd.read_csv(f'{DATA_TYPE}_data.csv', index_col='Unnamed: 0')
        if name in df.columns:
            data[name+'_'+DATA_TYPE] = df[name].dropna()
            data[name+'_'+DATA_TYPE+'_ratio'] = df[name].pct_change().add(1)
    
    return data

if __name__ == '__main__':
    df = table_algo('suffix_tree_lcs')