# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:50:03 2019

@author: DSU
"""

import sys
import experiments

from suffixtree import SuffixTree

# suffix search lcs with more output
def suffix_search_lcs(a, b):
    # left and right bounds, max sizes
    len_a, len_b = len(a) + 1, len(b) + 1
    short = min(len(a), len(b))
    
    tree = SuffixTree(False, [a])
    print('Completed suffix tree')
    
    # returns if there is a common substring of length m between a, b
    def found_common(m):
        return any(tree.findStringIdx(b[i-m:i]) for i in range(m, len_b))
    
    # exponentially increase l and r
    l, r = 0, 1
    print(l, r)
    while r < len_a and found_common(r):
        l, r = r + 1, r * 2
        print(l, r)
    r = min(r, short)
    print(l, r)
    
    # right-most binary search on if substring length is possible
    while l <= r:
        m = (l + r) // 2
        print(m)
        
        if found_common(m):
            l = m + 1
        else:
            r = m - 1
    
    print('Longest Common Substrings:')
    print('\n'.join(set(b[i-r:i] for i in range(r, len_b) if tree.findStringIdx(b[i-r:i]))))
    
    return r

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('There needs two be two file name arguments to find the lcs.')
        exit()
    
    # read in text files
    a = experiments.read_text(sys.argv[1])
    b = experiments.read_text(sys.argv[2])
    
    # get the length of lcs
    length = suffix_search_lcs(a, b)
    #length = lcs.exp_search_lcs(a, b)
    
    # print substrings
    #lcs.print_common(length, a, b)