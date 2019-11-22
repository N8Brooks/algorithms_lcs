# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:05:58 2019

@author: DSU
"""
from collections import defaultdict
from itertools import combinations
from array import array

# generate sets of every substring, find longest common - O(n^2)
def brute_force_lcs(a, b):
    a_substr = {a[x:y] for x, y in combinations(range(len(a)+1), 2)}
    b_substr = {b[x:y] for x, y in combinations(range(len(b)+1), 2)}
    both = a_substr.intersection(b_substr)
    return len(max(both, key=len)) if both else 0

# starts at highest possible then searches until it finds common - O(n^2)
def short_hi_lcs(a, b):
    # a should be shorter
    if len(b) < len(a):
        a, b = b, a
    
    # start at longest possible then drop
    for length in range(len(a) + 1, 0, -1):
        b_substrings = set(b[i-length:i] for i in range(length, len(b)+1))
        if any(a[i-length:i] in b_substrings for i in range(length, len(b)+1)):
            return length
    
    # didn't find any
    return 0

# starts at 1 then searches until it can't find one - O(n^2)
def short_lo_lcs(a, b):
    # a should be shorter
    if len(b) < len(a):
        a, b = b, a
    
    # start at 0 then increase
    for length in range(1, len(a) + 1):
        b_substrings = set(b[i-length:i] for i in range(length, len(b)+1))
        if not any(a[i-length:i] in b_substrings for i in\
                   range(length, len(b)+1)):
            return length - 1
    
    # max size hit
    return len(a)
    
# binary search of lenths - O(n^2)
def binary_search_lcs(a, b):
    # a should be shorter
    if len(b) < len(a):
        a, b = b, a
    
    # left and right bounds, max sizes
    l, r = 0, len(a)
    len_a, len_b = len(a) + 1, len(b) + 1
    
    # right-most binary search on if substring length is possible
    while l <= r:
        m = (l + r) // 2
        
        a_substrings = set(a[i-m:i] for i in range(m, len_a))
        if any(b[i-m:i] in a_substrings for i in range(m, len_b)):
            l = m + 1
        else:
            r = m - 1
    
    return r

# dynamic version from geeksforgeeks using suffix tree
def dynamic_lcs(a, b):
    # lengths and memoization table
    m, n = len(a) + 1, len(b) + 1
    dp = defaultdict(int)
    
    # build up suffix tree
    for i in range(1, m):
        for j in range(1, n):
            if a[i-1] == b[j-1]:
                dp[(i,j)] = dp[(i-1, j-1)] + 1
    
    # return highest (defaults to 0)
    return max(0, 0, *dp.values())

# recursive solution from geeksforgeeks
def recursive_lcs(a, b):
    # helper function for recursion
    def lcs(i, j, count) :
        if i is 0 or j is 0:  
            return count  
              
        if a[i - 1] == b[j - 1]: 
            count = lcs(i - 1, j - 1, count + 1)  
        
        return max(count, max(lcs( i, j - 1, 0), lcs( i - 1, j, 0))) 

    return lcs(len(a), len(b), 0)

# list of all algorithms
algorithms = [brute_force_lcs, short_hi_lcs, short_lo_lcs, binary_search_lcs,\
              dynamic_lcs, recursive_lcs]

# driver code for testing
if __name__ == '__main__':
    # test cases
    tests = [('abcdefg', 'ace', 1),
             ('oldsite: wiki', 'site:?', 5),
             ('lab num 7', 'b nu', 4),
             ('big test', 'big test', 8),
             ('one with none', 'zzz', 0),
             ('', '', 0)]
    
    # try each algorithm on tests and make sure they got the right answer
    for a, b, ans in tests:
        for algo in algorithms:
            assert algo(a, b) == ans
    
    # confirm tests passed if it makes it here
    print('All test cases passed')
            
            
























