# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:50:03 2019

@author: DSU
"""

import sys
import experiments
import longest_common_substring as lcs

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('There needs two be two file name arguments to find the lcs.')
        exit()
    
    # read in text files
    a = experiments.read_text(sys.argv[1])
    b = experiments.read_text(sys.argv[2])
    
    # get the length of lcs
    #length = lcs.suffix_search_lcs(a, b)
    length = lcs.exp_search_lcs(a, b)
    
    # print substrings
    lcs.print_common(length, a, b)