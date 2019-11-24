# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:05:58 2019

@author: DSU
"""

import numpy as np
import sys

from collections import defaultdict
from itertools import combinations
from itertools import accumulate, chain
from suffixtree import SuffixTree
from string import ascii_lowercase as alpha
from random import choices, randrange

# helper function to print the common substrings of length m between a, b
def print_common(m, a, b):
    len_a, len_b = len(a) + 1, len(b) + 1
    a_substr = set(a[i-m:i] for i in range(m, len_a))
    found = {b[i-m:i] for i in range(m, len_b) if b[i-m:i] in a_substr}
    print('\n'.join(found))

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
        
        a_substr = set(a[i-m:i] for i in range(m, len_a))
        if any(b[i-m:i] in a_substr for i in range(m, len_b)):
            l = m + 1
        else:
            r = m - 1
    
    return r

# exponentially increase, then binary search
def exp_search_lcs(a, b):
    # left and right bounds, max sizes
    len_a, len_b = len(a) + 1, len(b) + 1
    short = min(len(a), len(b))
    
    # returns if there is a common substring of length m between a, b
    def found_common(m):
        a_substr = set(a[i-m:i] for i in range(m, len_a))
        return any(b[i-m:i] in a_substr for i in range(m, len_b))
    
    # exponentially increase l and r
    l, r = 0, 1
    while r < len_a and found_common(r):
        l, r = r + 1, r * 2
    r = min(r, short)
    
    # right-most binary search on if substring length is possible
    while l <= r:
        m = (l + r) // 2
        
        if found_common(m):
            l = m + 1
        else:
            r = m - 1
    
    return r

# same as last one except uses a suffix tree to determine if substring of a
def suffix_search_lcs(a, b):
    # left and right bounds, max sizes
    len_a, len_b = len(a) + 1, len(b) + 1
    short = min(len(a), len(b))
    
    tree = SuffixTree(False, [a])
    
    # returns if there is a common substring of length m between a, b
    def found_common(m):
        return any(tree.findStringIdx(b[i-m:i]) for i in range(m, len_b))
    
    # exponentially increase l and r
    l, r = 0, 1
    while r < len_a and found_common(r):
        l, r = r + 1, r * 2
    r = min(r, short)
    
    # right-most binary search on if substring length is possible
    while l <= r:
        m = (l + r) // 2
        
        if found_common(m):
            l = m + 1
        else:
            r = m - 1
    
    return r

# dynamic version from geeksforgeeks - https://www.geeksforgeeks.org/longest-
# common-substring-dp-29/
def dynamic_lcs(a, b):
    # lengths and memoization table
    m, n = len(a) + 1, len(b) + 1
    dp = defaultdict(int)
    
    # build up memo
    for i in range(1, m):
        for j in range(1, n):
            if a[i-1] == b[j-1]:
                dp[(i,j)] = dp[(i-1, j-1)] + 1
    
    # return highest (defaults to 0)
    return max(0, 0, *dp.values())

# recursive solution from https://www.geeksforgeeks.org/longest-common-
# substring-dp-29/
def recursive_lcs(a, b):
    
    # helper function for recursion
    def lcs(i, j, count):
        if i is 0 or j is 0:  
            return count  
              
        if a[i - 1] == b[j - 1]: 
            count = lcs(i - 1, j - 1, count + 1)  
        
        return max(count, max(lcs( i, j - 1, 0), lcs( i - 1, j, 0))) 

    return lcs(len(a), len(b), 0)

# rosetta code indexes - https://rosettacode.org/wiki/Longest_Common_Substring
def iterative_lcs(s1, s2):
    if len(s2) > len(s1):
        s1, s2 = s2, s1
    len1, len2 = len(s1), len(s2)
    ir, jr = 0, 0
    for i1 in range(len1):
        i2 = s2.find(s1[i1])
        while i2 is not -1:
            j1, j2 = i1+1, i2+1
            while j1 < len1 and j2 < len2 and s2[j2] == s1[j1]:
                if j1-i1 > jr-ir:
                    ir, jr = i1, j1
                j1 += 1; j2 += 1
            i2 = s2.find(s1[i1], i2+1)
            
    #print (s1[ir:jr+1])
    return jr - ir + 1 if jr is not 0 else int(bool(set(s1).intersection(s2)))

# using functions - https://rosettacode.org/wiki/Longest_Common_Substring
def functional_lcs(a, b):
    # compose (<<<) :: (b -> c) -> (a -> b) -> a -> c
    def compose(g):
        return lambda f: lambda x: g(f(x))
     
    # concat :: [String] -> String
    def concat(xs):
        return ''.join(chain.from_iterable(xs))
     
    # concatMap :: (a -> [b]) -> [a] -> [b]
    def concatMap(f):
        return lambda xs: list(
            chain.from_iterable(
                map(f, xs)
            )
        )
     
    # inits :: [a] -> [[a]]
    def inits(xs):
        return scanl(lambda a, x: a + [x])(
            []
        )(list(xs))
     
    # intersect :: [a] -> [a] -> [a]
    def intersect(xs, ys):
        s = set(ys)
        return [x for x in xs if x in s]
     
    # map :: (a -> b) -> [a] -> [b]
    def map_(f):
        return lambda xs: list(map(f, xs))
     
    # scanl is like reduce, but returns a succession of
    # intermediate values, building from the left.
    # scanl :: (b -> a -> b) -> b -> [a] -> [b]
     
    def scanl(f):
        return lambda a: lambda xs: (
            list(accumulate([a] + list(xs), f))
        )
     
    # tail :: [a] -> [a]
    def tail(xs):
        return xs[1:]
     
     
    # tails :: [a] -> [[a]]
    def tails(xs):
        return list(map(
            lambda i: xs[i:],
            range(0, 1 + len(xs))
        ))
        
    def longestCommon(s1):
        return lambda s2: max(intersect(
            *map(lambda s: map(
                concat,
                concatMap(tails)(
                    compose(tail)(inits)(s)
                )
            ), [s1, s2])
        ), key=len)
                
    try:
        return len(longestCommon(a)(b))
    except:
        return 0

# using suffix tree - https://codereview.stackexchange.com/questions/183678/
# ukkonens-algorithm-for-longest-common-substring-search
def suffix_tree_lcs(s, t):
    """Return the length of the longest substring that appears in both s and t.
    This function builds a suffix tree of a combination of s and t using
    Ukkonen's algorithm. It assumes that the symbols $ and # appear in neither s
    nor t.
    """

    len_s = len(s)
    string = s + '#' + t + '$'
    max_len = 0

    class LeafNode():

        def __init__(self, from_first_word):
            self.from_first_word = from_first_word

        @property
        def has_s_leaves(self):
            return self.from_first_word

        @property
        def has_t_leaves(self):
            return not self.from_first_word

    class InternalNode():
        def __init__(self, root_length):
            self.edges = {}  # dictonary of edges keyed by first letter of edge
            self.link = None
            self.root_length = root_length
            self.has_s_leaves = False
            self.has_t_leaves = False
            self.already_counted = False

        def __getitem__(self, key):
            return self.edges[key]

        def __setitem__(self, key, edge):
            self.edges[key] = edge
            # Update leaf identity based on new child leaves
            # Using "or" is faster than "|=" (I guess |= doesn't short circuit)
            self.has_s_leaves = self.has_s_leaves or edge.dest.has_s_leaves
            self.has_t_leaves = self.has_t_leaves or edge.dest.has_t_leaves

        def __contains__(self, key):
            return key in self.edges

    class Edge():
        def __init__(self, dest, start, end):
            self.dest = dest
            self.start = start
            self.end = end
            self.length = self.end - self.start

    root = InternalNode(0)

    class Cursor():

        def __init__(self):
            self.node = root
            self.edge = None
            self.idx = 0
            self.lag = -1

        def is_followed_by(self, letter):
            if self.idx == 0:
                return letter in self.node
            return letter == string[self.node[self.edge].start + self.idx]

        def defer(self, letter):
            """When we defer the insertion of a letter,
            we need to advance the cursor one position.
            """
            self.idx += 1
            # We never want to leave the cursor at the end of an explicit edge.
            # If this is the case, move it to the beginning of the next edge.
            if self.edge is None:
                self.edge = letter
            edge = self.node[self.edge]
            if self.idx == edge.length:
                self.node = edge.dest
                self.edge = None
                self.idx = 0

        def post_insert(self, i):
            """When we are finished inserting a letter, we can pop
            it off the front of our queue and prepare the cursor for the
            next letter.
            """
            self.lag -= 1
            # Only when the current node is the root is the first letter (which
            # we must remove) part of the cursor edge and index. Otherwise it i
            # implicitly determined by the current node.
            if self.node is root:
                if self.idx > 1:
                    self.edge = string[i - self.lag]
                    self.idx -= 1
                else:
                    self.idx = 0
                    self.edge = None
            # Following an insert, we move the active node to the node
            # linked from our current active_node or root if there is none.
            self.node = self.node.link if self.node.link else root
            # When following a suffix link, even to root, it is possible to
            # end up with a cursor index that points past the end of the curre
            # edge. When that happens, follow the edges to a valid cursor
            # position. Note that self.idx might be zero and self.edge None.
            while self.edge and self.idx >= self.node[self.edge].length:
                edge = self.node[self.edge]
                self.node = edge.dest
                if self.idx == edge.length:
                    self.idx = 0
                    self.edge = None
                else:
                    self.idx -= edge.length
                    self.edge = string[i - self.lag + self.node.root_length]

        def split_edge(self):
            edge = self.node[self.edge]
            # Create a new node and edge
            middle_node = InternalNode(self.node.root_length + self.idx)
            midpoint = edge.start + self.idx
            next_edge = Edge(edge.dest, midpoint, edge.end)
            middle_node[string[midpoint]] = next_edge
            # Update the current edge to end at the new node
            edge.dest = middle_node
            edge.end = midpoint
            edge.length = midpoint - edge.start
            return middle_node


    cursor = Cursor()
    from_first_word = True
    dummy = InternalNode(0)

    for i, letter in enumerate(string):

        if from_first_word and i > len_s:
            from_first_word = False

        cursor.lag += 1
        prev = dummy  # dummy node to make suffix linking easier the first time

        while cursor.lag >= 0:

            if cursor.is_followed_by(letter):  # Suffix already exists in tree
                prev.link = cursor.node
                cursor.defer(letter)
                break

            elif cursor.idx != 0:  # We are part-way along an edge
                stem = cursor.split_edge()
            else:
                stem = cursor.node
            # Now we have an explicit node and can insert our new edge there.
            stem[letter] = Edge(LeafNode(from_first_word), i, sys.maxsize)
            # Whenever we update an internal node, we check for a new max_len
            # But not until we have started into the second input string
            if (i > len_s and not stem.already_counted
                and stem.has_s_leaves and stem.has_t_leaves):
                stem.already_counted = True
                if stem.root_length > max_len:
                    max_len = stem.root_length
            # Link the previously altered internal node to the new node and mak
            # the new node prev.
            prev.link = prev = stem
            cursor.post_insert(i)

    return max_len

# list of all algorithms
algorithms = [brute_force_lcs, short_hi_lcs, short_lo_lcs, binary_search_lcs,\
              suffix_search_lcs, exp_search_lcs, dynamic_lcs, recursive_lcs,\
              iterative_lcs, functional_lcs, suffix_tree_lcs]

# driver code for testing
if __name__ == '__main__':
    # hard test cases
    tests = [('abcdecfcg', 'ace', 1),
             ('oldsite: wiki', 'site:?', 5),
             ('lab num 7', 'b nu', 4),
             ('big test', 'big test', 8),
             ('one with none', 'zzz', 0),
             ('', '', 0)]
    
    # try each algorithm on tests and make sure they got the right answer
    for a, b, ans in tests:
        #print([algo(a, b) for algo in algorithms])
        for algo in algorithms:
            assert algo(a, b) == ans
            
    # creating large random test case where b has a portion of string a
    length = 1000
    i = randrange(length)
    j = randrange(i, length)
    insert = randrange(j-i)
    a = ''.join(choices(alpha, k=length))
    b = ''.join(choices(alpha, k=length-j+i))
    b = b[:insert] + a[i:j] + b[insert:]
    ans = brute_force_lcs(a, b)
    
    # hard coded algoriths that are fast enough
    for algo in [short_hi_lcs, short_lo_lcs, binary_search_lcs, 
                 suffix_search_lcs, exp_search_lcs, dynamic_lcs,
                 iterative_lcs, suffix_tree_lcs]:
        assert algo(a, b) == ans
    
    # confirm tests passed if it makes it here
    print('All test cases passed')
            
            
























