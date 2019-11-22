# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:52:37 2019

@author: DSU
"""
import time

class stopwatch:
    """
    when stopwatch starts, get and save the current CPU time for the current
        thread 
    we can always call this to reset the stopwatch 
    """
    def start(self) -> None:
        self.stopWatchStartTimeNanoSecs = time.clock()
    
    """
    get elapsed time in nanoseconds 
    Note: this will includes some time for the overhead of calling & executing
        this method itself as well as some of the time involved in executing
        and returning from the start() method 
    calculate elapsed time by getting current CPU time and substracting the CPU
        time from when we started the stopwatch this does not stop the
        stopwatch 
    """
    def time(self) -> float:
        return (time.clock() - self.stopWatchStartTimeNanoSecs) * 1e9

"""
Driver code which demonstrates timing a sleep of 1s
"""
if __name__ == '__main__':
    clock = stopwatch()
    clock.start()
    time.sleep(1)
    print(clock.time())