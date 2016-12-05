# -*- coding: utf-8 -*-
"""

Test if Python BKZ classes can be instantiated and run.
"""
from copy import copy

from fpylll import IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from yolobkz import YoloBKZ
from fpylll.util import set_random_seed
from fpylll import BKZ as fplll_bkz


n = 140
block_sizes = range(10, 80)
tours = 8


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
    return A


def compare():
    for bs in block_sizes:
        params = fplll_bkz.Param(block_size=bs, max_loops=tours, 
                                 flags=fplll_bkz.VERBOSE|fplll_bkz.GH_BND, strategies="default.json")
        A = make_integer_matrix(n)
        
        print
        print "======"
        print "BLOCKSIZE ", bs
        print "======"
        print 
        print "Good old BKZ2.0"

        B = copy(A)
        BKZ2(B)(params=params)

        print 
        print "yoloBKZ"
        print 

        B = copy(A)
        YoloBKZ(B)(b=bs, tours=tours)


compare()