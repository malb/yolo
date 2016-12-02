# -*- coding: utf-8 -*-
"""

Test if Python BKZ classes can be instantiated and run.
"""
from copy import copy

from fpylll import IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from autobkz import BKZReduction as aBKZ
from autobkz import AutoPreprocDecider
from fpylll.util import set_random_seed
from fpylll import BKZ as fplll_bkz


n = 80
block_sizes = range(40, 80)


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "ntrulike", bits=30)
    return A


def compare():
    for bs in block_sizes:
        params = fplll_bkz.Param(block_size=bs, max_loops=8, 
                                 flags=fplll_bkz.VERBOSE|fplll_bkz.GH_BND, strategies="default.json")
        A = make_integer_matrix(n)
        
        print
        print "======"
        print "BLOCKSIZE ", bs
        print "======"

        print 
        print "autoBKZ"
        print 
        decider = AutoPreprocDecider()
        B = copy(A)
        aBKZ(B, decider)(params=params)
        decider.report(bs)

        B = copy(A)
        aBKZ(B, decider)(params=params)
        decider.report(bs)

        print 
        print "Good old BKZ2.0"
        print 

        B = copy(A)
        BKZ2(B)(params=params)


compare()