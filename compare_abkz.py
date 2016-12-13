# -*- coding: utf-8 -*-
"""

Test if Python BKZ classes can be instantiated and run.
"""
from copy import copy

from fpylll import IntegerMatrix, GSO, LLL
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from yolobkz import YoloBKZ, Tuner, Timer
from fpylll.util import set_random_seed
from fpylll import BKZ as fplll_bkz


ns = 140
block_sizes = range(50, 80)
tours = 5


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
    return A


def compare():
    tuners = [Tuner(b) for b in range(100)]
    recycled_tuners = [Tuner(b) for b in range(100)]

    A = make_integer_matrix(n)
    M = GSO.Mat(A, flags=GSO.ROW_EXPO)
    lll_obj = LLL.Reduction(M, flags=LLL.DEFAULT)
    lll_obj()

    for n in ns:
        params = fplll_bkz.Param(block_size=bs, max_loops=tours, 
                                 flags=fplll_bkz.VERBOSE|fplll_bkz.GH_BND, strategies="default.json")

        print
        print "======"
        print "dim ", n
        print "======"

        # print 
        # print "yoloBKZ"
        # B = copy(A)
        # YoloBKZ(B, tuners=tuners, recycle=False)(b=bs, tours=tours)

        print 
        print "Recycled yoloBKZ"
        # B = copy(A)
        # YoloBKZ(B, tuners=recycled_tuners)(b=bs, tours=1)
        # print
        # print "Restart"
        # print

        B = copy(A)
        timer = Timer()
        YoloBKZ(B, tuners=recycled_tuners)(b=bs, tours=tours)
        print "Total: %.3f"%timer.elapsed()

        # print 
        # print "Good old BKZ2.0"

        # B = copy(A)
        # timer = Timer()
        # BKZ2(B)(params=params)
        # print "Total: %.3f"%timer.elapsed()


compare()