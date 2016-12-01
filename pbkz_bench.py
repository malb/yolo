import random
import time

from copy import copy
from fpylll import IntegerMatrix, BKZ, load_strategies_json
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.algorithms.pbkz import BKZReduction as ParallelBKZ


def run_timing_test(n, block_size=60, bits=30, ncores=8, min_success_probability=0.5, max_loops=4):
    A = IntegerMatrix.random(n, "qary", k=n//2, bits=bits)
    default_strategies = load_strategies_json(BKZ.DEFAULT_STRATEGY)

    param = BKZ.Param(block_size, strategies=default_strategies,
                      min_success_probability=min_success_probability,
                      max_loops=max_loops,
                      flags=BKZ.VERBOSE|BKZ.MAX_LOOPS)

    random.seed(1)
    bkz = ParallelBKZ(copy(A), ncores=ncores)
    t = time.time()
    bkz(param)
    t = time.time() - t
    print "Parallel(%d): %.2fs"%(ncores, t)

    random.seed(1)
    param = BKZ.Param(block_size, strategies=default_strategies,
                      min_success_probability=min_success_probability,
                      max_loops=max_loops,
                      flags=BKZ.VERBOSE|BKZ.MAX_LOOPS)
    t = time.time()
    BKZ2(copy(A))(param)
    t = time.time() - t
    print "  Sequential: %.2fs"%(t,)
