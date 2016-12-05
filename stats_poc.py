# -*- coding: utf-8 -*-
import copy
from fpylll import IntegerMatrix, BKZ
from fpylll.algorithms.bkz import BKZReduction as BKZ1
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

n = 100
A = IntegerMatrix.random(n, "qary", k=n//2, bits=40)
param = BKZ.Param(block_size=40, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.MAX_LOOPS|BKZ.VERBOSE)

bkz = BKZ1(copy.copy(A))
bkz(param)
print(bkz.trace.report())


bkz = BKZ2(copy.copy(A))
bkz(param)
print(bkz.trace.report())
stat = bkz.trace.tour[0].find("enumeration")["%"]
print stat.min, stat.max
