# -*- coding: utf-8 -*-
"""

Test if Python BKZ classes can be instantiated and run.
"""
from copy import copy

from multiprocessing import Process
from fpylll import IntegerMatrix, GSO, LLL, prune, Enumeration, EnumerationError
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.util import gaussian_heuristic
from yolobkz import YoloBKZ, Tuner, Timer
from fpylll.util import set_random_seed
from fpylll import BKZ as fplll_bkz
import sys
from time import sleep, clock, time
from plot import plot_and_save
from math import log

try:
    start = int(sys.argv[1])
except:
    start = 90

ns = range(start, 140, 2)
block_sizes = range(50, 80)
tours = 5


YOLO_PRUNER_PREC = 140


def load_basis(fin):
    f=open(fin, 'r')
    st = ""
    for l in f:
        st+=l
    # st = f.readline()
    X = eval(st)
    N = len(X[0])/2
    A = IntegerMatrix(2*N, 2*N)
    for i in xrange(0, 2*N):
        for j in xrange(0, 2*N):
            A[i, j] = X[i][j]
    f.close()
    return A


def make_integer_matrix(n):
    A = load_basis("svpchallenge_py/svpchallengedim%dseed0.txt"%n)
    return A

tuners = [Tuner(b) for b in range(140)]

GH_FACTOR = 1.1
NODE_PER_SEC = 2**24


def yolo_hsvp(n, A, gh_factor, core=0):
    timer = Timer()
    ybkz = YoloBKZ(A, tuners=tuners)

    start_from = None
    start_from_rec = None

    first_len = ybkz.M.get_r(0, 0)
    root_det = ybkz.M.get_root_det(0, n)

    gh_radius, ge = gaussian_heuristic(first_len, 0, n, root_det, 1.)
    gh_radius = abs(gh_radius * 2**ge)
    radius = gh_factor * gh_radius 

    target_prob = (1. / gh_factor)**(n/2)

    trial = 0
    count = 0
    restarted = 0
    ybkz.randomize(0, n, density=1)

    while True:
        timer.reset()
        max_efficiency = 0.
        for b in range(8, n/2, 4):
            ybkz.tour(b, target_prob=.50)

        restarted+=1
        for b in range(n/2, n-10, 2):
            count += 1
            ybkz.tour(b, target_prob=.10)
            overhead = NODE_PER_SEC * timer.elapsed()
            R = tuple([ybkz.M.get_r(i, i) for i in range(0, n)])

            title = "c=%d r=%d b=%d t=%.1fs"%(core, restarted, b, timer.elapsed())
            print title

            pruning = prune(radius, overhead, target_prob, [R], descent_method="hybrid", 
                            precision=53, start_from=start_from)
            start_from = pruning.coefficients
            print "c=%d  pruning approximated  t=%.1fs"%(core, timer.elapsed())

            pruning = prune(radius, overhead, target_prob, [R], 
                            descent_method="gradient", precision=YOLO_PRUNER_PREC, start_from=start_from)
            title = "c=%d r=%d b=%d t=%.1fs p=%1.2e e=%.1fs"%(core, restarted, b, timer.elapsed(), 
                    pruning.probability/target_prob, (target_prob*timer.elapsed())/pruning.probability)
            print title

            plot_and_save([log(x/gh_radius)/log(2.) for x in R], 
                          title, '%d/c%ds%d.png'%(n, core, count))

            start_from = pruning.coefficients
            try:
                enum_obj = Enumeration(ybkz.M)
                solution, _ = enum_obj.enumerate(0, n, radius, 0, pruning=pruning.coefficients)
                ybkz.insert(0, n, solution)
                print
                print list(A[0])
                return
            except EnumerationError:
                print "c=%d Enum failed  t=%.1fs"%(core, timer.elapsed())
                pass

            efficiency = (pruning.probability/timer.elapsed())

            #  RECYCLING
            r_start = count %10
            recycling_radius = ybkz.M.get_r(r_start, r_start) * .99
            pruning = prune(recycling_radius, overhead, target_prob, [R[r_start:]], 
                            descent_method="hybrid", precision=53)
            title = "REC c=%d r=%d b=%d t=%.1fs p=%1.2e e=%.1fs"%(core, restarted, b, timer.elapsed(), 
                    pruning.probability/target_prob, (target_prob*timer.elapsed())/pruning.probability)
            print title

            try:
                hints = []
                enum_obj = Enumeration(ybkz.M, n/2)
                solution, _ = enum_obj.enumerate(r_start, n, recycling_radius, r_start, 
                                                 pruning=pruning.coefficients, aux_sols=hints)
                hints=[sol for (sol, _) in hints[1:]]
                ybkz.insert(r_start, n, solution, hints=hints)
                print "c=%d Recycled %d t=%.1fs"%(core, len(hints)+1 , timer.elapsed())
                break 
            except EnumerationError:
                pass
            start_from_rec = pruning.coefficients
            #  END OF RECYCLING

            if 2*efficiency < max_efficiency:
                ybkz.randomize(0, n, density=1)
                ybkz.lll_obj(0, 0, n)
                break
            max_efficiency = max(efficiency, max_efficiency)
            timer.reset()


def proudly_parrallel(cores, f, args):
    procss = []
    for i in range(cores):
        args2 = list(copy(args))
        args2.append(i)
        procss.append(Process(target=f, args=tuple(args2)))
        procss[i].start()
    while True:
        sleep(.1)
        for proc in procss:
            if not proc.is_alive():
                for proc2 in procss:
                    proc2.terminate()
                return    
    while True:
        sleep(.1)
        some_alive = False
        for proc in procss:
            some_alive |= proc.is_alive()
        if not some_alive:
            return


def test():
    for n in ns:
        print
        print "++++ Dim ", n 
        A = make_integer_matrix(n)
        M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        lll_obj = LLL.Reduction(M, flags=LLL.DEFAULT)
        lll_obj()

        t_start = time()
        B = copy(A)
        timer = Timer()
        proudly_parrallel(3, yolo_hsvp, (n, B, 1.05**2))

        print "time: %.1f sec"%(time() - t_start)
test()