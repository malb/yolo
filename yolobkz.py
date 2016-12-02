# -*- coding: utf-8 -*-

from random import randint

from fpylll import LLL, BKZ, GSO, Enumeration, EnumerationError, IntegerMatrix, prune
from fpylll.util import gaussian_heuristic
from time import time, clock
from random import randint


AUTO_MAX_BLOCK_SIZE = 200
GH_FACTOR = 1.2
NODE_PER_SEC = 2**25


class Timer:
    def __init__(self):
        self.start = clock()

    reset = __init__

    def elapsed(self):
        return clock() - self.start


class Tuner(object):

    def __init__(self):
        pass

    def preprocess(self, M, k, b):
        return []

    def enum(self, M, k, b, target_prob, preproc_time):
        
        radius = M.get_r(k, k)
        root_det = M.get_root_det(k, k + b)
        gh_radius, ge = gaussian_heuristic(radius, 0, b, root_det, GH_FACTOR)
        radius = min(radius, gh_radius * 2**ge)
        R = tuple([M.get_r(i, i) for i in range(k, k+b)])
        overhead = preproc_time * NODE_PER_SEC
        print R
        print radius
        print overhead
        print target_prob
        pruning = prune(radius, overhead, target_prob, [R], descent_method="gradient", precision=53)

        return radius, pruning

    def enum_for_hints(self, M, k, b, preproc_time):
        return 0, None

    def feedback(self, M, b, preprocessing, pruning, time):
        pass


class YoloBKZ(object):

    def __init__(self, A, tuner=None):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        if isinstance(A, LLL.Reduction):
            L, M, B = A, A.M, A.M.B
        elif isinstance(A, GSO.Mat):
            L, M, B = None, A, A.B
        elif isinstance(A, IntegerMatrix):
            L, M, B = None, None, A
        else:
            raise TypeError("type of A must be in {IntegerMatrix, GSO.Mat, LLL.Reduction}, but got type '%s'"%type(A))

        if M is None and L is None:
            wrapper = LLL.Wrapper(B)
            wrapper()
        if M is None:
            M = GSO.Mat(B, flags=GSO.ROW_EXPO)
        if L is None:
            L = LLL.Reduction(M, flags=LLL.DEFAULT)

        self.lll_obj, self.M, self.B = L, M, B
        self.lll_obj()        
        self.enum_obj = Enumeration(self.M)
        self.enum_obj = Enumeration(self.M)

        if tuner is None:
            self.tuner = Tuner()
        else:
            self.tuner = tuner

    def tour(self, b, target_prob=0.5, begin=0, end=None):
        if end is None:
            end = self.M.d

        for k in range(begin, end):
            tmp_b = min(b, end - k)
            self.svp_reduce(k, b, target_prob)

    def preprocess(self, k, b, preprocessing):
        begin = k
        end = k + b
        for (preproc_b, preproc_target_prob) in preprocessing:
            self.tour(preproc_b, preproc_target_prob, begin, end)

    def filter_hints(self, hints):
        return [v for v in hints if sum([x*x for x in v]) > 1.5]      

    def insert(self, k, b, solution, hints):
        M = self.M

        if solution is not None:
            vectors = [solution]+hints
        else:
            if len(hints)==0:
                return 0
            fake_solution = tuple([1] + (b-1)*[0])
            vectors = [fake_solution]+hints
        l = len(vectors)

        for vector in vectors:
            M.create_row()
            with M.row_ops(M.d-1, M.d):
                for i in range(b):                    
                    M.row_addmul(M.d-1, k + i, vector[i])

        for i in reversed(range(l)):
            M.move_row(M.d-1, k)

        # with stats.context("lll"):
        #     self.lll_obj(k, k, k+b+l)

        for i in range(l):
            M.move_row(k+b, M.d-1)
            M.remove_last_row()
        M.update_gso()

        if solution is None:
            return l - 1
        else:
            return l

    def randomize(self, k, b):
        inserted_count = max(1, min(b, inserted_count))        
        assert False

    def enum(self, k, b, radius, pruning, for_hints=False):
        solutions = []
        try:
            solution, _ = self.enum_obj.enumerate(k, k + b, radius, 0, pruning=pruning)  # , aux_sols=solutions)
            print solution
            return solution, []
            return solutions[0], solutions[1:]
        except EnumerationError:
            return None, []

    def svp_reduce(self, k, b, target_prob):

        timer = Timer()
        rem_prob, inserted = 1.0, 1
        M = self.M

        while rem_prob > 1. - target_prob:
            tmp_target_prob =  1.01 * (target_prob - 1)/rem_prob + 1.01
            timer.reset()

            if inserted ==0:
                randomize(self, k+1, k+b)

            preprocessing = self.tuner.preprocess(M, k, b)
            self.preprocess(k, b, preprocessing)

            radius, pruning = self.tuner.enum(M, k, b, tmp_target_prob, timer.elapsed())
            solution, hints = self.enum(k, b, radius, pruning.coefficients)

            if pruning is None:
                rem_prob = 0
            else:
                rem_prob *= (1 - pruning.probability)
            self.tuner.feedback(M, b, preprocessing, pruning, timer.elapsed())

            radius, pruning = self.tuner.enum_for_hints(M, k, b, timer.elapsed())
            if radius>0:
                hints += self.enum(k, b, radius, pruning.coefficients, for_hints=False)

            hints = self.filter_hints(hints)
            inserted = self.insert(k, b, solution, hints)


n = 120
b = 20
A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
yBKZ = YoloBKZ(A)

t = time()
yBKZ.tour(b)
t = time() - t
print "  time: %.2fs"%(t,)
