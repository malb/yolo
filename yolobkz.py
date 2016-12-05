# -*- coding: utf-8 -*-

from random import randint

from fpylll import LLL, BKZ, GSO, Enumeration, EnumerationError, IntegerMatrix, prune
from fpylll.util import gaussian_heuristic
from fpylll.algorithms.bkz_stats import BKZTreeTracer
from time import time, clock
from random import randint


AUTO_MAX_BLOCK_SIZE = 200
GH_FACTOR = 1.1
NODE_PER_SEC = 2**25
RESTART_PENALTY = 0.01


class Timer:
    def __init__(self):
        self.start = clock()

    reset = __init__

    def elapsed(self):
        return clock() - self.start


class Tuner(object):

    def __init__(self):
        self.last_prunings = AUTO_MAX_BLOCK_SIZE * [None]

    def preprocess(self, M, k, b):
        return []

    def enum(self, M, k, b, target_prob, preproc_time):

        radius = M.get_r(k, k) * .99
        if b > 30:
            root_det = M.get_root_det(k, k + b - 1)
            gh_radius, ge = gaussian_heuristic(radius, 0, b, root_det, GH_FACTOR)
            radius = min(radius, gh_radius * 2**ge)

        R = tuple([M.get_r(i, i) for i in range(k, k+b)])
        overhead = (preproc_time + RESTART_PENALTY) * NODE_PER_SEC
        start_from = self.last_prunings[b]
        pruning = prune(radius, overhead, target_prob, [R], 
                        descent_method="gradient", precision=53, start_from=start_from)
        self.last_prunings[b] = pruning.coefficients
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

        self.lll_obj, self.M, self.A = L, M, B
        self.lll_obj()

        if tuner is None:
            self.tuner = Tuner()
        else:
            self.tuner = tuner
        self.tracer = BKZTreeTracer(self, verbosity=True)

    def tour(self, b, target_prob=0.5, begin=0, end=None):
        if end is None:
            end = self.M.d

        for k in range(begin, end - 2):
            tmp_b = min(b, end - k)
            self.svp_reduce(k, tmp_b, target_prob)

    def preprocess(self, k, b, preprocessing):
        begin = k
        end = k + b
        with self.tracer.context("lll"):
            self.lll_obj(k, k, k+b)        
        for (preproc_b, preproc_target_prob) in preprocessing:
            self.tour(preproc_b, preproc_target_prob, begin, end)

    def filter_hints(self, hints):
        return [v for v in hints if sum([x*x for x in v]) > 1.5]      

    def insert(self, k, b, solution, hints=[]):
        M = self.M

        if (solution is not None) and len(hints)==0:
            nonzero_vectors = len([x for x in solution if x])
            if nonzero_vectors == 1:
                first_nonzero_vector = None
                for i in range(b):
                    if abs(solution[i]) == 1:
                        first_nonzero_vector = i
                        break

                M.move_row(k + first_nonzero_vector, k)
                with self.tracer.context("lll"):
                    self.lll_obj.size_reduction(k, k + first_nonzero_vector + 1)
                return 1

        if solution is not None:
            vectors = [solution] + hints
        else:
            if len(hints)==0:
                return 0
            vectors = hints
        l = len(vectors)

        for vector in vectors:
            M.create_row()
            with M.row_ops(M.d-1, M.d):
                for i in range(b):                    
                    M.row_addmul(M.d-1, k + i, vector[i])

        for i in reversed(range(l)):
            M.move_row(M.d-1, k)

        with self.tracer.context("postproc"):
            self.lll_obj(k, k, k+b+l)

        for i in range(l):
            M.move_row(k+b, M.d-1)
            M.remove_last_row()

        return l

    def randomize(self, k, b):
        # assert False
        pass

    def enum(self, k, b, radius, pruning, for_hints=False):
        solutions = []
        try:
            enum_obj = Enumeration(self.M, b/2, always_update_radius=True)
            with self.tracer.context("enumeration", enum_obj=enum_obj, probability=pruning.probability):
                enum_obj.enumerate(k, k + b, radius, 0, pruning=pruning.coefficients, aux_sols=solutions)
            return solutions[0][0], [sol for (sol, _) in solutions[:1]]
        except EnumerationError:
            return None, []

    def svp_reduce(self, k, b, target_prob):

        timer = Timer()
        rem_prob, inserted = 1.0, 1
        M = self.M

        while rem_prob > 1. - target_prob:
            tmp_target_prob =  1.01 * (target_prob - 1)/rem_prob + 1.01            

            if inserted == 0:
                assert False
                self.randomize(k+1, k+b)

            with self.tracer.context("preprocessing"):
                preprocessing = self.tuner.preprocess(M, k, b)
                self.preprocess(k, b, preprocessing)

            with self.tracer.context("pruner"):
                radius, pruning = self.tuner.enum(M, k, b, tmp_target_prob, timer.elapsed())
            solution, hints = self.enum(k, b, radius, pruning)

            if pruning is None:
                rem_prob = 0
            else:
                rem_prob *= (1 - pruning.probability)
            self.tuner.feedback(M, b, preprocessing, pruning, timer.elapsed())

            # radius, pruning = self.tuner.enum_for_hints(M, k, b, timer.elapsed())
            # if radius>0:
            #     hints += self.enum(k, b, radius, pruning, for_hints=False)
            
            # hints = self.filter_hints(hints)[:b/2]
            timer.reset()
            with self.tracer.context("postprocessing"):
                inserted = self.insert(k, b, solution, [])

    def __call__(self, b, tours=8):
        self.M.discover_all_rows()
        
        for i in range(tours):
            with self.tracer.context("tour", i):
                self.tour(b)
            i += 1

        # self.tracer.exit()


# n = 160
# b = 45
# A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
# yBKZ = YoloBKZ(A)

# t = time()
# yBKZ(b)
# t = time() - t
# print "  time: %.2fs"%(t,)
