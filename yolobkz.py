# -*- coding: utf-8 -*-

from random import randint

from fpylll import LLL, BKZ, GSO, Enumeration, EnumerationError, IntegerMatrix, prune
from fpylll.util import gaussian_heuristic
from fpylll.algorithms.bkz_stats import BKZTreeTracer
from time import time, clock
from random import randint
from math import ceil
from fpylll import BKZ as fplll_bkz

YOLO_PREPROC_MIN_BLOCK_SIZE = 50
YOLO_PRUNER_MIN_BLOCK_SIZE = 50
YOLO_GAP_PREPROC_BLOCK_SIZE = 10
YOLO_MAX_BLOCK_SIZE = 200
YOLO_MEMORY_LENGTH = 6
GH_FACTOR = 1.1
NODE_PER_SEC = 2**26
RESTART_PENALTY = 0.01


class Timer:
    def __init__(self):
        self.start = clock()

    reset = __init__

    def elapsed(self):
        return clock() - self.start

DEFAULT_STRATEGIES = fplll_bkz.Param(block_size=1, strategies="default.json").strategies


class Tuner(object):
    def __init__(self, b):
        self.last_prunings = None
        self.data = {}
        self.counts = {}        
        self.b = b
        self.proba = .5
        if b>1 and b<max(YOLO_PRUNER_MIN_BLOCK_SIZE, YOLO_PREPROC_MIN_BLOCK_SIZE):
            self.strategy = DEFAULT_STRATEGIES[b]

    def get_variations(self, preprocessing):
        V = [preprocessing]
        
        minb = 10
        if len(preprocessing)==0:
            V.append(tuple([minb]))
            # V.append(tuple([(self.b/3, .5)]))
            return V
        if len(preprocessing)==1:
            b = preprocessing[0]
            if b<minb+6:
                V.append(tuple([]))
            for bb in reversed(range(max(b-2, minb), min(b+3, self.b - YOLO_GAP_PREPROC_BLOCK_SIZE))):
                V.append(tuple([bb]))
            return V
        assert False

    def preprocess(self):
        # return tuple()
        # self.count += 1
        if self.b < YOLO_PREPROC_MIN_BLOCK_SIZE:
            return self.strategy.preprocessing_block_sizes
        if len(self.data)==0:
            return tuple()
        best = max(self.data, key=self.data.get)
        best_efficiency = self.data[best]
        variations = self.get_variations(best)
        for variation in variations:
            if variation not in self.data:
                return variation
            if self.counts[variation]**2 < self.counts[best]:
                return variation

        variation = variations[randint(0, len(variations)-1)]
        variation_efficiency = self.data[variation]
        # print self.b, best, variations
        ratio = best_efficiency / variation_efficiency 
        p = ceil(ratio)
        if randint(0, p) == 0:
            return variation
        else:
            return best

    def enum(self, M, k, target_prob, preproc_time):
        b = self.b

        radius = M.get_r(k, k) * .99
        root_det = M.get_root_det(k, k + b - 1)
        gh_radius, ge = gaussian_heuristic(radius, 0, b, root_det, 1.)
        if b > 30:
            radius = min(radius, 1.21 * gh_radius * 2**ge)

        if b < YOLO_PRUNER_MIN_BLOCK_SIZE:
            return radius, self.strategy.get_pruning(radius, gh_radius * 2**ge)

        R = tuple([M.get_r(i, i) for i in range(k, k+b)])
        overhead = (preproc_time + RESTART_PENALTY) * NODE_PER_SEC
        start_from = self.last_prunings
        pruning = prune(radius, overhead, target_prob, [R], 
                        descent_method="gradient", precision=53, start_from=start_from)
        self.last_prunings = pruning.coefficients
        self.proba = (self.proba * YOLO_MEMORY_LENGTH) + pruning.probability
        self.proba /= YOLO_MEMORY_LENGTH + 1
        return radius, pruning

    def enum_for_hints(self, M, k, b, preproc_time):
        return 0, None

    def feedback(self, preprocessing, pruning, time):
        if pruning is None:
            efficiency = 1. / time
        else:
            efficiency = pruning.probability / time
        if preprocessing in self.data:
            x = self.data[preprocessing]
            c = self.counts[preprocessing]
            f = min(c, YOLO_MEMORY_LENGTH)
            x = f * efficiency + x            
            x /= (f+1)
            c += 1
        else:
            x = efficiency
            c = 1
        self.data[preprocessing] = x
        self.counts[preprocessing] = c


class YoloBKZ(object):

    def __init__(self, A, tuners=None, recycle=True):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        self.recycle = recycle
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

        if tuners is None:
            self.tuners = [Tuner(b) for b in range(YOLO_MAX_BLOCK_SIZE)]
        else:
            self.tuners = tuners
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
        for preproc_b in preprocessing:
            self.tour(preproc_b, .5, begin, end)

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

    def randomize(self, min_row, max_row, density=0):
        """Randomize basis between from ``min_row`` and ``max_row`` (exclusive)

            1. permute rows

            2. apply lower triangular matrix with coefficients in -1,0,1

            3. LLL reduce result

        :param min_row: start in this row
        :param max_row: stop at this row (exclusive)
        :param tracer: object for maintaining statistics
        :param density: number of non-zero coefficients in lower triangular transformation matrix
        """
        if max_row - min_row < 2:
            return  # there is nothing to do

        # 1. permute rows
        niter = 4 * (max_row-min_row)  # some guestimate
        with self.M.row_ops(min_row, max_row):
            for i in range(niter):
                b = a = randint(min_row, max_row-1)
                while b == a:
                    b = randint(min_row, max_row-1)
                self.M.move_row(b, a)

        # 2. triangular transformation matrix with coefficients in -1,0,1
        with self.M.row_ops(min_row, max_row):
            for a in range(min_row, max_row-2):
                for i in range(density):
                    b = randint(a+1, max_row-1)
                    s = randint(0, 1)
                    self.M.row_addmul(a, b, 2*s-1)

        return

    def enum(self, k, b, radius, pruning, for_hints=False):
        solutions = []
        try:
            if self.recycle:
                enum_obj = Enumeration(self.M, b/2, always_update_radius=True)
            else:
                enum_obj = Enumeration(self.M, 1, always_update_radius=True)
            if pruning is None:
                with self.tracer.context("enumeration", enum_obj=enum_obj, probability=1.):
                    enum_obj.enumerate(k, k + b, radius, 0, aux_sols=solutions)
            else:
                with self.tracer.context("enumeration", enum_obj=enum_obj, probability=pruning.probability):
                    enum_obj.enumerate(k, k + b, radius, 0, pruning=pruning.coefficients, aux_sols=solutions)
            return solutions[0][0], [sol for (sol, _) in solutions[1:]]
        except EnumerationError:
            return None, []

    def svp_reduce(self, k, b, target_prob, stop_at_gh=None):

        timer = Timer()
        rem_prob, inserted = 1.0, 1
        M = self.M

        while rem_prob > 1. - target_prob:
            tmp_target_prob =  1.01 * (target_prob - 1)/rem_prob + 1.01            

            if inserted == 0:
                with self.tracer.context("randomize"):
                    self.randomize(k+1, k+b)

            with self.tracer.context("preprocessing"):
                preprocessing = self.tuners[b].preprocess()
                self.preprocess(k, b, preprocessing)

            with self.tracer.context("pruner"):
                radius, pruning = self.tuners[b].enum(M, k, tmp_target_prob, timer.elapsed())
            solution, hints = self.enum(k, b, radius, pruning)

            if pruning is None:
                rem_prob = 0
            else:
                rem_prob *= (1 - pruning.probability)

            # radius, pruning = self.tuner.enum_for_hints(M, k, b, timer.elapsed())
            # if radius>0:
            #     hints += self.enum(k, b, radius, pruning, for_hints=False)
            # hints = self.filter_hints(hints)[:b/2]

            self.tuners[b].feedback(preprocessing, pruning, timer.elapsed())
            timer.reset()
            with self.tracer.context("postprocessing"):
                inserted = self.insert(k, b, solution, hints)

    def __call__(self, b, tours=8):
        self.M.discover_all_rows()

        for i in range(tours):
            print
            with self.tracer.context("tour", i):
                self.tour(b)
            print "proba %.4f"%self.tuners[b].proba, 
            i += 1
            # best = max(self.tuners[b].data, key=self.tuners[b].data.get)
            for x in sorted(self.tuners[b].data.keys()):
                try:
                    print x, "\t %d \t %.2f "%(self.tuners[b].counts[x], self.tuners[b].data[x])
                except:
                    pass
            print

        # self.tracer.exit()


# n = 160
# b = 45
# A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
# yBKZ = YoloBKZ(A)

# t = time()
# yBKZ(b)
# t = time() - t
# print "  time: %.2fs"%(t,)
