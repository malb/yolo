# -*- coding: utf-8 -*-

from random import randint
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz_stats import DummyStats
from fpylll.util import gaussian_heuristic
from fpylll import prune
from time import time
from random import randint


AUTO_MIN_BLOCK_SIZE = 40
AUTO_MAX_BLOCK_SIZE = 200
AUTO_OVERRIDE_RATIO = .2


AUTO_MIN_PREPROC_BLOCK_SIZE = 2
AUTO_GAP_PREPROC_BLOCK_SIZE = 10
AUTO_OVERRIDE_RATIO = .2

AUTO_NODE_PER_SEC = 2**25


class LocalTimer:
    def __init__(self):
        self.val = 0.


class LocalTimerContext:
    def __init__(self, T):
        self.T = T

    def __enter__(self):
        self.start = time()
        
    def __exit__(self, type, value, traceback):
        self.T.val = time() - self.start


class AutoPreprocDecider():
    def __init__(self):
        self.table = AUTO_MAX_BLOCK_SIZE* [AUTO_MAX_BLOCK_SIZE *[0.]]

    def decide(self, block_size):
        V = self.table[block_size]
        best_efficiency = 0.
        best_i = 0  # block_size/2 - 10

        if block_size < 40:
            return 0

        # Find the best efficiency so far
        for i in range(AUTO_MIN_PREPROC_BLOCK_SIZE, block_size - AUTO_GAP_PREPROC_BLOCK_SIZE):
            if V[i] > best_efficiency:
                best_i = i
                best_efficiency = V[i]

        best_i = max(best_i, 2)
        # If the next one is not tested yet, try that
        if V[best_i + 2] == 0.:
            best_i += 2

        elif V[best_i - 2] == 0.:
            best_i -= 2
        else:
            if best_i > 2:
                best_i += randint(0, 4) - 2
            else:
                if randint(0, 20) == 0:
                    best_i += randint(0, 2)   

        return best_i

    def feedback(self, block_size, preproc_block_size, proba_per_sec):
        x = self.table[block_size][preproc_block_size]
        if x==0:
            x = proba_per_sec
        else:
            x = AUTO_OVERRIDE_RATIO * proba_per_sec + (1. - AUTO_OVERRIDE_RATIO) * x
        self.table[block_size][preproc_block_size] = x

    def report(self, bs):
        V = self.table[bs]
        for i in range(bs -20):
            print "%d:%.2f |"%(i, V[i]),
        print


class BKZReduction(BKZBase):

    def __init__(self, A, preproc_decider):
        """Create new BKZ object.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        BKZBase.__init__(self, A)
        self.M.discover_all_rows()  # TODO: this belongs in __call__ (?)
        self.preproc_decider = preproc_decider
        self.last_pruning = AUTO_MAX_BLOCK_SIZE * [None]
            
    def decide_enumeration(self, kappa, block_size, param, stats=None, preproc_time=0.1, target_probability=.5):

        radius = self.M.get_r(kappa, kappa)
        root_det = self.M.get_root_det(kappa, kappa + block_size)
        gh_radius, ge = gaussian_heuristic(radius, 0, block_size, root_det, 1.0)

        if block_size < AUTO_MIN_BLOCK_SIZE:
            strategy = param.strategies[block_size]
            return radius, strategy.get_pruning(radius, gh_radius * 2**ge)
        else:
            with stats.context("pruner"):
                R = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
                overhead = preproc_time * AUTO_NODE_PER_SEC
                start_from = self.last_pruning[block_size]
                pruning = prune(radius, overhead, target_probability, [R], descent_method="gradient", 
                                precision=53, start_from=start_from)
                self.last_pruning[block_size] = pruning.coefficients
                return radius, pruning

    def randomize_block(self, min_row, max_row, stats, density=0):
        """Randomize basis between from ``min_row`` and ``max_row`` (exclusive)

            1. permute rows

            2. apply lower triangular matrix with coefficients in -1,0,1

            3. LLL reduce result

        :param min_row: start in this row
        :param max_row: stop at this row (exclusive)
        :param stats: object for maintaining statistics
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

    def svp_preprocessing(self, kappa, block_size, param, stats):
        clean = True

        clean &= BKZBase.svp_preprocessing(self, kappa, block_size, param, stats)

        if block_size < AUTO_MIN_BLOCK_SIZE:
            preprocessing_block_sizes = param.strategies[block_size].preprocessing_block_sizes
        else:
            preprocessing_block_sizes = [self.preproc_decider.decide(block_size)]
            if preprocessing_block_sizes[0] <= 2:
                return 2

        for preproc in preprocessing_block_sizes:
            prepar = param.__class__(block_size=preproc, strategies=param.strategies, flags=BKZ.GH_BND)
            clean &= self.tour(prepar, kappa, kappa + block_size)

        try:
            return preprocessing_block_sizes[-1]
        except:
            return 0

    def svp_reduction(self, kappa, block_size, param, stats):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param stats:

        """
        if stats is None:
            stats = DummyStats(self)

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False
        preproc_timer = LocalTimer()
        enum_timer = LocalTimer()
        pruner_timer = LocalTimer()
        preproc_block_size = 0

        while remaining_probability > 1. - param.min_success_probability:

            with LocalTimerContext(preproc_timer):
                with stats.context("preproc"):
                    if rerandomize:
                        self.randomize_block(kappa+1, kappa+block_size,
                                             density=param.rerandomization_density, stats=stats)
                    preproc_block_size = self.svp_preprocessing(kappa, block_size, param, stats)
                    self.lll_obj(kappa, kappa, kappa + block_size)

            target_probability = 1.01 * ((param.min_success_probability - 1)/remaining_probability + 1.)
            
            with LocalTimerContext(pruner_timer):
                radius, pruning = self.decide_enumeration(kappa, block_size, param, stats=stats, 
                                                          preproc_time=preproc_timer.val,
                                                          target_probability=target_probability)
            try:
                enum_obj = Enumeration(self.M)
                with LocalTimerContext(enum_timer):
                    with stats.context("svp", E=enum_obj):
                        solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, 
                                                                radius, 0, pruning=pruning.coefficients)
                self.svp_postprocessing(kappa, block_size, solution, stats)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            if block_size >= AUTO_MIN_BLOCK_SIZE:
                probability_per_second = pruning.probability / (preproc_timer.val+enum_timer.val+pruner_timer.val)
                self.preproc_decider.feedback(block_size, preproc_block_size, probability_per_second)

            remaining_probability *= (1 - pruning.probability)

        self.lll_obj.size_reduction(0, kappa + 1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean
