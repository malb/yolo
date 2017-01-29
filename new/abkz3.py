# -*- coding: utf-8 -*-

from sys import stderr
from copy import copy

from random import randint
from fpylll import GSO, LLL, BKZ, Enumeration, EvaluatorStrategy, EnumerationError, IntegerMatrix, prune
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.util import gaussian_heuristic
from fpylll.algorithms.bkz_stats import BKZTreeTracer
from fpylll.algorithms.bkz_stats import dummy_tracer
from math import ceil
import time

YOLO_PREPROC_MIN_BLOCK_SIZE = 50
YOLO_PRUNER_MIN_BLOCK_SIZE = 50
YOLO_GAP_PREPROC_BLOCK_SIZE = 5
YOLO_MAX_BLOCK_SIZE = 200
YOLO_MEMORY_LENGTH = 10
NODE_PER_SEC = 2**26
RETRY_PENALTY = 0.01
APPROX_FACTOR = 1.21


class BKZ3Param():
    def __init__(self, block_size, strategies=None,
                 delta=LLL.DEFAULT_DELTA, flags=BKZ.DEFAULT,
                 max_loops=0, max_time=0,
                 auto_abort=None,
                 gh_factor=None,
                 min_success_probability=BKZ.DEFAULT_MIN_SUCCESS_PROBABILITY,
                 rerandomization_density=BKZ.DEFAULT_RERANDOMIZATION_DENSITY,
                 dump_gso_filename=None):
        """
        Create BKZ parameters object.

        :param block_size: an integer from 1 to ``nrows``
        :param strategies: a filename or a list of Strategies
        :param delta: LLL parameter `0.25 < δ < 1.0`
        :param flags: flags
        :param max_loops: maximum number of full loops
        :param max_time: stop after time seconds (up to loop completion)
        :param auto_abort: heuristic, stop when the average slope of `\log(||b_i^*||)` does not
            decrease fast enough.  If a tuple is given it is parsed as ``(scale, max_iter)`` such
            that the algorithm will terminate if for ``max_iter`` loops the slope is not smaller
            than ``scale * old_slope`` where ``old_slope`` was the old minimum.  If ``True`` is
            given, this is equivalent to providing ``(1.0,5)`` which is fpLLL's default.
        :param gh_factor: heuristic, if set then the enumeration bound will be set to
            ``gh_factor`` times the Gaussian Heuristic.  If ``True`` then ``gh_factor`` is set to
            1.1, which is fpLLL's default.
        :param min_success_probability: minimum success probability in an SVP reduction (when using
            pruning)
        :param rerandomization_density: density of rerandomization operation when using extreme
            pruning
        :param dump_gso_filename: if this is not ``None`` then the logs of the norms of the
            Gram-Schmidt vectors are written to this file after each BKZ loop.
        """
        self.bkz_param = BKZ.Param(block_size, strategies,
                                   delta, flags,
                                   max_loops, max_time,
                                   auto_abort,
                                   gh_factor,
                                   min_success_probability,
                                   rerandomization_density,
                                   dump_gso_filename)
        self.lll_eta = 0.51   # For weaker inner LLL
        self.nr_hints = 0.5   # Relative to block_size. For enumerating more than the SVP to do multiple insertions
        self.hints_bound = 1  # Keep the solution whose norm are <= hints_bound * norm(SVP)
        self.rampup = False   # Activate or not a progressive-BKZ to begin

    def set_lll_eta(self, eta):
        self.lll_eta = eta

    def with_rampup(self):
        self.rampup = True


class Timer:
    def __init__(self):
        self.start = time.clock()

    reset = __init__

    def elapsed(self):
        return time.clock() - self.start

DEFAULT_STRATEGIES = BKZ.Param(block_size=1, strategies="default.json").strategies


def strength_to_preprocessing(s):
    L = []
    s = max(0, s)
    # print "A", s, range(s, 0, -5)
    for ps in range(s, 10, -5):
        if ps < 20:
            L += [("LLL", ps)] 
        else:
            L += [("BKZ", ps)]
    return reversed(L)


class Tuner(object):
    def __init__(self, block_size, tuners=None):
        self.last_pruning = None
        self.data = {}
        self.counts = {}
        self.block_size = block_size
        self.proba = .5
        self.tuners = tuners
        try:
            self.strategy = DEFAULT_STRATEGIES[block_size]
        except:
            pass

    def get_variations(self, strength):
        V = [strength] + range(max(0, strength-2), min(strength+3, self.block_size - YOLO_GAP_PREPROC_BLOCK_SIZE))
        return V

    def copy_data_form_closest_tuners(self):
        best_i = -9999
        for i in range(len(self.tuners)):
            if abs(self.block_size-i) < abs(self.block_size-best_i):
                if self.tuners[i] is not None and len(self.tuners[i].data) > 0:
                    best_i = i

        if best_i<0:
            return False

        x = max(self.tuners[best_i].data, key=self.tuners[best_i].data.get)
        self.data[x] = self.tuners[best_i].data[x]
        self.counts[x] = 1

        # for x in self.tuners[best_i].data.keys():
        #     if self.tuners[best_i].counts[x] > 1:
        #         self.data[x] = self.tuners[best_i].data[x]
        #         self.counts[x] = 1 + self.tuners[best_i].counts[x]/10
        return True

    def get_preprocessing_strength(self):
        if len(self.data) == 0:
            self.copy_data_form_closest_tuners()

        if len(self.data) == 0:
            return 10

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
        ratio = best_efficiency / variation_efficiency
        p = ceil(ratio**2)
        if randint(0, p) == 0:
            return variation
        else:
            return best

    def get_pruning(self, M, kappa, preproc_time):
        block_size = self.block_size

        
        r = tuple([M.get_r(i, i) for i in range(kappa, kappa+block_size)])
        radius = r[0] * .99
        print
        print kappa, self.block_size
        print r
        print
        gh_radius = gaussian_heuristic(r)
        if block_size > 30:
            radius = max(radius, APPROX_FACTOR * gh_radius)

        if block_size < YOLO_PRUNER_MIN_BLOCK_SIZE:
            return radius, self.strategy.get_pruning(radius, gh_radius)

        solutions = 8
        overhead = (preproc_time + RETRY_PENALTY) * NODE_PER_SEC
        self.last_pruning = prune(radius, overhead, solutions, [r],
                                  descent_method="gradient", metric="solutions", 
                                  float_type="double", pruning=self.last_pruning)

        return radius, self.last_pruning

    def enum_for_hints(self, M, kappa, block_size, preproc_time):
        return 0, None

    def feedback(self, preprocessing, pruning, time):

        if pruning is not None:
            self.proba = (self.proba * YOLO_MEMORY_LENGTH) + pruning.expectation
            self.proba /= YOLO_MEMORY_LENGTH + 1

        efficiency = 1. / time

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


# class Tuners(object):
#     def __init__(self, block_size):


class BKZReduction(BKZ2):
    def __init__(self, A, tuners=None, recycle=True):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        BKZBase.__init__(self, A)
        self.recycle = recycle

        self.tuners = YOLO_MAX_BLOCK_SIZE * [None]
        if tuners is None:
            for i in range(YOLO_MAX_BLOCK_SIZE):
                self.tuners[i] = Tuner(i, self.tuners)
        else:
            self.tuners = tuners

    def __call__(self, params, min_row=0, max_row=-1):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)

        """
        self.ith_block = 0
        tracer = BKZTreeTracer(self, verbosity=params.bkz_param.flags & BKZ.VERBOSE, start_clocks=True)
        self.params = params

        self.M = GSO.Mat(A, float_type="dd")
        self.lll_objs = 20*[None]
        for i in range(20):
            eta = .51 + (19-i)/100.
            self.lll_objs[i] = LLL.Reduction(self.M, flags=LLL.DEFAULT, eta=eta)

        cputime_start = time.clock()

        self.M.discover_all_rows()
        with tracer.context("lll"):
            for i in range(20):
                self.lll_objs[i]()

        if params.rampup:
            with tracer.context("rampup", -1):
                self.preprocessing(params.bkz_param.block_size, min_row, max_row, start=10, step=1, tracer=tracer)

        i = 0
        self.ith_tour = 0
        while True:
            with tracer.context("tour", i):
                self.ith_block = 0
                self.ith_tour += 1
                clean = self.tour(params.bkz_param, min_row, max_row, tracer=tracer, top_level=True)
            print "proba %.4f" % self.tuners[params.bkz_param.block_size].proba
            # for x in sorted(self.tuners[params.bkz_param.block_size].data.keys()):
            #    try:
            #        print x, "\t %d \t %.2f " % (self.tuners[params.bkz_param.block_size].counts[x], self.tuners[params.bkz_param.block_size].data[x])
            #    except:
            #        pass
            print
            i += 1
            if (not clean) or params.bkz_param.block_size >= self.A.nrows:
                break
            if (params.bkz_param.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.bkz_param.flags & BKZ.MAX_LOOPS) and i >= params.bkz_param.max_loops:
                break
            if (params.bkz_param.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= params.bkz_param.max_time:
                break

        self.trace = tracer.trace
        return clean

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer, top_level=False):
        """One BKZ loop over all indices.

        :param params: BKZ parameters
        :param min_row: start index ≥ 0
        :param max_row: last index ≤ n

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if max_row == -1:
            max_row = self.A.nrows

        clean = True

        for kappa in range(min_row, max_row-2):
            block_size = min(params.block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, block_size, params, tracer, top_level=top_level)

        tuner = self.tuners[params.block_size]        
        if (min_row==0) and (max_row==self.A.nrows):
            print ("Tuner %d \t Proba %.5f"%(params.block_size, tuner.proba))
            print "-------------------------"
            print ("s \t| cnt \t| eff \t")
            print "-------------------------"
            for x in tuner.data:
                print ("%d \t| %d \t| %.2f \t"%(x, tuner.counts[x], tuner.data[x]))
            print "-------------------------"

        return clean

    def preprocessing(self, block_size_max, min_row, max_row, start=0, step=10, tracer=dummy_tracer):
        block_sizes = range(start, block_size_max, step)
        # block_sizes.append(block_size_max-5)
        print "Ramp up with blocksizes:" + str(block_sizes)
        self.ith_tour = -len(block_sizes)
        total_timer = Timer()
        for i in block_sizes:
            p = self.params.bkz_param.__class__(block_size=i, strategies=self.params.bkz_param.strategies, flags=self.params.bkz_param.flags)
            with tracer.context("tour", self.ith_tour):
                self.tour(p, min_row, max_row, tracer=tracer)
            slope = self.M.get_current_slope(0, self.A.nrows)
            print ">> %d : %.1fs \ %.5f"%(i, total_timer.elapsed(), slope)
            print
            self.ith_tour += 1

    def svp_call(self, kappa, block_size, radius, pruning, nr_hints=0, tracer=dummy_tracer):
        """Call SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: Coordinates of SVP solution or ``None`` if none was found.

        ..  note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.
        """
        solutions = []
        try:
            enum_obj = Enumeration(self.M, nr_hints+1, EvaluatorStrategy.OPPORTUNISTIC_N_SOLUTIONS)
            if pruning is None:
                with tracer.context("enumeration", enum_obj=enum_obj, probability=1., full=block_size==self.params.bkz_param.block_size):
                    solutions = enum_obj.enumerate(kappa, kappa + block_size, radius, 0)
            else:
                with tracer.context("enumeration", enum_obj=enum_obj, probability=pruning.expectation, full=block_size==self.params.bkz_param.block_size):
                    solutions = enum_obj.enumerate(kappa, kappa + block_size, radius, 0, pruning=pruning.coefficients)
            return solutions
        except EnumerationError:
            return []

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer, top_level=False):
        """Find shortest vector in projected lattice of dimension ``block_size`` and insert into
        current basis.

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """

        # if top_level:
        #     # do a full LLL up to kappa + block_size
        #     with tracer.context("lll"):
        #         self.lll_obj(0, kappa, kappa+block_size, 0)

        inserted = 1
        if (block_size == 60):
            self.ith_block += 1

        total_timer = Timer()
        timer = Timer()
        extra_strength = 0

        self.M.discover_all_rows()
        self.lll_objs[19].size_reduction(kappa, kappa + block_size, kappa)
        self.lll_objs[19](kappa, kappa, kappa + block_size, kappa)

        r = tuple([self.M.get_r(i, i) for i in range(kappa, kappa+block_size)])
        print "X"
        print kappa, block_size
        print r
        print "X"

        gh_radius = gaussian_heuristic(r)
        strength = self.tuners[block_size].get_preprocessing_strength()

        pruning = None

        trial = 0
        while r[0] > gh_radius * APPROX_FACTOR:
            
            if inserted == 0:
                extra_strength += 2
                # with tracer.context("randomize"):
                #    self.randomize_block(kappa+1, kappa+block_size)

            total_strength = min(strength + extra_strength, block_size-2)
            with tracer.context("preprocessing"):              
                # if len(preprocessing) > 0 and block_size == 60:
                #     print >> stderr, self.ith_tour, self.ith_block, block_size, preprocessing

                self.svp_preprocessing(kappa, block_size, params, total_strength, tracer)

            with tracer.context("pruner"):
                self.M.discover_all_rows()
                self.lll_objs[19].size_reduction(kappa, kappa + block_size, kappa)
                self.lll_objs[19](kappa, kappa, kappa + block_size, kappa)

                radius, pruning = self.tuners[block_size].get_pruning(self.M, kappa, timer.elapsed())

            solutions = self.svp_call(kappa, block_size, radius, pruning, nr_hints=0, tracer=tracer)
            if len(solutions) == 0:
                solution = None
            else:
                solution = solutions[0][0]
            if len(solutions) <= 1:
                hints = []
            else:
                hints = self.filter_hints(solutions[1:], self.params.hints_bound*solutions[0][1])

            timer.reset()
            with tracer.context("postprocessing"):
                inserted = self.svp_postprocessingg(kappa, block_size, solution, tracer)
            r = tuple([self.M.get_r(i, i) for i in range(kappa, kappa+block_size)])
            if block_size <= 40:
                break
            trial += 1
            if trial>1:
                print "trial", trial, r[0] / (gh_radius * APPROX_FACTOR)

        self.tuners[block_size].feedback(strength, pruning, timer.elapsed())
        return True

    def svp_preprocessing(self, kappa, block_size, param, strength=None, tracer=dummy_tracer):
        clean = True

        # TODO validate, size_reduction seems needed
        self.M.discover_all_rows()
        self.lll_objs[19].size_reduction(kappa, kappa + block_size, kappa)
        self.lll_objs[19](kappa, kappa, kappa + block_size, kappa)

        for (alg, val) in strength_to_preprocessing(strength):
            if alg == "LLL":
                with tracer.context("lll"):
                    self.lll_objs[val](kappa, kappa, kappa + block_size, kappa)
            
            elif alg == "BKZ":
                prepar = param.__class__(block_size=val, strategies=param.strategies, flags=BKZ.GH_BND|BKZ.BOUNDED_LLL)
                clean &= self.tour(prepar, kappa, kappa + block_size, tracer=tracer)

            else:
                assert False

        return clean

    def svp_postprocessingg(self, kappa, block_size, solution, tracer):
        """Insert SVP solution into basis and LLL reduce.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if solution is None:
            return True

        nonzero_vectors = len([x for x in solution if x])
        if nonzero_vectors == 1:
            first_nonzero_vector = None
            for i in range(block_size):
                if abs(solution[i]) == 1:
                    first_nonzero_vector = i
                    break

            self.M.move_row(kappa + first_nonzero_vector, kappa)
            with tracer.context("lll"):
                self.lll_objs[19].size_reduction(kappa, kappa + first_nonzero_vector + 1)

        else:
            d = self.M.d
            self.M.create_row()

            with self.M.row_ops(d, d+1):
                for i in range(block_size):
                    self.M.row_addmul(d, kappa + i, solution[i])

            self.M.move_row(d, kappa)
            with tracer.context("lll"):
                self.lll_objs[0](kappa, kappa, kappa + block_size + 1)
            self.M.move_row(kappa + block_size, d)
            self.M.remove_last_row()

        return False


for i in range(1):
    n = 90
    bs = 80
    loops = 3
    A = IntegerMatrix.random(n, "qary", k=n//2, bits=20)

    M = GSO.Mat(A, float_type="dd")
    lll_obj = LLL.Reduction(M, flags=LLL.DEFAULT)
    lll_obj()

    p = BKZ3Param(bs, max_loops=loops, min_success_probability=0.5, flags=BKZ.VERBOSE | BKZ.BOUNDED_LLL)
    p.with_rampup()
    p.set_lll_eta(0.71)
    # p.nr_hints = 0.25
    # p.hints_bound = 1.25
    yBKZ = BKZReduction(copy(A))
    print "Go!"

    t = time.time()
    yBKZ(p)
    t = time.time() - t
    print "  time: %.2fs" % (t,)
    print
    # print yBKZ.trace.report()

    # p = BKZ.Param(bs, strategies="default.json", max_loops=loops, min_success_probability=0.5, flags=BKZ.VERBOSE | BKZ.BOUNDED_LLL | BKZ.GH_BND)
    # bkz2o = BKZ2(copy(A))
    # print "Go!"
    # 
    # t = time.time()
    # bkz2o(p)
    # t = time.time() - t
    # print "  time: %.2fs" % (t,)
    # print
