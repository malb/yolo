# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
from collections import OrderedDict
from math import log

from fpylll import IntegerMatrix, GSO, LLL
from fpylll import BKZ
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.util import gaussian_heuristic


def dict_str(d, keyword_width=None, round_bound=4096):
    if d is None:
        return
    s = []
    for k in d:
        v = d[k]
        if keyword_width:
            fmt = u"\"%%%ds\"" % keyword_width
            k = fmt % k
        else:
            k = "\"%s\""%k
        if 1./round_bound < v < round_bound or v == 0:
            if v < 2.0 and v >= 0.0:
                s.append(u"%s: %9.7f" % (k, v))
            else:
                s.append(u"%s: %9.4f" % (k, v))
        else:
            t = u"2^%.1f" % log(v, 2)
            s.append(u"%s: %9s" % (k, t))
    return u"{" + u",  ".join(s) + u"}"


class StatsContext(object):
    def __init__(self, stats, what, **kwds):
        self.stats = stats
        self.what = what
        self.kwds = kwds

    def __enter__(self):
        self.stats.enter(self.what, **self.kwds)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.stats.exit(self.what, **self.kwds)


class Node(object):
    """
    A simple tree implementation with labels and associated data.
    """
    def __init__(self, label, parent=None, data=None):
        """Create a new node.

        :param label:
        :param parent:
        :param data:
        :returns:
        :rtype:

        """

        self.label = label
        if data is None:
            self.data = OrderedDict()
        self.parent = parent
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

    def child(self, label):
        for child in self.children:
            if child.label == label:
                return child
        return self.add_child(Node(label))

    def __str__(self):
        return u"{\"%s\": %s"%(self.label, dict_str(self.data))

    __repr__ = __str__

    def report(self, indentation=0):
        s = []
        s.append(" "*indentation + str(self))
        for child in self.children:
            s.append(child.report(indentation+2))
        return "\n".join(s)

    def sum(self, what):
        """Sum over all data times with name ``what`` in any child node of this node."

        :param what: a string

        """
        r = self.data.get(what, 0)
        for child in self.children:
            r += child.sum(what)
        return r

    def mean(self, what):
        """Mean over all data times with name ``what`` in any child node of this node."

        :param what: a string

        """
        r = [self.data.get(what, 0)]
        for child in self.children:
            r.append(child.sum(what))
        return sum(r)/len(r)

    def __getattr__(self, what):
        r = []
        for child in self.children:
            if child.label == what:
                return child
            if isinstance(child.label, tuple) and child.label[0] == what:
                r.append(child)
        if r:
            return tuple(r)
        else:
            raise AttributeError("'Node' object has no attribute '%s'"%(what))


def require_mutability(func):
    def func_wrapper(self, *args, **kwds):
        if not self.mutable:
            raise ValueError("This stats object is immutable.")
        return func(self, *args, **kwds)
    func_wrapper.__doc__ = func.__doc__
    return func_wrapper


def context(what):
    def context_wrapper(func):
        def func_wrapper(self, *args, **kwds):

            for arg in args:
                if isinstance(arg, Stats):
                    with arg.context(what):
                        return func(self, *args, **kwds)

            for k, v in kwds.iteritems():
                if k == "stats" and isinstance(v, Stats):
                    with v.context(what):
                        return func(self, *args, **kwds)
        func_wrapper.__doc__ = func.__doc__
        return func_wrapper
    return context_wrapper


class Stats(object):
    def __init__(self, instance, debug=False, verbose=False):
        pass

    def context(self, what, **kwds):
        return StatsContext(self, what, **kwds)

    @require_mutability
    def enter(self, what, **kwds):
        pass

    def exit(self, what, **kwds):
        pass

    def finalize(self):
        pass


dummy_stats = Stats(None)


class TimeStats(Stats):

    entries = (("cputime", time.clock), ("walltime", time.time))

    def __init__(self, instance, debug=False, verbose=False):
        self.instance = instance
        self.verbose = verbose
        self.debug = debug
        self.mutable = True
        self.d = Node("root")
        self.current = self.d

    def context(self, what, **kwds):
        return StatsContext(self, what, **kwds)

    @require_mutability
    def enter(self, what, **kwds):
        """Documentation

        :param what:
        :returns:
        :rtype:

        """
        if self.debug:
            print("=entering=", what, kwds)
        node = self.current.child(what)

        for t, f in TimeStats.entries:
            node.data[t] = node.data.get(t, 0) - f()

        self.current = node

    @require_mutability
    def exit(self, what, **kwds):
        if self.debug:
            print("=exiting=", what, kwds)

        node = self.current

        for t, f in TimeStats.entries:
            node.data[t] = node.data.get(t, 0) + f()

        # special cases
        if what == "svp":
            node.data["#enum"] = node.data.get("#enum", 1) + kwds["enum_obj"].get_nodes()
        if isinstance(what, tuple) and what[0] == "tour" and self.verbose:
            print(self.current)

        self.current = self.current.parent

    def finalize(self):
        self.mutable = False


class BKZReduction:
    def __init__(self, A):
        self.A = A
        self.M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
        self.lll_obj()

    def __call__(self, params, min_row=0, max_row=-1):
        stats = TimeStats(self, verbose=True)

        auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        with stats.context("setup"):
            with stats.context("lll"):
                self.lll_obj()

        i = 0
        while True:
            with stats.context(("tour", i)):
                self.tour(params, min_row, max_row, stats)
            i += 1
            if auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break

        stats.finalize()
        return stats

    def tour(self, params, min_row=0, max_row=-1, stats=dummy_stats):
        if max_row == -1:
            max_row = self.A.nrows

        for kappa in range(min_row, max_row-2):
            block_size = min(params.block_size, max_row - kappa)
            self.svp_reduction(kappa, block_size, params, stats)

    @context("preproc")
    def svp_preprocessing(self, kappa, block_size, params, stats):
        with stats.context("lll"):
            self.lll_obj(0, 0, kappa + block_size)

    def svp_call(self, kappa, block_size, params, stats=None):
        max_dist, expo = self.M.get_r_exp(kappa, kappa)
        delta_max_dist = self.lll_obj.delta * max_dist

        root_det = self.M.get_root_det(kappa, kappa+block_size)
        max_dist, expo = gaussian_heuristic(max_dist, expo, block_size, root_det, params.gh_factor)

        try:
            enum_obj = Enumeration(self.M)
            with stats.context("svp", enum_obj=enum_obj):
                solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, max_dist, expo)
        except EnumerationError:
            return None

        if max_dist >= delta_max_dist:
            return None
        else:
            return solution

    @context("postproc")
    def svp_postprocessing(self, kappa, block_size, solution, stats):
        """Test

        :param kappa:
        :param block_size:
        :param solution:
        :param stats:
        :returns:
        :rtype:

        """
        if solution is None:
            return

        nonzero_vectors = len([x for x in solution if x])
        if nonzero_vectors == 1:
            first_nonzero_vector = None
            for i in range(block_size):
                if abs(solution[i]) == 1:
                    first_nonzero_vector = i
                    break

            self.M.move_row(kappa + first_nonzero_vector, kappa)
            with stats.context("lll"):
                self.lll_obj.size_reduction(kappa, kappa + first_nonzero_vector + 1)

        else:
            d = self.M.d
            self.M.create_row()

            with self.M.row_ops(d, d+1):
                for i in range(block_size):
                    self.M.row_addmul(d, kappa + i, solution[i])

            self.M.move_row(d, kappa)
            with stats.context("lll"):
                self.lll_obj(kappa, kappa, kappa + block_size + 1)
            self.M.move_row(kappa + block_size, d)
            self.M.remove_last_row()

    def svp_reduction(self, kappa, block_size, params, stats=dummy_stats):
        self.svp_preprocessing(kappa, block_size, params, stats)
        solution = self.svp_call(kappa, block_size, params, stats)
        self.svp_postprocessing(kappa, block_size, solution, stats)


A = IntegerMatrix.random(80, "qary", k=40, bits=30)
stats = BKZReduction(A)(BKZ.Param(block_size=30, max_loops=6, flags=BKZ.MAX_LOOPS|BKZ.VERBOSE))
print(stats.d.report())
print(stats.d.sum("#enum"))
print(stats.d.mean("#enum"))
print(stats.d.tour[1].sum("walltime"))
