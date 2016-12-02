# -*- coding: utf-8 -*-
"""
Collecting traces of BKZ-like computations

.. moduleauthor:: Martin R. Albrecht <fplll-devel@googlegroups.com>

"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
import copy
import random
from collections import OrderedDict
from math import log

from fpylll import IntegerMatrix, GSO, LLL
from fpylll import BKZ
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.util import gaussian_heuristic


def pretty_dict(d, keyword_width=None, round_bound=4096):
    """Return 'pretty' string representation of the dictionary ``d``.

    :param d: a dictionary
    :param keyword_width: width allocated for keywords
    :param round_bound: values beyond this bound are shown as `2^x`

    """
    s = []
    for k in d:

        v = d[k]

        if keyword_width:
            fmt = u"\"%%%ds\"" % keyword_width
            k = fmt % k
        else:
            k = "\"%s\""%k

        if isinstance(v, (int, long)):
            if v > round_bound:
                s.append(u"%s: %9s" %(k,  u"2^%.1f" % log(v, 2)))
            else:
                s.append(u"%s: %9d"%(k, v))
            continue
        elif not isinstance(v, float):
            try:
                v = float(v)
            except TypeError:
                s.append(u"%s: %s"%(k, v))
                continue

        if v < 2.0 and v >= 0.0:
            s.append(u"%s: %9.7f"%(k, v))
        elif v < round_bound:
            s.append(u"%s: %9.4f"%(k, v))
        else:
            s.append(u"%s: %9s" %(k,  u"2^%.1f" % log(v, 2)))

    return u"{" + u",  ".join(s) + u"}"


class Stat(object):
    """
    A ``stat`` collects information about some random variable (such as running time) such as
    minima, maxima, mean and 2nd momentum.
    """

    def __init__(self, value=None, main="sum"):
        """
        Create a new instance, optionally providing an initial ``value``.

        :param value: some initial value
        :param main: represent this object by either its sum or its mean when converting to float etc.

        """

        if value is not None:
            self._min = value
            self._max = value
            self._sum = value
            self._sqr = value*value
            self._ctr = 1
        else:
            self._ctr = 0
        self._main = main

    def add(self, value):
        """
        Add value to this statistic.

        :param value: some value
        """
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._sum += value
        self._sqr += value*value
        self._ctr += 1
        return self

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def avg(self):
        return self._sum/self._ctr

    @property
    def sum(self):
        return self._sum

    @property
    def second_moment(self):
        return self._sqrt/self._ctr

    def __add__(self, other):
        """
        Addition semantics are:

        - ``stat + None`` returns ``stat``
        - ``stat + stat`` returns the sum of their underlying values
        - ``stat + value`` inserts ``value`` into ``stat``
        """
        if other is None:
            return copy.copy(self)
        elif not isinstance(other, Stat):
            ret = copy.copy(self)
            return ret.add(other)
        else:
            if self._main != other._main:
                raise ValueError("%s != %s"%(self._main, other._main))
            ret = Stat()
            ret._min = min(self.min, other.min)
            ret._max = max(self.max, other.max)
            ret._sum = self._sum + other._sum
            ret._sqr = self._sqr + other._sqr
            ret._ctr = self._ctr + other._ctr
            ret._main = self._main
            return ret

    def __radd__(self, other):
        """
        Revert to normal addition.
        """
        return self + other

    def __float__(self):
        """
        Reduce this stats object down a float depending on strategy chosen in constructor.
        """
        if self._main == "sum":
            return float(self.sum)
        elif self._main == "avg":
            return float(self.avg)
        else:
            raise NotImplementedError

    def __str__(self):
        """
        String representation of this stat object.
        """
        if self._main == "sum":
            return str(self.sum)
        elif self._main == "avg":
            return str(self.avg)
        else:
            raise NotImplementedError


class TraceContext(object):
    """
    A trace context collects data on entry and exit about the underlying process.
    """
    def __init__(self, trace, *args, **kwds):
        """Create a new context for gathering statistics.

        :param trace: a trace object
        :param args: all args form a label for the trace context
        :param kwds: all kwds are considered auxiliary data

        """
        self.trace = trace
        self.what = args if len(args)>1 else args[0]
        self.kwds = kwds

    def __enter__(self):
        """
        Call ``enter`` on trace object
        """
        self.trace.enter(self.what, **self.kwds)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Call ``exit`` on trace object
        """
        self.trace.exit(self.what, **self.kwds)


class Trace(object):
    """
    A trace object which does nothing.
    """
    def __init__(self, instance, verbose=False):
        """
        Create a new trace instance.

        :param instance: BKZ-like object instance
        :param verbose: print information, integers ≥ 0 are also accepted

        """
        self.instance = instance
        self.verbose = int(verbose)

    def context(self, *args, **kwds):
        """
        Return ``TraceCotext``
        """
        return TraceContext(self, *args, **kwds)

    def enter(self, label, **kwds):
        pass

    def exit(self, label, **kwds):
        pass


dummy_trace = Trace(None)


class Node(object):
    """
    A simple tree implementation with labels and associated data.
    """
    def __init__(self, label, parent=None, data=None):
        """Create a new node.

        :param label: some label such as a string or a tuple
        :param parent: nodes know their parents
        :param data: nodes can have associated data

        """

        self.label = label
        if data is None:
            self.data = OrderedDict()
        self.parent = parent
        self.children = []

    def add_child(self, child):
        """Add a child.

        :param child: a node
        :returns: the child

        """
        child.parent = self
        self.children.append(child)
        return child

    def child(self, label):
        """
        If node has a child labelled ``label`` return it, otherwise add a new child.

        :param label: a label
        :returns: a node
        """
        for child in self.children:
            if child.label == label:
                return child

        return self.add_child(Node(label))

    def __str__(self):
        """

        """
        return u"{\"%s\": %s}"%(self.label, pretty_dict(self.data))

    __repr__ = __str__

    def report(self, indentation=0, depth=None):
        """
        Return detailed string representation of thisbtree.

        :param indentation: add spaces to the left of the string representation
        :param depth: stop at this depth

        """
        s = [" "*indentation + str(self)]
        if depth is None or depth > 0:
            for child in self.children:
                depth = None if depth is None else depth-1
                s.append(child.report(indentation+2, depth=depth))
        return "\n".join(s)

    def sum(self, tag, include_self=True, raise_keyerror=False, label=None):
        """
        Return sum over all items tagged ``tag`` in associated data within this tree.

        :param tag: a string
        :param include_self: include data in this node
        :param raise_keyerror: if a node does not have an item tagged with ``tag`` raise a
            ``KeyError``
        :param label: filter by ``label``
        """
        if include_self and (label is None or self.label == label):
            if raise_keyerror:
                r = self.data[tag]
            else:
                r = self.data.get(tag, 0)
        else:
            r = 0
        for child in self.children:
            r = r + child.sum(tag, include_self=True, raise_keyerror=raise_keyerror, label=label)
        return r

    def __getattr__(self, label):
        """Return first child node with label ``label``

        :param label: label

        """
        r = []
        for child in self.children:
            if child.label == label:
                return child
            if isinstance(child.label, tuple) and child.label[0] == label:
                r.append(child)
        if r:
            return tuple(r)
        else:
            raise AttributeError("'Node' object has no attribute '%s'"%(label))

    def __getitem__(self, tag):
        """Return associated data tagged ``tag```

        :param tag:

        """
        return self.data[tag]

    @property
    def level(self):
        """
        Return level of this node, i.e. how many steps it takes to reach a node with parent
        ``None``.
        """
        node, level = self, 0
        while node.parent is not None:
            level += 1
            node = node.parent
        return level


class TimeTreeTrace(Trace):
    """
    Collect CPU and wall time for every context visited, creating a tree structure along the way.
    """

    entries = (("cputime", time.clock), ("walltime", time.time))

    def __init__(self, instance, verbose=False):
        Trace.__init__(self, instance, verbose)
        self.d = Node("root")
        self.current = self.d

    def enter(self, label, **kwds):
        """
        Enter context, start tracking time.

        :param label: if a child with given label already exits, it is modified, otherwise a new
            label is created.
        """
        node = self.current.child(label)

        for t, f in TimeTreeTrace.entries:
            node.data[t] = node.data.get(t, 0) - f()

        self.current = node

    def exit(self, label, **kwds):
        """
        Leave context, record time spent.

        :param label: if a child with given label already exits, it is modified, otherwise a new
            label is created.

        .. note :: If verbosity ≥ to the current level, also print the current node.

        """
        node = self.current

        for t, f in TimeTreeTrace.entries:
            node.data[t] = node.data.get(t, 0) + f()

        if self.verbose and self.verbose >= self.current.level:
            print(self.current)

        self.current = self.current.parent


class BKZTreeTrace(Trace):
    def __init__(self, instance, verbose=False):
        Trace.__init__(self, instance, verbose)
        self.d = Node("root")
        self.current = self.d

    def enter(self, label, **kwds):
        node = self.current.child(label)
        node.data["cputime"] = node.data.get("cputime", 0) - time.clock()
        node.data["walltime"] = node.data.get("walltime", 0) - time.time()
        self.current = node

    def exit(self, label, **kwds):
        node = self.current
        node.data["cputime"]  += time.clock()
        node.data["walltime"] += time.time()
        node.data["r_0"] = self.instance.M.get_r(0, 0)
        node.data["/"] = self.instance.M.get_current_slope(0, self.instance.A.nrows)

        if label == "enumeration":
            node.data["#enum"] = Stat(kwds["enum_obj"].get_nodes(), main="sum") + node.data.get("#enum", None)
            node.data["%"] = Stat(kwds["probability"], main="avg") + node.data.get("%", None)

        if self.verbose and label[0] == "tour":
            i = label[1]
            print(pretty_dict(OrderedDict((("i", i),
                                           ("cputime", node["cputime"]),
                                           ("walltime", node["walltime"]),
                                           ("preproc", node.svp.preprocessing["cputime"]),
                                           ("svp", node.svp.enumeration["cputime"]),
                                           ("lll", node.svp.sum("cputime", label="lll")),
                                           ("postproc", node.svp.postprocessing["cputime"]),
                                           ("r_0", node["r_0"]),
                                           ("/", node["/"]),
                                           ("#enum", node.sum("#enum"))))))

        self.current = self.current.parent


class BKZReduction:
    """
    An implementation of the BKZ algorithm in Python.
    """
    def __init__(self, A):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        if isinstance(A, GSO.Mat):
            L = None
            M = A
            A = M.B
        elif isinstance(A, LLL.Reduction):
            L = A
            M = L.M
            A = M.B
        elif isinstance(A, IntegerMatrix):
            L = None
            M = None
            A = A
        else:
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'"%type(A))

        if M is None and L is None:
            # run LLL first, but only if a matrix was passed
            wrapper = LLL.Wrapper(A)
            wrapper()

        self.A = A
        if M is None:
            self.M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        else:
            self.M = M
        if L is None:
            self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
        else:
            self.lll_obj = L

    def __call__(self, params, min_row=0, max_row=-1):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)

        """
        trace = BKZTreeTrace(self, verbose=1)

        self.lll_obj()

        auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        i = 0
        while True:
            with trace.context("tour", i):
                self.tour(params, min_row, max_row, trace)
            i += 1
            if auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break

        return trace

    def tour(self, params, min_row=0, max_row=-1, trace=dummy_trace):
        """One BKZ loop over all indices.

        :param params: BKZ parameters
        :param min_row: start index ≥ 0
        :param max_row: last index ≤ n

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if max_row == -1:
            max_row = self.A.nrows

        for kappa in range(min_row, max_row-2):
            block_size = min(params.block_size, max_row - kappa)
            with trace.context("svp"):
                self.svp_reduction(kappa, block_size, params, trace)

    def get_pruning(self, kappa, block_size, param, trace=None):
        strategy = param.strategies[block_size]

        radius, re = self.M.get_r_exp(kappa, kappa)
        root_det = self.M.get_root_det(kappa, kappa + block_size)
        gh_radius, ge = gaussian_heuristic(radius, re, block_size, root_det, 1.0)
        return strategy.get_pruning(radius  * 2**re, gh_radius * 2**ge)

    def randomize_block(self, min_row, max_row, trace, density=0):
        """Randomize basis between from ``min_row`` and ``max_row`` (exclusive)

            1. permute rows

            2. apply lower triangular matrix with coefficients in -1,0,1

            3. LLL reduce result

        :param min_row: start in this row
        :param max_row: stop at this row (exclusive)
        :param trace: object for maintaining statistics
        :param density: number of non-zero coefficients in lower triangular transformation matrix
        """
        if max_row - min_row < 2:
            return  # there is nothing to do

        # 1. permute rows
        niter = 4 * (max_row-min_row)  # some guestimate
        with self.M.row_ops(min_row, max_row):
            for i in range(niter):
                b = a = random.randint(min_row, max_row-1)
                while b == a:
                    b = random.randint(min_row, max_row-1)
                self.M.move_row(b, a)

        # 2. triangular transformation matrix with coefficients in -1,0,1
        with self.M.row_ops(min_row, max_row):
            for a in range(min_row, max_row-2):
                for i in range(density):
                    b = random.randint(a+1, max_row-1)
                    s = random.randint(0, 1)
                    self.M.row_addmul(a, b, 2*s-1)

    def svp_preprocessing(self, kappa, block_size, params, trace=dummy_trace):
        """Perform preprocessing for calling the SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param trace: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise

        .. note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.

        """
        with trace.context("lll"):
            self.lll_obj(0, 0, kappa + block_size)

        for preproc in params.strategies[block_size].preprocessing_block_sizes:
            prepar = params.__class__(block_size=preproc, strategies=params.strategies, flags=BKZ.GH_BND)
            self.tour(prepar, kappa, kappa + block_size)

    def svp_reduction(self, kappa, block_size, param, trace=dummy_trace):

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - param.min_success_probability:
            with trace.context("preprocessing"):
                if rerandomize:
                    with trace.context("randomization"):
                        self.randomize_block(kappa+1, kappa+block_size,
                                             density=param.rerandomization_density, trace=trace)
                with trace.context("reduction"):
                    self.svp_preprocessing(kappa, block_size, param, trace)

            radius, expo = self.M.get_r_exp(kappa, kappa)
            radius *= self.lll_obj.delta

            if param.flags & BKZ.GH_BND and block_size > 30:
                root_det = self.M.get_root_det(kappa, kappa + block_size)
                radius, expo = gaussian_heuristic(radius, expo, block_size, root_det, param.gh_factor)

            with trace.context("pruning"):
                pruning = self.get_pruning(kappa, block_size, param, trace)

            try:
                enum_obj = Enumeration(self.M)
                with trace.context("enumeration", enum_obj=enum_obj, probability=pruning.probability):
                    solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                            pruning=pruning.coefficients)
                with trace.context("postprocessing"):
                    self.svp_postprocessing(kappa, block_size, solution, trace)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            remaining_probability *= (1 - pruning.probability)

    def svp_postprocessing(self, kappa, block_size, solution, trace=dummy_trace):
        """Insert SVP solution into basis and LLL reduce.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param trace: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
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
            with trace.context("lll"):
                self.lll_obj.size_reduction(kappa, kappa + first_nonzero_vector + 1)

        else:
            d = self.M.d
            self.M.create_row()

            with self.M.row_ops(d, d+1):
                for i in range(block_size):
                    self.M.row_addmul(d, kappa + i, solution[i])

            self.M.move_row(d, kappa)
            with trace.context("lll"):
                self.lll_obj(kappa, kappa, kappa + block_size + 1)
            self.M.move_row(kappa + block_size, d)
            self.M.remove_last_row()


n = 100
A = IntegerMatrix.random(n, "qary", k=n//2, bits=40)
param = BKZ.Param(block_size=64, max_loops=6, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.MAX_LOOPS|BKZ.VERBOSE)
trace = BKZReduction(A)(param)
print(trace.d.report())
