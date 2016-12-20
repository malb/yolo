# -*- coding: utf-8 -*-
"""
BKZ 2.0 variant which ensures that the basis quality does not decrease.
"""

from fpylll.algorithms.bkz_stats import dummy_tracer
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
# from fpylll.algorithms.bkz2 import BKZReduction as BKZ1
from fpylll import BKZ, IntegerMatrix, Enumeration, EnumerationError
from fpylll.util import gaussian_heuristic


class BKZReduction(BKZ2):
    def copy_block(self, kappa, block_size):
        for i in range(block_size):
            self.M.create_row()

        with self.M.row_ops(self.M.d-block_size, self.M.d):
            for i in range(block_size):
                self.M.row_addmul(self.M.d-block_size+i, kappa+i, 1)

    def delete_copy_block(self, kappa, block_size, restore):
        if restore:
            for i in range(block_size):
                # this implement a swap
                self.M.move_row(self.M.d-block_size+i, kappa+i)
                self.M.move_row(kappa+i+1, self.M.d-block_size+i)
        for i in range(block_size):
            self.M.remove_last_row()

    # def svp_preprocessing(self, kappa, block_size, param, tracer=dummy_tracer):
    #     clean = True

    #     clean &= BKZ1.svp_preprocessing(self, kappa, block_size, param, tracer)

    #     for preproc in param.strategies[block_size].preprocessing_block_sizes:
    #         prepar = param.__class__(block_size=preproc, strategies=param.strategies,
    #                                  flags=BKZ.GH_BND|BKZ.BOUNDED_LLL)
    #         clean &= self.tour(prepar, kappa, kappa + block_size)

    #     return clean

    def svp_reduction(self, kappa, block_size, param, tracer=dummy_tracer):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - param.min_success_probability:
            with tracer.context("preprocessing"):
                if rerandomize:
                    with tracer.context("randomization"):
                        # make a copy of the local block to restore in case rerandomisation decreases quality
                        self.copy_block(kappa, block_size)
                        self.randomize_block(kappa+1, kappa+block_size,
                                             density=param.rerandomization_density, tracer=tracer)
                with tracer.context("reduction"):
                    self.svp_preprocessing(kappa, block_size, param, tracer=tracer)

            radius, expo = self.M.get_r_exp(kappa, kappa)
            radius *= self.lll_obj.delta

            if param.flags & BKZ.GH_BND and block_size > 30:
                root_det = self.M.get_root_det(kappa, kappa + block_size)
                radius, expo = gaussian_heuristic(radius, expo, block_size, root_det, param.gh_factor)

            pruning = self.get_pruning(kappa, block_size, param, tracer)

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.probability,
                                    full=block_size==param.block_size):
                    solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                            pruning=pruning.coefficients)[0]
                with tracer.context("postprocessing"):
                    self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)
                    if rerandomize:
                        self.delete_copy_block(kappa, block_size, restore=False)
                rerandomize = False

            except EnumerationError:
                with tracer.context("postprocessing"):
                    if rerandomize:
                        self.delete_copy_block(kappa, block_size, restore=True)
                rerandomize = True

            remaining_probability *= (1 - pruning.probability)

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean


import copy
n = 120
block_size = 68
A = IntegerMatrix.random(n, "qary", k=n//2, bits=36)
bkz = BKZReduction(copy.copy(A))
bkz(BKZ.Param(block_size=block_size, max_loops=2, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.MAX_LOOPS|BKZ.VERBOSE))

print

bkz = BKZ2(copy.copy(A))
bkz(BKZ.Param(block_size=block_size, max_loops=2, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.MAX_LOOPS|BKZ.VERBOSE))
