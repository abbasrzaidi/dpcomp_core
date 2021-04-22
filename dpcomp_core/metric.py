from __future__ import division
from dpcomp_core.mixins import Cacheable
import hashlib
from dpcomp_core.mixins import Marshallable
from dpcomp_core import util
from numpy import linalg as la
import numpy as np


class Metric(Marshallable, Cacheable):

    def __init__(self, E):
        # move this to the highest subclass
        self.init_params = util.init_params_from_locals(locals())
        print('creating a new metric')
        self.E = E
        self.X = self.E.X
        self.X_hat = None
        self.W = self.E.W

    def compute(self, update_payload=False):
        # print('in the compute func of metric')
        self.E = self.maybe(self.E, self.E.hash, 'run') # this func calculates X_hat which is used
        # print('after maybe called in compute of metric')       
        self.X_hat = self.E.X_hat
        return self

    def asDict(self):
        d = util.class_to_dict(self)

        return d

    def analysis_payload(self):
        return util.class_to_dict(self, ignore_list=['E', 'X', 'X_hat', 'W'])

    @property
    def key(self):
        return self.hash[:24]

    @property
    def hash(self):
        m = hashlib.sha1()

        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(self.E.hash))

        return m.hexdigest()


class SampleError(Metric):
    """
    Calculate sampled based error measures, comparing estimated workload answers with true workload answers on
    sampled input data.
    """
    def compute(self, update_payload=False):
        super(SampleError, self).compute(update_payload)
        # print('this is the important computing function')
        scale = self.X.scale
        # print('calculating the real answer-------')
        true_ans = self.W.evaluate(self.X.payload) # dot prod with some const matrix
        # print('true answer:', true_ans)
        # print('calculating the estimated answer------')
        est_ans = self.W.evaluate(self.X_hat) # dot prod with some const matrix
        # x_hat is array of answers to all the range queries 
        # print('estimated answer:', est_ans)
        diff = true_ans - est_ans
        self.error_payload = calculate_error('TypeI', diff, scale)

        return self


class PopulationError(Metric):
    """
    Calculate population-based error measures, comparing estimated workload answers with workload answers on
    scaled population
    """

    def compute(self, update_payload=False):
        super(PopulationError, self).compute(update_payload)

        scale = self.X.scale
        scaled_dist_ans = self.W.evaluate( self.X.dist * scale )  # scale up distribution and evaluate query answers
        est_ans = self.W.evaluate(self.X_hat)
        diff = scaled_dist_ans - est_ans
        self.error_payload = calculate_error('TypeII', diff, scale)

        return self


def calculate_error(prefix, diff, norm_factor):
    """
    Error calculations are implemented once here
    Error calculations are performed on a vector of query differences
        'diff' should be a vector, not a matrix or norms will be different.
    For L1 and L2, per-query error is reported

    """
    assert len(diff) == diff.size, 'diff should be a vector'
    d = {}
    d[prefix + '.Linf'] = util.old_div(la.norm(diff, np.inf), norm_factor)       #
    d[prefix + '.L1'] = util.old_div((util.old_div(la.norm(diff,1), float(diff.size))), norm_factor)
    d[prefix + '.L2'] = util.old_div((util.old_div(la.norm(diff), float(diff.size))), norm_factor)
    return d


