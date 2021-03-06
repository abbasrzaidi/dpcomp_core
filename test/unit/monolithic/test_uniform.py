"""Unit test for uniform.py"""
from __future__ import division

from builtins import range
import numpy
from dpcomp_core.algorithm import uniform 
from dpcomp_core import workload 
from dpcomp_core import dataset
from dpcomp_core import util
import unittest

class PriveletTests(unittest.TestCase):


    def setUp(self):
        n = 1024
        scale = 1E5
        self.hist = numpy.array( list(range(n)))
        self.d = dataset.Dataset(self.hist, None)
        self.dist = numpy.random.exponential(1,n)
        self.dist = util.old_div(self.dist, float(self.dist.sum()))
    
        self.epsilon = 0.1
        self.w1 = workload.Identity.oneD(1024 , weight=1.0)
        self.w2 = workload.Prefix1D(1024)
        self.eng = uniform.uniform_noisy_engine()

    def testRandom(self):
        seed =1 

        h1 = self.eng.Run(self.w1,self.d.payload,self.epsilon,seed)
        h2 = self.eng.Run(self.w1,self.d.payload,self.epsilon,seed)
        self.assertSequenceEqual(list(h1),list(h2))

        numpy.random.seed(100)#see if setting numpy seed would affect random state object prng


        h3 = self.eng.Run(self.w1,self.d.payload,self.epsilon,seed)
        self.assertSequenceEqual(list(h1),list(h3))


if __name__ == "__main__":
    unittest.main(verbosity=2)   
