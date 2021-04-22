from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dpcomp_core.algorithm import *
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload
import numpy as np
import argparse

'''
An example execution of one single algorithm. 
CLI args: 
    python examples/algorithm_execution.py -a [algorithm_choice] -c show
        a (algorithm) flag lets you select algorithm (required)
        c (comparison) flag lets you choose to show true vs noisy counts (optional)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select an algorithm')
    parser.add_argument("-a", "--algorithm",
            type=str, 
            choices=[
                'identity', 'privelet', 'h', 'hb', 'greedyH',
                'uniform', 'mwem', 'mwem*', 'ahp', 'ahp*', 'dpcube', 'dawa', 'php', 'efpa', 'sf'
            ],
            help="select an algorithm to run")
    parser.add_argument("-c", "--comparison",
            type=str, 
            choices=['show'],
            help="use -c show to show running list of true vs noisy counts")
    args = parser.parse_args()

    domain = (201,) # 201 buildings in our data

    # epsilon = 1.0
    epsilon = 0.1 

    # nickname = 'UPC_DATA_1' # for APs
    nickname = 'UPC_DATA_2' # for buildings

    sample = 201 # not sure what to put this number at 
    seed = 0
    shape_list = [(1, )] # unary tuple for 1d data
    size = 1000 # query workload size 

    # Instantiate algorithm
    ''' data independent algorithms ''' 
    a = None

    if args.algorithm == 'identity':
        a = identity.identity_engine()
    if args.algorithm == 'privelet':
        a = privelet.privelet_engine()
    if args.algorithm == 'h':
        a = HB.H2_engine()  
    if args.algorithm == 'hb':
        a = HB.HB_engine()
    if args.algorithm == 'greedyH':
        a = dawa.greedyH_only_engine()

    ''' data dependent algorithms '''
    if args.algorithm == 'uniform':
        a = uniform.uniform_noisy_engine()
    if args.algorithm == 'mwem':
        a = mwemND.mwemND_engine()
    if args.algorithm == 'mwem*':
        a = mwemND.mwemND_adaptive_engine() 
    if args.algorithm == 'ahp':
        a = ahp.ahpND_engine()
    if args.algorithm == 'ahp*':
        a = ahp.ahpND_adaptive_engine() 
    if args.algorithm == 'dpcube':
        a = DPcube1D.DPcube1D_engine()
    if args.algorithm == 'dawa':
        a = dawa.dawa_engine()  
    if args.algorithm == 'php':
        a = thirdparty.php_engine()
    if args.algorithm == 'efpa':
        a = thirdparty.efpa_engine()
    if args.algorithm == 'sf':
        a = thirdparty.StructureFirst_engine()
    if args.algorithm == None:
        print('Select an algorithm type')

    # this takes SAMPLES from input data
    d = dataset.DatasetSampledFromFile(nickname=nickname, 
                                        sample_to_scale=sample, 
                                        reduce_to_dom_shape=domain, 
                                        seed=111) # the 111 was a constant already in the repo

    # this takes our ACTUAL data
    # d= dataset.DatasetFromFile(nickname=nickname, 
    #                             reduce_to_dom_shape=domain)

    # Instantiate workload
    w = workload.RandomRange(shape_list=shape_list, 
                            domain_shape=domain, 
                            size=size, 
                            seed=seed)

    # Calculate noisy estimate for x
    x = d.payload
    x_hat = a.Run(w, x, epsilon, seed)

    if args.comparison == 'show':
        # printing counts vs noisy counts
        for i in range(0, len(x)):
            print(i, x[i], x_hat[i])

    # Compute error between true x and noisy estimate
    diff = w.evaluate(x) - w.evaluate(x_hat)
    print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))

    # computing average relative error 
    true_results = w.evaluate(x)
    abs_values = abs(diff)
    delta = 0.001 * 7323 # 0.001 * number of users - should this be * 201 since 201 buildings? 
    rel_error_sum = 0
    for i in range(0, len(abs_values)):
        rel_error_sum += abs_values[i]/max(true_results[i], delta)
    print('Per Query Relative Error:', old_div(rel_error_sum, 1000))