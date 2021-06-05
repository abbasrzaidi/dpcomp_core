from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dpcomp_core.algorithm import *
from dpcomp_core import dataset
from dpcomp_core import util
from dpcomp_core import workload
import numpy as np
import argparse
import pandas as pd

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
    parser.add_argument("-dp", "--paper_results", 
            type=str,
            choices=['show'],
            help="")
    parser.add_argument("-g", "--graphing",
            type=str,
            choices=["y"],
            help="")
    parser.add_argument("-t", "--time",
            type=int,
            choices=[0, 1, 2],
            help="")
    args = parser.parse_args()

    epsilon = 0.2

    # temporal runs
    nickname = 'time_0' # DOMAINS: 0 = 118, 1 = 138, 2 = 139
    domain = (118,) # 226 buildings in new data
    if args.time == 0:
        nickname = 'time_0'
        domain = (118, 0)
    elif args.time == 1:
        nickname = 'time_1'
        domain = (138, 0)
    elif args.time == 2:
        nickname = 'time_2'
        domain = (139, 0)

    # nickname = 'UPC_DATA_1' # for APs
    # nickname = 'UPC_DATA_2' # for buildings
    # nickname = 'BIDS-FJ' # their 1D data: domain = 4096

    seed = 0
    shape_list = [(1,)] # unary tuple for 1d data
    size = 2000 # query workload size 

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
    # d = dataset.DatasetSampledFromFile(nickname=nickname, 
    #                                     sample_to_scale=sample, 
    # # #                                     # reduce_to_dom_shape=domain, # can remove this 
    # # #                                     # seed=111) # the 111 was a constant already in the repo
    #                                 )
    # this takes our ACTUAL data


    d= dataset.DatasetFromFile(nickname=nickname) 
                                # reduce_to_dom_shape=domain)

    # Instantiate workload
    w = workload.RandomRange(shape_list=shape_list, 
                            domain_shape=domain, 
                            size=size, 
                            seed=seed)

    # Prefix workload (dpbench paper uses this)
    # w = workload.Prefix1D(domain_shape_int=domain[0], pretty_name=nickname)

    if args.graphing == 'y':
        # Calculate noisy estimate for x
        epsilons = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        for eps in epsilons: # just to make getting the results easier
            print('RESULTS FOR EPSILON =', eps)
            x = d.payload
            x_hat = a.Run(w, x, eps, seed)

            # negative removal: if n
            # oisy count < 0, set it to 0
            for i in range(len(x_hat)):
                if x_hat[i] < 0: x_hat[i] = 0
            
            # printing counts vs noisy counts - i think these are QUERY RESULTS not real vs noisy... 
            if args.comparison == 'show':
                for i in range(0, len(x)):
                    print(i, x[i], x_hat[i])

            # Compute error between true x and noisy estimate
            diff = w.evaluate(x) - w.evaluate(x_hat)
                # take first 5 results, compute your own error 
                # see what the results are if they match the calculated error 
            # print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))
            print('Per Query Average Absolute Error:', np.divide(np.linalg.norm(diff,1), float(diff.size)))
            avg_abs_err = np.divide(np.linalg.norm(diff,1), float(diff.size))

            # computing average relative error 
            true_results = w.evaluate(x)
            abs_values = abs(diff)
            delta = domain[0] * 0.001 # 0.001 * domain size (226 on new data, 201 on old, 4096 on theirs)
            rel_error_sum = 0
            for i in range(0, len(abs_values)):
                rel_error_sum += abs_values[i]/max(true_results[i], delta)
            print('Per Query Relative Error:', old_div(rel_error_sum, size))

            # calculating avg noise
            noise = 0
            for i in range(len(x)):
                noise += (abs(x_hat[i] - x[i]))
            
            avg_noise = noise/len(x)
            # print('Average Noise:', avg_noise)

            # computing variance
            variance = 0
            for x in diff:
                variance += (x - avg_abs_err) ** 2
            
            variance = variance / (size - 1)
    else:
        epsilon = 1.0
        x = d.payload
        x_hat = a.Run(w, x, epsilon, seed)

        # negative removal: if noisy count < 0, set it to 0
        for i in range(len(x_hat)):
            if x_hat[i] < 0: x_hat[i] = 0
        
        ''' now we have the noisy values in x_hat ''' 
        noisy_vals = x_hat
        if args.paper_results == 'show':
            df = pd.DataFrame(noisy_vals)
            df.to_csv('time_' + str(args.time) + '.csv', index=False, header=False)
        else:
            diff = w.evaluate(x) - w.evaluate(x_hat)
                # take first 5 results, compute your own error 
                # see what the results are if they match the calculated error 
            # print('Per Query Average Absolute Error:', old_div(np.linalg.norm(diff,1), float(diff.size)))
            print('Per Query Average Absolute Error:', np.divide(np.linalg.norm(diff,1), float(diff.size)))
            avg_abs_err = np.divide(np.linalg.norm(diff,1), float(diff.size))

            # computing average relative error 
            true_results = w.evaluate(x)
            abs_values = abs(diff)
            delta = domain[0] * 0.001 # 0.001 * domain size (226 on new data, 201 on old, 4096 on theirs)
            rel_error_sum = 0
            for i in range(0, len(abs_values)):
                rel_error_sum += abs_values[i]/max(true_results[i], delta)
            print('Per Query Relative Error:', old_div(rel_error_sum, size))
