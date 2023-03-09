import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from stocksGAN import stocksGAN
from get_data import get_data
from Searches._param_dicts import *
from sys import argv
from RandomSearch import RandomSearch


import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ticker', '-t', type=str, help="Ticker symbol of the stock")
    parser.add_argument('--start_date', '-sd', type=str, help="Format: YYYY-MM-DD")
    parser.add_argument('--end_date', '-ed', type=str, help="Format: YYYY-MM-DD")
    # parser.add_argument('--close', '-c', type=bool) # TODO: implement this later
    parser.add_argument('--n_iter', '-n', type=int, default=None, help="Number of iterations for the random search")
    parser.add_argument('--verbose', '-v', type=int, default=0, help="0, 1, or 2")
    parser.add_argument('--default', '-d', type=bool, default=False, help="Use default parameters(dict_main), overrides n_iter")
    parser.add_argument('--save', '-s', type=str, default=None, help="Path to save the history")
    parser.add_argument('--num_samples', '-ns', type=int, default=1000, help="Number of samples to generate")

    args = parser.parse_args()

    data = get_data(args.ticker, args.start_date, args.end_date)

    if args.default:
        param_dict = default_params
    else: # Random Search using dict_main
        params = dict_main
        model = RandomSearch(data=data, param_dict=params)
        model.fit(n_iter=args.n_iter, verbose=args.verbose, save_history=False)
        param_dict = model.best_params_



    model = stocksGAN(data=data, **param_dict)

    model.train(verbose=args.verbose)

    samps = model.sample(args.num_samples, plot=True)
    print(samps)

    if args.save:
        model.save_model(args.save)
    
        
if __name__ == '__main__':
    main()

