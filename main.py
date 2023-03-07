import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from stocksGAN import stocksGAN
from get_data import get_data
from Searches._param_dicts import *
from sys import argv


import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ticker', '-t', type=str, help="Ticker symbol of the stock")
    parser.add_argument('--start_date', '-sd', type=str, help="Format: YYYY-MM-DD")
    parser.add_argument('--end_date', '-ed', type=str, help="Format: YYYY-MM-DD")
    # parser.add_argument('--close', '-c', type=bool) # TODO: implement this later
    parser.add_argument('--n_iter', '-n', type=int, help="Number of iterations for the random search")
    parser.add_argument('--verbose', '-v', type=int, help="0, 1, or 2")
    parser.add_argument('--default', '-d', type=bool, help="Use default parameters(dict_main), overrides n_iter")
    parser.add_argument('--save', '-s', type=str, help="Path to save the history")
    parser.add_argument('--num_samples', '-ns', type=int, help="Number of samples to generate")

    args = parser.parse_args()

    data = get_data(args.ticker, args.start_date, args.end_date)

    if args.default:
        param_dict = default_params
    else: # TODO: implement this later
        param_dict = default_params

    model = stocksGAN(data=data, **param_dict)

    model.train(verbose=args.verbose)

    samps = model.sample(args.num_samples, plot=True)
    print(samps)

    if args.save:
        model.save_model(args.save)
    
        
if __name__ == '__main__':
    main()

