#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Final Project
main.py: Run script for Glossary Term Extraction
Usage:
    main.py sanity-check
    main.py experiment [options]
    main.py something [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --expt=<name>                           experiment name 
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
"""

import utils
import LSTM
import CNN

import random

import numpy as np
import sys
import torch
import experiments

from docopt import docopt

def sanity_check(args):
    '''
    Runs through a short test suite, running each test or not depending
    on the arguments specified in the command line (or through defaults!)

    Note to selves: Add any sanity check testing we want here.
    '''
    print("doing the sanity check thing")

def experiment(args):
    """
    Fits model to predict glossary terms based on textbook
    chapter content.

    Analyzes these models' efficacy on a test set, and reports
    model performance.
    """
    experiment_name = args["--expt"]
    experiments.run_experiment(experiment_name, args)


def main():
    args = docopt(__doc__)

	# Check Python & PyTorch Versions
    assert(sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    print("hello, world!")

    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['sanity-check']:
        sanity_check(args)
    elif args['experiment']:
        experiment(args)
    elif args['something']:
        print(args['--expt'])
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
	main()