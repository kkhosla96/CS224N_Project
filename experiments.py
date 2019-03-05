#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import json
import experiments
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def run_experiment(experiment_name, args):
	experiment_function = eval(experiment_name)
	print('Running experiment: {}'.format(experiment_name))
	experiment_function(args)

def first_experiment(args):
	print("running the first_experiment")