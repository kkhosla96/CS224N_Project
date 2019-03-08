#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import pickle
import json
import experiments
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from CotrainingPipeline import CotrainingPipeline

def run_experiment(experiment_name, args):
	experiment_function = eval(experiment_name)
	print('Running experiment: {}'.format(experiment_name))
	experiment_function(args)

def general_experiment(unlabelled_file, seed_file, output_data_files, output_labels_files, models, g=5, p=500, num_iterations=200, num_epochs=75):
	'''
	This covers a general experiment. Other methods will set up models and only pass to this method parameters it wants to change.
	@param unlabelled_file (str): the location of a file that has the candidates. file should be of type .txt.
	@param seed_file (str): the location of a file that has the seed terms. file should be of type .txt.
	@param output_data_files (List[str]): the names of files to save the final labeled set for each model. must have same length as output_labels_files and models. will output as .pkl files.
	@param output_labels_kfiles (List[str]): the names of files to save the labels for the final labeled set for each model. must have same length as output_data_files and models. will output as .pkl files.
	@param models (List[Models]): a list of models that the cotrainer will use. 
	@param g (int): the growth rate for the cotraining algorithm.
	@param p (int): the size of the subset of the candidates over which the models will evaluate probabilities.
	@param num_iterations (int): the number of iterations the cotraining algorithm should run.
	@param num_epochs (int): the number of epochs each models is trained for inside one iteration of the cotraining algorithm.
	'''

	assert len(output_data_files) == len(output_labels_files) == len(models)

	number_models = len(models)

	# get data
	unlabelled = [line.split() for line in open(unlabelled_file)]
	seed_terms = []
	seed_labels = []
	with open(seed_file) as f:
		for line in f:
			split = line.split()
			seed_terms.append(split[:-1])
			seed_labels.append(int(split[-1]))

	cotrainer = CotrainingPipeline(seed_terms, seed_labels, unlabelled, models, g, p)
	cotrainer.train(num_iterations, num_epochs)

	for index in range(number_models):
		with open(output_data_files[index], 'wb') as f:
			pickle.dump(cotrainer.labelled_data[index], f)
		with open(output_labels_files[index], 'wb') as f:
			pickle.dump(cotrainer.labels[index], 'wb', f)



def first_experiment(args):
	print("running the first_experiment")