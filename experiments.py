#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import pickle
import json
import os
import experiments
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from CotrainingPipeline import CotrainingPipeline
from CNN import CNN
from WordVectorParser import WordVectorParser

def run_experiment(experiment_name, args):
	experiment_function = eval(experiment_name)
	print('Running experiment: {}'.format(experiment_name))
	experiment_function(args)

def general_experiment(unlabelled_file, seed_file, output_data_files, output_label_files, models, g=5, p=500, num_iterations=200, num_epochs=75):

	'''
	This covers a general experiment. Other methods will set up models and only pass to this method parameters it wants to change. 
	@param unlabelled_file (str): the location of a file that has the candidates. file should be of type .txt.
	@param seed_file (str): the location of a file that has the seed terms. file should be of type .txt.
	@param output_data_files (List[str]): the names of files to save the final labeled set for each model. must have same length as output_labels_files and models. will output as .pkl files.
	@param output_label_files (List[str]): the names of files to save the labels for the final labeled set for each model. must have same length as output_data_files and models. will output as .pkl files.
	@param models (List[Models]): a list of models that the cotrainer will use. 
	@param g (int): the growth rate for the cotraining algorithm.
	@param p (int): the size of the subset of the candidates over which the models will evaluate probabilities.
	@param num_iterations (int): the number of iterations the cotraining algorithm should run.
	@param num_epochs (int): the number of epochs each models is trained for inside one iteration of the cotraining algorithm.
	'''

	assert len(output_data_files) == len(output_label_files) == len(models)


	# with this, even if the user specifies a location that does not exist, it will create the path.
	for output_data_file in output_data_files:
		directory = os.path.dirname(output_data_file)
		if not os.path.exists(directory):
			os.makedirs(directory)
	for output_label_file in output_label_files:
		directory = os.path.dirname(output_label_file)
		if not os.path.exists(directory):
			os.makedirs(directory)

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
		with open(output_label_files[index], 'wb') as f:
			pickle.dump(cotrainer.labels[index], f)



def various_gs_with_chapters123(args):
	'''
	This will be a simple experiment to test out our new word vectors.
	@param args (dict): will be a dictionary as described in the docopt in main.py. for this experiment, will only contain growth-sizes.
	'''
	unlabelled_file = "./data/candidates/openstax_biology_chapters_123_sentences_simple_lemmatized_ngram.txt"
	seed_file = "./data/seed_sets/openstax_biology_chapters123_seed.txt"

	growth_sizes = args["--growth-sizes"]
	growth_sizes = list(map(int, growth_sizes.split(";")))

	word_vector_file = "./data/vectors/openstax_biology_chapters123_simple_lemmatized_vectors.vec"
	wvp = WordVectorParser(word_vector_file)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()


	for g in growth_sizes:
		output_data_files = ["./experiment_results/various_gs_with_chapters123/data_files/g_" + str(g)]
		output_label_files = ["./experiment_results/various_gs_with_chapters123/label_files/g_" + str(g)]
		models = [CNN(vocab, embedding_layer, gpu=args["--cuda"])]
		if args["--cuda"]:
			for model in models:
				model.cuda()
		general_experiment(unlabelled_file, seed_file, output_data_files, output_label_files, models, g=g, num_iterations=200)

	print("running the first_experiment")
