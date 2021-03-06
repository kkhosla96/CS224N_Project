#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import utils
import pickle
import json
import time
from copy import copy
import os
import experiments
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time
import torch
from CotrainingPipeline import CotrainingPipeline
from CNN import CNN
from LSTM import LSTM
from CNN import DeepCNN
from FullyConnected import FullyConnected
from WordVectorParser import WordVectorParser
from pdb import set_trace as debug

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

	return cotrainer



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

def deepcnn_fullyconnected_cotraining(args):
	'''
	see if we can accomplish something by using cotraining with both the fully connected and deep cnn.
	'''

	unlabelled_file = "./data/candidates/openstax_microbiology/all_candidates_preprocessed.txt"
	seed_file = "./data/seed_sets/openstax_microbiology_seed.txt"

	word_vector_file = "./data/vectors/openstax_microbiology/microbiology_bertvectors.vec"
	wvp = WordVectorParser(word_vector_file, word_vector_length=768)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()

	cnn = DeepCNN(vocab, embedding_layer, gpu=args["--cuda"])
	net = FullyConnected(vocab, embedding_layer, gpu=args["--cuda"])
	models = [cnn, net]

	file_stem = "./paper_results/cotraining_microbiology/"
	output_data_files = [file_stem + "cnn/data_files/final_set.pkl", file_stem + "fc/data_files/final_set.pkl"]
	output_label_files = [file_stem + "cnn/label_files/final_labels.pkl", file_stem + "fc/label_files/final_labels.pkl"]

	if args["--cuda"]:
		for model in models:
			model.cuda()

	cotrainer = general_experiment(unlabelled_file, seed_file, output_data_files, output_label_files, models, num_iterations=150)

	torch.save(models[0].state_dict(), file_stem + "cnn/cnn.pt")
	torch.save(models[1].state_dict(), file_stem + "fc/fc.pt")

	cnn_labeled = cotrainer.labelled_data[0]
	cnn_labeled = set([' '.join(term) for term in cnn_labeled])
	net_labeled = cotrainer.labelled_data[1]
	net_labeled = set([' '.join(term) for term in net_labeled])
	cotraining_labeled_set = cnn_labeled | net_labeled

	gold_file = "./data/gold/openstax_microbiology/all_golds_preprocessed.txt"
	seed_set = set([line.strip()[:-1].strip() for line in open(seed_file)])

	positive = set([line.strip() for line in open(gold_file)])
	negative = set([line.strip() for line in open(unlabelled_file)])
	negative = negative - positive

	negative = negative - seed_set
	negative = negative - cotraining_labeled_set
	positive = positive - seed_set
	positive = positive - cotraining_labeled_set

	positive_set = copy(positive)
	negative_set = copy(negative)

	negative = list(negative)[int(.9 * len(negative)):]
	positive = list(positive)[int(.9 * len(positive)):]

	# we are going to evaluate on the rest of the data
	data = positive + negative
	data = [x.split() for x in data]
	labels = [1] * len(positive) + [0] * len(negative)
	all_results = [models[i].predict(data) for i in range(len(models))]
	cnn_results = all_results[0]
	fc_results = all_results[1]

	for num_model in range(len(models)):
		results = all_results[num_model]
		for i in range(len(results)):
			t = results[i]
			is_positive = ' '.join(t[0]) in positive_set
			results[i] = (*t, 1 if is_positive else 0)

	with open(file_stem + "cnn/cnn_results.txt", 'w') as f:
		for t in cnn_results:
			f.write(str(t) + "\n")
	pickle.dump(cnn_results, open(file_stem + "cnn/cnn_results.pkl", 'wb'))

	with open(file_stem + "fc/fc_results.txt", 'w') as f:
		for t in fc_results:
			f.write(str(t) + "\n")
	pickle.dump(fc_results, open(file_stem + "fc/fc_results.pkl", 'wb'))

	cotraining_results = []
	for i in range(len(cnn_results)):
		cnn_confidence = cnn_results[i][2]
		fc_confidence = fc_results[i][2]
		if cnn_confidence > fc_confidence:
			cotraining_results.append(cnn_results[i])
		else:
			cotraining_results.append(fc_results[i])

	classes = [t[2] for t in cotraining_results]
	number_predicted_positive = sum(classes)
	accuracy_count = 0
	precision_recall_count = 0
	for i in range(len(classes)):
		if classes[i] == labels[i]:
			accuracy_count += 1
		if classes[i] == 1 and labels[i] == 1:
			precision_recall_count += 1

	accuracy = accuracy_count / len(classes)
	precision = precision_recall_count / number_predicted_positive if number_predicted_positive > 0 else 0
	recall = precision_recall_count / len(positive)

	with open(file_stem + "cotraining_results.txt", 'w') as f:
		for t in cotraining_results:
			f.write(str(t) + "\n")
	pickle.dump(cotraining_results, open(file_stem + "cotraining_results.pkl", 'wb'))

	print(accuracy)
	print(precision)
	print(recall)

def supervised_learning(args):
	candidates = "./data/candidates/openstax_micro_and_bio/all_candidates_preprocessed.txt"
	gold_file = "./data/gold/openstax_micro_and_bio/all_golds_preprocessed.txt"

	negative = set([line.strip() for line in open(candidates)])
	positive = set([line.strip() for line in open(gold_file)])
	negative = negative - positive

	positive_set = copy(positive)
	negative_set = copy(negative)

	negative = list(negative)
	positive = list(positive)

	random.seed(42)
	random.shuffle(negative)
	random.shuffle(positive)
	random.seed(time.time())

	number_positive_in_train = int(.9 * len(positive))
	number_positive_in_test = len(positive) - number_positive_in_train
	positive_train = positive[:number_positive_in_train]
	positive_test = positive[number_positive_in_train:]

	number_negative_in_train = int(.9 * len(negative))
	number_negative_in_test = len(negative) - number_negative_in_train
	negative_train = negative[:number_negative_in_train]
	negative_test = negative[number_negative_in_train : number_negative_in_train + number_negative_in_test]

	positive_train = [x.split() for x in positive_train]
	positive_test = [x.split() for x in positive_test]
	negative_train = [x.split() for x in negative_train]
	negative_test = [x.split() for x in negative_test]

	X_train = positive_train + negative_train
	y_train = [1] * number_positive_in_train + [0] * number_negative_in_train

	X_test = positive_test + negative_test
	y_test = [1] * number_positive_in_test + [0] * number_negative_in_test

	word_vector_file = "./data/vectors/openstax_micro_and_bio/micro_and_bio_bertvectors.vec"
	wvp = WordVectorParser(word_vector_file, word_vector_length=768)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()

	cnn = DeepCNN(vocab, embedding_layer, gpu=args["--cuda"])
	start = time.time()
	losses = cnn.train_on_data(X_train, y_train, lr=.01, num_epochs=150, verbose=True)
	end = time.time()
	print("it took %s seconds to train the data" % str(end - start))

	file_stem = "./paper_results/supervised_deep_micro_and_bio/"
	save_file_txt = file_stem + "predictions.txt" 
	save_file_pkl = file_stem + "predictions.pkl"
	directory = os.path.dirname(save_file_txt)
	if not os.path.exists(directory):
		os.makedirs(directory)

	torch.save(cnn.state_dict(), file_stem + "model.pt")

	results = cnn.predict(X_test)
	for i in range(len(results)):
		is_positive = ' '.join(results[i][0]) in positive_set
		t = results[i]
		results[i] = (*t, 1 if is_positive else 0)
	with open(save_file_txt, 'w') as f:
		for t in results:
			f.write(str(t) + "\n")
	pickle.dump(results, open(save_file_pkl, 'wb'))


	classes = [t[2] for t in results]
	number_predicted_positive = sum(classes)
	accuracy_count = 0
	precision_recall_count = 0
	for i in range(len(classes)):
		if classes[i] == y_test[i]:
			accuracy_count += 1
		if classes[i] == 1 and y_test[i] == 1:
			precision_recall_count += 1

	accuracy = accuracy_count / len(classes)
	precision = precision_recall_count / number_predicted_positive
	recall = precision_recall_count / number_positive_in_test

	stats_file = file_stem + "stats.txt"
	with open(stats_file, 'w') as f:
		f.write(str(accuracy) + "\n")
		f.write(str(precision) + "\n")
		f.write(str(recall) + "\n")

	print(accuracy)
	print(precision)
	print(recall)

	save_plot = file_stem + "training_loss.png"
	if os.path.isfile(save_plot):
		os.remove(save_plot)
	fig, ax  = plt.subplots(nrows=1, ncols=1)
	ax.plot(losses)
	fig.savefig(save_plot)

	# plt.plot(losses)
	# plt.show()

def supervised_learning_lstm(args):
	candidates = "./data/candidates/openstax_biology/openstax_biology_sentences_np.txt"
	gold_file = "./data/gold/openstax_biology/openstax_biology_gold_lemmatized.txt"

	negative = set([line.strip() for line in open(candidates)])
	positive = set([line.strip() for line in open(gold_file)])
	negative = negative - positive

	positive_set = copy(positive)
	negative_set = copy(negative)

	negative = list(negative)
	positive = list(positive)

	number_positive_in_train = int(.9 * len(positive))
	number_positive_in_test = len(positive) - number_positive_in_train
	positive_train = positive[:number_positive_in_train]
	positive_test = positive[number_positive_in_train:]

	number_negative_in_train = int(.9 * len(negative))
	number_negative_in_test = len(negative) - number_negative_in_train
	negative_train = negative[:number_negative_in_train]
	negative_test = negative[number_negative_in_train : number_negative_in_train + number_negative_in_test]

	positive_train = [x.split() for x in positive_train]
	positive_test = [x.split() for x in positive_test]
	negative_train = [x.split() for x in negative_train]
	negative_test = [x.split() for x in negative_test]

	X_train = positive_train + negative_train
	y_train = [1] * number_positive_in_train + [0] * number_negative_in_train

	X_test = positive_test + negative_test
	y_test = [1] * number_positive_in_test + [0] * number_negative_in_test

	remove_indices = [i for i in range(len(X_train)) if len(X_train[i]) == 0]
	X_train = [X_train[i] for i in range(len(X_train)) if i not in remove_indices]
	y_train = [y_train[i] for i in range(len(y_train)) if i not in remove_indices]

	remove_indices = [i for i in range(len(X_test)) if len(X_test[i]) == 0]
	X_test = [X_test[i] for i in range(len(X_test)) if i not in remove_indices]
	y_test = [y_test[i] for i in range(len(y_test)) if i not in remove_indices]	

	word_vector_file = "./data/vectors/openstax_biology_vectors.vec"
	wvp = WordVectorParser(word_vector_file)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()
	lstm = LSTM(vocab=vocab, bidirectional=True, embeddings=embedding_layer, gpu=args["--cuda"])
	losses = lstm.train_on_data(X_train, y_train, lr=.001, num_epochs=50, verbose=True)

	save_file_txt = "./experiment_results/supervised_learning/LSTM_predictions.txt"
	save_file_pkl = "./experiment_results/supervised_learning/LSTM_predictions.pkl"
	directory = os.path.dirname(save_file_txt)
	if not os.path.exists(directory):
		os.makedirs(directory)

	results = lstm.predict(X_test)
	for i in range(len(results)):
		is_positive = ' '.join(results[i][0]) in positive_set
		t = results[i]
		results[i] = (*t, 1 if is_positive else 0)
	with open(save_file_txt, 'w') as f:
		for t in results:
			f.write(str(t) + "\n")
	pickle.dump(results, open(save_file_pkl, 'wb'))


	classes = [t[2] for t in results]
	number_predicted_positive = sum(classes)
	accuracy_count = 0
	precision_recall_count = 0
	for i in range(len(classes)):
		if classes[i] == y_test[i]:
			accuracy_count += 1
		if classes[i] == 1 and y_test[i] == 1:
			precision_recall_count += 1
	accuracy = accuracy_count / len(classes)
	precision = precision_recall_count / number_predicted_positive
	recall = precision_recall_count / number_positive_in_test
	print(accuracy)
	print(precision)
	print(recall)
	plt.plot(losses)
	plt.show()

def supervised_learning_fullyconnected(args):
	candidates = "./data/candidates/sadava_life/all_candidates_preprocessed.txt"
	gold_file = "./data/gold/sadava_life/all_golds_preprocessed.txt"

	negative = set([line.strip() for line in open(candidates)])
	positive = set([line.strip() for line in open(gold_file)])
	negative = negative - positive

	positive_set = copy(positive)
	negative_set = copy(negative)

	negative = list(negative)
	positive = list(positive)

	number_positive_in_train = int(.9 * len(positive))
	number_positive_in_test = len(positive) - number_positive_in_train
	positive_train = positive[:number_positive_in_train]
	positive_test = positive[number_positive_in_train:]

	number_negative_in_train = int(.9 * len(negative))
	number_negative_in_test = len(negative) - number_negative_in_train
	negative_train = negative[:number_negative_in_train]
	negative_test = negative[number_negative_in_train : number_negative_in_train + number_negative_in_test]

	positive_train = [x.split() for x in positive_train]
	positive_test = [x.split() for x in positive_test]
	negative_train = [x.split() for x in negative_train]
	negative_test = [x.split() for x in negative_test]

	X_train = positive_train + negative_train
	y_train = [1] * number_positive_in_train + [0] * number_negative_in_train

	X_test = positive_test + negative_test
	y_test = [1] * number_positive_in_test + [0] * number_negative_in_test

	word_vector_file = "./data/vectors/sadava_life/sadava_bertvectors.vec"
	wvp = WordVectorParser(word_vector_file, word_vector_length=768)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()

	net = FullyConnected(vocab, embedding_layer, gpu=args["--cuda"])
	start = time.time()
	losses = net.train_on_data(X_train, y_train, lr=.01, num_epochs=200, verbose=True)
	end = time.time()
	print("it took %s seconds to train the data" % str(end - start))

	file_stem = "./paper_results/supervised_fc_sadava/"
	save_file_txt = file_stem + "predictions.txt"
	save_file_pkl = file_stem + "predictions.pkl"
	directory = os.path.dirname(save_file_txt)
	if not os.path.exists(directory):
		os.makedirs(directory)

	torch.save(net.state_dict(), file_stem + "model.pt")

	results = net.predict(X_test)
	for i in range(len(results)):
		is_positive = ' '.join(results[i][0]) in positive_set
		t = results[i]
		results[i] = (*t, 1 if is_positive else 0)
	with open(save_file_txt, 'w') as f:
		for t in results:
			f.write(str(t) + "\n")
	pickle.dump(results, open(save_file_pkl, 'wb'))


	classes = [t[2] for t in results]
	number_predicted_positive = sum(classes)
	accuracy_count = 0
	precision_recall_count = 0
	for i in range(len(classes)):
		if classes[i] == y_test[i]:
			accuracy_count += 1
		if classes[i] == 1 and y_test[i] == 1:
			precision_recall_count += 1

	accuracy = accuracy_count / len(classes)
	precision = precision_recall_count / number_predicted_positive if number_predicted_positive > 0 else 0
	recall = precision_recall_count / number_positive_in_test

	stats_file = file_stem + "stats.txt"
	with open(stats_file, 'w') as f:
		f.write(str(accuracy) + "\n")
		f.write(str(precision) + "\n")
		f.write(str(recall) + "\n")

	print(accuracy)
	print(precision)
	print(recall)

	save_plot = "training_loss.png"
	if os.path.isfile(save_plot):
		os.remove(save_plot)
	fig, ax  = plt.subplots(nrows=1, ncols=1)

	ax.plot(losses)
	fig.savefig(save_plot)
	# plt.plot()
	# plt.show()

def transfer_learning(args):
	model_to_use = "./paper_results/supervised_deep_biology/model.pt"
	word_vector_file = "./data/vectors/openstax_biology/biology_bertvectors.vec"
	wvp = WordVectorParser(word_vector_file, word_vector_length=768)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()
	cnn = DeepCNN(vocab, embedding_layer)
	cnn.load_state_dict(torch.load(model_to_use))

	candidates = "./data/candidates/sadava_life/all_candidates_preprocessed.txt"
	gold_file = "./data/gold/sadava_life/all_golds_preprocessed.txt"

	negative = set([line.strip() for line in open(candidates)])
	positive = set([line.strip() for line in open(gold_file)])
	negative = negative - positive

	positive_set = copy(positive)
	negative_set = copy(negative)

	negative = list(negative)
	positive = list(positive)

	random.seed(42)
	random.shuffle(negative)
	random.shuffle(positive)
	random.seed(time.time())

	number_positive_in_train = int(.9 * len(positive))
	number_positive_in_test = len(positive) - number_positive_in_train
	positive_train = positive[:number_positive_in_train]
	positive_test = positive[number_positive_in_train:]

	number_negative_in_train = int(.9 * len(negative))
	number_negative_in_test = len(negative) - number_negative_in_train
	negative_train = negative[:number_negative_in_train]
	negative_test = negative[number_negative_in_train : number_negative_in_train + number_negative_in_test]

	positive_train = [x.split() for x in positive_train]
	positive_test = [x.split() for x in positive_test]
	negative_train = [x.split() for x in negative_train]
	negative_test = [x.split() for x in negative_test]

	X_train = positive_train + negative_train
	y_train = [1] * number_positive_in_train + [0] * number_negative_in_train

	X_test = positive_test + negative_test
	y_test = [1] * number_positive_in_test + [0] * number_negative_in_test

	word_vector_file = "./data/vectors/sadava_life/sadava_bertvectors.vec"
	wvp = WordVectorParser(word_vector_file, word_vector_length=768)
	vocab = wvp.get_vocab()
	embedding_layer = wvp.get_embedding_layer()

	cnn.vocab = vocab
	cnn.embedding_layer = embedding_layer

	file_stem = "./paper_results/transfer_learning_bio_to_sadava/"
	save_file_txt = file_stem + "predictions.txt"
	save_file_pkl = file_stem + "predictions.pkl"
	directory = os.path.dirname(save_file_txt)
	if not os.path.exists(directory):
		os.makedirs(directory)

	results = cnn.predict(X_test)
	for i in range(len(results)):
		is_positive = ' '.join(results[i][0]) in positive_set
		t = results[i]
		results[i] = (*t, 1 if is_positive else 0)
	with open(save_file_txt, 'w') as f:
		for t in results:
			f.write(str(t) + "\n")
	pickle.dump(results, open(save_file_pkl, 'wb'))


	classes = [t[2] for t in results]
	number_predicted_positive = sum(classes)
	accuracy_count = 0
	precision_recall_count = 0
	for i in range(len(classes)):
		if classes[i] == y_test[i]:
			accuracy_count += 1
		if classes[i] == 1 and y_test[i] == 1:
			precision_recall_count += 1

	accuracy = accuracy_count / len(classes)
	precision = precision_recall_count / number_predicted_positive if number_predicted_positive > 0 else 0
	recall = precision_recall_count / number_positive_in_test

	stats_file = file_stem + "stats.txt"
	with open(stats_file, 'w') as f:
		f.write(str(accuracy) + "\n")
		f.write(str(precision) + "\n")
		f.write(str(recall) + "\n")

	print(accuracy)
	print(precision)
	print(recall)	


