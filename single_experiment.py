import sys
import pickle
from CNN import CNN
from CotrainingPipeline import CotrainingPipeline
from WordVectorParser import WordVectorParser
import time

'''
Usage: python single_experiment.py path_to_word_vectors path_to_unlabelled_data path_to_seed
'''

# get files
word_vector_file = sys.argv[1]
unlabelled_file = sys.argv[2]
seed_file = sys.argv[3]

# get data
unlabelled = [line.split() for line in open(unlabelled_file)]
seed_terms = []
seed_labels = []
with open(seed_file) as f:
	for line in f:
		split = line.split()
		seed_terms.append(split[:-1])
		seed_labels.append(int(split[-1]))

# create learning mechanisms
out_channels = 3
wvp = WordVectorParser(word_vector_file)
vocab = wvp.get_vocab()
embedding_layer = wvp.get_embedding_layer()
cnn = CNN(vocab, out_channels, embedding_layer)

# create cotrainer
g = 20
p = 500
cotrainer = CotrainingPipeline(seed_terms, seed_labels, unlabelled, [cnn], g, p)

# train the models
num_iterations = 2
num_epochs = 20
now = time.time()
cotrainer.train(num_iterations, num_epochs)
print("it took %f seconds to train for %d iterations." % (time.time() - now, num_iterations))

# save the results
output_labeled_data_file = "./data/output/single_cnn_labeled_data.pkl"
with open(output_labeled_data_file, 'wb') as f:
	pickle.dump(cotrainer.labelled_data[0], f)
output_label_file = "./data/output/single_cnn_labels.pkl"
with open(output_label_file, 'wb') as f:
	pickle.dump(cotrainer.labels[0], f)




