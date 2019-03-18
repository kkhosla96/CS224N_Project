import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from numpy.random import permutation
from CNN import ListModule

class FullyConnected(nn.Module):
	'''
	Simple, fully connected neural network.
	'''

	def __init__(self, vocab, embedding_layer, gpu=False):
		super(FullyConnected, self).__init__()
		self.vocab = vocab
		self.length_of_term = vocab.get_term_length()
		self.word_embed_size = embedding_layer.weight.size()[1]
		self.gpu = gpu

		self.embedding_layer = embedding_layer
		first_layer_neurons = self.length_of_term * 300
		second_layer_neurons = int(1.25 * first_layer_neurons)
		third_layer_neurons = int(first_layer_neurons)
		fourth_layer_neurons = int(.66 * first_layer_neurons)
		layer_sizes = [first_layer_neurons, second_layer_neurons, third_layer_neurons, fourth_layer_neurons]
		self.model = nn.Sequential()
		self.model.add_module("dimensionality_reduction", nn.Linear(self.word_embed_size * self.length_of_term, first_layer_neurons))
		self.model.add_module("dropout1", nn.Dropout(p=.5))
		self.model.add_module("linear1", nn.Linear(first_layer_neurons, second_layer_neurons))
		self.model.add_module("relu1", nn.ReLU())
		self.model.add_module("dropout2", nn.Dropout(p=.5))
		self.model.add_module("linear2", nn.Linear(second_layer_neurons, third_layer_neurons))
		self.model.add_module("relu2", nn.ReLU())
		self.model.add_module("dropout3", nn.Dropout(p=.5))
		self.model.add_module("linear3", nn.Linear(third_layer_neurons, fourth_layer_neurons))
		self.model.add_module("linear4", nn.Linear(fourth_layer_neurons, 1))
		self.model.add_module("sigmoid", nn.Sigmoid())
		if self.gpu:
			self.embedding_layer.cuda()
			self.model.cuda()

	def forward(self, terms):
		indices = self.vocab.to_input_tensor(terms)
		if self.gpu:
			indices = indices.cuda()
		embeddings = self.embedding_layer(indices)
		embeddings = embeddings.view(embeddings.size()[0], -1)
		probs = self.model(embeddings)
		probs = probs.squeeze(dim=1)
		return probs

	def predict(self, terms):
		probs = self.forward(terms)
		ret = []
		for index in range(len(probs)):
			prob = probs[index].item()
			if prob >= .5:
				ret.append((terms[index], prob, 1))
			else:
				ret.append((terms[index], 1 - prob, 0))
		return ret

	def train_on_data(self, X_train, y_train, num_epochs=20, lr=.01, momentum=.9, batch_size=32, verbose=False):
		self.X_train = X_train
		self.y_train = torch.tensor(y_train, dtype=torch.float)
		if (self.gpu):
			self.y_train = self.y_train.cuda()

		loss_function = nn.BCELoss()
		optimizer = optim.SGD(self.parameters(), lr, momentum)

		batch_starting_index = 0
		number_examples = len(self.X_train)
		num_iterations = math.ceil(number_examples / batch_size)
		losses = []
		for epoch in range(num_epochs):
			running_loss = 0.0
			batch_starting_index = 0
			permu = permutation(number_examples)
			self.X_train = [self.X_train[permu[i]] for i in range(number_examples)]
			self.y_train = self.y_train[permu]
			for iteration in range(num_iterations):
				inputs = self.X_train[batch_starting_index : min(number_examples, batch_starting_index + batch_size)]
				labels = self.y_train[batch_starting_index : min(number_examples, batch_starting_index + batch_size)]

				optimizer.zero_grad()

				outputs = self.forward(inputs)
				loss = loss_function(outputs, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()

				batch_starting_index = (batch_starting_index + batch_size) % number_examples

			losses.append(float(running_loss))
			
			if verbose and epoch % 10 == 0: print('Finished epoch %d' % (epoch + 1))
		
		if verbose: print('Finished training')
		return losses
