import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from numpy.random import permutation



class ListModule(nn.Module):
	"""
	Was running into issues overfitting on small training-data. turns out the convolution layers were not training
	because they were in a list. to overcome this, use this small module in order to actually make the convolutional
	layers trainable. can't link source right now, but this is ripped from Pytorch boards.
	"""

	def __init__(self, *args):
		super(ListModule, self).__init__()
		idx = 0
		for module in args:
			self.add_module(str(idx), module)
			idx += 1

	def __getitem__(self, idx):
		if idx < 0 or idx >= len(self._modules):
			raise IndexError('index %s is out of range' % idx)
		it = iter(self._modules.values())
		for i in range(idx):
			next(it)
		return next(it)

	def __iter__(self):
		return iter(self._modules.values())

	def __len__(self):
		return len(self._modules)



class CNN(nn.Module):
	"""
	A simple CNN network to determine if a term is a glossary term.
	"""

	def __init__(self, vocab, embedding_layer, out_channels=3, kernel_sizes=None, gpu=False):
		'''
		Need to supply the hyper-parameters that define the CNN network architecture.
		
		@param vocab (Vocab): a class to help get embeddings from words
		@param out_channels (int): how many filters to have for the convolution of each size. corresponds to n in the original paper.
		@param kernel_size (List[int]): list of the sizes for the convolutions (for example, look at bigrams, trigrams, quadgrams etc.).
		the length of this list corresponds to r in the original paper.
		@param embedding_layer (nn.Embedding): pass in the embedding layer for our vocabulary
		'''
		super(CNN, self).__init__()
		self.vocab = vocab
		self.length_of_term = vocab.get_term_length()
		self.out_channels = out_channels
		self.word_embed_size = embedding_layer.weight.size()[1]
		self.in_channels = self.word_embed_size

		if kernel_sizes is None:
			self.kernel_sizes = [s for s in range(2, self.length_of_term + 1)]
		else:
			self.kernel_sizes = kernel_sizes

		self.gpu = gpu	

		convs = [nn.Conv1d(self.in_channels, self.out_channels, kernel_size) for kernel_size in self.kernel_sizes]
		self.convs = ListModule(*convs)
		self.linear = nn.Linear(len(self.kernel_sizes) * self.out_channels, 1)
		self.embedding_layer = embedding_layer
		if (self.gpu):
			self.convs.cuda()
			self.linear.cuda()
			self.embedding_layer.cuda()

	def forward(self, terms):
		'''	
		@param terms (List[List[str]]): a list of terms to evaluate, each of which are a list of strings. 
		@returns probabilities (Tensor): a tensor of shape (batch_size). 
		'''

		# indices is of size (batch_size, self.max_term_length)
		indices = self.vocab.to_input_tensor(terms)
		if (self.gpu):
			indices = indices.cuda()
		# embeddings is of size (batch_size, self.max_term_length, self.word_embed_size)
		embeddings = self.embedding_layer(indices)
		# we transpose the two dimensions because the convolution networks expect an input
		# of size (batch_size, self.word_embed_size, self.max_term_length)
		embeddings = torch.transpose(embeddings, 1, 2)
		# each thing in feature_maps is of size (batch_size, self.out_channels, h_out), where the last two terms are computed
		# using the formula here: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d.
		feature_maps = [self.convs[idx].forward(embeddings) for idx in range(len(self.convs))]
		# we take the maximum for each fm, each element in maxes has shape (batch_size, self.out_channels)
		maxes = [torch.max(fm, dim=2)[0] for fm in feature_maps]
		# concat all of them to send to the final layer, should have size (batch_size, len(self.kernel_size) * self.out_channels)
		cat = torch.cat(maxes, dim=1)
		# multiply by linear layer and put through sigmoid. vals, since we squeeze, should have shape (batch_size) 
		vals = torch.squeeze(self.linear(cat), dim=-1)
		# finally, we pass it through a sigmoid to get a probability. prob should also have shape (batch_size)
		probs = torch.sigmoid(vals)
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

	def train_on_data(self, X_train, y_train, num_epochs=20, lr=.001, momentum=.9, batch_size=32, verbose=False):
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


class DeepCNN(nn.Module):
	"""
	A CNN that is deeper than the last. Will allow more expressive power.
	"""

	def __init__(self, vocab, embedding_layer, gpu=False):
		super(DeepCNN, self).__init__()
		self.vocab = vocab
		self.length_of_term = vocab.get_term_length()
		self.word_embed_size = embedding_layer.weight.size()[1]
		self.in_channels = self.word_embed_size
		self.initial_out_channels = 64 
		self.embedding_layer = embedding_layer
		self.gpu = gpu

		grams = [nn.Conv1d(self.in_channels, self.initial_out_channels, kernel_size) for kernel_size in range(2, self.length_of_term + 1)]
		self.grams = ListModule(*grams)
		self.max1 = nn.MaxPool2d(kernel_size=2, stride=1)
		self.relu = nn.ReLU()
		self.second_out_channels = 64
		self.conv = nn.Conv2d(in_channels=1, out_channels=self.second_out_channels, kernel_size=2)
		self.max2 = nn.MaxPool2d(kernel_size=2, stride=1)
		gram_output_dimension_total = sum(range(1, self.length_of_term))
		self.linear = nn.Linear((gram_output_dimension_total - 3) * (self.initial_out_channels - 3) * self.second_out_channels, 1)
		self.dropout = nn.Dropout(p=.5)
		if self.gpu:
			self.grams.cuda()
			self.max1.cuda()
			self.relu.cuda()
			self.conv.cuda()
			self.max2.cuda()
			self.linear.cuda()
			self.dropout.cuda()
			self.embedding_layer.cuda()

	def forward(self, terms):
		indices = self.vocab.to_input_tensor(terms)
		if self.gpu:
			indices = indices.cuda()
		embeddings = self.embedding_layer(indices)
		embeddings = torch.transpose(embeddings, 1, 2)
		feature_maps = [self.grams[idx].forward(embeddings) for idx in range(len(self.grams))]
		cat = torch.cat(feature_maps, dim=2)
		cat = torch.transpose(cat, 1, 2)
		maxd = self.max1(cat)
		relud = self.relu(maxd)
		unsqueezed = torch.unsqueeze(relud, dim=1)
		convd = self.conv(unsqueezed)
		maxd2 = self.max2(convd)
		flat = maxd2.view(maxd2.size()[0], -1)
		dropped = self.dropout(flat)
		scores = self.linear(dropped)
		scores = scores.squeeze(dim=1)
		probs = torch.sigmoid(scores) 
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
		optimizer = optim.SGD(self.parameters(), lr, momentum, weight_decay=1e-3)

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





