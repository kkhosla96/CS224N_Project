import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	"""
	A simple CNN network to determine if a term is a glossary term.
	"""

	def __init__(self, vocab, out_channels, embeddings, kernel_sizes=None):
		'''
		Need to supply the hyper-parameters that define the CNN network architecture.
		
		@param vocab (Vocab): a class to help get embeddings from words
		@param out_channels (int): how many filters to have for the convolution of each size. corresponds to n in the original paper.
		@param kernel_size (List[int]): list of the sizes for the convolutions (for example, look at bigrams, trigrams, quadgrams etc.).
		the length of this list corresponds to r in the original paper.
		@param embeddings (Tensor): a tensor of (num_words, embedding_size) that contains the word embeddings we are going to use.
		'''
		super(CNN, self).__init__()
		self.vocab = vocab
		self.length_of_term = vocab.getTermLength()
		self.out_channels = out_channels
		self.word_embed_size = len(embeddings[0])
		self.in_channels = self.word_embed_size

		if kernel_sizes is None:
			self.kernel_sizes = [s for s in range(2, self.length_of_term + 1)]
		else:
			self.kernel_sizes = kernel_sizes
		
		self.convs = [nn.Conv1d(self.in_channels, self.out_channels, kernel_size) for kernel_size in self.kernel_sizes]
		self.linear = nn.Linear(len(self.kernel_sizes) * self.out_channels, 1)
		self.embeddings = nn.Embedding.from_pretrained(embeddings)

	def forward(self, terms):
		'''	
		@param terms (List[List[str]]): a list of terms to evaluate, each of which are a list of strings. 
		@returns probabilities (Tensor): a tensor of shape (len(terms),) that represents the probability each term is a key-phrase.
		'''

		# indices is of size (batch_size, self.max_term_length)
		indices = self.vocab.to_input_tensor(terms)
		# embeddings is of size (batch_size, self.max_term_length, self.word_embed_size)
		embeddings = self.embeddings(indices)
		# we transpose the two dimensions because the convolution networks expect an input
		# of size (batch_size, self.word_embed_size, self.max_term_length)
		embeddings = torch.transpose(embeddings, 1, 2)
		# each thing in feature_maps is of size (batch_size, self.out_channels, h_out), where the last two terms are computed
		# using the formula here: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d.
		feature_maps = [conv(embeddings) for conv in self.convs]
		# we take the maximum for each fm, each element in maxes has shape (batch_size, self.out_channels)
		maxes = [torch.max(fm, dim=2)[0] for fm in feature_maps]
		# concat all of them to send to the final layer, should have size (batch_size, len(self.kernel_size) * self.out_channels)
		cat = torch.cat(maxes, dim=1)
		# multiply by linear layer and put through sigmoid. vals, since we squeeze, should have shape (batch_size) 
		vals = torch.squeeze(self.linear(cat))
		# finally, we pass it through a sigmoid to get a probability. prob should also have shape (batch_size)
		probs = torch.sigmoid(vals)
		return probs

	def train(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
