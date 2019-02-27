import torch
import torch.nn as nn
import numpy as np

from Vocab import Vocab


class WordVectorParser(object):

	def __init__(self, text_file, max_term_length=None):
		'''
		Class to read a .vec file that is generated by FastText. This will allow us to get both the Vocab object as well as the tensor for the words that we then pass to our models.
		@param text_file (str): location of a .vec file. the first line should have the format (num_vectors, length_of_vector) and each subsequent line should be of the form (word v_1 v_2 ... v_{length_of_vector}).
		unfortunately, num_vectors might be an overestimate, since the .vec file might contain duplicates, so we cant use it.
		@param max_term_length (int): the Vocab class, which turns terms into index tensors, needs to know the maximum
		length of a term. because we construct a Vocab object in this class, we need to supply it with this max length. if it is passed in as None, we will just use the default from the vocab class.
		'''

		if max_term_length is None:
			self.vocab = Vocab(None)
		else:
			self.vocab = Vocab(None, max_term_length)
		self.text_file = text_file
		vecs = []
		with open(text_file) as f:
			line = f.readline()
			split = line.split()
			self.word_vector_length = int(split[1])
			# this corresponds to the padding vector, it always goes first since
			# in every vocab object we have vocab.pad_id = 0
			vecs.append([0.] * self.word_vector_length)
			line = f.readline()
			while line:
				split = line.split()
				word = split[0]
				if word not in self.vocab:
					word_index = self.vocab.add(word)
					vec = split[1:]
					vec = [float(val) for val in vec]
					vecs.append(vec)
				line = f.readline()
		vecs = np.array(vecs)
		self.embeddings = torch.from_numpy(vecs)
		self.embedding_layer = nn.Embedding.from_pretrained(self.embeddings)
		self.embedding_layer.weight.requires_grad = False

	def get_vocab(self):
		'''
		@return vocab (Vocab): the vocab created in this class
		'''

		return self.vocab

	def get_embeddings(self):
		'''
		@return embeddings (Tensor): the embeddings created in this class
		'''

		return self.embeddings

	def get_embedding_layer(self):
		'''
		@return embedding_layer (nn.Embedding): the nn.Embedding layer created in this class
		'''
		return self.embedding_layer










