import torch
from utils import pad_terms

from typing import List

class Vocab(object):
	"""
	Allows us to turn words into vectors to input them into our networks.
	"""

	def __init__(self, vocabFile, max_term_length=4):
		'''
		Initializes the Vocab object by creating a map from words to indices.
		@param vocabFile (str): location of a file from which to read words.
		this file should have one word per line.
		'''
		self.word2id = dict()
		self.id2word = dict()
		self.max_term_length = max_term_length
		self.word2id['<pad>'] = 0
		self.pad_id = self.word2id['<pad>']
		self.id2word[self.pad_id] = '<pad>'
		with open(vocabFile) as f:	
			line = f.readline()
			count = self.pad_id + 1
			while line:
				without_newline = line[:-1]
				self.word2id[without_newline] = count
				self.id2word[count] = without_newline
				line = f.readline()
				count += 1

	def __getitem__(self, word):
		return self.word2id[word]

	def __contains__(self, word):
		return word in self.word2id

	def __setitem__(self, key, value):
		raise ValueError("Should not be writing to the vocabulary.")

	def __len__(self):
		return len(self.word2id)

	def getTermLength(self):
		return self.max_term_length

	def id2word(self, id):
		return self.id2word[id]

	def words2indices(self, terms):
		'''
		Converts either a list of terms into list of list of indices or
		a list of words into a list of indices.
		@param terms: (List[List[str]] or List[str]): multiple terms or just one term
		@return word_ids (List[List[int] or List[int]]): the input except all words replaced
		by their indices
		'''
		# in this case, we are converting multiple terms
		if type(terms[0]) == list:
			return [[self[w] for w in t] for t in terms]
		# in this case, we are converting a single term (which is represented as a list of str)
		elif type(terms[0]) == str:
			return [self[w] for w in terms]
		else:
			ValueError("Invalid input to words2indices, please make sure input is of type List[List[str]] or List[str]")

	def indices2words(self, word_ids):
		'''
		Converts list of word ids into words.
		@param word_ids (List[int]): list of ids for words
		@return term (List[str]) list of words
		'''
		return [self.id2word[word_id] for word_id in word_ids]
	
	def to_input_tensor(self, terms: List[List[str]]) -> torch.Tensor:
		'''
		Convert a list of terms into a tensor with shape (batch_size, self.max_term_length)
		@param terms (List[List[str]]) a batch of terms each of which we have to get their
		index representations for their words
		# @param device: device on which to load the tensor
		@returns terms_as_ints (tensor): tensor of size (batch_size, self.max_term_length)
		'''
		word_ids = self.words2indices(terms)
		terms_padded = pad_terms(word_ids, self['<pad>'], self.max_term_length)
		tens = torch.tensor(terms_padded, dtype=torch.long)
		return tens









