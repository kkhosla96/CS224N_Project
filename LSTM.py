import pdb
import math
import torch
from torch import nn
from typing import List
import torch.optim as optim
import torch.nn.functional as F
from numpy.random import permutation


HIDDEN_SIZE = 256


class LSTM(nn.Module):
	def __init__(self, hidden_size=HIDDEN_SIZE, bidirectional=False, vocab=None, embeddings=None, gpu=False):
		""" Init NMT Model.

		@param hidden_size (int): Hidden Size (dimensionality)
		@param bidirectional (bool): Indicates whether the LSTM is bidirectional
		@param vocab (Vocab): Vocabulary object. See Vocab.py for documentation.
		@param embeddings (Embedding): Embedding object storing the word embedings
		"""
		
		super(LSTM, self).__init__()
		self.embeddings = embeddings
		self.embed_size = embeddings.embedding_dim
		self.LSTM = nn.LSTM(self.embed_size, hidden_size, bidirectional=bidirectional)
		self.vocab = vocab
		self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
		self.gpu = gpu
		if self.gpu:
			self.embeddings.cuda()
			self.LSTM.cuda()
			self.linear.cuda()

	
	def forward(self, candidates: List[List[str]]) -> torch.Tensor:
		""" Takes a mini-batch of candidates and computes the log-likelihood that
		they are glossary terms
		
		@param candidates (List[List[str]]): Batch of candidates (need to be padded)
		
		@returns scores (Tensor): a tensor of shape (batch_size, ) representing the
		log-likelihood that a candidate is a glossary term
		"""
		# pdb.set_trace()
		candidate_lengths = sorted([min(len(candidate), self.vocab.get_term_length()) for candidate in candidates], reverse = True)
		candidates = sorted(candidates, key= lambda candidate : len(candidate), reverse=True)
		candidates_padded = self.vocab.to_input_tensor(candidates).permute(1, 0)   # Tensor: (max_length, batch_size))
		if self.gpu:
			candidates_padded.cuda()
		enc_hiddens = self.encode(candidates_padded, candidate_lengths)

		# this code taken from https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/
		idx = (torch.LongTensor(candidate_lengths) - 1).view(-1, 1).expand(len(candidate_lengths), enc_hiddens.size(2))
		idx = idx.unsqueeze(1)
		if self.gpu:
			idx.cuda()
		last_hiddens = enc_hiddens.gather(1, idx).squeeze(1)
		probs = torch.sigmoid(self.linear(last_hiddens).squeeze(-1))
		return probs

	def encode(self, candidates_padded: torch.Tensor, candidate_lengths: List[int]) -> torch.Tensor:
		""" Apply the encoder to candidates to obtain encoder hidden states.

		@param source_padded (Tensor): Tensor of padded candidates with shape (max_length, batch_size), where
										b = batch_size, max_length = maximum source sentence length. Note that 
									   these have already been sorted in order of longest to shortest sentence.
		@param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
		@returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h or 2h), where
										b = batch size, src_len = maximum source sentence length, h = hidden size.
										If the LSTM is bidirectional, then the final dimension is 2h
		"""
		enc_hiddens, dec_init_state = None, None
		X = self.embeddings(candidates_padded)
		X = nn.utils.rnn.pack_padded_sequence(X, candidate_lengths)
		enc_hiddens, _ = self.LSTM(X)
		enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)
		return enc_hiddens

	# unfortunately, the next two methods are copied exactly from the CNN class. would be better to abstract that. however,
	# maybe not if they two models have to be trained different.y
	
	def predict(self, terms):
		probs = self.forward(terms)
		max_probs, labels = torch.max(probs, dim=1)
		ret = [(terms[index], max_probs[index].item(), labels[index].item()) for index in range(len(terms))]
		return ret

	def train_on_data(self, X_train, y_train, num_epochs=20, lr=.0001, betas=(.9, .999), batch_size=32, verbose=False):
		self.X_train = X_train
		self.y_train = torch.tensor(y_train, dtype=torch.float)
		if (self.gpu):
			self.y_train = self.y_train.cuda()

		loss_function = nn.BCELoss()
		optimizer = optim.Adam(self.parameters(), lr, betas)

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

				# print(*self.LSTM.parameters())
				# print(self.linear.weight)

			losses.append(float(running_loss))
			
			if verbose: print('Finished epoch %d' % (epoch + 1))
		
		if verbose: print('Finished training')
		return losses

