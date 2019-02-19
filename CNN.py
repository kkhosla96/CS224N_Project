import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
	"""
	A simple CNN network to determine if a term is a glossary term.
	"""

	def __init__(self, height, word_embed_size, kernel_size=3, max_pool_kernel=2):
		'''
		Need to supply the hyper-parameters that define the CNN network architecture.
		Right now, this will be a simple CNN.
		'''
		super(CNN, self).__init__()
		self.word_embed_size = word_embed_size
		self.kernel_size = kernel_size
		self.height = height

		self.in_channels = 1
		self.out_channels = 3
		self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
		conv_H_out = self.height - kernel_size + 1
		conv_W_out = self.word_embed_size - kernel_size + 1
		conv_out_size = (-1, self.out_channels, conv_H_out, conv_W_out)
		self.max_pool = nn.MaxPool2d(max_pool_kernel)
		max_pool_H_out = int((conv_H_out - max_pool_kernel) / (max_pool_kernel) + 1)
		max_pool_W_out = int((conv_W_out - max_pool_kernel) / (max_pool_kernel) + 1)
		max_pool_size = (-1, self.out_channels, max_pool_H_out, max_pool_W_out)
		self.linear = nn.Linear(self.out_channels * max_pool_H_out * max_pool_W_out, 1)

	'''
	mat has size (batch_size, 1, height, embed_size)
	'''
	def forward(self, mat):
		conv = self.conv(mat)
		rel = F.relu(conv)
		pooled = self.max_pool(rel)
		flat = pooled.view(pooled.size()[0], -1)
		out = self.linear(flat)
		prob = F.sigmoid(out)
		return prob.squeeze()

	def train(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
