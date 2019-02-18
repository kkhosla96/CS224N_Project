import torch.nn as nn
import torch.nn.functional as F

from Model import Model

class CNN(nn.Module):
	"""
	A simple CNN network to determine if a term is a glossary term.
	"""

	def __init__(self, word_embed_size, kernel_size=3, max_pool_kernel=2):
		'''
		Need to supply the hyper-parameters that define the CNN network architecture.
		Right now, this will be a simple CNN.
		'''
		super(CNN, self).__init__()
		self.word_embed_size = word_embed_size
		self.kernel_size = kernel_size

		self.conv = nn.Conv2d(1, 3, self.kernel_size)
		self.max_pool = nn.MaxPool2d(max_pool_kernel)

	'''
	mat has size (batch_size, 1, height, embed_size)
	'''
	def forward(self, mat):
		conv = self.conv(mat)
		rel = F.relu(conv)
		pooled = self.max_pool(rel)
		flat = pooled.view(pooled.size()[0], -1)
		return flat

	def train(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train