import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	"""
	Abstract class. All of our models will inherit from this class.
	Methods include __init__, forward and training
	"""

	def __init__(self):
		'''
		Set some flags.
		'''
		self.trained = False


	def forward(self, *args):
		'''
		The forward method as specified by the PyTorch model class.
		:param args:
		:return:
		'''
		raise NotImplementedError

	def train(self, *args):
		'''
		Method to supply data to actually train the model.
		:param args:
		:return:
		'''
		if not self.trained:
			raise ValueError("Aye wat u doin I haven't been trained yet.")
		raise NotImplementedError
