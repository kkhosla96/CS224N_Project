import torch.nn as nn
import torch.nn.functional as F

from Model import Model

class CNN(Model):
	"""
	A simple CNN network to determine if a term is a glossary term.
	"""

	def __init__(self):
		'''
		Need to supply the hyper-parameters that define the CNN network architecture.
		'''