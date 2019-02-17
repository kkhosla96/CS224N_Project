import numpy as np
import random
from Typing import List, Tuple, Dict, Set


class CotrainingPipeline:
	'''
	Class to keep track of labeled and unlabeled data,
	as well as hyperparameters and models.
	'''

	"""
	@param L: list of labeled data
	@param U: list of candidate terms
	@param models: list of models to train in the cotraining algorithm
	@param g: number of predictions to add to L for each model
	@param p: size of subset of U on which we run the models to predict terms
	"""
	def __init__(self, L: Set[str], U: List[str], models: List[Model], g: int, p: int):
		self.L = L
		self.U = random.shuffle(U)
		self.models = models
		self.g = g
		self.p = p
		self.U_prime = None

	"""
	@param num_iterations: for how many iterations we should train the models
	"""
	def train(self, num_iterations=500: int):
		self.U_prime = self.U[:p]
		self.U = self.U[p:]
		for iteration in range(num_iterations):
			for model in models:
				model.train(L)
			for model in models:
				predictions = model.predict(self.U_prime)
				sorted_predictions = sorted(predictions, key=lambda tup: tup[1], reverse=True)
				top_g = [t[0] for t in sorted_predictions[:g]]
				self.L.update(set(top_g))
			self.U_prime += self.U[:len(models) * g]
			self.U = self.U[len(models) * g:]





