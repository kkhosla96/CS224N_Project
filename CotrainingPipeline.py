import numpy as np
import random
from typing import List, Tuple, Dict, Set
from torch.nn import Module


class CotrainingPipeline:
	def __init__(self, L: List[List[str]], labels: List[int], U: List[List[str]], models: List[Module], g: int, p: int):

		self.labelled_data = [L.copy() for _ in range(len(models))]
		self.labels = [labels.copy() for _ in range(len(models))]
		self.U = U
		random.shuffle(self.U)
		self.models = models
		self.g = g
		self.p = p

	def train(self, num_iterations=200, num_epochs=75):
		U_prime = self.U[:self.p]
		self.U = self.U[self.p:]
		for iteration in range(num_iterations):
			for i, model in enumerate(self.models):
				model.train_on_data(self.labelled_data[i], self.labels[i], num_epochs=num_epochs)
			terms_to_remove = []
			for i, model in enumerate(self.models):
				predictions = model.predict(U_prime) # [(term, probability, label)]
				sorted_predictions = sorted(predictions, key=lambda tup: tup[1], reverse=True)
				top_g = [(t[0], t[2]) for t in sorted_predictions[:self.g]]
				for term in top_g:
					self.labelled_data[i - 1].append(term[0])
					self.labels[i - 1].append(term[1])
					terms_to_remove.append(term[0])
			self.update_Uprime(U_prime, terms_to_remove)
			U_prime += self.U[:len(terms_to_remove)]
			self.U = self.U[len(terms_to_remove):]
			print("Done with iteration %d" % iteration)


	def update_Uprime(self, U_prime, terms_to_remove):
		for term in terms_to_remove:
			try:
				U_prime.remove(term) # currently O(n) time
			except:
				pass
