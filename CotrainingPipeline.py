import numpy as np
import random
from Typing import List, Tuple, Dict, Set


class CotrainingPipeline:
	def __init__(self, L: List[List[str]], labels: List[int], U: List[List[str]], models: List[Model], g: int, p: int):
        '''
	Class to keep track of labeled and unlabeled data,
	as well as hyperparameters and models.
	'''

	"""
	@param L: List[List[str]] of labeled data
        @param labels: List[int] of labels where 0 = not glossary term, 1 = glossary term
	@param U: list of candidate terms
	@param models: list of models to train in the cotraining algorithm
	@param g: number of predictions to add to L for each model
	@param p: size of subset of U on which we run the models to predict terms
	"""
		self.labelled_data = [L.copy() for _ in range(len(models))]
                self.labels = [labels.copy() for _ in range(len(models))]
		self.U = random.shuffle(U)
		self.models = models
		self.g = g
		self.p = p

	def train(self, num_iterations=500):
        """
        Runs the cotraining algorithm specified in Algorithm 1 of the paper
	@param num_iterations: for how many iterations we should run the pipeline
	"""
		U_prime = self.U[:p]
		self.U = self.U[p:]
		for iteration in range(num_iterations):
			for i, model in enumerate(self.models):
				model.train_on_data(self.labelled_data[i], self.labels[i])
                        terms_to_remove = set()
			for i, model in enumerate(self.models):
				predictions = model.predict(self.U_prime) # [(term, probability, label)]
				sorted_predictions = sorted(predictions, key=lambda tup: tup[1], reverse=True)
				top_g = [(t[0], t[2]) for t in sorted_predictions[:g]]
                                for term in top_g:
                                        self.labelled_data[i - 1].append(term[0])
                                        self.labels[i - 1].append(term[2])
                                        terms_to_remove.add(term[0])
                        update_Uprime(U_prime, terms_to_remove)
			U_prime += self.U[:len(terms_to_remove)]
			self.U = self.U[len(terms_to_remove):]

        def update_Uprime(self, U_prime, terms_to_remove):
        """
        Factors out the ugly try-except logic to remove from U_prime
        @param U_prime: Current unlabelled terms that we are making predictions on
        @param terms_to_remove: set of terms to remove from U'
        """
                for term in terms_to_remove:
                        try:
                                U_prime.remove(term) # currently O(n) time
                        except:
                                pass
