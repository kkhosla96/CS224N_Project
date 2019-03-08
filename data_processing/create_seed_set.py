import random
import pickle

num_positive = 20
num_negative = 1000

gold_set_file = "../data/gold/openstax_biology_chapters123_gold_simple_lemmatized.pkl"
candidates_set_file = "../data/candidates/openstax_biology_chapters_123_sentences_simple_lemmatized_ngram.pkl"
seed_set_path = "../data/seed_sets/openstax_biology_chapters123_seed.txt"
seed_set_pkl = "../data/seed_sets/openstax_biology_chapters123_seed.pkl"

gold_fh = open(gold_set_file, "rb")
candidate_fh = open(candidates_set_file, "rb")

gold = pickle.load(gold_fh)
candidates = pickle.load(candidate_fh)

seed_set = list()

random_positive = random.sample(gold, num_positive)
for pos in random_positive:
	seed_set.append(pos + " 1")

random_negative = random.sample(candidates - gold, num_negative)
for neg in random_negative:
	seed_set.append(neg + " 0")

random.shuffle(seed_set)
seed_set = set(seed_set)

with open(seed_set_path, "w") as text_file:
	with open(seed_set_pkl, "wb") as pickle_file:
		for line in seed_set:
			text_file.write(line + "\n")
		pickle.dump(seed_set, pickle_file)


gold_fh.close()
candidate_fh.close()