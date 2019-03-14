candidate_files = ["../data/candidates/openstax_biology/openstax_biology_sentences_part_%s_np.txt" % str(i) for i in range(1, 4)]
gold_file = "../data/gold/openstax_biology/openstax_biology_gold_lemmatized.txt"
combined_file = "../data/candidates_gold_together/openstax_biology_candidates_and_gold.txt"

golds = set([line.strip() for line in open(gold_file).readlines()])

words = set()
for candidate_file in candidate_files:
	candidates = set([line.strip() for line in open(candidate_file).readlines()])
	for candidate in candidates:
		for word in candidate.split():
			words.add(word)
for gold in golds:
	for word in gold.split():
		words.add(word)

with open(combined_file, 'w') as f:
	for word in words:
		f.write(word + "\n")


