candidates_file = "../data/candidates/openstax_biology_chapters_123_sentences_simple_lemmatized_ngram.txt"
gold_file = "../data/gold/openstax_biology_chapters123_gold_simple_lemmatized.txt"
combined_file = "../data/candidates_gold_together/openstax_biology_chapters123_simple_lemmatized_candidates_gold_together.txt"

candidates = set(line.strip() for line in open(candidates_file))
golds = set(line.strip() for line in open(gold_file))

words = set()
for candidate in candidates:
	for word in candidate.split():
		words.add(word)
for gold in golds:
	for word in gold.split():
		words.add(word)

with open(combined_file, 'w') as f:
	for word in words:
		f.write(word + "\n")


