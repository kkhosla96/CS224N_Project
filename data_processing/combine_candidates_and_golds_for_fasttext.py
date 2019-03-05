candidates_file = "../data/candidates/openstax_biology_candidates.txt"
gold_file = "../data/gold/openstax_biology_gold.txt"
combined_file = "../data/candidates_gold_together/openstax_biology_candidates_gold_together.txt"

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


