from nltk import RegexpParser, word_tokenize, pos_tag
import pickle



data = "../data/textbook-sentences.txt"
outputPickle = "../data/candidates.pickle"
outputFile = "../data/candidates.txt"

grammar = "Candidate: {<JJ.*>*<NN.*>+}"
cp = RegexpParser(grammar)
candidates = set()

with open(data) as f:
	line = f.readline()
	count = 0
	while line:
		sentence = line.split("\t")[1].lower()
		tokens = word_tokenize(sentence)
		pos = pos_tag(tokens)
		tree = cp.parse(pos)
		for subtree in tree.subtrees():
			if subtree.label() == "Candidate":
				candidates.add(" ".join(map(lambda x: x[0], subtree.leaves())))
		count += 1
		line = f.readline()

with open(outputFile, 'w') as f:
	for candidate in candidates:
		f.write(candidate + "\n")
with open(outputPickle, 'wb') as handle:
	pickle.dump(candidates, handle)


