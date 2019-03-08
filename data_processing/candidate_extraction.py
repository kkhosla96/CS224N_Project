from nltk import RegexpParser, word_tokenize, pos_tag, ngrams
import pickle
import sys
import os
import string
import pdb

DATA_FOLDER = "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"


NGRAM_LENGTHS = [1, 2, 3, 4, 5]
STOP_WORDS = pickle.load(open("stopwords.pkl", 'rb'))

def noun_phrase_chunk(sentence_file_name, output_pickle_name, output_file_name):
	grammar = "Candidate: {<JJ.*>*<NN.*>+}"
	cp = RegexpParser(grammar)
	candidates = set()

	with open(sentence_file_name, 'r') as f:
		line = f.readline()
		count = 0
		while line:
			sentence = line.lower()
			tokens = word_tokenize(sentence)
			pos = pos_tag(tokens)
			tree = cp.parse(pos)
			for subtree in tree.subtrees():
				if subtree.label() == "Candidate":
					candidates.add(" ".join(map(lambda x: x[0], subtree.leaves())))
			count += 1
			line = f.readline()

	with open(output_file_name, 'w') as f:
		for candidate in candidates:
			f.write(candidate + "\n")
	with open(output_pickle_name, 'wb') as handle:
		pickle.dump(candidates, handle)

def should_add(ngram):
	for gram in ngram:
		if gram in STOP_WORDS:
			return False
	if ngram[0] == "-" or ngram[-1] == "-":
		return False
	return True

def is_word(word):
	for char in word:
		if char.isalpha() or char == '-':
			return True
	return False


def ngram_chunk(sentence_file_name, output_pickle_name, output_file_name):
	print("ngram chunking...")
	sentence_file = open(sentence_file_name, 'r')
	candidates = set()
	for line in sentence_file:
		words = [word for word in word_tokenize(line) if is_word(word)]
		for length in NGRAM_LENGTHS:
			curr_ngrams = ngrams(words, length)
			for ngram in curr_ngrams:
				if should_add(ngram):
					candidates.add(" ".join(ngram))
	pickle.dump(candidates, open(output_pickle_name, "wb"))
	with open(output_file_name, 'w') as output_file:
		for candidate in candidates:
			output_file.write(candidate + "\n")
	sentence_file.close()

def main(extraction_type, input_file_name):
	sentence_file_name = os.path.join(DATA_FOLDER, input_file_name + ".txt")
	input_file_name += "_" + extraction_type
	output_pickle_name = os.path.join(CANDIDATES_FOLDER, input_file_name + ".pkl")
	output_file_name = os.path.join(CANDIDATES_FOLDER, input_file_name + ".txt")
	if extraction_type == "ngram":
		ngram_chunk(sentence_file_name, output_pickle_name, output_file_name)
	elif extraction_type == "np":
		noun_phrase_chunk(sentence_file_name, output_pickle_name, output_file_name)
	else:
		print("Not a valid chunking technique!")


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python candidate_extraction [ngram or np] [sentences file name]")
	else:
		main(*sys.argv[1:])
