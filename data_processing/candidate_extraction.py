from nltk import RegexpParser, word_tokenize, pos_tag, ngrams
import pickle
import sys
import os
import string
import pdb
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.vocab import Vocab
import sys
if ".." not in sys.path:
	sys.path.append("..")
import utils
import io
from utils import normalize

DATA_FOLDER = "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"


NGRAM_LENGTHS = [1, 2, 3, 4, 5]
STOP_WORDS = pickle.load(open("stopwords.pkl", 'rb'))

def noun_phrase_chunk(text_file_name, output_pickle_name, output_file_name):
	# grammar = "Candidate: {<JJ.*>*<NN.*>+}"
	# cp = RegexpParser(grammar)
	candidates = set()

	# with open(sentence_file_name, 'r') as f:
	# 	line = f.readline()
	# 	count = 0
	# 	while line:
	# 		sentence = line.lower()
	# 		tokens = word_tokenize(sentence)
	# 		pos = pos_tag(tokens)
	# 		tree = cp.parse(pos)
	# 		for subtree in tree.subtrees():
	# 			if subtree.label() == "Candidate":
	# 				candidates.add(" ".join(map(lambda x: x[0], subtree.leaves())))
	# 		count += 1
	# 		line = f.readline()
	nlp = spacy.load('en')
	matcher = Matcher(nlp.vocab)
	pattern = [{'POS': 'ADJ', 'OP': '*'}, {'POS': 'NOUN', 'OP': '+'}]
	matcher.add('candidate', None, pattern)
	pattern = [{'POS': 'ADJ', 'OP': '*'}, {'POS': 'PROPN', 'OP': '+'},{'POS': 'PART', 'OP': '?'},{'POS': 'PROPN', 'OP': '*'}]
	matcher.add('candidate2', None, pattern)
	pattern = [{'POS': 'NOUN'}, {'ORTH': '-'}, {'POS': 'VERB'}, {'POS': 'NOUN', 'OP': '+'}]
	matcher.add('candidate3', None, pattern)
	#catches 1, adds 500 (falsifiable)
	pattern = [{'POS': 'ADJ'}]
	matcher.add('candidate4', None, pattern)
	#catches 1, add 50 (ph scale)
	pattern = [{'POS': 'PROPN'}, {'POS': 'NOUN','OP': '+'}]
	matcher.add('candidate5', None, pattern)
	pattern = [{'POS': 'NOUN'}, {'POS': 'PROPN','OP': '+'}]
	matcher.add('candidate52', None, pattern)
	pattern = [{'POS':'NOUN'}, {'POS':'ADP'}, {'POS':'ADJ', 'OP': '*'}, {'POS':'NOUN'}]
	matcher.add('candidate6', None, pattern)
	pattern = [{'POS':'NOUN'}, {'POS':'CCONJ'}, {'POS':'ADJ', 'OP': '*'}, {'POS':'NOUN'}]
	matcher.add('candidate7', None, pattern)
	pattern = [{'POS': 'VERB'}, {'POS': 'NOUN', 'OP': '+'}]
	matcher.add('candidate8', None, pattern)
	pattern = [{'POS': 'ADV'}, {'ORTH': '-'}, {'POS':'VERB'}]
	matcher.add('candidate9', None, pattern)
	pattern = [{'POS': 'ADV'}, {'ORTH': '-'}, {'POS':'NOUN'}]
	matcher.add('candidate10', None, pattern)
	doc_text = io.open(text_file_name, "r", encoding='utf-8').read().replace('\n', ' ')
	doc = nlp(doc_text)

	matches = matcher(doc)
	for match_id, start, end in matches:
		span = doc[start:end]
		candidates.add(span.text)

	new_candidates = set()
	for candidate in candidates:
		cand_doc = nlp(candidate)
		new_cand = " ".join([normalize(word.lemma_).strip() for word in cand_doc])
		if "-PRON-" not in new_cand:
			new_candidates.add(new_cand)



	with open(output_file_name, 'w') as f:
		for candidate in new_candidates:
			f.write(candidate + "\n")
	with open(output_pickle_name, 'wb') as handle:
		pickle.dump(new_candidates, handle)

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
