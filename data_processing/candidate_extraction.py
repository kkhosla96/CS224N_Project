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
from utils import normalize, clean_tokens_gen

import pdb

DATA_FOLDER = "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"


NGRAM_LENGTHS = [1, 2, 3, 4, 5]
STOP_WORDS = pickle.load(open("stopwords.pkl", 'rb'))

def init_matcher(vocab, candidates):
	matcher = Matcher(vocab)

	pattern = [{'POS': 'ADJ', 'OP': '*'}, {'POS': 'NOUN', 'OP': '+'}]
	matcher.add('candidate', None, pattern)
	pattern = [{'POS': 'ADJ', 'OP': '*'}, {'POS': 'PROPN', 'OP': '+'},{'POS': 'PART', 'OP': '?'},{'POS': 'PROPN', 'OP': '*'}]
	matcher.add('candidate2', None, pattern)
	pattern = [{'POS': 'NOUN'}, {'ORTH': '-'}, {'POS': 'VERB'}, {'POS': 'NOUN', 'OP': '+'}]
	matcher.add('candidate3', None, pattern)
	#catches 1, adds 500 (falsifiable)
	pattern = [{'POS':'ADV', 'OP':'*'},{'POS': 'ADJ'}]
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
	pattern = [{'POS': 'VERB'}, {'POS':'NOUN', 'OP':'+'}]
	matcher.add('candidate8', None, pattern)
	pattern = [{'POS': 'ADV'}, {'ORTH': '-'}, {'POS':'VERB'}, {'POS':'NOUN', "OP":"*"}]
	matcher.add('candidate9', None, pattern)
	pattern = [{'POS': 'ADV'}, {'ORTH': '-'}, {'POS':'NOUN'}]
	matcher.add('candidate10', None, pattern)
	pattern = [{'POS': 'ADJ'}, {'ORTH': '-'}, {'POS':'ADJ', 'OP':'*'}, {'POS': 'NOUN', 'OP': '*'}]
	matcher.add('candidate11', None, pattern)
	pattern = [{'POS': 'NOUN'}, {'ORTH': '-'}, {'POS':'ADJ'}]
	matcher.add('candidate12', None, pattern)
	pattern = [{'POS': 'NOUN','OP':'+'}, {'ORTH': '-'}, {'POS':'VERB'}, {'POS': 'NOUN','OP':'+'}]
	matcher.add('candidate13', None, pattern)
	pattern = [{'POS': 'NOUN'}, {'ORTH': '-'}, {'POS':'NOUN'}]
	matcher.add('candidate14', None, pattern)

	return matcher

def is_valid(start, end, doc):
	for token in doc[start + 1:end]:
		if token.is_sent_start:
			return False
	return True

def noun_phrase_chunk(doc, vocab):
	candidates = set()
	matcher = init_matcher(vocab, candidates)

	matches = matcher(doc)
	for _, start, end in matches:
		if not is_valid(start, end, doc):
			continue
		if "- pron -" in clean_tokens_gen(doc[start:end]):
			continue
		
		candidate = clean_tokens_gen(doc[start:end])
		candidate_string = " ".join(candidate).strip()
		if len(candidate_string) > 0:
			candidates.add(candidate_string)

	return candidates


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
