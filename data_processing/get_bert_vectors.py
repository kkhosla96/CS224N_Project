import os
import sys
import pickle
from bert_serving.client import BertClient
from pdb import set_trace as debug

SENTENCES_FOLDER = "../data/textbook_sentences"
SENTENCES_FILE = "all_sentences_preprocessed.pkl"
CANDIDATES_FOLDER = "../data/candidates"
CANDIDATES_FILE = "all_candidates_preprocessed.pkl"
GOLD_FOLDER = "../data/gold"
GOLD_FILE = "all_golds_preprocessed.pkl"
VECTORS_FOLDER = "../data/vectors"
BATCH_SIZE = 256

def get_pickle_data(pickle_file_path):
	pickle_data = None
	with open(pickle_file_path, 'rb') as file:
		pickle_data = pickle.load(file)
	return type(pickle_data)(filter(lambda x : len(x) > 0, pickle_data))

def write_vectors(sentences, unigrams, vector_file_path):
	vector_file = open(vector_file_path, 'w')
	bc = BertClient()
	found_unigrams = set()
	i = 0
	while i < len(sentences):
		sentence_batch = sentences[i:i + BATCH_SIZE]
		vectors = bc.encode(sentence_batch, is_tokenized=True)                
		for sentence_num, sentence in enumerate(sentence_batch):
			for word_index, word in enumerate(sentence):
				if word in unigrams and word not in found_unigrams:
					found_unigrams.add(word)
					word_vector = vectors[sentence_num][word_index]
					string_vector = " ".join(map(str, word_vector))
					vector_file.write(word + ' ' + string_vector + '\n')
		print("Finished writing vectors through sentence number %d" % (i + BATCH_SIZE))
		i += BATCH_SIZE
	vector_file.close()

def get_unigrams(all_terms):
	unigrams = set()
	for term in all_terms:
		split_term = term.split()
		for unigram in split_term:
			unigrams.add(unigram)
	return unigrams

def main(textbook_name, vector_output_file):

	sentence_file_path = os.path.join(SENTENCES_FOLDER, textbook_name, SENTENCES_FILE)
	sentences = get_pickle_data(sentence_file_path)
	sentences = [sentence.split() for sentence in sentences]
	
	candidates_file_path = os.path.join(CANDIDATES_FOLDER, textbook_name, CANDIDATES_FILE)
	candidates = get_pickle_data(candidates_file_path)
	
	gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, GOLD_FILE)
	gold = get_pickle_data(gold_file_path)
	
	all_terms = gold | candidates
	unigrams = get_unigrams(all_terms)
	
	vector_file_path = os.path.join(VECTORS_FOLDER, vector_output_file)
	write_vectors(sentences, unigrams, vector_file_path)
	


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python get_bert_vectors.py <textbook name> <vectors_output_file>")
	else:
		main(*sys.argv[1:])
