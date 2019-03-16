import os
import sys
import pickle
from bert_serving.client import BertClient
from pdb import set_trace as debug

SENTENCES_FOLDER = "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"
GOLD_FOLDER = "../data/gold"
VECTORS_FOLDER = "../data/vectors"
BATCH_SIZE = 256

def get_sentences(sentence_file_path):
	sentences = None
	with open(sentence_file_path, 'r') as file:
		sentences = [line.strip().split() for line in file]
		sentences = list(filter(lambda sentence : len(sentence) > 0, sentences))
	return sentences

def get_pickle_data(pickle_file_path):
	pickle_data = None
	with open(pickle_file_path, 'rb') as file:
		pickle_data = pickle.load(file)
	return type(pickle_data)(filter(lambda x : len(x), pickle_data))

def filter_sentences(sentences, all_terms):
	keep_indices = []
	for i, sentence in enumerate(sentences):
		remove_terms = set()
		keep = False
		for term in all_terms:
			if term in sentence:
				if not keep:
					keep_indices.append(i)
					keep = True
				remove_terms.add(term)
		for term in remove_terms:
			all_terms.remove(term)
	return [sentences[i] for i in keep_indices]

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

def main(sentence_file, candidates_file, gold_file, vector_file):
	sentence_file_path = os.path.join(SENTENCES_FOLDER, sentence_file)
	sentences = get_sentences(sentence_file_path)
	
	candidates_file_path = os.path.join(CANDIDATES_FOLDER, candidates_file)
	candidates = get_pickle_data(candidates_file_path)
	
	gold_file_path = os.path.join(GOLD_FOLDER, gold_file)
	gold = get_pickle_data(gold_file_path)
	
	all_terms = gold | candidates
	unigrams = get_unigrams(all_terms)
	#sentences = filter_sentences(sentences, all_terms.copy())
	
	vector_file_path = os.path.join(VECTORS_FOLDER, vector_file)
	write_vectors(sentences, unigrams, vector_file_path)
	


if __name__ == "__main__":
	if len(sys.argv) < 5:
		print("Usage: python get_bert_vectors.py <sentence_file> <candidates_file> <gold_file> <vectors_file>")
	else:
		main(*sys.argv[1:])
