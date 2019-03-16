import os
import io
import sys
import spacy
import pickle
if ".." not in sys.path:
	sys.path.append("..")
from utils import normalize, clean_tokens_gen
from candidate_extraction import noun_phrase_chunk

SENTENCES_FOLDER =  "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"
GOLD_FOLDER = "../data/gold"


def get_sentences(chapter_doc):
	sentences = []
	for sent in chapter_doc.sents:

		new_sent = " ".join(clean_tokens_gen(sent))
		sentences.append(new_sent)
	return sentences

def get_golds(gold_terms, nlp):
	new_golds = set()
	for term in gold_terms:
		parsed_gold = nlp(term)
		new_gold = " ".join(clean_tokens_gen(parsed_gold))
		new_golds.add(new_gold)
	return new_golds


def process_chapter(textbook_name, file_prefix, nlp):
	sentence_file_name = file_prefix + "sentences.txt"
	sentence_file_path = os.path.join(SENTENCES_FOLDER, textbook_name, sentence_file_name)
	gold_file_name = file_prefix + "gold.txt"
	gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, gold_file_name)

	chapter_text = io.open(sentence_file_path, "r", encoding='utf-8').read().replace('\n', ' ')
	chapter_doc = nlp(chapter_text)
	gold_terms = [term.strip() for term in io.open(gold_file_path, "r", encoding='utf-8')]

	new_candidates = noun_phrase_chunk(chapter_doc, nlp.vocab)
	new_sentences = get_sentences(chapter_doc)
	new_golds = get_golds(gold_terms, nlp)

	return (new_candidates, new_sentences, new_golds)

def write_output(output_type, data_folder, textbook_name, data):
	file = "all_" + output_type + "_preprocessed"
	txt_file_path = os.path.join(data_folder, textbook_name, file + ".txt")
	with open(txt_file_path, "w") as f:
		for item in data:
			f.write(item + "\n")
	pkl_file_path = os.path.join(data_folder, textbook_name, file + ".pkl")
	with open(pkl_file_path, "wb") as f:
		pickle.dump(data, f)

def main(textbook_name, num_chapters):
	num_chapters = int(num_chapters)
	nlp = spacy.load('en')
	all_candidates = set()
	all_sents = []
	all_golds = set()

	for i in range(1,num_chapters+1):
		file_prefix =  "chapter_" + str(i) + "_"
		new_candidates, new_sentences, new_golds = process_chapter(textbook_name, file_prefix, nlp)
		all_candidates.update(new_candidates)
		all_sents.extend(new_sentences)
		all_golds.update(new_golds)
		print("Processed chapter %d" % i)

	write_output("candidates", CANDIDATES_FOLDER, textbook_name, all_candidates)
	write_output("sentences", SENTENCES_FOLDER, textbook_name, all_sents)
	write_output("golds", GOLD_FOLDER, textbook_name, all_golds)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print("python preprocessing_pipeline.py <textbook_name> <num_chapters>")
	else:
		main(*sys.argv[1:])
