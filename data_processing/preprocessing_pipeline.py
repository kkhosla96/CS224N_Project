"""
CS224N 2018-19: Final Project
preprocessing_pipeline.py: Logic for preprocessing textbook data
Usage:
    preprocessing_pipeline.py generate <textbook_name> (--num-chapters=<num_chapters> | --chapter-num=<chapter_num>)
    preprocessing_pipeline.py glossary print-missing <textbook_name> (--num-chapters=<num_chapters> | --chapter-num=<chapter_num>) [--show-plurals]
	preprocessing_pipeline.py glossary replace-plurals <textbook_name> <num-chapters> [--replace-file]
	preprocessing_pipeline.py validate <textbook_name> [--gold | --candidates]
"""

import os
import io
import sys
import spacy
import re
import pickle
from docopt import docopt
if ".." not in sys.path:
	sys.path.append("..")
from utils import normalize, clean_tokens_gen, term_to_regex, custom_seg
from candidate_extraction import noun_phrase_chunk
from glossary_processing import find_missing_gold_chapter
import pdb

SENTENCES_FOLDER =  "../data/textbook_sentences"
CANDIDATES_FOLDER = "../data/candidates"
GOLD_FOLDER = "../data/gold"


def get_sentences(chapter_doc):
	sentences = []
	for sent in chapter_doc.sents:
		new_sent = " ".join(clean_tokens_gen(sent))
		sentences.append(new_sent)
	return sentences

# def get_golds(gold_terms, nlp):
# 	new_golds = set()
# 	for term in gold_terms:
# 		parsed_gold = nlp(term)
# 		new_gold = " ".join(clean_tokens_gen(parsed_gold))
# 		new_golds.add(new_gold)
# 	return new_golds

def get_gold_lemma(term, chapter_doc):
	no_space_term = term.replace(' ', '').lower()
	for i in range(len(chapter_doc)):
		if no_space_term.startswith(chapter_doc[i].text.lower()):
			curr_string = chapter_doc[i].text.lower()
			if curr_string == no_space_term:
				return normalize(chapter_doc[i].lemma_)
			curr_idx = len(curr_string)				
			for j in range(i + 1, len(chapter_doc)):
				if no_space_term.startswith(chapter_doc[j].text.lower(), curr_idx):
					curr_string += chapter_doc[j].text.lower()
					if curr_string == no_space_term:
						return " ".join(clean_tokens_gen(chapter_doc[i:j+1]))
					curr_idx = len(curr_string)
				else:
					break
	assert False, "Couldn't find term {}".format(term)


def get_golds(gold_terms, chapter_doc):
	new_golds = set()
	for term in gold_terms:
		gold_lemma = get_gold_lemma(term, chapter_doc)
		if len(gold_lemma) > 0:
			new_golds.add(gold_lemma.strip())
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
	new_golds = get_golds(gold_terms, chapter_doc)

	return (new_candidates, new_sentences, new_golds)

def write_output(file, data_folder, textbook_name, data):
	txt_file_path = os.path.join(data_folder, textbook_name, file + ".txt")
	with open(txt_file_path, "w") as f:
		for item in data:
			f.write(item + "\n")
	pkl_file_path = os.path.join(data_folder, textbook_name, file + ".pkl")
	with open(pkl_file_path, "wb") as f:
		pickle.dump(data, f)

def generate(textbook_name, num_chapters, chapter_num):
	nlp = spacy.load('en')
	nlp.add_pipe(custom_seg, before="parser")
	all_candidates = set()
	all_sents = []
	all_golds = set()
	if num_chapters is not None:
		num_chapters = int(num_chapters)
		
		for i in range(1,num_chapters+1):
			file_prefix =  "chapter_" + str(i) + "_"
			new_candidates, new_sentences, new_golds = process_chapter(textbook_name, file_prefix, nlp)
			all_candidates.update(new_candidates)
			all_sents.extend(new_sentences)
			all_golds.update(new_golds)
			print("Processed chapter %d" % i)

		write_output("all_candidates_preprocessed", CANDIDATES_FOLDER, textbook_name, all_candidates)
		write_output("all_sentences_preprocessed", SENTENCES_FOLDER, textbook_name, all_sents)
		write_output("all_golds_preprocessed", GOLD_FOLDER, textbook_name, all_golds)

	else:
		file_prefix =  "chapter_{}_".format(chapter_num)
		new_candidates, new_sentences, new_golds = process_chapter(textbook_name, file_prefix, nlp)
		all_candidates.update(new_candidates)
		all_sents.extend(new_sentences)
		all_golds.update(new_golds)
		print("Processed chapter {}".format(chapter_num))
		write_output("chapter_{}_candidates_preprocessed".format(chapter_num), CANDIDATES_FOLDER, textbook_name, all_candidates)
		write_output("chapter_{}_sentences_preprocessed".format(chapter_num), SENTENCES_FOLDER, textbook_name, all_sents)
		write_output("chapter_{}_golds_preprocessed".format(chapter_num), GOLD_FOLDER, textbook_name, all_golds)

def print_missing_gold_chapter(textbook_name, chapter_num, show_plurals):
	print("MISSING GLOSSARY TERMS FOR CHAPTER %d\n" % chapter_num)

	missing, plurals = find_missing_gold_chapter(textbook_name, chapter_num, show_plurals)
	for term in missing:
		extra = ""
		if show_plurals and (term + 's') in plurals:
			extra = " (But we found %s)" % (term + 's')
		elif show_plurals and (term + 'es') in plurals:
			extra = " (But we found %s)" % (term + "es")
		elif show_plurals and (term + 'e') in plurals:
			extra = " (But we found %s)" % (term + 'e')
		print (term + extra)
	
	print("\nTOTAL MISSING TERMS FOR CHAPTER {}: {}".format(chapter_num, len(missing)))
	
	if show_plurals:
		print("Of the missing terms, {} were plural.".format(len(plurals)))
	print("-" * 80)
	
	return len(missing), len(plurals)

def print_missing_gold(textbook_name, num_chapters, chapter_num, show_plurals):
	print("-" * 80)
	if num_chapters is not None:
		total_missing = 0
		total_plural = 0
		num_chapters = int(num_chapters)

		for i in range(1, num_chapters + 1):
			missing, plural = print_missing_gold_chapter(textbook_name, i, show_plurals)
			total_missing += missing
			total_plural += plural

		print("TOTAL MISSING TERMS: {}".format(total_missing))
		if show_plurals:
			print("Of these, {} were plural.".format(total_plural))
			

	else:
		chapter_num = int(chapter_num)
		print_missing_gold_chapter(textbook_name, chapter_num, show_plurals)

def replace_plural_gold_chapter(textbook_name, chapter_num):
	gold_file_name = "chapter_{}_gold.txt".format(chapter_num)
	gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, gold_file_name)
	gold_terms = [term.strip() for term in io.open(gold_file_path, "r", encoding='utf-8')]
	missing, plurals = find_missing_gold_chapter(textbook_name, chapter_num, True)
	new_golds = set()
	for term in gold_terms:
		if term in missing and term + 's' in plurals:
			new_golds.add(term + 's')
		elif term in missing and term + "es" in plurals:
			new_golds.add(term + 'es')
		elif term in missing and term + 'e' in plurals:
			new_golds.add(term + 'e')
		else:
			new_golds.add(term)
	output_file = "chapter_{}_gold".format(chapter_num)
	write_output(output_file, GOLD_FOLDER, textbook_name, new_golds)


def replace_plural_gold(textbook_name, num_chapters):
	for i in range(1,num_chapters+1):
		replace_plural_gold_chapter(textbook_name, i)

def verify_text_equal_pickles(textbook_name, data_folder, file_name):
	file_path = os.path.join(data_folder, textbook_name, file_name)
	txt = set([term.strip() for term in io.open(file_path + ".txt", "r", encoding='utf-8')])
	pkl = pickle.load(open(file_path + ".pkl", 'rb'))
	assert txt == pkl, "txt and pkl mismatch for file {}".format(file_name)

def term_in_sentences(sentences, term):
	re_string = term_to_regex(term)
	pattern = re.compile(re_string)
	for sentence in sentences:
		if pattern.search(sentence):
			return True
	return False

def verify_terms_in_sentences(sentences, textbook_name, data_folder, file_name):
	print("Verifying all terms in {} show up in the sentences".format(file_name))
	file_path = os.path.join(data_folder, textbook_name, file_name)
	terms = pickle.load(open(file_path + ".pkl", 'rb'))
	for term_num, term in enumerate(terms):
		assert term_in_sentences(sentences, term), "Could not find term: {}".format(term)
		if (term_num + 1) % 100 == 0:
			print("Finished {} out of {} terms".format(term_num + 1, len(terms)))

def validate(textbook_name, just_gold, just_candidates):
	both = not just_gold and not just_candidates
	print("Validating preprocessed files for {}".format(textbook_name))
	print("-" * 80)

	print("Verifying pickle and text files match")
	if both or just_candidates:
		verify_text_equal_pickles(textbook_name, CANDIDATES_FOLDER, "all_candidates_preprocessed")
	if both or just_gold:
		verify_text_equal_pickles(textbook_name, GOLD_FOLDER, "all_golds_preprocessed")
	print("They do!")
	print("-" * 80)	

	print("Verifying all of the preprocessed terms show up in the sentences")
	sentence_file_path = os.path.join(SENTENCES_FOLDER, textbook_name, "all_sentences_preprocessed.pkl")
	sentences = pickle.load(open(sentence_file_path, 'rb'))
	if both or just_candidates:
		verify_terms_in_sentences(sentences, textbook_name, CANDIDATES_FOLDER, "all_candidates_preprocessed")
	if both or just_gold:
		verify_terms_in_sentences(sentences, textbook_name, GOLD_FOLDER, "all_golds_preprocessed")
	print("They do!")
	print("-" * 80)	

	print("Bazinga! All the tests pass")


def main(args):
	textbook_name = args["<textbook_name>"]
	if args["glossary"]:
		if args["print-missing"]:
			num_chapters = args["--num-chapters"]
			chapter_num = args["--chapter-num"]
			print_missing_gold(textbook_name, num_chapters, chapter_num, args["--show-plurals"])
		elif args["replace-plurals"]:
			num_chapters = int(args["<num-chapters>"])
			replace_plural_gold(textbook_name, num_chapters)
	elif args["generate"]:
		num_chapters = args["--num-chapters"]
		chapter_num = args["--chapter-num"]
		generate(textbook_name, num_chapters, chapter_num)
	elif args["validate"]:
		validate(textbook_name, args["--gold"], args["--candidates"])

if __name__ == '__main__':
	args = docopt(__doc__)
	main(args)
