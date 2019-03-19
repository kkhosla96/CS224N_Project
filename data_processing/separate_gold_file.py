import sys
import io
import os
import re
if ".." not in sys.path:
	sys.path.append("..")
from utils import term_to_regex
import pickle


SENTENCES_FOLDER =  "../data/textbook_sentences"
GOLD_FOLDER = "../data/gold"

def in_chapter_text(term, chapter_text):
	return re.search(term_to_regex(term), chapter_text)

def assign_golds_to_chapter(textbook_name, chapter_num, golds, gold_chapter_dict, changes, find_plurals=False):
	file_prefix =  "chapter_" + str(chapter_num) + "_"
	sentence_file_name = file_prefix + "sentences.txt"
	sentence_file_path = os.path.join(SENTENCES_FOLDER, textbook_name, sentence_file_name)

	chapter_text = io.open(sentence_file_path, "r", encoding='utf-8').read().replace('\n', ' ').lower()
	
	gold_chapter_dict[chapter_num] = list()
	to_remove = set()

	for term in golds:
		if in_chapter_text(term, chapter_text):
			to_remove.add(term)
			gold_chapter_dict[chapter_num].append(term)
		elif find_plurals:
			if in_chapter_text(term + 's', chapter_text):
				to_remove.add(term)
				changes[term] = term + 's'
				gold_chapter_dict[chapter_num].append(term + 's')
			elif in_chapter_text(term + 'es', chapter_text):
				to_remove.add(term)
				changes[term] = term + 'es'
				gold_chapter_dict[chapter_num].append(term + 'es')
			elif in_chapter_text(term + 'e', chapter_text):
				to_remove.add(term)
				changes[term] = term + 'e'
				gold_chapter_dict[chapter_num].append(term + 'e')
	new_golds = golds - to_remove
	return new_golds, gold_chapter_dict, changes

def build_chapter_dict(textbook_name, total_chapters, changes, find_plurals=False):
	file_prefix = textbook_name + "_"
	gold_chapter_dict = {}
	gold_file_name = file_prefix + "gold.txt"
	gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, gold_file_name)
	gold_terms = set([term.strip() for term in io.open(gold_file_path, "r", encoding='utf-8')])
	for i in range(1,total_chapters+1):
		print("Assigning chapter %d" % i)
		gold_terms, gold_chapter_dict, changes = assign_golds_to_chapter(textbook_name, i, gold_terms, gold_chapter_dict, changes, find_plurals)
	return gold_chapter_dict, changes, gold_terms

def dump_terms_to_files(term_dict, textbook_name, num_chapters):
	for chapter_num in range(1,num_chapters+1):
		print("Dumping chapter %d" % chapter_num)
		file_prefix =  "chapter_" + str(chapter_num) + "_"
		gold_file_name = file_prefix + "gold.txt"
		gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, gold_file_name)
		with open(gold_file_path, "w") as f:
			for term in term_dict[chapter_num]:
				f.write(term + "\n")
		gold_file_name = file_prefix + "gold.pkl"
		gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, gold_file_name)
		term_set = set(term_dict[chapter_num])
		with open(gold_file_path, "wb") as f:
			pickle.dump(term_set, f)

def main():
	gold_chapter_dict, changes, gold_terms = build_chapter_dict("sadava_life", 59, {}, True)
	dump_terms_to_files(gold_chapter_dict, "sadava_life", 59)

if __name__ == '__main__':
	main()