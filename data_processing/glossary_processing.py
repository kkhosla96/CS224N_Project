import sys
import io
import os
import re
if ".." not in sys.path:
	sys.path.append("..")
from utils import term_to_regex


SENTENCES_FOLDER =  "../data/textbook_sentences"
GOLD_FOLDER = "../data/gold"

def in_chapter_text(term, chapter_text):
	return re.search(term_to_regex(term), chapter_text)

def find_missing_gold_chapter(textbook_name, chapter_num, find_plurals=False):
	file_prefix =  "chapter_" + str(chapter_num) + "_"
	sentence_file_name = file_prefix + "sentences.txt"
	sentence_file_path = os.path.join(SENTENCES_FOLDER, textbook_name, sentence_file_name)
	gold_file_name = file_prefix + "gold.txt"
	gold_file_path = os.path.join(GOLD_FOLDER, textbook_name, gold_file_name)

	chapter_text = io.open(sentence_file_path, "r", encoding='utf-8').read().replace('\n', ' ').lower()
	gold_terms = [term.strip() for term in io.open(gold_file_path, "r", encoding='utf-8')]
	
	missing = set()
	plural = set()

	for term in gold_terms:
		if not in_chapter_text(term, chapter_text):
			missing.add(term)
			if find_plurals:
				if in_chapter_text(term + 's', chapter_text):
					plural.add(term + 's')
				elif find_plurals and in_chapter_text(term + 'es', chapter_text):
					plural.add(term + 'es')
				elif find_plurals and in_chapter_text(term + 'e', chapter_text):
					plural.add(term + 'e')
	return missing, plural